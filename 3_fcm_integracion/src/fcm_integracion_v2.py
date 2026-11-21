# -*- coding: utf-8 -*-
"""Integración FCM + Q-learning REHECHA tras la reunión del 23/10/2026.

Cambios respecto a fcm_integracion.py (versión anterior):

  1) V2 (FCM en la decisión) es ahora un proceso de DECISIÓN tipo
     explorar/explotar que sigue la descripción de José, no un desempate
     con tolerancia. Para cada paso:
        - Ordena las acciones por softmax(Q) = probabilidad de elegir cada una.
        - Para la mejor, infiere s_{t+1} con el FCM de INFERIR.
        - Si el estado inferido es "aceptable" (el contagio no empeora de
          nivel), la ejecuta.
        - Si no, prueba la 2ª y la 3ª mejor.
        - Si ninguna de las tres convence y la probabilidad de la siguiente
          (4ª) es baja (< p_min), EXPLORA (acción aleatoria).

  2) V3 (FCM en la recompensa) usa ahora la matriz de PREDECIR (sólo la
     columna E1), no la completa. Centra la predicción respecto al escenario
     "sin acción" para quitar el sesgo de la sigmoide: el shaping premia /
     castiga sólo lo que la acción DIFERENCIA respecto a no hacer nada.
     Esto es más honesto y más cercano a "potencial-based shaping".

  3) Admite --fuente {expertos,llm,datos} y --pais {ES,GB,BR,generico}.
     Para datos, además --pesos-datos {corr,aprendidos} eligen el método de
     pesos (corr es la versión vieja, aprendidos es la nueva por regresión).

  4) Todos los hiperparámetros del moldeado y de la decisión exponen por CLI.

Salidas:
  - comparacion_fcm_v2.csv : 1 fila por versión x fuente x país.
  - fig_comparacion_v2.png : gráfica (si se pide).
"""
import argparse
import csv
import os
import random

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import qlearning_sencillo as ql

CONC = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "A1", "A2", "A3", "A4"]
IDX = {n: i for i, n in enumerate(CONC)}
AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(AQUI, "..", "datos")

_CMAX = ql.coste_max()
_LAM = ql.CONFIG["lambda_fcm"]



# ----------------------------------------------------------------------------
# Escenarios (presets de parámetros del simulador)
# ----------------------------------------------------------------------------
# Pandemia: epidemia agresiva (crecimiento alto, imp alta, menos eficacia de medidas)
# Normal:   situación estable (crecimiento bajo, poca imp, hospital calmado)
# Viral:   brotes periódicos (imp alta como choques recurrentes)
# Los demás parámetros quedan en los PARAMS del EntornoSimple.
ESCENARIOS = {
    "pandemia": {"r0": 0.30, "rmax": 0.30, "imp": 0.008, "A_h": 0.95, "B_h": 0.20},
    "normal":   {"r0": 0.08, "rmax": 0.30, "imp": 0.001, "A_h": 0.85, "B_h": 0.10},
    "viral":    {"r0": 0.20, "rmax": 0.30, "imp": 0.010, "A_h": 0.88, "B_h": 0.15},
    "generico": {},  # los PARAMS por defecto del entorno
}

# ----------------------------------------------------------------------------
# Carga de matrices (usa los CSV generados por construir_matrices_v2.py)
# ----------------------------------------------------------------------------
def cargar_matriz(ruta):
    """Lee un CSV 11x11 con cabecera."""
    M = np.zeros((11, 11))
    with open(ruta) as f:
        r = csv.reader(f)
        header = next(r)
        cols = {c: i for i, c in enumerate(header[1:])}
        for row in r:
            ei = row[0]
            for c, val in zip(header[1:], row[1:]):
                M[IDX[ei], IDX[c]] = float(val)
    return M


def ruta_matriz(fuente, tipo):
    return os.path.join(DATOS, f"fcm_W_{fuente}_{tipo}.csv")


# ----------------------------------------------------------------------------
# Predicciones del FCM
# ----------------------------------------------------------------------------
def _vector_estado(c, h, preo, a_prev, a):
    """Vector de activación de los 11 conceptos."""
    x = np.empty(11)
    x[0] = c            # E1 contagio
    x[1] = 0.5          # E2 tend. contagio
    x[2] = h            # E3 hospital
    x[3] = 0.5          # E4 tend. hospital
    x[4] = ql.coste(a_prev) / _CMAX   # E5 restricción vigente
    x[5] = 0.5          # E6 tend. socioeconómica
    x[6] = preo         # E7 preocupación
    x[7] = a[0] / 2.0; x[8] = a[1] / 2.0
    x[9] = a[2] / 2.0; x[10] = a[3] / 2.0
    return x


def fcm_inferir(W_inf, c, h, preo, a_prev, a):
    """Predicción COMPLETA del FCM de inferir. Devuelve (c_hat, h_hat, preo_hat)."""
    x = _vector_estado(c, h, preo, a_prev, a)
    c_n = 1.0 / (1.0 + np.exp(-_LAM * (c + float(W_inf[:, 0] @ x))))
    h_n = 1.0 / (1.0 + np.exp(-_LAM * (h + float(W_inf[:, 2] @ x))))
    preo_n = 1.0 / (1.0 + np.exp(-_LAM * (preo + float(W_inf[:, 6] @ x))))
    return c_n, h_n, preo_n


def fcm_predecir_contagio(W_pre, c, h, preo, a_prev, a):
    """Predicción SOLO del contagio con el FCM de predecir (columna E1).
    No aplica sigmoide: devuelve el peso agregado + c para hacerlo comparable."""
    x = _vector_estado(c, h, preo, a_prev, a)
    return c + float(W_pre[:, 0] @ x)


# ----------------------------------------------------------------------------
# V2: elegir acción con proceso de decisión explorar/explotar
# ----------------------------------------------------------------------------
def softmax(x, tau):
    """Softmax con temperatura tau."""
    z = (x - x.max()) / max(tau, 1e-6)
    e = np.exp(z)
    return e / e.sum()


def nivel_contagio(c, disc):
    """0 Bajo, 1 Medio, 2 Alto, según el discretizador."""
    return disc.nivel(c, "c")


def elegir_con_fcm_v2(ag, env, disc, rng, W_inf,
                      tau=0.5, k_top=3, p_min=0.3,
                      aceptar='mismo_o_mejor_nivel'):
    """V2 nuevo. Lógica explorar/explotar con FCM de INFERIR.

    - Ordena las N_ACCIONES por softmax(Q) = probabilidad.
    - Para cada una de las top-k, infiere s_{t+1} con el FCM.
      Acepta si el nivel de contagio inferido NO empeora respecto al actual
      (criterio 'mismo_o_mejor_nivel') o si cumple el criterio pedido.
    - Si ninguna de las top-k convence Y la probabilidad de la siguiente (k+1) es
      < p_min, EXPLORA (acción aleatoria).
    - Si la probabilidad es ≥ p_min, acepta la top-1 (hay suficiente confianza).

    Devuelve (accion, info) donde info es dict con la lógica tomada para debug.
    """
    s = env.estado_discreto()
    fila = ag.Q[ql.indice_de_estado(s)]
    probs = softmax(fila, tau)
    # Ordenar acciones por probabilidad descendente
    orden = np.argsort(-probs)

    c_actual = env.c
    nivel_actual = nivel_contagio(c_actual, disc)
    probadas = []

    for rank in range(k_top):
        i = int(orden[rank])
        a = ql.accion_de_indice(i)
        c_hat, h_hat, _ = fcm_inferir(W_inf, env.c, env.h, env.preo, env.a_prev, a)
        nivel_hat = nivel_contagio(c_hat, disc)
        probadas.append((i, probs[i], nivel_hat))
        if aceptar == 'mismo_o_mejor_nivel':
            ok = nivel_hat <= nivel_actual
        elif aceptar == 'estrictamente_mejor':
            ok = nivel_hat < nivel_actual
        elif aceptar == 'no_pasar_de_alto':
            ok = nivel_hat < 2
        else:
            ok = True
        if ok:
            return a, {'lograda_en_rank': rank, 'probadas': probadas,
                        'nivel_actual': nivel_actual}

    # Ninguna de las top-k pasó el filtro. Mirar la probabilidad de la k+1.
    if k_top < len(orden):
        p_siguiente = probs[int(orden[k_top])]
    else:
        p_siguiente = 0.0

    if p_siguiente < p_min:
        # Baja confianza en lo que viene: EXPLORAR
        idx_rand = rng.randrange(ql.N_ACCIONES)
        return ql.accion_de_indice(idx_rand), {
            'lograda_en_rank': -1, 'exploro': True, 'probadas': probadas,
            'p_siguiente': p_siguiente}
    else:
        # Confianza suficiente: ejecutar la top-1 aunque no convenza al FCM
        i = int(orden[0])
        return ql.accion_de_indice(i), {
            'lograda_en_rank': 0, 'forzado': True, 'probadas': probadas,
            'p_siguiente': p_siguiente}


# ----------------------------------------------------------------------------
# Entrenamiento (con V3 corregido)
# ----------------------------------------------------------------------------
def entrenar(env, disc, rng, episodios, horizonte,
             W_pre=None, lam_f=1.0, mu_f=0.6, nu_f=0.2, shaping_centrado=True):
    """Entrena Q-learning. Si se pasa W_pre, modula la recompensa con el FCM de
    PREDECIR (V3/V4). Con shaping_centrado=True, compara la predicción con la
    acción vs predicción SIN acción (baseline), no con c_t (quita el sesgo de
    la sigmoide y hace el shaping marginal más fiel)."""
    K, T = episodios, horizonte
    ag = ql.AgenteQ(rng)
    eps = ql.CONFIG["eps0"]
    accion_nula = (0, 0, 0, 0)  # 'no hacer nada'
    for k in range(K):
        s = env.reset(amplio=True)
        s_idx = ql.indice_de_estado(s)
        for t in range(T):
            a_idx = ag.elegir(s_idx, eps)
            a = ql.accion_de_indice(a_idx)
            c_t, h_t, preo_t, a_prev = env.c, env.h, env.preo, env.a_prev
            s1 = env.paso(a)
            r, _ = ql.recompensa_tramos(s, c_t, env.c, h_t, env.h, a, a_prev, disc)
            if W_pre is not None:
                c_hat = fcm_predecir_contagio(W_pre, c_t, h_t, preo_t, a_prev, a)
                if shaping_centrado:
                    c_base = fcm_predecir_contagio(W_pre, c_t, h_t, preo_t, a_prev, accion_nula)
                    # marginal: cuánto baja/sube la predicción POR la acción
                    r += (mu_f * max(0.0, c_base - c_hat)     # premia si baja respecto a base
                          - lam_f * max(0.0, c_hat - c_base)) # castiga si sube
                else:
                    r += (-lam_f * max(0.0, c_hat - c_t)
                          + mu_f * max(0.0, c_t - c_hat)
                          - nu_f * c_hat)
            s1_idx = ql.indice_de_estado(s1)
            ag.actualizar(s_idx, a_idx, r, s1_idx)
            s, s_idx = s1, s1_idx
        eps = max(ql.CONFIG["eps_min"], eps * ql.CONFIG["decay"])
    ag.consolidar()
    return ag


# ----------------------------------------------------------------------------
# Evaluación
# ----------------------------------------------------------------------------
def evaluar(env, disc, ag, rng, W_inf=None, fcm_dec=False, episodios=30,
            horizonte=None, **v2_kwargs):
    T = horizonte or ql.CONFIG["horizonte"]
    tot = {"retorno": 0.0, "c": 0.0, "h": 0.0, "coste": 0.0, "bajo": 0.0}
    rank_hist = {-1: 0, 0: 0, 1: 0, 2: 0}  # para V2: contador de ranks escogidos
    for _ in range(episodios):
        s = env.reset()
        for t in range(T):
            if fcm_dec:
                a, info = elegir_con_fcm_v2(ag, env, disc, rng, W_inf, **v2_kwargs)
                rk = info['lograda_en_rank']
                rank_hist[rk] = rank_hist.get(rk, 0) + 1
            else:
                a = ag.accion_greedy(s)
            c_t, h_t, a_prev = env.c, env.h, env.a_prev
            s1 = env.paso(a)
            r, _ = ql.recompensa_tramos(s, c_t, env.c, h_t, env.h, a, a_prev, disc)
            tot["retorno"] += r
            tot["c"] += env.c
            tot["h"] += env.h
            tot["coste"] += ql.coste(a)
            if disc.nivel(env.c, "c") == 0:
                tot["bajo"] += 1
            s = s1
    n = episodios * T
    res = {"contagio": tot["c"] / n, "hospital": tot["h"] / n,
           "coste": tot["coste"] / n, "pct_bajo": 100.0 * tot["bajo"] / n,
           "retorno": tot["retorno"] / episodios}
    if fcm_dec:
        total_r = sum(rank_hist.values())
        res["v2_rank_dist"] = {k: round(100 * v / max(total_r, 1), 1)
                                for k, v in rank_hist.items()}
    return res


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", choices=["expertos", "llm", "datos"], default="expertos")
    ap.add_argument("--pais", choices=["ES", "GB", "BR", "generico"], default="generico")
    ap.add_argument("--escenario", choices=["pandemia","normal","viral","generico"], default="generico")
    ap.add_argument("--episodios", type=int, default=30000)
    ap.add_argument("--horizonte", type=int, default=None)
    ap.add_argument("--eval", type=int, default=30)
    ap.add_argument("--semilla", type=int, default=0)
    # V2
    ap.add_argument("--tau", type=float, default=0.5, help="Temperatura softmax de V2")
    ap.add_argument("--k-top", type=int, default=3, help="Nº de acciones top a evaluar en V2")
    ap.add_argument("--p-min", type=float, default=0.3, help="Prob. mínima para no explorar")
    ap.add_argument("--aceptar", choices=["mismo_o_mejor_nivel", "estrictamente_mejor",
                                           "no_pasar_de_alto"],
                    default="mismo_o_mejor_nivel")
    # V3
    ap.add_argument("--lam-f", type=float, default=1.0)
    ap.add_argument("--mu-f", type=float, default=0.6)
    ap.add_argument("--nu-f", type=float, default=0.2)
    ap.add_argument("--sin-centrar", action="store_true",
                    help="V3: shaping viejo (no centrado) para comparar")
    args = ap.parse_args()

    T = args.horizonte or ql.CONFIG["horizonte"]
    res_dir = os.path.join(AQUI, "..", "resultados")
    os.makedirs(res_dir, exist_ok=True)

    rng = random.Random(args.semilla)
    np.random.seed(args.semilla)

    def hacer_env(disc):
        params = {}
        if args.pais != "generico" and hasattr(ql, "PARAMS_PAIS"):
            params.update(ql.PARAMS_PAIS[args.pais])
        if args.escenario != "generico":
            params.update(ESCENARIOS[args.escenario])  # escenario SOBREESCRIBE pais
        return ql.EntornoSimple(disc, rng, params=params if params else None)

    print(f"Calibrando discretización (país={args.pais})...")
    disc = ql.calibrar_discretizador(hacer_env, rng)

    print(f"Cargando FCM (fuente={args.fuente})...")
    W_inf = cargar_matriz(ruta_matriz(args.fuente, "inferir"))
    W_pre = cargar_matriz(ruta_matriz(args.fuente, "predecir"))
    print(f"  INFERIR: {int((np.abs(W_inf)>1e-6).sum())} celdas no nulas")
    print(f"  PREDECIR: {int((np.abs(W_pre)>1e-6).sum())} celdas no nulas")

    # Entrenamientos
    print(f"\nEntrenando V1 (puro)... ({args.episodios} episodios)")
    env = hacer_env(disc)
    ag_puro = entrenar(env, disc, rng, args.episodios, T, W_pre=None)

    print(f"Entrenando V3 (shaping con FCM predecir, "
          f"{'centrado' if not args.sin_centrar else 'viejo'})...")
    env = hacer_env(disc)
    ag_mod = entrenar(env, disc, rng, args.episodios, T, W_pre=W_pre,
                      lam_f=args.lam_f, mu_f=args.mu_f, nu_f=args.nu_f,
                      shaping_centrado=not args.sin_centrar)

    # Evaluaciones
    print("\nEvaluando 4 versiones...")
    env = hacer_env(disc)
    v2_kw = dict(tau=args.tau, k_top=args.k_top, p_min=args.p_min, aceptar=args.aceptar)
    versiones = [
        ("V1", "Q-learning puro",             ag_puro, False),
        ("V2", "+ FCM decisión (exp/expl)",   ag_puro, True),
        ("V3", "+ FCM predecir en reward",    ag_mod,  False),
        ("V4", "Híbrido V2+V3",                ag_mod,  True),
    ]
    filas = []
    for ver, nombre, ag, dec in versiones:
        m = evaluar(env, disc, ag, rng, W_inf=W_inf, fcm_dec=dec,
                    episodios=args.eval, horizonte=T, **v2_kw)
        filas.append((ver, nombre, m))
        extra = ""
        if "v2_rank_dist" in m:
            extra = f"  rank_dist={m['v2_rank_dist']}"
        print(f"  {ver} {nombre:28s}  c={m['contagio']:.3f}  h={m['hospital']:.3f}"
              f"  coste={m['coste']:.2f}  %bajo={m['pct_bajo']:.1f}"
              f"  R={m['retorno']:+.1f}{extra}")

    # CSV
    sufijo = f"{args.fuente}_{args.pais}_{args.escenario}"
    ruta_csv = os.path.join(res_dir, f"comparacion_v2_{sufijo}.csv")
    with open(ruta_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "nombre", "contagio", "hospital", "coste",
                    "pct_bajo", "retorno"])
        for ver, nom, m in filas:
            w.writerow([ver, nom, f"{m['contagio']:.4f}", f"{m['hospital']:.4f}",
                        f"{m['coste']:.3f}", f"{m['pct_bajo']:.1f}",
                        f"{m['retorno']:.2f}"])
    print(f"\nEscrito: {ruta_csv}")


if __name__ == "__main__":
    main()
