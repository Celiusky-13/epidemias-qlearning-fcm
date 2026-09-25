# -*- coding: utf-8 -*-
"""Integración del FCM con el Q-learning: las 4 versiones del plan de José.

Compara, sobre el mismo simulador y con las mismas métricas:
  V1  Q-learning puro (baseline, sin FCM).
  V2  Q-learning + FCM de próximo estado (el FCM decide cuando la política duda).
  V3  Q-learning + FCM predictivo del RoS en la recompensa (modula el premio/castigo
      durante el entrenamiento).
  V4  Híbrido: los dos FCM juntos (V3 al entrenar + V2 al decidir).

El FCM que se usa es la matriz PROMEDIO del panel de expertos (la fuente de
referencia del Capítulo 5). El mecanismo sigue la formalización del Capítulo 6:
  - Predictivo (V3): estima el contagio del paso siguiente y ajusta la recompensa
    (premia si mejora, castiga si empeora), Ec. de recompensa del FCM.
  - Inferencia (V2): cuando entre las mejores acciones hay casi empate en la tabla Q,
    usa el FCM para desempatar quedándose con la que lleva al estado más deseable.

Salidas (en ../resultados y ../figuras):
  - comparacion_fcm.csv : una fila por versión con sus métricas.
  - fig_comparacion_fcm.png : gráfica comparando las 4 versiones.

Uso:
  python3 fcm_integracion.py                       # valores por defecto
  python3 fcm_integracion.py --episodios 50000 --eval 50
"""
import argparse
import csv
import os
import random

# Un solo hilo por proceso de BLAS (operaciones pequeñas; evita contención).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import qlearning_sencillo as ql

CONC = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "A1", "A2", "A3", "A4"]
IDX = {n: i for i, n in enumerate(CONC)}

RUTA_EXPERTOS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "2_fcm_construccion", "datos", "FCM EXPERTOS.xlsx")
HOJA_EXPERTOS = "MATRIZ DE PESOS PROXIMOS ESTADO"


# ---------------------------------------------------------------------
def cargar_fcm_promedio(ruta, hoja):
    """Carga la matriz 11x11 promediando TODOS los bloques (un experto por bloque)
    apilados en la hoja. Cada bloque empieza en una fila 'Desde \\ Hacia'."""
    import openpyxl
    wb = openpyxl.load_workbook(ruta, data_only=True)
    ws = wb[hoja]
    filas = list(ws.iter_rows(values_only=True))
    cabeceras = [r for r, f in enumerate(filas)
                 if f and isinstance(f[0], str) and f[0].strip().startswith("Desde")]
    matrices = []
    for cab in cabeceras:
        col2c = {}
        for c, x in enumerate(filas[cab]):
            e = str(x).strip()[:2] if x else ""
            if e in CONC:
                col2c[c] = e
        M = np.zeros((11, 11))
        for rr in range(cab + 1, cab + 1 + 11):
            if rr >= len(filas) or not filas[rr] or not filas[rr][0]:
                continue
            ei = str(filas[rr][0]).strip()[:2]
            if ei not in CONC:
                continue
            for c, ej in col2c.items():
                v = filas[rr][c] if c < len(filas[rr]) else None
                if isinstance(v, (int, float)):
                    M[IDX[ei], IDX[ej]] = float(v)
        matrices.append(M)
    W = np.mean(np.stack(matrices), axis=0)
    np.fill_diagonal(W, 0.0)               # la diagonal (inercia) no se usa como arista
    for a in ("A1", "A2", "A3", "A4"):
        W[:, IDX[a]] = 0.0                  # las acciones solo emiten
    return W, len(matrices)


# ---------------------------------------------------------------------
_CMAX = ql.coste_max()
_LAM = ql.CONFIG["lambda_fcm"]


def fcm_predice(W, disc, rng, c, h, preo, a_prev, a):
    """Un paso del FCM (sin ruido) desde el estado continuo dado aplicando la acción a.
    Devuelve el estado siguiente estimado (contagio, hospital, preocupación).

    Versión rápida en numpy, equivalente a EntornoFCM.paso con ruido 0: al predecir un
    solo paso las tendencias valen 0 (nodos E2, E4, E6 = 0.5) y el retardo hospitalario
    toma el contagio actual. Se evita crear un objeto por llamada (se llama millones de
    veces al entrenar)."""
    x = np.empty(11)
    x[0] = c                                   # E1 contagio
    x[1] = 0.5                                  # E2 tend. contagio (0 -> 0.5)
    x[2] = h                                    # E3 hospital
    x[3] = 0.5                                  # E4 tend. hospital
    x[4] = ql.coste(a_prev) / _CMAX            # E5 restricción vigente
    x[5] = 0.5                                  # E6 tend. socioeconómica
    x[6] = preo                                 # E7 preocupación
    x[7] = a[0] / 2.0                           # A1 trabajo
    x[8] = a[1] / 2.0                           # A2 colegios
    x[9] = a[2] / 2.0                           # A3 transporte
    x[10] = a[3] / 2.0                          # A4 confinamiento
    c_n = 1.0 / (1.0 + np.exp(-_LAM * (c + float(W[:, 0] @ x))))
    h_n = 1.0 / (1.0 + np.exp(-_LAM * (h + float(W[:, 2] @ x))))
    preo_n = 1.0 / (1.0 + np.exp(-_LAM * (preo + float(W[:, 6] @ x))))
    return c_n, h_n, preo_n


# ---------------------------------------------------------------------
def entrenar(env, disc, rng, episodios, horizonte,
             W=None, lam_f=1.0, mu_f=0.6, nu_f=0.2):
    """Entrena el Q-learning. Si se pasa W, aplica la MODULACIÓN de la recompensa con
    el FCM predictivo (V3/V4): premia si el contagio estimado baja, castiga si sube."""
    K, T = episodios, horizonte
    ag = ql.AgenteQ(rng)
    eps = ql.CONFIG["eps0"]
    for k in range(K):
        s = env.reset(amplio=True)
        s_idx = ql.indice_de_estado(s)
        for t in range(T):
            a_idx = ag.elegir(s_idx, eps)
            a = ql.accion_de_indice(a_idx)
            c_t, h_t, preo_t, a_prev = env.c, env.h, env.preo, env.a_prev
            s1 = env.paso(a)
            r, _ = ql.recompensa_tramos(s, c_t, env.c, h_t, env.h, a, a_prev, disc)
            if W is not None:
                c_hat, _, _ = fcm_predice(W, disc, rng, c_t, h_t, preo_t, a_prev, a)
                r += (-lam_f * max(0.0, c_hat - c_t)
                      + mu_f * max(0.0, c_t - c_hat)
                      - nu_f * c_hat)
            s1_idx = ql.indice_de_estado(s1)
            ag.actualizar(s_idx, a_idx, r, s1_idx)
            s, s_idx = s1, s1_idx
        eps = max(ql.CONFIG["eps_min"], eps * ql.CONFIG["decay"])
    ag.consolidar()
    return ag


# ---------------------------------------------------------------------
def elegir_con_fcm(ag, env, disc, rng, W, tol=1.0, omega_h=0.3, omega_k=0.1):
    """Decisión con desempate por FCM (V2/V4). Entre las acciones cuya Q está a menos
    de 'tol' de la mejor, elige la que el FCM estima que deja el estado más deseable
    (menos contagio y hospital, con un empujoncito hacia menos coste)."""
    s = env.estado_discreto()
    fila = ag.Q[ql.indice_de_estado(s)]
    maxq = fila.max()
    cand = [i for i in range(ql.N_ACCIONES) if fila[i] >= maxq - tol]
    if len(cand) == 1:
        return ql.accion_de_indice(cand[0])
    cmax = ql.coste_max()
    mejor, mejor_val = cand[0], float("inf")
    for i in cand:
        a = ql.accion_de_indice(i)
        c_hat, h_hat, _ = fcm_predice(W, disc, rng, env.c, env.h, env.preo, env.a_prev, a)
        val = c_hat + omega_h * h_hat + omega_k * ql.coste(a) / cmax
        if val < mejor_val:
            mejor_val, mejor = val, i
    return ql.accion_de_indice(mejor)


# ---------------------------------------------------------------------
def evaluar(env, disc, ag, rng, W=None, fcm_dec=False, episodios=30, horizonte=None,
            tol=1.0):
    T = horizonte or ql.CONFIG["horizonte"]
    tot = {"retorno": 0.0, "c": 0.0, "h": 0.0, "coste": 0.0, "bajo": 0.0}
    for _ in range(episodios):
        s = env.reset()
        for t in range(T):
            if fcm_dec:
                a = elegir_con_fcm(ag, env, disc, rng, W, tol)
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
    return {"contagio": tot["c"] / n, "hospital": tot["h"] / n,
            "coste": tot["coste"] / n, "pct_bajo": 100.0 * tot["bajo"] / n,
            "retorno": tot["retorno"] / episodios}


# ---------------------------------------------------------------------
def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodios", type=int, default=30000)
    ap.add_argument("--horizonte", type=int, default=None)
    ap.add_argument("--eval", type=int, default=30, help="episodios de evaluación")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1.0, help="tolerancia de empate en Q")
    args = ap.parse_args()

    T = args.horizonte or ql.CONFIG["horizonte"]
    res_dir = os.path.join(aqui, "..", "resultados")
    fig_dir = os.path.join(aqui, "..", "figuras")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    rng = random.Random(args.semilla)
    np.random.seed(args.semilla)

    def hacer_env(disc):
        return ql.EntornoSimple(disc, rng)

    print("Calibrando discretización...")
    disc = ql.calibrar_discretizador(hacer_env, rng)

    print("Cargando FCM (promedio de expertos)...")
    W, nblq = cargar_fcm_promedio(RUTA_EXPERTOS, HOJA_EXPERTOS)
    print(f"  matriz de {nblq} expertos; A4->E1={W[IDX['A4'],IDX['E1']]:.2f}, "
          f"E1->E3={W[IDX['E1'],IDX['E3']]:.2f}")

    # --- Entrenamientos ---
    print(f"Entrenando V1 (Q-learning puro), {args.episodios} episodios...")
    env = hacer_env(disc)
    ag_puro = entrenar(env, disc, rng, args.episodios, T, W=None)

    print(f"Entrenando V3 (recompensa modulada por FCM), {args.episodios} episodios...")
    env = hacer_env(disc)
    ag_mod = entrenar(env, disc, rng, args.episodios, T, W=W)

    # --- Evaluaciones (mismo entorno de evaluación) ---
    print("Evaluando las 4 versiones...")
    env = hacer_env(disc)
    versiones = [
        ("V1  Q-learning puro",            ag_puro, False),
        ("V2  + FCM próximo estado",       ag_puro, True),
        ("V3  + FCM RoS en recompensa",    ag_mod,  False),
        ("V4  Híbrido (los dos)",          ag_mod,  True),
    ]
    filas = []
    for nombre, ag, dec in versiones:
        m = evaluar(env, disc, ag, rng, W=W, fcm_dec=dec,
                    episodios=args.eval, horizonte=T, tol=args.tol)
        filas.append((nombre, m))
        print(f"  {nombre:32s} contagio={m['contagio']:.3f}  hospital={m['hospital']:.3f}"
              f"  coste={m['coste']:.2f}  %días bajo={m['pct_bajo']:.1f}  "
              f"retorno={m['retorno']:.1f}")

    # --- Guardar CSV ---
    ruta_csv = os.path.join(res_dir, "comparacion_fcm.csv")
    with open(ruta_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "contagio", "hospital", "coste", "pct_dias_bajo", "retorno"])
        for nombre, m in filas:
            w.writerow([nombre, f"{m['contagio']:.4f}", f"{m['hospital']:.4f}",
                        f"{m['coste']:.3f}", f"{m['pct_bajo']:.1f}", f"{m['retorno']:.2f}"])
    print(f"Escrito: {ruta_csv}")

    # --- Figura ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        etiquetas = [n.split("  ")[0] for n, _ in filas]
        AZUL = "#0054B4"
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for ax, clave, titulo in zip(
                axes,
                ["contagio", "coste", "retorno"],
                ["Contagio medio (menos = mejor)", "Coste medio", "Retorno (más = mejor)"]):
            vals = [m[clave] for _, m in filas]
            ax.bar(etiquetas, vals, color=AZUL)
            ax.set_title(titulo, fontsize=10)
            ax.grid(True, axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        ruta_fig = os.path.join(fig_dir, "fig_comparacion_fcm.png")
        fig.savefig(ruta_fig, dpi=150)
        print(f"Escrito: {ruta_fig}")
    except Exception as e:
        print("No se pudo generar la figura:", e)


if __name__ == "__main__":
    main()
