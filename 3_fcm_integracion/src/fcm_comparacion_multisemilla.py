# -*- coding: utf-8 -*-
"""Comparación ROBUSTA de las 4 versiones (V1 puro, V2 FCM decisión, V3 FCM recompensa,
V4 híbrido) sobre VARIAS semillas independientes, para que el resultado no dependa del
azar de una sola ejecución. Pensado para lanzarse en el servidor con nohup.

Para cada semilla se entrena V1 y V3 y se evalúan las cuatro; luego se promedian los
resultados de todas las semillas y se da la media con su intervalo de confianza al 95%.

Salidas (en ../resultados):
  - comparacion_multisemilla_detalle.csv : una fila por (semilla, versión).
  - comparacion_multisemilla_resumen.csv : media e IC 95% por versión y métrica.

Uso:
  python3 fcm_comparacion_multisemilla.py --semillas 30 --episodios 50000 --eval 30 --procesos 4
"""
import argparse
import csv
import os
import random

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool

import numpy as np

import qlearning_sencillo as ql
import fcm_integracion as fi

VERSIONES = ["V1", "V2", "V3", "V4"]
METRICAS = ["contagio", "hospital", "coste", "pct_bajo", "retorno"]
_ETQ = {"V1": "V1 Q-learning puro", "V2": "V2 + FCM decision",
        "V3": "V3 + FCM recompensa", "V4": "V4 hibrido"}


def cargar_W(fuente="expertos"):
    """Carga la matriz de pesos 11x11 de la fuente indicada (expertos/llm/datos)
    desde su CSV etiquetado (cabecera + primera columna con los nombres de concepto),
    generado por construir_matrices_fuentes.py. Así funciona igual en el servidor,
    que no tiene los Excel."""
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "datos", f"fcm_W_{fuente}.csv")
    filas = []
    with open(ruta, newline="") as f:
        r = csv.reader(f)
        next(r)  # cabecera
        for row in r:
            filas.append([float(x) for x in row[1:]])  # se salta la etiqueta de fila
    return np.array(filas, dtype=float)


def _correr_semilla(args):
    seed, W, episodios, T, eval_n, tol = args
    rng = random.Random(seed)
    np.random.seed(seed)

    def hacer_env(disc):
        return ql.EntornoSimple(disc, rng)

    disc = ql.calibrar_discretizador(hacer_env, rng)
    env = hacer_env(disc)
    ag_puro = fi.entrenar(env, disc, rng, episodios, T, W=None)
    env = hacer_env(disc)
    ag_mod = fi.entrenar(env, disc, rng, episodios, T, W=W)
    out = {}
    for nombre, ag, dec in [("V1", ag_puro, False), ("V2", ag_puro, True),
                            ("V3", ag_mod, False), ("V4", ag_mod, True)]:
        # Números aleatorios comunes: las cuatro versiones se evalúan sobre
        # EXACTAMENTE los mismos episodios (mismo rng de evaluación, independiente
        # del entrenamiento), para una comparación pareada. Así, además, V1 sale
        # idéntico entre fuentes, porque no usa el FCM.
        rng_ev = random.Random(seed + 10_000_019)
        env_ev = ql.EntornoSimple(disc, rng_ev)
        out[nombre] = fi.evaluar(env_ev, disc, ag, rng_ev, W=W, fcm_dec=dec,
                                 episodios=eval_n, horizonte=T, tol=tol)
    return seed, out


def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--semillas", type=int, default=30)
    ap.add_argument("--episodios", type=int, default=50000)
    ap.add_argument("--horizonte", type=int, default=None)
    ap.add_argument("--eval", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--semilla-base", type=int, default=3000)
    ap.add_argument("--fuente", choices=["expertos", "llm", "datos"],
                    default="expertos",
                    help="qué matriz del FCM usar (por defecto, expertos)")
    ap.add_argument("--normalizar", action="store_true",
                    help="escala la matriz para que su peso máximo absoluto sea 1, "
                         "de modo que las tres fuentes se comparen a igualdad de escala "
                         "(estructura, no rango numérico de cada elicitación)")
    args = ap.parse_args()

    T = args.horizonte or ql.CONFIG["horizonte"]
    res_dir = os.path.join(aqui, "..", "resultados")
    os.makedirs(res_dir, exist_ok=True)
    W = cargar_W(args.fuente)
    if args.normalizar:
        m = float(np.max(np.abs(W)))
        if m > 0:
            W = W / m
    suf = f"_{args.fuente}" + ("_norm" if args.normalizar else "")
    print(f"FCM ({args.fuente}{' normalizado' if args.normalizar else ''}) cargado. "
          f"Lanzando {args.semillas} semillas x 4 versiones "
          f"(episodios={args.episodios}, eval={args.eval}, {args.procesos} procesos)...",
          flush=True)

    semillas = [args.semilla_base + i for i in range(args.semillas)]
    tareas = [(s, W, args.episodios, T, args.eval, args.tol) for s in semillas]

    resultados = []
    with Pool(processes=args.procesos) as pool:
        for i, res in enumerate(pool.imap_unordered(_correr_semilla, tareas), 1):
            resultados.append(res)
            print(f"  semilla {i}/{args.semillas} terminada (semilla={res[0]})",
                  flush=True)
    resultados.sort(key=lambda r: r[0])

    # --- detalle ---
    ruta_det = os.path.join(res_dir, f"comparacion_multisemilla_detalle{suf}.csv")
    with open(ruta_det, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["semilla", "version"] + METRICAS)
        for seed, out in resultados:
            for v in VERSIONES:
                w.writerow([seed, v] + [f"{out[v][m]:.4f}" for m in METRICAS])
    print(f"Escrito: {ruta_det}")

    # --- resumen (media e IC 95%) ---
    # t de Student (no la normal 1.96): con n semillas pequeñas es lo correcto.
    _T95 = {5: 2.776, 10: 2.262, 15: 2.145, 20: 2.093, 25: 2.064, 30: 2.045, 50: 2.010}

    def _tcrit(n):
        if n <= 1:
            return 0.0
        df = n - 1
        return _T95.get(n, _T95[min(_T95, key=lambda k: abs(k - n))])

    def media_ic(valores):
        a = np.array(valores, dtype=float)
        media = a.mean()
        n = len(a)
        if n > 1:
            sem = a.std(ddof=1) / np.sqrt(n)
            tcrit = _tcrit(n)
        else:
            sem, tcrit = 0.0, 0.0
        return media, media - tcrit * sem, media + tcrit * sem

    ruta_res = os.path.join(res_dir, f"comparacion_multisemilla_resumen{suf}.csv")
    print("\n=== RESUMEN (media sobre {} semillas) ===".format(args.semillas))
    with open(ruta_res, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "metrica", "media", "ic95_lo", "ic95_hi"])
        for v in VERSIONES:
            linea = [_ETQ[v]]
            for m in METRICAS:
                vals = [out[v][m] for _, out in resultados]
                mu, lo, hi = media_ic(vals)
                w.writerow([v, m, f"{mu:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
                linea.append(f"{m}={mu:.3f}")
            print("  " + "  ".join(linea))
    print(f"\nEscrito: {ruta_res}")


if __name__ == "__main__":
    main()
