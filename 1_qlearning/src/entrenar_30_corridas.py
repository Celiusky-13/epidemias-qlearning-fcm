# -*- coding: utf-8 -*-
"""Corridas independientes para las dos figuras de calidad del aprendizaje
(Cap. 4, Sec. "Calidad del proceso de aprendizaje" y "Velocidad de
convergencia") pedidas en la reunion del 23/07: no una sola ejecucion con
media movil, sino el promedio (y el IC 95%) de VARIAS corridas independientes
con la tabla Q inicializada al azar, tal como se discutio.

Para cada corrida (semilla distinta):
  - Se recalibra la discretizacion (rodaje aleatorio propio de la corrida).
  - Se entrena un agente cuya Q se inicializa al azar (Uniform(-1,1)), NO a
    cero: el entrenamiento "oficial" (el que produce Q.npy y la politica del
    TFG) sigue usando Q_0=0 como manda la teoria de Watkins & Dayan; esta Q
    aleatoria es SOLO para el diagnostico de robustez de la convergencia que
    pidio el director ("arranco aleatoriamente mi matriz").
  - Se registra el retorno y el error TD por episodio (para promediarlos).
  - Se registra, para dos estados representativos ya usados en la Tabla 4.2
    del TFG (propagacion rapida -> (2,2,2,2); calma sostenida -> (0,0,0,0)),
    en que episodio la accion greedy del agente empieza a coincidir con la
    accion ideal y YA NO CAMBIA el resto del entrenamiento ("aprender de
    forma sostenida", como se pidio).

Salidas (en --salidas, por defecto ./salidas):
  - curva_aprendizaje_30corridas.csv : episodio, retorno_medio,
    retorno_ic95_lo/hi, td_medio, td_ic95_lo/hi (agregado de las N corridas).
  - convergencia_estados.csv : corrida, episodio_convergencia_prop_rapida,
    episodio_convergencia_calma (una fila por corrida; vacio si esa corrida
    no llega a converger de forma sostenida).

Uso:
  python3 entrenar_30_corridas.py                     # 30 corridas, K=50000
  python3 entrenar_30_corridas.py --corridas 30 --episodios 50000 --procesos 8
"""
import argparse
import csv
import os
import random
import sys

# IMPORTANTE: fijar ANTES de "import numpy". Sin esto, cada uno de los N
# procesos crea su propio pool de hilos BLAS (OpenBLAS/Accelerate); con 8
# procesos x varios hilos cada uno se sobre-suscriben los nucleos (mas hilos
# que cores) y el sistema pasa a hacer thread-thrashing: procesos que parecen
# "colgados" al 0% CPU (bloqueados en un lock del BLAS) mientras otros van al
# 95-99%, y el tiempo total se dispara. Nuestras operaciones son vectores de
# 81 y tablas de 2187x81: no se benefician nada de BLAS multihilo.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool

import numpy as np

import qlearning_sencillo as ql

# Estados representativos = los mismos de la Tabla 4.2 (tab:politica) del TFG.
# s = (c, dc, h, dh, L, dL, preo), niveles {0,1,2}={B,M,A}, tendencias {-1,0,+1}.
ESTADOS_OBJETIVO = {
    "prop_rapida": {
        # Contagio Medio(sube), hosp. Bajo(sube), restr. Bajo(baja), cumpl. Bajo
        "estado": (1, +1, 0, +1, 0, -1, 0),
        "accion": (2, 2, 2, 2),
    },
    "calma": {
        # Contagio Bajo(baja), hosp. Bajo(baja), restr. Alto(sube), cumpl. Bajo
        "estado": (0, -1, 0, -1, 2, +1, 0),
        "accion": (0, 0, 0, 0),
    },
}


def _correr_una(args):
    semilla, K, T = args
    rng = random.Random(semilla)
    np.random.seed(semilla)

    def hacer_env(disc):
        return ql.EntornoSimple(disc, rng)

    disc = ql.calibrar_discretizador(hacer_env, rng)
    env = hacer_env(disc)

    agente = ql.AgenteQ(rng)
    agente.Q = np.random.uniform(-1.0, 1.0, size=agente.Q.shape)  # Q al azar

    coincide = {nombre: np.zeros(K, dtype=bool) for nombre in ESTADOS_OBJETIVO}

    def callback(k, ag):
        for nombre, obj in ESTADOS_OBJETIVO.items():
            coincide[nombre][k] = (ag.accion_greedy(obj["estado"]) == obj["accion"])

    _, historial = ql.entrenar(env, disc, rng, episodios=K, horizonte=T,
                               verbose=False, agente=agente, callback=callback)

    retorno = np.array([h[1] for h in historial])
    td = np.array([h[5] for h in historial])

    conv = {}
    for nombre, serie in coincide.items():
        # episodio de convergencia SOSTENIDA: el primero a partir del cual la
        # accion greedy YA NO cambia en el resto del entrenamiento.
        idx_falsos = np.flatnonzero(~serie)
        if len(idx_falsos) == 0:
            conv[nombre] = 0            # coincide desde el primer episodio
        elif idx_falsos[-1] == K - 1:
            conv[nombre] = None         # no converge de forma sostenida
        else:
            conv[nombre] = int(idx_falsos[-1] + 1)

    return semilla, retorno, td, conv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corridas", type=int, default=30)
    ap.add_argument("--episodios", type=int, default=None)
    ap.add_argument("--horizonte", type=int, default=None)
    ap.add_argument("--semilla-base", type=int, default=1000,
                    help="distinta de la semilla=0 del entrenamiento oficial")
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--salidas", type=str, default=None)
    args = ap.parse_args()

    K = args.episodios or ql.CONFIG["episodios"]
    T = args.horizonte or ql.CONFIG["horizonte"]
    N = args.corridas
    carpeta = args.salidas or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "salidas")
    os.makedirs(carpeta, exist_ok=True)

    semillas = [args.semilla_base + i for i in range(N)]
    print(f"Lanzando {N} corridas independientes (K={K}, T={T}, "
          f"{args.procesos} procesos)...")

    tareas = [(s, K, T) for s in semillas]
    resultados = []
    with Pool(processes=args.procesos) as pool:
        for i, res in enumerate(pool.imap_unordered(_correr_una, tareas), 1):
            resultados.append(res)
            print(f"  corrida {i}/{N} terminada (semilla={res[0]})")

    resultados.sort(key=lambda r: r[0])   # orden estable por semilla
    retornos = np.stack([r[1] for r in resultados])   # (N, K)
    tds = np.stack([r[2] for r in resultados])        # (N, K)

    def media_ic95(mat):
        media = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        return media, media - 1.96 * sem, media + 1.96 * sem

    ret_media, ret_lo, ret_hi = media_ic95(retornos)
    td_media, td_lo, td_hi = media_ic95(tds)

    ruta_curva = os.path.join(carpeta, "curva_aprendizaje_30corridas.csv")
    with open(ruta_curva, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episodio", "retorno_medio", "retorno_ic95_lo",
                    "retorno_ic95_hi", "td_medio", "td_ic95_lo", "td_ic95_hi"])
        for k in range(K):
            w.writerow([k, f"{ret_media[k]:.4f}", f"{ret_lo[k]:.4f}",
                        f"{ret_hi[k]:.4f}", f"{td_media[k]:.6f}",
                        f"{td_lo[k]:.6f}", f"{td_hi[k]:.6f}"])
    print(f"Escrito: {ruta_curva}  ({N} corridas, K={K})")

    ruta_conv = os.path.join(carpeta, "convergencia_estados.csv")
    with open(ruta_conv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corrida", "semilla", "episodio_convergencia_prop_rapida",
                    "episodio_convergencia_calma"])
        for semilla, _, _, conv in resultados:
            w.writerow([semilla, semilla,
                        conv["prop_rapida"] if conv["prop_rapida"] is not None else "",
                        conv["calma"] if conv["calma"] is not None else ""])
    print(f"Escrito: {ruta_conv}")

    for nombre in ESTADOS_OBJETIVO:
        eps = [conv[nombre] for _, _, _, conv in resultados if conv[nombre] is not None]
        n_no_conv = N - len(eps)
        if eps:
            print(f"  {nombre}: mediana={int(np.median(eps))} episodios "
                  f"(min={min(eps)}, max={max(eps)}); no convergieron: {n_no_conv}/{N}")
        else:
            print(f"  {nombre}: NINGUNA corrida convergio de forma sostenida")

    print(f"\nRetorno final (media de las {N} corridas, ultimo episodio): "
          f"{ret_media[-1]:.1f}")
    print(f"Error TD final (media de las {N} corridas, ultimo episodio): "
          f"{td_media[-1]:.4f}")


if __name__ == "__main__":
    main()
