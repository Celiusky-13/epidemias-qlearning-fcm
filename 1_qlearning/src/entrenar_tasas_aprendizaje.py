# -*- coding: utf-8 -*-
"""Compara distintas tasas de aprendizaje iniciales (alpha_0) del Q-learning,
pedido en la reunion del 23/07 ("podriamos probar con diferentes tasas de
aprendizaje"). Pendiente hasta ahora.

Para cada tasa alpha_0 candidata se corren R corridas independientes (Q_0=0,
la inicializacion OFICIAL, NO la Q aleatoria de entrenar_30_corridas.py: aqui
solo queremos aislar el efecto de alpha_0, sin mezclarlo con otro cambio) y se
promedia el retorno y el residual de Bellman por episodio, igual que en
entrenar_30_corridas.py. kappa (decaimiento) y alpha_min se dejan como estan
en CONFIG: solo cambia la tasa INICIAL.

Salidas (en --salidas):
  - tasas_aprendizaje.csv : tasa, episodio, retorno_medio, td_medio (formato
    largo, una fila por tasa x episodio, para graficar las curvas juntas).
  - tasas_aprendizaje_resumen.csv : tasa, retorno_final, td_final (medias de
    las R corridas en el ultimo episodio).

Uso:
  python3 entrenar_tasas_aprendizaje.py
  python3 entrenar_tasas_aprendizaje.py --tasas 0.1,0.5,0.9 --corridas 10
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


def _correr_una(args):
    semilla, K, T, alpha0 = args
    rng = random.Random(semilla)
    np.random.seed(semilla)

    def hacer_env(disc):
        return ql.EntornoSimple(disc, rng)

    disc = ql.calibrar_discretizador(hacer_env, rng)
    env = hacer_env(disc)

    agente = ql.AgenteQ(rng, alpha=alpha0)   # Q_0 = 0 (por defecto), solo cambia alpha0

    _, historial = ql.entrenar(env, disc, rng, episodios=K, horizonte=T,
                               verbose=False, agente=agente)

    retorno = np.array([h[1] for h in historial])
    td = np.array([h[5] for h in historial])
    return alpha0, semilla, retorno, td


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasas", type=str, default="0.1,0.5,0.9",
                     help="valores de alpha_0 a comparar, separados por coma")
    ap.add_argument("--corridas", type=int, default=10,
                     help="corridas independientes POR tasa")
    ap.add_argument("--episodios", type=int, default=None)
    ap.add_argument("--horizonte", type=int, default=None)
    ap.add_argument("--semilla-base", type=int, default=2000)
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--salidas", type=str, default=None)
    args = ap.parse_args()

    tasas = [float(x) for x in args.tasas.split(",")]
    K = args.episodios or ql.CONFIG["episodios"]
    T = args.horizonte or ql.CONFIG["horizonte"]
    R = args.corridas
    carpeta = args.salidas or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "salidas")
    os.makedirs(carpeta, exist_ok=True)

    print(f"Lanzando {len(tasas)} tasas x {R} corridas = {len(tasas)*R} "
          f"entrenamientos (K={K}, T={T}, {args.procesos} procesos)...")
    print(f"Tasas alpha_0 a comparar: {tasas}  (actual del TFG: "
          f"{ql.CONFIG['alpha0']})")

    tareas = []
    semilla = args.semilla_base
    for alpha0 in tasas:
        for _ in range(R):
            tareas.append((semilla, K, T, alpha0))
            semilla += 1

    resultados = []
    with Pool(processes=args.procesos) as pool:
        for i, res in enumerate(pool.imap_unordered(_correr_una, tareas), 1):
            resultados.append(res)
            print(f"  entrenamiento {i}/{len(tareas)} terminado "
                  f"(alpha0={res[0]}, semilla={res[1]})")

    por_tasa = {a: [] for a in tasas}
    for alpha0, semilla, retorno, td in resultados:
        por_tasa[alpha0].append((retorno, td))

    ruta_curva = os.path.join(carpeta, "tasas_aprendizaje.csv")
    ruta_resumen = os.path.join(carpeta, "tasas_aprendizaje_resumen.csv")
    with open(ruta_curva, "w", newline="") as fc, \
         open(ruta_resumen, "w", newline="") as fr:
        wc = csv.writer(fc)
        wc.writerow(["alpha0", "episodio", "retorno_medio", "td_medio"])
        wr = csv.writer(fr)
        wr.writerow(["alpha0", "n_corridas", "retorno_final", "td_final",
                     "episodio_retorno_80pct"])
        for alpha0 in tasas:
            corridas = por_tasa[alpha0]
            retornos = np.stack([c[0] for c in corridas])
            tds = np.stack([c[1] for c in corridas])
            ret_media = retornos.mean(axis=0)
            td_media = tds.mean(axis=0)
            for k in range(K):
                wc.writerow([alpha0, k, f"{ret_media[k]:.4f}",
                             f"{td_media[k]:.6f}"])
            # episodio en el que el retorno medio alcanza el 80% de su valor final
            objetivo = 0.8 * ret_media[-1]
            sobre = np.flatnonzero(ret_media >= objetivo)
            ep80 = int(sobre[0]) if len(sobre) else -1
            wr.writerow([alpha0, len(corridas), f"{ret_media[-1]:.2f}",
                         f"{td_media[-1]:.4f}", ep80])
            print(f"  alpha0={alpha0}: retorno final={ret_media[-1]:.1f}  "
                  f"TD final={td_media[-1]:.4f}  "
                  f"episodio al 80% del retorno final={ep80}")

    print(f"\nEscrito: {ruta_curva}")
    print(f"Escrito: {ruta_resumen}")


if __name__ == "__main__":
    main()
