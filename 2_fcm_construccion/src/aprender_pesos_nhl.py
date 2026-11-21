#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NHL: aprendizaje Hebbiano No-Lineal de pesos del FCM.

Implementa dos algoritmos canónicos de aprendizaje de FCM (lo que José llamó
"meterle su algoritmo" y "Hebbiano"):

  1) NHL  (Nonlinear Hebbian Learning) - Papageorgiou, Stylios, Groumpos (2003)
     "Fuzzy Cognitive Map Learning Based on Nonlinear Hebbian Rule", LNAI 2903.

     Regla:  w_ji(t+1) = gamma * w_ji(t) + eta * A_i(t) * [A_j(t+1) - w_ji(t) * A_i(t)]

     Donde:
       eta   = learning rate (decrece con el tiempo)
       gamma = weight decay (típico 0.98)
       A_i(t) = activación del concepto i en t (normalizada a [0,1])

  2) DHL  (Differential Hebbian Learning) - Dickerson, Kosko (1994)
     "Virtual Worlds as Fuzzy Cognitive Maps", Presence 3(2).

     Regla:  w_ji(t+1) = w_ji(t) + c(t) * [dA_j(t) * A_i(t) - w_ji(t)]

     Donde:
       dA_j(t) = A_j(t) - A_j(t-1)   (cambio, no nivel)
       c(t)    = learning rate decreciente.

El método del WORKFLOW ya implementado (regresión lag-1 estandarizada controlando
por los demás padres) sigue existiendo como opción `--metodo regresion`. Este
módulo añade `--metodo nhl` y `--metodo dhl`. Todos parten de la MISMA estructura
del consenso (consenso_datos.py); sólo cambia cómo se aprenden los pesos.

Salida: mismos CSV que aprender_pesos_datos.py pero con sufijo `_nhl` o `_dhl`
  - pesos_aprendidos_datos_inferir_nhl.csv
  - pesos_aprendidos_datos_predecir_nhl.csv
  (idem por país)

Y un informe comparativo con la calidad del ajuste (MSE y correlación).
"""
import argparse
import csv
import os

import numpy as np

CONC = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "A1", "A2", "A3", "A4"]
IDX = {n: i for i, n in enumerate(CONC)}

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(AQUI, "..", "datos")
RESULTS = os.path.join(AQUI, "..", "resultados")
DEST = os.path.join(AQUI, "..", "..", "3_fcm_integracion", "datos")
PAISES = ["ES", "GB", "BR"]


def cargar_series(pais):
    ruta = os.path.join(DATOS_DIR, f"fcm_{pais}.csv")
    cols = {c: [] for c in CONC}
    with open(ruta, newline="") as f:
        for row in csv.DictReader(f):
            for c in CONC:
                clave = next(k for k in row if k.startswith(c + "_"))
                cols[c].append(float(row[clave]))
    return {c: np.array(v) for c, v in cols.items()}


def cargar_consenso(ruta):
    cons = {}
    with open(ruta) as f:
        for row in csv.DictReader(f):
            cons[(row["desde"], row["hacia"])] = int(row["apariciones"])
    return cons


def normalizar(x):
    """[0,1] para alimentar la sigmoide del FCM (min-max)."""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-9:
        return np.full_like(x, 0.5)
    return (x - mn) / (mx - mn)


def nhl_aprender(series, mascara, episodios=400, eta_0=0.05, eta_min=1e-4,
                  gamma=0.98, decaimiento=0.995, W_inicial=None):
    """NHL con pasada online por los T-1 pasos temporales. Repite `episodios`
    veces (pasadas por los datos) con eta decreciente.

    series: dict concepto -> array normalizado [0,1]
    mascara: matriz 11x11 booleana; W[i,j] se actualiza sólo si mascara[i,j]=True
    """
    T = len(next(iter(series.values())))
    if W_inicial is None:
        W = np.zeros((11, 11))
    else:
        W = W_inicial.copy()

    A = np.stack([series[c] for c in CONC], axis=1)  # TxN
    eta = eta_0

    for ep in range(episodios):
        for t in range(T - 1):
            Ai = A[t]
            Aj1 = A[t + 1]
            # Regla NHL vectorizada: w_ji(t+1) = gamma*w_ji + eta*A_i*(A_j(t+1) - w_ji*A_i)
            # (nótese: notación w_ji aquí = W[i,j] = influencia de i sobre j)
            for i in range(11):
                for j in range(11):
                    if not mascara[i, j]:
                        continue
                    delta = eta * Ai[i] * (Aj1[j] - W[i, j] * Ai[i])
                    W[i, j] = gamma * W[i, j] + delta
        eta = max(eta_min, eta * decaimiento)

    return W


def dhl_aprender(series, mascara, episodios=400, eta_0=0.05, decaimiento=0.995,
                 W_inicial=None):
    """DHL: usa el cambio ΔA_j(t) = A_j(t) - A_j(t-1)."""
    T = len(next(iter(series.values())))
    W = (np.zeros((11, 11)) if W_inicial is None else W_inicial.copy())
    A = np.stack([series[c] for c in CONC], axis=1)
    dA = np.diff(A, axis=0, prepend=A[0:1])  # dA[t] = A[t]-A[t-1]
    eta = eta_0
    for ep in range(episodios):
        for t in range(1, T):
            Ai = A[t - 1]
            dAj = dA[t]
            for i in range(11):
                for j in range(11):
                    if not mascara[i, j]:
                        continue
                    delta = eta * (dAj[j] * Ai[i] - W[i, j])
                    W[i, j] = W[i, j] + delta
        eta = eta * decaimiento
    return W


def calidad_ajuste(W, series):
    """Para cada concepto destino j, calcula el MSE y correlación entre
    la predicción FCM y la serie real. Media sobre conceptos con aristas."""
    A = np.stack([series[c] for c in CONC], axis=1)
    T, N = A.shape
    mse_por_concepto = {}
    corr_por_concepto = {}
    for j in range(N):
        pred = []
        obs = []
        for t in range(T - 1):
            # Predicción FCM (una capa, sin sigmoide para medir ajuste lineal)
            pred.append(A[t] @ W[:, j])
            obs.append(A[t + 1, j])
        pred = np.array(pred); obs = np.array(obs)
        mse_por_concepto[CONC[j]] = float(np.mean((pred - obs) ** 2))
        if pred.std() > 1e-9 and obs.std() > 1e-9:
            corr_por_concepto[CONC[j]] = float(np.corrcoef(pred, obs)[0, 1])
    return mse_por_concepto, corr_por_concepto


def construir_mascara(cons):
    """Convierte el consenso (dict de aristas) en matriz 11x11 booleana."""
    M = np.zeros((11, 11), dtype=bool)
    for (i, j) in cons:
        M[IDX[i], IDX[j]] = True
    return M


def guardar_matriz(W, ruta):
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + CONC)
        for i, ei in enumerate(CONC):
            w.writerow([ei] + [f"{W[i, j]:.4f}" for j in range(11)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metodo", choices=["nhl", "dhl"], default="nhl")
    ap.add_argument("--episodios", type=int, default=400)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.98)
    args = ap.parse_args()

    series_raw = {p: cargar_series(p) for p in PAISES}
    # Normalizar [0,1] para que las activaciones sean válidas en el FCM
    series = {p: {c: normalizar(v) for c, v in sp.items()}
              for p, sp in series_raw.items()}

    cons_inf = cargar_consenso(os.path.join(RESULTS, "consenso_datos_inferir.csv"))
    cons_pre = cargar_consenso(os.path.join(RESULTS, "consenso_datos_predecir.csv"))

    algo = {"nhl": nhl_aprender, "dhl": dhl_aprender}[args.metodo]
    nombre_algo = args.metodo.upper()
    print(f"Aprendiendo pesos con {nombre_algo} "
          f"(episodios={args.episodios}, eta={args.eta})\n")

    informe = [f"ALGORITMO: {nombre_algo}\n"]

    for tipo, cons in [("inferir", cons_inf), ("predecir", cons_pre)]:
        informe.append(f"\n===== {tipo.upper()} ({len(cons)} aristas) =====")
        print(f"--- {tipo.upper()} ---")

        Ws_pais = {}
        for pais in PAISES:
            mascara = construir_mascara(cons)
            if tipo == "predecir":
                # Sólo mantener aristas que apuntan a E1
                for j in range(11):
                    if j != IDX["E1"]:
                        mascara[:, j] = False
            W = algo(series[pais], mascara,
                      episodios=args.episodios, eta_0=args.eta,
                      **({"gamma": args.gamma} if args.metodo == "nhl" else {}))
            Ws_pais[pais] = W
            # Calidad del ajuste
            mse, corr = calidad_ajuste(W, series[pais])
            mse_med = np.mean([mse[c] for c in CONC if any(mascara[:, IDX[c]])])
            corr_med = np.mean([corr.get(c, 0) for c in CONC if any(mascara[:, IDX[c]])])
            informe.append(f"  {pais}: MSE medio={mse_med:.4f}  "
                            f"corr medio={corr_med:.3f}  "
                            f"celdas no nulas={int((np.abs(W) > 1e-6).sum())}")
            guardar_matriz(W, os.path.join(DEST,
                f"pesos_aprendidos_datos_{tipo}_{args.metodo}_{pais}.csv"))

        # Agregado: media simple entre países (NHL ya es no-lineal, no conviene
        # ponderar por R² aquí porque la correlación se ha movido mucho)
        W_agg = np.mean([Ws_pais[p] for p in PAISES], axis=0)
        guardar_matriz(W_agg, os.path.join(DEST,
            f"pesos_aprendidos_datos_{tipo}_{args.metodo}.csv"))
        informe.append(f"  AGREGADO: celdas no nulas = {int((np.abs(W_agg) > 1e-6).sum())}")

    print("\n".join(informe))
    with open(os.path.join(RESULTS, f"informe_pesos_{args.metodo}.txt"), "w") as f:
        f.write("\n".join(informe))
    print(f"\nInforme en {RESULTS}/informe_pesos_{args.metodo}.txt")


if __name__ == "__main__":
    main()
