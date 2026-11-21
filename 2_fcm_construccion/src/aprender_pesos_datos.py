#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aprende los PESOS del FCM de datos a partir de la estructura del consenso.

NO usa correlación de Pearson lag-1 (como el código antiguo). En su lugar, para
cada concepto destino j, hace una regresión multivariante lag-1 estandarizada
con los demás padres como regresores de control. Esto es lo que José pidió:
'el descubrimiento causal no da peso... tiene que entrenar'.

Modelo por concepto j (una regresión por país):
  z_j(t+1) ~ Σ_i β_{ij} * z_i(t)      (i: padres de j en el consenso)

donde z = estandarizado (media 0, desviación 1). Los coeficientes β_{ij} son
los pesos del FCM. Son comparables entre sí (misma escala) y controlan por el
efecto de los demás padres, a diferencia de la correlación bivariada.

Se agrega entre países por media ponderada por R² (los países con mejor ajuste
pesan más). También se guardan los pesos por país separados.

Entrada:
  - 2_fcm_construccion/resultados/consenso_datos_inferir.csv (estructura)
  - 2_fcm_construccion/resultados/consenso_datos_predecir.csv (estructura)
  - 2_fcm_construccion/datos/fcm_{ES,GB,BR}.csv (series)

Salida (en 3_fcm_integracion/datos/, que es donde las lee construir_matrices_v2):
  - pesos_aprendidos_datos_inferir.csv       (matriz 11x11 agregada)
  - pesos_aprendidos_datos_predecir.csv      (matriz 11x11 agregada)
  - pesos_aprendidos_datos_inferir_<pais>.csv  (por país)
  - pesos_aprendidos_datos_predecir_<pais>.csv (por país)

Y un report:
  - 2_fcm_construccion/resultados/informe_pesos_aprendidos.txt
"""
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
    """Devuelve dict concepto -> np.array."""
    ruta = os.path.join(DATOS_DIR, f"fcm_{pais}.csv")
    cols = {c: [] for c in CONC}
    with open(ruta, newline="") as f:
        for row in csv.DictReader(f):
            for c in CONC:
                clave = next(k for k in row if k.startswith(c + "_"))
                cols[c].append(float(row[clave]))
    return {c: np.array(v) for c, v in cols.items()}


def cargar_consenso(ruta):
    """Devuelve dict (i,j) -> {apariciones, paises, algoritmos}."""
    cons = {}
    with open(ruta) as f:
        for row in csv.DictReader(f):
            cons[(row["desde"], row["hacia"])] = {
                "apariciones": int(row["apariciones"]),
                "paises": row["paises"].split("|"),
                "algoritmos": row["algoritmos"].split("|"),
            }
    return cons


def padres_de(cons, j):
    """Lista de i tal que (i,j) está en el consenso."""
    return sorted({i for (i, jj) in cons if jj == j})


def regresion_ols(X, y):
    """β = (X'X)^-1 X'y. Devuelve coeficientes, R² y residuos."""
    try:
        beta, residuos, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        return beta, r2
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1]), 0.0


def estandarizar(x):
    """z = (x - mean)/std. Si std=0, devuelve ceros (serie constante)."""
    s = x.std()
    if s < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def aprender_pesos_por_pais(cons, series_pais, ventana=None):
    """Para cada concepto destino j con padres, regresión lag-1 estandarizada.
    Devuelve matriz 11x11 de pesos y dict de R² por concepto."""
    W = np.zeros((11, 11))
    r2 = {}
    for j in CONC:
        padres = padres_de(cons, j)
        if not padres:
            continue
        # y = j(t+1), z-estandarizado. Omitir el último paso temporal.
        y_series = series_pais[j][1:]
        y = estandarizar(y_series)
        # X = matriz de padres en t, estandarizados
        Xs = []
        valid = []
        for i in padres:
            xi = series_pais[i][:-1]
            xi_z = estandarizar(xi)
            if xi_z.std() > 1e-9 or np.all(xi_z == 0):
                Xs.append(xi_z)
                valid.append(i)
        if not Xs:
            continue
        X = np.column_stack(Xs)
        beta, r2_j = regresion_ols(X, y)
        r2[j] = r2_j
        for k, i in enumerate(valid):
            W[IDX[i], IDX[j]] = float(beta[k])
    return W, r2


def guardar_matriz(W, ruta):
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + CONC)
        for i, ei in enumerate(CONC):
            w.writerow([ei] + [f"{W[i, j]:.4f}" for j in range(11)])


def main():
    os.makedirs(DEST, exist_ok=True)

    series = {p: cargar_series(p) for p in PAISES}
    cons_inf = cargar_consenso(os.path.join(RESULTS, "consenso_datos_inferir.csv"))
    cons_pre = cargar_consenso(os.path.join(RESULTS, "consenso_datos_predecir.csv"))
    print(f"Consenso INFERIR: {len(cons_inf)} aristas")
    print(f"Consenso PREDECIR: {len(cons_pre)} aristas")

    informe_lineas = []

    for tipo, cons in [("inferir", cons_inf), ("predecir", cons_pre)]:
        informe_lineas.append(f"\n===== {tipo.upper()} =====")
        print(f"\n--- Aprendiendo pesos {tipo} ---")

        # Por país
        Ws = {}
        r2s = {}
        for p in PAISES:
            W, r2 = aprender_pesos_por_pais(cons, series[p])
            Ws[p] = W
            r2s[p] = r2
            guardar_matriz(W, os.path.join(DEST, f"pesos_aprendidos_datos_{tipo}_{p}.csv"))
            r2_mean = np.mean(list(r2.values())) if r2 else 0.0
            informe_lineas.append(f"  {p}: R² medio = {r2_mean:.3f}, "
                                   f"R² por concepto = "
                                   f"{ {k: round(v,3) for k,v in r2.items()} }")

        # Agregado por media ponderada por R² (país con mejor ajuste pesa más)
        W_agg = np.zeros((11, 11))
        for j in CONC:
            pesos_pais = []
            matrices_pais = []
            for p in PAISES:
                r2_p = r2s[p].get(j, 0.0)
                w_pais = max(0.0, r2_p)  # peso = R² clipado a 0
                if w_pais > 0:
                    pesos_pais.append(w_pais)
                    matrices_pais.append(Ws[p])
            if not pesos_pais:
                continue
            pesos_pais = np.array(pesos_pais) / np.sum(pesos_pais)
            for pp, M in zip(pesos_pais, matrices_pais):
                W_agg[:, IDX[j]] += pp * M[:, IDX[j]]

        guardar_matriz(W_agg, os.path.join(DEST, f"pesos_aprendidos_datos_{tipo}.csv"))
        informe_lineas.append(f"  AGREGADO (media ponderada por R²):")
        informe_lineas.append(f"    celdas no nulas: {int((np.abs(W_agg) > 1e-6).sum())}")

    # Guardar informe
    informe = "\n".join(informe_lineas)
    with open(os.path.join(RESULTS, "informe_pesos_aprendidos.txt"), "w") as f:
        f.write("INFORME DE PESOS APRENDIDOS (regresión lag-1 estandarizada)\n")
        f.write("=" * 60 + "\n")
        f.write("Modelo: z_j(t+1) ~ Σ_i β_{ij} z_i(t), donde los i son los\n")
        f.write("padres de j según el consenso automático. z = estandarizado.\n")
        f.write("Agregación entre países: media ponderada por R².\n")
        f.write(informe + "\n")
    print(informe)
    print(f"\nInforme guardado en {RESULTS}/informe_pesos_aprendidos.txt")


if __name__ == "__main__":
    main()
