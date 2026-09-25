#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construye las matrices de pesos 11x11 (próximos estados) de las tres fuentes del
FCM y las guarda como CSV, para usarlas en la comparación de las 4 versiones.

  - EXPERTOS: promedio de los bloques del Excel FCM EXPERTOS.xlsx.
  - LLM:      promedio de los bloques del Excel FCM LLM.xlsx (Claude, Gemini, ChatGPT).
  - DATOS:    estructura del consenso (FCM DATOS.xlsx) con peso = correlación de Pearson
              con desfase de un día (variable i en t frente a j en t+1), promediada
              sobre los tres países. Es el mismo criterio de peso que usa el TFG.

Guarda: fcm_W_expertos.csv, fcm_W_llm.csv, fcm_W_datos.csv (en programas/).
"""
import csv
import os

import numpy as np
import openpyxl

CONC = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "A1", "A2", "A3", "A4"]
IDX = {n: i for i, n in enumerate(CONC)}

AQUI = os.path.dirname(os.path.abspath(__file__))
# Los Excel y las series por país están en los datos de la fase 2.
BASE_FCM = os.path.join(AQUI, "..", "..", "2_fcm_construccion", "datos")
# Las matrices de pesos se guardan en los datos de esta fase.
DATOS = os.path.join(AQUI, "..", "datos")
HOJA_PE = "MATRIZ DE PESOS PROXIMOS ESTADO"


def cargar_promedio_bloques(ruta, hoja):
    """Promedia todos los bloques 'Desde \\ Hacia' apilados en una hoja."""
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
    return W, len(matrices)


def limpiar(W):
    """Diagonal a cero (persistencia, no arista) y columnas de acción a cero."""
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    for a in ("A1", "A2", "A3", "A4"):
        W[:, IDX[a]] = 0.0
    return W


def estructura_consenso(ruta):
    """Lee la hoja PROXIMOS ESTADOS - CONSENSO: devuelve el conjunto de aristas (i,j)
    presentes (celda no vacía) y su recuento."""
    wb = openpyxl.load_workbook(ruta, data_only=True)
    ws = wb["PROXIMOS ESTADOS - CONSENSO"]
    filas = list(ws.iter_rows(values_only=True))
    cab = next(r for r, f in enumerate(filas)
              if f and isinstance(f[0], str) and f[0].strip().startswith("Desde"))
    col2c = {c: str(x).strip()[:2] for c, x in enumerate(filas[cab])
             if x and str(x).strip()[:2] in CONC}
    aristas = {}
    for rr in range(cab + 1, cab + 1 + 11):
        if rr >= len(filas) or not filas[rr] or not filas[rr][0]:
            continue
        ei = str(filas[rr][0]).strip()[:2]
        if ei not in CONC:
            continue
        for c, ej in col2c.items():
            v = filas[rr][c] if c < len(filas[rr]) else None
            if isinstance(v, (int, float)) and v > 0:
                aristas[(ei, ej)] = int(v)
    return aristas


def corr_lag1_media(paises):
    """Para cada par (i,j), correlación de Pearson entre i(t) y j(t+1), promediada
    sobre los países dados. Devuelve dict (i,j) -> correlación media con signo."""
    series = {}
    for p in paises:
        ruta = os.path.join(BASE_FCM, f"fcm_{p}.csv")
        cols = {c: [] for c in CONC}
        with open(ruta, newline="") as f:
            for row in csv.DictReader(f):
                for c in CONC:
                    # las columnas del csv son E1_contagio, A1_work, etc.
                    clave = next(k for k in row if k.startswith(c + "_"))
                    cols[c].append(float(row[clave]))
        series[p] = {c: np.array(v) for c, v in cols.items()}
    corr = {}
    for i in CONC:
        for j in CONC:
            vals = []
            for p in paises:
                xi = series[p][i][:-1]   # i en t
                yj = series[p][j][1:]    # j en t+1
                if xi.std() > 1e-9 and yj.std() > 1e-9:
                    vals.append(float(np.corrcoef(xi, yj)[0, 1]))
            if vals:
                corr[(i, j)] = float(np.mean(vals))
    return corr


def matriz_datos():
    """Estructura del consenso, pesos = correlación de Pearson lag-1 media entre países."""
    aristas = estructura_consenso(os.path.join(BASE_FCM, "FCM DATOS.xlsx"))
    corr = corr_lag1_media(["ES", "GB", "BR"])
    W = np.zeros((11, 11))
    for (i, j) in aristas:
        W[IDX[i], IDX[j]] = corr.get((i, j), 0.0)
    return limpiar(W), len(aristas)


def guardar(W, ruta):
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + CONC)
        for i, ei in enumerate(CONC):
            w.writerow([ei] + [f"{W[i, j]:.4f}" for j in range(11)])


def main():
    # EXPERTOS
    os.makedirs(DATOS, exist_ok=True)
    We, ne = cargar_promedio_bloques(os.path.join(BASE_FCM, "FCM EXPERTOS.xlsx"), HOJA_PE)
    We = limpiar(We)
    guardar(We, os.path.join(DATOS, "fcm_W_expertos.csv"))
    # LLM
    Wl, nl = cargar_promedio_bloques(os.path.join(BASE_FCM, "FCM LLM.xlsx"), HOJA_PE)
    Wl = limpiar(Wl)
    guardar(Wl, os.path.join(DATOS, "fcm_W_llm.csv"))
    # DATOS
    Wd, nd = matriz_datos()
    guardar(Wd, os.path.join(DATOS, "fcm_W_datos.csv"))

    print(f"EXPERTOS: {ne} bloques promediados")
    print(f"LLM:      {nl} bloques promediados")
    print(f"DATOS:    {nd} aristas del consenso, pesos por correlación lag-1\n")

    # --- Verificación contra el TFG ---
    def cel(W, i, j):
        return W[IDX[i], IDX[j]]
    print("Verificación (celdas conocidas del TFG):")
    print(f"  EXPERTOS A4->E1 (esperado -0,72): {cel(We,'A4','E1'):+.2f}")
    print(f"  EXPERTOS E1->E3 (esperado  0,73): {cel(We,'E1','E3'):+.2f}")
    print(f"  LLM      E2->E1 (esperado  0,62): {cel(Wl,'E2','E1'):+.2f}")
    print(f"  LLM      A4->E1 (esperado -0,77): {cel(Wl,'A4','E1'):+.2f}")
    print(f"  LLM      A4->E5 (esperado  0,90): {cel(Wl,'A4','E5'):+.2f}")
    print(f"  DATOS    E1->E1 persistencia (diag, debe ser 0 tras limpiar): {cel(Wd,'E1','E1'):+.2f}")
    print(f"  DATOS    E1->E3 (contagio->hospital, +): {cel(Wd,'E1','E3'):+.2f}")
    print(f"  DATOS    A4->E1 (confinamiento->contagio, ~0 esperado): {cel(Wd,'A4','E1'):+.2f}")


if __name__ == "__main__":
    main()
