#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construye, por cada fuente (expertos, LLM, datos), DOS matrices 11x11:
  - fcm_W_<fuente>_inferir.csv   (matriz "próximos estados", usada por V2 para
    inferir s_{t+1} completo al decidir qué acción tomar)
  - fcm_W_<fuente>_predecir.csv  (matriz "predecir contagio", usada por V3 para
    moldear la recompensa; sólo la columna E1 es relevante en la práctica)

Lee las hojas "MATRIZ DE PESOS PROXIMOS ESTADO" y "MATRIZ DE PESOS DE PREDECIR CON"
de FCM EXPERTOS.xlsx y FCM LLM.xlsx (promedio de bloques). Para datos: la
estructura se toma del CONSENSO y los pesos del script de pesos aprendidos
(aprender_pesos_datos.py); si ese fichero no existe aún, cae a correlación como
antes y lo avisa por consola (se puede regenerar con aprender_pesos_datos.py).

Cambios respecto al script original:
  1. DOS matrices por fuente (antes sólo "próximos estados").
  2. Para datos: ahora admite pesos aprendidos (tarea 5 del plan), no sólo
     correlación. Parámetro --pesos {corr, aprendidos}.
  3. Diagnóstico de cobertura: cuántas celdas no nulas tiene cada matriz.
"""
import argparse
import csv
import os
import sys

import numpy as np
import openpyxl

CONC = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "A1", "A2", "A3", "A4"]
IDX = {n: i for i, n in enumerate(CONC)}

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE_FCM = os.path.join(AQUI, "..", "..", "2_fcm_construccion", "datos")
DATOS = os.path.join(AQUI, "..", "datos")

HOJA_INFERIR = "MATRIZ DE PESOS PROXIMOS ESTADO"
HOJA_PREDECIR = "MATRIZ DE PESOS DE PREDECIR CON"

# Hojas de datos (una por algoritmo+tipo) y consensos
HOJAS_DATOS_INFERIR = ["PROXIMOS ESTADOS - PC", "PROXIMOS ESTADOS - FCI",
                       "PROXIMOS ESTADOS - GES"]
HOJAS_DATOS_PREDECIR = ["PREDECIR CONTAGIO - PC", "PREDECIR CONTAGIO - FCI",
                        "PREDECIR CONTAGIO - GES"]


def cargar_promedio_bloques(ruta, hoja):
    """Promedia todos los bloques 'Desde \\ Hacia' apilados en una hoja."""
    wb = openpyxl.load_workbook(ruta, data_only=True)
    if hoja not in wb.sheetnames:
        raise RuntimeError(f"Hoja '{hoja}' no existe en {ruta}. Hojas: {wb.sheetnames}")
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
    if not matrices:
        raise RuntimeError(f"Ningún bloque 'Desde' en hoja '{hoja}' de {ruta}")
    W = np.mean(np.stack(matrices), axis=0)
    return W, len(matrices)


def limpiar_inferir(W):
    """FCM 'inferir': diagonal a cero (la persistencia va implícita por f(u)=sigmoid),
    columnas de acción a cero (las acciones son causa del agente, no efecto)."""
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    for a in ("A1", "A2", "A3", "A4"):
        W[:, IDX[a]] = 0.0
    return W


def limpiar_predecir(W):
    """FCM 'predecir': sólo importa la columna E1 (el contagio del día siguiente).
    Fuera de esa columna ponemos a cero para que quede claro para qué se usa."""
    W = W.copy()
    salida = np.zeros_like(W)
    salida[:, IDX["E1"]] = W[:, IDX["E1"]]
    # La diagonal E1->E1 queda, representa la persistencia del contagio.
    return salida


def estructura_consenso(ruta, hoja):
    """Lee una hoja de consenso (ej. 'PROXIMOS ESTADOS - CONSENSO') y devuelve
    el conjunto de aristas presentes con su recuento."""
    wb = openpyxl.load_workbook(ruta, data_only=True)
    ws = wb[hoja]
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
    """Peso = correlación de Pearson lag-1 entre i(t) y j(t+1), media entre países."""
    series = {}
    for p in paises:
        ruta = os.path.join(BASE_FCM, f"fcm_{p}.csv")
        cols = {c: [] for c in CONC}
        with open(ruta, newline="") as f:
            for row in csv.DictReader(f):
                for c in CONC:
                    clave = next(k for k in row if k.startswith(c + "_"))
                    cols[c].append(float(row[clave]))
        series[p] = {c: np.array(v) for c, v in cols.items()}
    corr = {}
    for i in CONC:
        for j in CONC:
            vals = []
            for p in paises:
                xi = series[p][i][:-1]
                yj = series[p][j][1:]
                if xi.std() > 1e-9 and yj.std() > 1e-9:
                    vals.append(float(np.corrcoef(xi, yj)[0, 1]))
            if vals:
                corr[(i, j)] = float(np.mean(vals))
    return corr


def estructura_consenso_csv(tipo):
    """Lee el consenso automático generado por consenso_datos.py.
       tipo: 'inferir' o 'predecir'. Devuelve dict (i,j) -> apariciones."""
    ruta = os.path.join(AQUI, "..", "..", "2_fcm_construccion", "resultados",
                        f"consenso_datos_{tipo}.csv")
    aristas = {}
    import csv as _csv
    with open(ruta) as f:
        for row in _csv.DictReader(f):
            aristas[(row["desde"], row["hacia"])] = int(row["apariciones"])
    return aristas

def matriz_datos_inferir(metodo_pesos="corr"):
    """Construye la matriz 'próximos estados' de la fuente datos.
       metodo_pesos: 'corr' (vieja, correlación) o 'aprendidos' (nuevo, regresión
       lag-1 estandarizada controlando persistencia - cargado desde CSV)."""
    aristas = estructura_consenso_csv("inferir")
    if metodo_pesos in ("aprendidos", "nhl", "dhl"):
        sufijo = "" if metodo_pesos == "aprendidos" else "_" + metodo_pesos
        ruta = os.path.join(DATOS, f"pesos_aprendidos_datos_inferir{sufijo}.csv")
        if not os.path.exists(ruta):
            print(f"  AVISO: no existe {ruta}. Cae a 'corr'. Lanza "
                  f"aprender_pesos_datos.py primero.")
            metodo_pesos = "corr"
    W = np.zeros((11, 11))
    if metodo_pesos == "corr":
        corr = corr_lag1_media(["ES", "GB", "BR"])
        for (i, j) in aristas:
            W[IDX[i], IDX[j]] = corr.get((i, j), 0.0)
    else:
        pesos = {}
        with open(ruta) as f:
            r = csv.reader(f)
            header = next(r)
            for row in r:
                ei = row[0]
                for c, val in zip(header[1:], row[1:]):
                    pesos[(ei, c)] = float(val)
        for (i, j) in aristas:
            W[IDX[i], IDX[j]] = pesos.get((i, j), 0.0)
    return limpiar_inferir(W), len(aristas)


def matriz_datos_predecir(metodo_pesos="corr"):
    """Construye la matriz 'predecir contagio' de la fuente datos."""
    aristas = estructura_consenso_csv("predecir")
    if metodo_pesos in ("aprendidos", "nhl", "dhl"):
        sufijo = "" if metodo_pesos == "aprendidos" else "_" + metodo_pesos
        ruta = os.path.join(DATOS, f"pesos_aprendidos_datos_predecir{sufijo}.csv")
        if not os.path.exists(ruta):
            print(f"  AVISO: no existe {ruta}. Cae a 'corr'. Lanza "
                  f"aprender_pesos_datos.py primero.")
            metodo_pesos = "corr"
    W = np.zeros((11, 11))
    if metodo_pesos == "corr":
        corr = corr_lag1_media(["ES", "GB", "BR"])
        for (i, j) in aristas:
            if j != "E1":
                continue  # 'predecir' sólo cuenta la columna E1
            W[IDX[i], IDX[j]] = corr.get((i, j), 0.0)
    else:
        pesos = {}
        with open(ruta) as f:
            r = csv.reader(f)
            header = next(r)
            for row in r:
                ei = row[0]
                for c, val in zip(header[1:], row[1:]):
                    pesos[(ei, c)] = float(val)
        for (i, j) in aristas:
            if j != "E1":
                continue
            W[IDX[i], IDX[j]] = pesos.get((i, j), 0.0)
    return limpiar_predecir(W), len(aristas)


def guardar(W, ruta):
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + CONC)
        for i, ei in enumerate(CONC):
            w.writerow([ei] + [f"{W[i, j]:.4f}" for j in range(11)])


def no_nulos(W, tol=1e-6):
    return int(np.sum(np.abs(W) > tol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", choices=["corr", "aprendidos", "nhl", "dhl"], default="corr",
                    help="Método para pesos de la fuente datos (default: corr). "
                         "'aprendidos' requiere lanzar antes aprender_pesos_datos.py.")
    args = ap.parse_args()

    os.makedirs(DATOS, exist_ok=True)

    print(f"Construyendo 2 matrices por fuente (pesos datos: {args.pesos})\n")

    # EXPERTOS (dos hojas)
    We_inf, ne_inf = cargar_promedio_bloques(
        os.path.join(BASE_FCM, "FCM EXPERTOS.xlsx"), HOJA_INFERIR)
    We_inf = limpiar_inferir(We_inf)
    guardar(We_inf, os.path.join(DATOS, "fcm_W_expertos_inferir.csv"))

    We_pre, ne_pre = cargar_promedio_bloques(
        os.path.join(BASE_FCM, "FCM EXPERTOS.xlsx"), HOJA_PREDECIR)
    We_pre = limpiar_predecir(We_pre)
    guardar(We_pre, os.path.join(DATOS, "fcm_W_expertos_predecir.csv"))
    print(f"EXPERTOS inferir: {ne_inf} bloques  |  {no_nulos(We_inf)} celdas no nulas")
    print(f"EXPERTOS predecir: {ne_pre} bloques |  {no_nulos(We_pre)} celdas no nulas")

    # LLM (dos hojas)
    Wl_inf, nl_inf = cargar_promedio_bloques(
        os.path.join(BASE_FCM, "FCM LLM.xlsx"), HOJA_INFERIR)
    Wl_inf = limpiar_inferir(Wl_inf)
    guardar(Wl_inf, os.path.join(DATOS, "fcm_W_llm_inferir.csv"))

    Wl_pre, nl_pre = cargar_promedio_bloques(
        os.path.join(BASE_FCM, "FCM LLM.xlsx"), HOJA_PREDECIR)
    Wl_pre = limpiar_predecir(Wl_pre)
    guardar(Wl_pre, os.path.join(DATOS, "fcm_W_llm_predecir.csv"))
    print(f"LLM inferir:      {nl_inf} bloques  |  {no_nulos(Wl_inf)} celdas no nulas")
    print(f"LLM predecir:     {nl_pre} bloques  |  {no_nulos(Wl_pre)} celdas no nulas")

    # DATOS (dos consensos)
    Wd_inf, nd_inf = matriz_datos_inferir(args.pesos)
    guardar(Wd_inf, os.path.join(DATOS, "fcm_W_datos_inferir.csv"))
    Wd_pre, nd_pre = matriz_datos_predecir(args.pesos)
    guardar(Wd_pre, os.path.join(DATOS, "fcm_W_datos_predecir.csv"))
    print(f"DATOS inferir:    {nd_inf} aristas en consenso  |  "
          f"{no_nulos(Wd_inf)} celdas no nulas  (pesos: {args.pesos})")
    print(f"DATOS predecir:   {nd_pre} aristas en consenso  |  "
          f"{no_nulos(Wd_pre)} celdas no nulas  (pesos: {args.pesos})")

    # Verificación de celdas conocidas del TFG
    def cel(W, i, j): return W[IDX[i], IDX[j]]
    print("\nVerificación (celdas conocidas del TFG - matriz INFERIR):")
    print(f"  EXPERTOS A4->E1 (esperado -0,72): {cel(We_inf,'A4','E1'):+.2f}")
    print(f"  EXPERTOS E1->E3 (esperado  0,73): {cel(We_inf,'E1','E3'):+.2f}")
    print(f"  LLM      A4->E1 (esperado -0,77): {cel(Wl_inf,'A4','E1'):+.2f}")


if __name__ == "__main__":
    main()
