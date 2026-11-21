#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CONSENSO AUTOMÁTICO de la fuente 'datos' para el FCM.

Combina las aristas detectadas por los TRES algoritmos (PC, FCI, GES) en los
TRES países (ES, GB, BR) = 9 grafos totales por tipo (inferir / predecir).

Fuentes de datos:
  - PC y FCI: leídas de FCM DATOS.xlsx, hojas "PROXIMOS ESTADOS - PC/FCI" y
    "PREDECIR CONTAGIO - PC/FCI".
  - GES: leído del CSV resultados/ges/tablas/GES_desfasado.csv para INFERIR
    (filtra relacion=cruzada, es decir aristas i(t)->j(t+1)), y de las hojas
    "PREDECIR CONTAGIO - GES" del Excel para PREDECIR. GES desfasado NO está
    en las hojas del Excel; sólo en el CSV.

Consenso: aristas que aparecen en al menos --umbral grafos (por defecto 2).

ANTES (incorrecto): el consenso se hacía a mano en una hoja del Excel. Añadía
aristas que ningún algoritmo detectó (11 aristas inventadas vs los 14 que PC/FCI
sí detectan) y borraba 6 aristas que GES sí devuelve pero no estaban pegadas
al Excel. Esto es lo que José dijo que había que corregir.

Salida: dos CSV en 2_fcm_construccion/resultados/:
  - consenso_datos_inferir.csv   (próximos estados)
  - consenso_datos_predecir.csv  (predecir contagio)
Columnas: desde, hacia, apariciones (de 1 a 9), paises, algoritmos.
"""
import argparse
import csv
import os

import openpyxl

CONC = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "A1", "A2", "A3", "A4"]
IDX = {n: i for i, n in enumerate(CONC)}

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(AQUI, "..", "datos")
RESULTS = os.path.join(AQUI, "..", "resultados")
GES_CSV_DIR = os.path.join(RESULTS, "ges", "tablas")

HOJAS_INFERIR_EXCEL = ["PROXIMOS ESTADOS - PC", "PROXIMOS ESTADOS - FCI"]
HOJAS_PREDECIR_EXCEL = ["PREDECIR CONTAGIO - PC", "PREDECIR CONTAGIO - FCI",
                        "PREDECIR CONTAGIO - GES"]


def extraer_bloques_excel(ruta, hoja, algoritmo):
    """Extrae los 3 bloques (ES, GB, BR) de una hoja del Excel."""
    wb = openpyxl.load_workbook(ruta, data_only=True)
    ws = wb[hoja]
    filas = list(ws.iter_rows(values_only=True))
    bloques = []
    for r, f in enumerate(filas):
        if not f or not isinstance(f[0], str):
            continue
        if not f[0].strip().startswith("Desde"):
            continue
        pais = None
        for look in range(1, 5):
            if r - look < 0:
                break
            cand = filas[r - look][0] if filas[r - look] else None
            if isinstance(cand, str) and cand.strip() in ("ES", "GB", "BR"):
                pais = cand.strip()
                break
        if pais is None:
            continue
        col2c = {c: str(x).strip()[:2] for c, x in enumerate(f)
                 if x and str(x).strip()[:2] in CONC}
        aristas = {}
        for rr in range(r + 1, r + 1 + 11):
            if rr >= len(filas) or not filas[rr] or not filas[rr][0]:
                continue
            ei = str(filas[rr][0]).strip()[:2]
            if ei not in CONC:
                continue
            for c, ej in col2c.items():
                v = filas[rr][c] if c < len(filas[rr]) else None
                if isinstance(v, (int, float)) and abs(v) > 1e-9:
                    aristas[(ei, ej)] = float(v)
        bloques.append({"pais": pais, "algoritmo": algoritmo, "aristas": aristas})
    return bloques


def extraer_ges_desfasado_csv():
    """Lee GES_desfasado.csv. Devuelve 3 bloques (ES, GB, BR) con aristas
    cruzadas (excluye la diagonal persistencia porque la persistencia se trata
    aparte en el FCM, pero conservamos la información)."""
    ruta = os.path.join(GES_CSV_DIR, "GES_desfasado.csv")
    if not os.path.exists(ruta):
        print(f"  AVISO: {ruta} no existe. Saltando GES desfasado.")
        return []
    por_pais = {"ES": {}, "GB": {}, "BR": {}}
    with open(ruta) as f:
        for row in csv.DictReader(f):
            if row["arista"] != "dirigida":
                continue
            p = row["pais"]
            ei, ej = row["desde"], row["hacia"]
            if ei in CONC and ej in CONC:
                por_pais[p][(ei, ej)] = 1.0
    return [{"pais": p, "algoritmo": "GES", "aristas": a}
            for p, a in por_pais.items() if a]


def consenso(bloques, umbral):
    conteo = {}
    for b in bloques:
        for (i, j) in b["aristas"]:
            conteo.setdefault((i, j), {"paises": set(), "algoritmos": set()})
            conteo[(i, j)]["paises"].add(b["pais"])
            conteo[(i, j)]["algoritmos"].add(b["algoritmo"])
    salida = {}
    for (i, j), info in conteo.items():
        # Apariciones = en cuántos grafos aparece (país x algoritmo)
        apariciones = 0
        for b in bloques:
            if (i, j) in b["aristas"]:
                apariciones += 1
        if apariciones >= umbral:
            salida[(i, j)] = {"apariciones": apariciones,
                              "paises": sorted(info["paises"]),
                              "algoritmos": sorted(info["algoritmos"])}
    return salida


def guardar_csv(cons, ruta):
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["desde", "hacia", "apariciones", "paises", "algoritmos"])
        for (i, j), info in sorted(cons.items(),
                                    key=lambda x: (-x[1]["apariciones"], x[0])):
            w.writerow([i, j, info["apariciones"],
                        "|".join(info["paises"]),
                        "|".join(info["algoritmos"])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=int, default=2)
    args = ap.parse_args()

    ruta = os.path.join(DATOS_DIR, "FCM DATOS.xlsx")
    os.makedirs(RESULTS, exist_ok=True)

    # INFERIR: PC + FCI (Excel) + GES desfasado (CSV)
    bloques_inf = []
    for h in HOJAS_INFERIR_EXCEL:
        algo = h.split(" - ")[-1]
        bs = extraer_bloques_excel(ruta, h, algo)
        bloques_inf += bs
        print(f"[INFERIR] {h}: {len(bs)} bloques, "
              f"{sum(len(b['aristas']) for b in bs)} aristas totales")
    bs_ges = extraer_ges_desfasado_csv()
    bloques_inf += bs_ges
    print(f"[INFERIR] GES desfasado (CSV): {len(bs_ges)} bloques, "
          f"{sum(len(b['aristas']) for b in bs_ges)} aristas")

    cons_inf = consenso(bloques_inf, args.umbral)
    print(f"  Consenso INFERIR (umbral={args.umbral}): {len(cons_inf)} aristas")
    guardar_csv(cons_inf, os.path.join(RESULTS, "consenso_datos_inferir.csv"))

    # PREDECIR: PC + FCI + GES (todas del Excel)
    bloques_pre = []
    for h in HOJAS_PREDECIR_EXCEL:
        algo = h.split(" - ")[-1]
        bs = extraer_bloques_excel(ruta, h, algo)
        bloques_pre += bs
        print(f"[PREDECIR] {h}: {len(bs)} bloques, "
              f"{sum(len(b['aristas']) for b in bs)} aristas totales")

    cons_pre = consenso(bloques_pre, args.umbral)
    print(f"  Consenso PREDECIR (umbral={args.umbral}): {len(cons_pre)} aristas")
    guardar_csv(cons_pre, os.path.join(RESULTS, "consenso_datos_predecir.csv"))

    # Diagnóstico
    print("\nDiagnóstico (INFERIR):")
    sin_salida = [c for c in CONC if not any((c, j) in cons_inf for j in CONC if j != c)]
    sin_entrada = [c for c in CONC if not any((i, c) in cons_inf for i in CONC if i != c)
                   and not c.startswith("A")]
    print(f"  Conceptos sin salida: {sin_salida or 'ninguno'}")
    print(f"  Conceptos sin entrada (excepto acciones): {sin_entrada or 'ninguno'}")

    # Top 10 aristas por apariciones
    print("\nTop 15 aristas por apariciones (INFERIR):")
    top = sorted(cons_inf.items(), key=lambda x: -x[1]["apariciones"])[:15]
    for (i, j), info in top:
        print(f"  {i}->{j}: {info['apariciones']}/9  "
              f"[{','.join(info['algoritmos'])}]  "
              f"[{','.join(info['paises'])}]")


if __name__ == "__main__":
    main()
