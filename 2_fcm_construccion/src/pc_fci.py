#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descubrimiento causal (PC y FCI) por país sobre las 11 variables del FCM.
- Datos crudos, SIN normalizar ni discretizar (como pide Diego).
- Se ejecuta un país cada vez (no todos los datos juntos).
- Test de independencia: Fisher-Z (datos continuos).
- Salida: el CPDAG impreso como lista de aristas + matriz.

Uso:
    python3 pc_fci.py            # los tres países: ES, GB, BR
    python3 pc_fci.py GB         # solo un país
"""
import sys
import os
import numpy as np
import pandas as pd

# carpeta de datos: ../datos respecto a este script (FCM ALGORITMOS/datos)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

# columnas de las 11 variables (en orden), sin 'date' ni 'country'
VARS = ["E1_contagio", "E2_tend_contagio", "E3_presion_hosp", "E4_tend_hosp",
        "E5_restriccion", "E6_tend_socioec", "E7_preocupacion",
        "A1_work", "A2_school", "A3_ptrans", "A4_conf"]
# etiquetas cortas para imprimir
SHORT = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "A1", "A2", "A3", "A4"]

ALPHA = 0.05  # nivel de significación de los tests de independencia


def cargar(cc):
    df = pd.read_csv(os.path.join(DATA_DIR, f"fcm_{cc}.csv")).sort_values("date")
    # quitar variables constantes (sin varianza -> inútiles para el descubrimiento causal)
    usar, fuera = [], []
    for v in VARS:
        if df[v].nunique(dropna=True) <= 1:
            fuera.append(v)
        else:
            usar.append(v)
    df = df.dropna(subset=usar)          # PC/FCI no admiten NaN
    X = df[usar].to_numpy(dtype=float)
    labels = [SHORT[VARS.index(v)] for v in usar]
    return X, labels, fuera, len(df)


def aristas_pc(cg, labels):
    """Extrae las aristas del CPDAG de PC: '->' dirigida, '--' sin dirección."""
    G = cg.G.graph
    n = G.shape[0]
    dirigidas, sin_dir = [], []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = G[i, j], G[j, i]
            # convención causal-learn: G[i,j]=-1 & G[j,i]=1  => i -> j
            if a == -1 and b == 1:
                dirigidas.append(f"{labels[i]} -> {labels[j]}")
            elif a == 1 and b == -1:
                dirigidas.append(f"{labels[j]} -> {labels[i]}")
            elif a == -1 and b == -1:
                sin_dir.append(f"{labels[i]} -- {labels[j]}")
    return dirigidas, sin_dir


def aristas_fci(g, labels):
    """Aristas del PAG de FCI: ->, <->, o--o, o->, --."""
    G = g.graph
    n = G.shape[0]
    out = []
    M = {1: 'o', -1: '-', 2: '>'}  # marca de extremo (arrow/circle/tail) en causal-learn
    for i in range(n):
        for j in range(i + 1, n):
            a, b = G[j, i], G[i, j]  # extremo en i, extremo en j
            if a == 0 and b == 0:
                continue
            li = {1: 'o', -1: '-', 2: '<'}.get(a, '?')
            rj = {1: 'o', -1: '-', 2: '>'}.get(b, '?')
            out.append(f"{labels[i]} {li}-{rj} {labels[j]}")
    return out


def run(cc):
    print("=" * 64)
    print(f"  PAÍS: {cc}")
    print("=" * 64)
    X, labels, fuera, n = cargar(cc)
    print(f"Filas usadas: {n}   Variables: {len(labels)}  ({', '.join(labels)})")
    if fuera:
        print(f"Excluidas por ser constantes (sin varianza): "
              f"{', '.join(SHORT[VARS.index(v)] for v in fuera)}")
    print(f"Test: Fisher-Z   alpha={ALPHA}\n")

    # ---------- PC -> CPDAG ----------
    cg = pc(X, alpha=ALPHA, indep_test=fisherz, show_progress=False)
    dr, sd = aristas_pc(cg, labels)
    print("----- PC (CPDAG) -----")
    print(f"Aristas dirigidas (causal con sentido):  {len(dr)}")
    for e in dr:
        print("   " + e)
    print(f"Aristas sin dirección (— , sentido indeterminado):  {len(sd)}")
    for e in sd:
        print("   " + e)
    if not dr and not sd:
        print("   (ninguna relación encontrada)")

    # ---------- FCI -> PAG ----------
    g, _ = fci(X, independence_test_method=fisherz, alpha=ALPHA, show_progress=False)
    fe = aristas_fci(g, labels)
    print("\n----- FCI (PAG) -----")
    print(f"Aristas: {len(fe)}   ( -> dirigida | <-> confusor latente | o-o / o-> indeterminada | -- )")
    for e in fe:
        print("   " + e)
    if not fe:
        print("   (ninguna relación encontrada)")
    print()


if __name__ == "__main__":
    paises = [sys.argv[1]] if len(sys.argv) > 1 else ["ES", "GB", "BR"]
    for cc in paises:
        run(cc)
