#!/usr/bin/env python3
"""Genera figuras comparativas a partir de barrido_validacion.csv."""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "..", "resultados")
FIG = os.path.join(AQUI, "..", "figuras")
os.makedirs(FIG, exist_ok=True)

# Leer CSV
datos = defaultdict(lambda: defaultdict(list))  # {(fuente, escenario, version): {metrica: [vals]}}
with open(os.path.join(RES, "barrido_validacion.csv")) as f:
    for row in csv.DictReader(f):
        clave = (row["fuente"], row["escenario"], row["version"])
        for m in ("contagio", "hospital", "coste", "pct_bajo", "retorno"):
            datos[clave][m].append(float(row[m]))

# Promediar
agg = {}
for clave, metricas in datos.items():
    agg[clave] = {m: (np.mean(v), np.std(v)) for m, v in metricas.items()}

# Figura: 4 versiones x 3 fuentes, escenario=generico
fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=False)
fuentes = ["expertos", "llm", "datos"]
versiones = ["V1", "V2", "V3", "V4"]
color_v = {"V1": "#777", "V2": "#0054B4", "V3": "#F2A900", "V4": "#50C878"}

for idx, (metrica, titulo) in enumerate(
        [("retorno", "Retorno (media ± std)"),
         ("contagio", "Contagio medio"),
         ("pct_bajo", "% días en contagio bajo"),
         ("coste", "Coste medio")]):
    ax = axes[idx]
    x = np.arange(len(fuentes))
    w = 0.2
    for i, v in enumerate(versiones):
        valores = []
        errores = []
        for f in fuentes:
            clave = (f, "generico", v)
            if clave in agg:
                valores.append(agg[clave][metrica][0])
                errores.append(agg[clave][metrica][1])
            else:
                valores.append(0); errores.append(0)
        ax.bar(x + i*w - 1.5*w, valores, w, yerr=errores, label=v,
                color=color_v[v], capsize=3, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(fuentes)
    ax.set_title(titulo, fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    if idx == 0:
        ax.legend(loc="best", fontsize=8)

fig.suptitle("Barrido de validación · escenario genérico · 5 semillas",
              fontsize=12, y=1.02)
fig.tight_layout()
ruta = os.path.join(FIG, "fig_barrido_generico.png")
fig.savefig(ruta, dpi=150, bbox_inches="tight")
print(f"Escrito: {ruta}")

# Repetir para pandemia
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for idx, (metrica, titulo) in enumerate(
        [("retorno", "Retorno"),
         ("contagio", "Contagio medio"),
         ("pct_bajo", "% días bajo"),
         ("coste", "Coste medio")]):
    ax = axes[idx]
    x = np.arange(len(fuentes))
    w = 0.2
    for i, v in enumerate(versiones):
        valores = [agg.get((f, "pandemia", v), {metrica: (0, 0)})[metrica][0] for f in fuentes]
        errores = [agg.get((f, "pandemia", v), {metrica: (0, 0)})[metrica][1] for f in fuentes]
        ax.bar(x + i*w - 1.5*w, valores, w, yerr=errores, label=v,
                color=color_v[v], capsize=3, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(fuentes)
    ax.set_title(titulo, fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    if idx == 0:
        ax.legend(loc="best", fontsize=8)
fig.suptitle("Barrido de validación · escenario PANDEMIA · 5 semillas",
              fontsize=12, y=1.02)
fig.tight_layout()
ruta = os.path.join(FIG, "fig_barrido_pandemia.png")
fig.savefig(ruta, dpi=150, bbox_inches="tight")
print(f"Escrito: {ruta}")

print("\n=== RESUMEN NUMÉRICO ===")
print("Escenario genérico, retorno medio por versión x fuente:")
for v in versiones:
    print(f"  {v}:", end=" ")
    for f in fuentes:
        clave = (f, "generico", v)
        m, s = agg.get(clave, {}).get("retorno", (0, 0))
        print(f"{f}={m:+.1f}±{s:.0f}", end="  ")
    print()
