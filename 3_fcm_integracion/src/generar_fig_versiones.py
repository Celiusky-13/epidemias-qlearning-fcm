#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redibuja la figura de comparacion de las 4 versiones (V1-V4) a partir del CSV
de resultados multisemilla, colocando las etiquetas de valor POR ENCIMA del
extremo superior del intervalo de confianza, para que el numero no se solape
con la barra de error. Es solo dibujo: no re-simula nada, lee el resumen ya
calculado.

Uso:
    python3.12 generar_fig_versiones.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)  # carpeta de la fase (3_fcm_integracion)
CSV = os.path.join(RAIZ, "resultados", "comparacion_multisemilla_resumen_expertos_norm.csv")

# La figura se guarda en la carpeta figuras/ de esta fase.
DESTINOS = [
    os.path.join(RAIZ, "figuras", "fig_fcm_versiones.png"),
]

AZUL = "#003DA5"   # azul corporativo UAH (Pantone 293)
VERSIONES = ["V1", "V2", "V3", "V4"]

# Paneles: (clave en el csv, titulo, formato de la etiqueta)
PANELES = [
    ("contagio", "Contagio medio (menos = mejor)", "{:.2f}"),
    ("coste",    "Coste medio",                    "{:.2f}"),
    ("pct_bajo", "% días en nivel Bajo (más = mejor)", "{:.2f}"),
    ("retorno",  "Retorno (más = mejor)",          "{:.2f}"),
]


def cargar(csv_path):
    """Devuelve datos[version][metrica] = (media, ic95_lo, ic95_hi)."""
    datos = {}
    with open(csv_path, newline="") as f:
        for fila in csv.DictReader(f):
            v = fila["version"]
            datos.setdefault(v, {})[fila["metrica"]] = (
                float(fila["media"]), float(fila["ic95_lo"]), float(fila["ic95_hi"])
            )
    return datos


def main():
    datos = cargar(CSV)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))

    for ax, (clave, titulo, fmt) in zip(axes, PANELES):
        medias = [datos[v][clave][0] for v in VERSIONES]
        los    = [datos[v][clave][1] for v in VERSIONES]
        his    = [datos[v][clave][2] for v in VERSIONES]
        # barras de error asimetricas: distancia de la media a cada extremo
        err_lo = [m - lo for m, lo in zip(medias, los)]
        err_hi = [hi - m for m, hi in zip(medias, his)]

        ax.bar(VERSIONES, medias, color=AZUL,
               yerr=[err_lo, err_hi], capsize=4,
               error_kw=dict(ecolor="#52514e", elinewidth=1.1))
        ax.set_title(titulo, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

        # margen superior: dejar hueco para las etiquetas sobre el IC
        tope = max(his)
        ax.set_ylim(0, tope * 1.20)
        hueco = tope * 0.025  # separacion etiqueta-bigote

        # etiqueta POR ENCIMA del extremo superior del intervalo (no del borde)
        for i, (m, hi) in enumerate(zip(medias, his)):
            ax.text(i, hi + hueco, fmt.format(m),
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("Comparación de las 4 versiones (media de 30 semillas, con IC 95%)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for destino in DESTINOS:
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        fig.savefig(destino, dpi=150)
        print("Escrito:", destino)


if __name__ == "__main__":
    main()
