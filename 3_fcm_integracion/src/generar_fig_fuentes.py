#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura que compara las 4 versiones (V1-V4) usando las TRES fuentes del FCM
(expertos, LLM y datos), a partir de los CSV de resultados multisemilla.

Cuatro paneles (contagio, coste, % días en nivel bajo, retorno). En cada panel,
para cada versión hay tres barras (una por fuente), con su intervalo de confianza
al 95%. Solo dibuja: lee los resúmenes ya calculados.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "..", "resultados")
DESTINOS = [
    os.path.join(AQUI, "..", "figuras", "fig_fcm_fuentes.png"),
]

VERSIONES = ["V1", "V2", "V3", "V4"]
# Paleta: azul corporativo UAH (Pantone 293) + terracota + teal.
FUENTES = [("expertos", "Expertos", "#003DA5"),
           ("llm", "LLM", "#D2691E"),
           ("datos", "Datos", "#0E8A6B")]
PANELES = [("contagio", "Contagio medio (menos = mejor)"),
           ("coste", "Coste medio"),
           ("pct_bajo", "% días en nivel Bajo (más = mejor)"),
           ("retorno", "Retorno (más = mejor)")]


def cargar(fuente):
    d = {}
    with open(os.path.join(RES, f"comparacion_multisemilla_resumen_{fuente}_norm.csv")) as f:
        for r in csv.DictReader(f):
            d[(r["version"], r["metrica"])] = (
                float(r["media"]), float(r["ic95_lo"]), float(r["ic95_hi"]))
    return d


def main():
    datos = {clave: cargar(clave) for clave, _, _ in FUENTES}
    x = np.arange(len(VERSIONES))
    ancho = 0.26

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (metrica, titulo) in zip(axes, PANELES):
        tope = 0.0
        for k, (clave, etq, color) in enumerate(FUENTES):
            medias = [datos[clave][(v, metrica)][0] for v in VERSIONES]
            lo = [datos[clave][(v, metrica)][1] for v in VERSIONES]
            hi = [datos[clave][(v, metrica)][2] for v in VERSIONES]
            err = [np.array(medias) - np.array(lo), np.array(hi) - np.array(medias)]
            ax.bar(x + (k - 1) * ancho, medias, ancho, label=etq, color=color,
                   yerr=err, capsize=3, error_kw=dict(ecolor="#52514e", elinewidth=0.9))
            tope = max(tope, max(hi))
        ax.set_title(titulo, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(VERSIONES)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if metrica == "retorno":
            ax.axhline(0, color="#888888", lw=0.8)
            ax.set_ylim(min(-120, ax.get_ylim()[0]), tope * 1.12)
        else:
            ax.set_ylim(0, tope * 1.15)

    axes[1].legend(title="Fuente del FCM", fontsize=8, title_fontsize=8,
                   loc="upper left", framealpha=0.95)
    fig.suptitle("Las 4 versiones según la fuente del FCM (media de 30 semillas, IC 95%)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for destino in DESTINOS:
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        fig.savefig(destino, dpi=150)
        print("Escrito:", destino)


if __name__ == "__main__":
    main()
