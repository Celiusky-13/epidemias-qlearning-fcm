#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dibuja el grafo final de cada fuente (expertos / LLM / datos) y tipo
(inferir / predecir) a partir de las matrices de pesos.

Nodos de estado (E1-E7) en azul, acciones (A1-A4) en naranja. Aristas:
grosor = |peso|, color: azul si positivo, rojo si negativo. Umbral mínimo
configurable con --umbral (por defecto 0.05 para no saturar).

Guarda PNG en 2_fcm_construccion/figuras/ y copia a 3_fcm_integracion/figuras/
y docs/figuras/ para la web.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONC = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "A1", "A2", "A3", "A4"]
IDX = {n: i for i, n in enumerate(CONC)}
LABEL = {
    "E1": "Contagio", "E2": "ΔCont", "E3": "Hospital", "E4": "ΔHosp",
    "E5": "Restricción", "E6": "ΔEcon", "E7": "Preocup.",
    "A1": "Trabajo", "A2": "Colegios", "A3": "Transp.", "A4": "Confin.",
}

AQUI = os.path.dirname(os.path.abspath(__file__))
FCM_DATA = os.path.join(AQUI, "..", "..", "3_fcm_integracion", "datos")
FIG_DIR = os.path.join(AQUI, "..", "figuras")
WEB_DIR = os.path.join(AQUI, "..", "..", "docs", "figuras")


def cargar_matriz(ruta):
    M = np.zeros((11, 11))
    with open(ruta) as f:
        r = csv.reader(f); header = next(r)
        for row in r:
            ei = row[0]
            for c, val in zip(header[1:], row[1:]):
                M[IDX[ei], IDX[c]] = float(val)
    return M


def posiciones_circulares(n):
    """11 nodos repartidos en un círculo. Estados arriba, acciones abajo."""
    pos = {}
    # Estados E1-E7 en semicírculo superior
    for i, c in enumerate(["E1", "E2", "E3", "E4", "E5", "E6", "E7"]):
        angulo = np.pi * (1 - i / 6)   # de pi a 0
        pos[c] = (np.cos(angulo), np.sin(angulo) * 0.85 + 0.15)
    # Acciones A1-A4 en línea inferior
    for i, c in enumerate(["A1", "A2", "A3", "A4"]):
        pos[c] = (-0.75 + 0.5 * i, -0.75)
    return pos


def dibujar(W, titulo, ruta_png, umbral=0.05):
    fig, ax = plt.subplots(figsize=(8, 7))
    pos = posiciones_circulares(11)

    # Dibujar aristas primero (para que queden debajo de los nodos)
    for i, ci in enumerate(CONC):
        for j, cj in enumerate(CONC):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) < umbral:
                continue
            x0, y0 = pos[ci]
            x1, y1 = pos[cj]
            color = "#0054B4" if w > 0 else "#C8102E"
            lw = 0.5 + 3 * min(abs(w), 1.0)
            # Flecha con curvatura ligera para no solaparse
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=lw, alpha=0.65,
                                        connectionstyle="arc3,rad=0.1"))

    # Nodos
    for c in CONC:
        x, y = pos[c]
        col = "#003DA5" if c.startswith("E") else "#F2A900"
        circle = plt.Circle((x, y), 0.08, color=col, zorder=5,
                            ec="white", lw=1.5)
        ax.add_patch(circle)
        ax.text(x, y, c, ha="center", va="center", color="white",
                fontsize=8, fontweight="bold", zorder=6)
        # Etiqueta debajo
        ax.text(x, y - 0.14, LABEL[c], ha="center", va="top",
                fontsize=7, color="#333")

    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.1, 1.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(titulo, fontsize=11)
    # Leyenda
    ax.plot([], [], color="#0054B4", lw=2, label="Positivo (aumenta)")
    ax.plot([], [], color="#C8102E", lw=2, label="Negativo (reduce)")
    ax.scatter([], [], color="#003DA5", s=60, label="Estado")
    ax.scatter([], [], color="#F2A900", s=60, label="Acción")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(WEB_DIR, exist_ok=True)

    fuentes = ["expertos", "llm", "datos"]
    tipos = ["inferir", "predecir"]
    generados = []
    for fuente in fuentes:
        for tipo in tipos:
            ruta_csv = os.path.join(FCM_DATA, f"fcm_W_{fuente}_{tipo}.csv")
            if not os.path.exists(ruta_csv):
                print(f"  (saltado) no existe {ruta_csv}")
                continue
            W = cargar_matriz(ruta_csv)
            titulo = f"FCM {fuente.upper()} - {tipo.upper()} ({int((np.abs(W) > args.umbral).sum())} aristas sobre umbral)"
            png = f"grafo_{fuente}_{tipo}.png"
            ruta_png = os.path.join(FIG_DIR, png)
            dibujar(W, titulo, ruta_png, args.umbral)
            # Copia a web
            import shutil
            shutil.copy(ruta_png, os.path.join(WEB_DIR, png))
            generados.append((fuente, tipo, ruta_png, int((np.abs(W) > args.umbral).sum())))
            print(f"  [{fuente}/{tipo}] {ruta_png} -> {(np.abs(W) > args.umbral).sum()} aristas")

    print(f"\nTotal: {len(generados)} grafos generados")


if __name__ == "__main__":
    main()
