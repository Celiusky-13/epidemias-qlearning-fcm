#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera FIGURAS y TABLAS del descubrimiento causal, separadas por algoritmo:

  resultados/
    ├── pc/   (figuras/ cpdag_*.png   +  tablas/ aristas_PC.csv, comparacion..., resumen...)
    └── fci/  (figuras/ pag_*.png     +  tablas/ aristas_FCI.csv)

PC  -> CPDAG:  → dirigida | — sin dirección
FCI -> PAG :   →/←  punta de flecha | o  círculo (indeterminado) | —  cola
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from pc_fci import cargar, aristas_pc, VARS, SHORT, ALPHA

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "resultados")
DIRS = {
    "pc_fig": os.path.join(OUT, "pc", "figuras"),
    "pc_tab": os.path.join(OUT, "pc", "tablas"),
    "fci_fig": os.path.join(OUT, "fci", "figuras"),
    "fci_tab": os.path.join(OUT, "fci", "tablas"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

PAISES = ["ES", "GB", "BR"]
STATES = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
R = 0.30          # radio de nodo
OFF = 0.34        # offset del borde para los extremos de arista
POS = {
    "E1": (0.0, 2.0), "E2": (-2.0, 1.4), "E3": (2.0, 1.4),
    "E4": (2.8, 0.2), "E5": (2.0, -1.2), "E6": (-2.0, -1.2),
    "E7": (-2.8, 0.2),
    "A1": (-1.4, -2.4), "A2": (-0.5, -2.6), "A3": (0.5, -2.6), "A4": (1.4, -2.4),
}


def _endpoint_marker(ax, e, tip_dir, kind, color):
    """Dibuja el extremo de una arista en el punto e. tip_dir apunta HACIA el nodo."""
    if kind == "arrow":
        start = (e[0] - 0.16 * tip_dir[0], e[1] - 0.16 * tip_dir[1])
        ax.add_patch(FancyArrowPatch(start, e, arrowstyle="-|>",
                     mutation_scale=18, color=color, lw=0, zorder=4))
    elif kind == "circle":
        ax.add_patch(Circle(e, 0.10, facecolor="white", edgecolor=color,
                     lw=1.8, zorder=5))
    # kind == "tail": nada


def dibujar(cc, edges, titulo, path, excl):
    """edges: lista de (a, b, mark_a, mark_b, color, dashed)."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    for a, b, ma, mb, color, dashed in edges:
        p1, p2 = np.array(POS[a]), np.array(POS[b])
        d = (p2 - p1) / np.linalg.norm(p2 - p1)
        e1, e2 = p1 + OFF * d, p2 - OFF * d
        ax.plot([e1[0], e2[0]], [e1[1], e2[1]], color=color, lw=2,
                ls="--" if dashed else "-", zorder=2)
        _endpoint_marker(ax, tuple(e1), -d, ma, color)   # extremo en a
        _endpoint_marker(ax, tuple(e2),  d, mb, color)   # extremo en b
    # nodos encima
    for n, (x, y) in POS.items():
        estado = n in STATES
        fc = "#c9daf8" if estado else "#d9ead3"
        ec = ("#999999" if n in excl else ("#1f4e79" if estado else "#38761d"))
        ax.add_patch(Circle((x, y), R, facecolor=fc, edgecolor=ec, lw=2, zorder=6))
        ax.text(x, y, n, ha="center", va="center", fontsize=11,
                fontweight="bold", zorder=7)
    ax.set_title(titulo, fontsize=13, fontweight="bold", color="#1f4e79")
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.4, 2.9)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- FCI: extraer aristas con marcas de extremo ----------
MARK = {1: "circle", -1: "tail", 2: "arrow"}
GLYPH = {("tail", "arrow"): "->", ("arrow", "tail"): "<-", ("arrow", "arrow"): "<->",
         ("circle", "arrow"): "o->", ("arrow", "circle"): "<-o",
         ("circle", "circle"): "o-o", ("tail", "circle"): "-o",
         ("circle", "tail"): "o-", ("tail", "tail"): "--"}


def fci_edges(g, labels):
    G = g.graph
    n = G.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = MARK.get(G[j, i]), MARK.get(G[i, j])  # marca en i, marca en j
            if mi is None or mj is None:
                continue
            if G[j, i] == 0 and G[i, j] == 0:
                continue
            out.append((labels[i], labels[j], mi, mj))
    return out


FCM_REF = [
    ("E2->E1", "tendencia contagio → contagio"),
    ("E4->E3", "tendencia hosp → presión hosp"),
    ("E1->E3", "contagio → presión hospitalaria"),
    ("E6->E5", "tendencia socioec → restricción"),
    ("E1->E7", "contagio → preocupación"),
    ("E3->E7", "presión hosp → preocupación"),
    ("E7->E1", "preocupación → contagio"),
    ("E5->E1", "restricción → contagio"),
    ("A1->E5", "trabajo → restricción"),
    ("A4->E5", "confinamiento → restricción"),
    ("A4->E1", "confinamiento → contagio"),
]


def main():
    filas_pc, filas_fci = [], []
    recup = {r[0]: {} for r in FCM_REF}

    for cc in PAISES:
        X, labels, fuera, n = cargar(cc)
        excl = [SHORT[VARS.index(v)] for v in fuera]
        extra = f"   [excluidas: {', '.join(excl)}]" if excl else ""

        # ---- PC ----
        cg = pc(X, alpha=ALPHA, indep_test=fisherz, show_progress=False)
        dr, sd = aristas_pc(cg, labels)
        edges_pc = []
        for e in dr:
            a, b = e.split(" -> ")
            edges_pc.append((a, b, "tail", "arrow", "#1f4e79", False))
            filas_pc.append({"pais": cc, "desde": a, "hacia": b, "tipo": "dirigida"})
        for e in sd:
            a, b = e.split(" -- ")
            edges_pc.append((a, b, "tail", "tail", "#c00000", True))
            filas_pc.append({"pais": cc, "desde": a, "hacia": b, "tipo": "sin_direccion"})
        dibujar(cc, edges_pc, f"CPDAG (PC)  —  {cc}{extra}",
                os.path.join(DIRS["pc_fig"], f"cpdag_{cc}.png"), excl)

        # ---- FCI ----
        g, _ = fci(X, independence_test_method=fisherz, alpha=ALPHA, show_progress=False)
        fe = fci_edges(g, labels)
        edges_fci = []
        for a, b, ma, mb in fe:
            col = "#7030a0" if (ma == "arrow" and mb == "arrow") else "#38761d"
            edges_fci.append((a, b, ma, mb, col, False))
            filas_fci.append({"pais": cc, "desde": a, "hacia": b,
                              "arista": f"{a} {GLYPH.get((ma, mb), '?')} {b}"})
        dibujar(cc, edges_fci, f"PAG (FCI)  —  {cc}{extra}",
                os.path.join(DIRS["fci_fig"], f"pag_{cc}.png"), excl)

        # ---- recuperación FCM (según PC) ----
        dir_set = set((x.split(" -> ")[0], x.split(" -> ")[1]) for x in dr)
        und = set()
        for x in sd:
            a, b = x.split(" -- "); und |= {(a, b), (b, a)}
        for rel, _ in FCM_REF:
            s, t = rel.split("->")
            recup[rel][cc] = ("sí (→)" if (s, t) in dir_set else
                              "invertida" if (t, s) in dir_set else
                              "sin dirección" if (s, t) in und else "no")

    # ---- tablas ----
    pd.DataFrame(filas_pc).to_csv(os.path.join(DIRS["pc_tab"], "aristas_PC.csv"), index=False)
    pd.DataFrame(filas_fci).to_csv(os.path.join(DIRS["fci_tab"], "aristas_FCI.csv"), index=False)
    comp = pd.DataFrame([{"relacion_FCM": r, "descripcion": d,
                          "ES": recup[r]["ES"], "GB": recup[r]["GB"], "BR": recup[r]["BR"]}
                         for r, d in FCM_REF])
    comp.to_csv(os.path.join(DIRS["pc_tab"], "comparacion_FCM_vs_datos.csv"), index=False)
    res = (pd.DataFrame(filas_pc).query("tipo=='dirigida'").groupby("pais").size()
           .reindex(PAISES).fillna(0).astype(int).rename("aristas_dirigidas").reset_index())
    res.to_csv(os.path.join(DIRS["pc_tab"], "resumen_por_pais.csv"), index=False)

    print("resultados/")
    print("  pc/figuras/:  cpdag_ES.png  cpdag_GB.png  cpdag_BR.png")
    print("  pc/tablas/ :  aristas_PC.csv  comparacion_FCM_vs_datos.csv  resumen_por_pais.csv")
    print("  fci/figuras/: pag_ES.png  pag_GB.png  pag_BR.png")
    print("  fci/tablas/ : aristas_FCI.csv")
    print("\nComparación FCM vs datos (PC):")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
