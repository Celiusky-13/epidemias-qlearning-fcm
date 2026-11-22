#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analiza barrido_final.csv: figura de 4 paneles + tabla pivote + stats Wilcoxon."""
import csv
import os
import statistics
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "..", "resultados")
FIG = os.path.join(AQUI, "..", "figuras")

AZUL = "#003DA5"; NARA = "#F2A900"; VERDE = "#0E8A6B"; ROJO = "#C8102E"
COLOR_V = {"V1": "#777", "V2": AZUL, "V3": NARA, "V4": VERDE}


def leer(csv_path):
    """Devuelve dict (fuente, pais, escenario, version) -> dict metrica -> [vals]."""
    d = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            k = (row["fuente"], row["pais"], row["escenario"], row["version"])
            for m in ("contagio", "hospital", "coste", "pct_bajo", "retorno"):
                d[k][m].append(float(row[m]))
    return d


def bootstrap_ic(vals, rng, n=1000):
    if len(vals) < 2:
        return (float(vals[0]), float(vals[0])) if vals else (0, 0)
    a = np.array(vals, float)
    boot = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def fig_por_pais_escenario(d, metrica, titulo, ruta):
    """3 paneles (ES, GB, BR) x N escenarios, barras por fuente y versión."""
    rng = np.random.default_rng(42)
    paises = ["ES", "GB", "BR"]
    escenarios = sorted({k[2] for k in d})
    fuentes = ["expertos", "llm", "datos"]
    versiones = ["V1", "V2", "V3", "V4"]

    fig, axes = plt.subplots(len(escenarios), len(paises),
                              figsize=(4.5 * len(paises), 3.5 * len(escenarios)),
                              squeeze=False, sharey=True)
    for ei, esc in enumerate(escenarios):
        for pi, p in enumerate(paises):
            ax = axes[ei][pi]
            x = np.arange(len(fuentes)); w = 0.2
            for vi, v in enumerate(versiones):
                medias = []; errs_lo = []; errs_hi = []
                for f in fuentes:
                    k = (f, p, esc, v)
                    vals = d.get(k, {}).get(metrica, [])
                    m = statistics.mean(vals) if vals else 0.0
                    lo, hi = bootstrap_ic(vals, rng) if vals else (0, 0)
                    medias.append(m); errs_lo.append(m - lo); errs_hi.append(hi - m)
                ax.bar(x + vi * w - 1.5 * w, medias, w,
                        yerr=[errs_lo, errs_hi], label=v if ei == 0 and pi == 0 else "",
                        color=COLOR_V[v], capsize=2, edgecolor='white', linewidth=0.5)
            ax.set_xticks(x); ax.set_xticklabels(fuentes, fontsize=9)
            if ei == 0: ax.set_title(f"País: {p}", fontsize=10)
            if pi == 0: ax.set_ylabel(f"{esc.capitalize()}", fontsize=10, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{titulo} (IC95% bootstrap)", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {ruta}")


def tabla_pivote(d, metrica, ruta_csv):
    """Guarda tabla long con medias + IC95 bootstrap."""
    rng = np.random.default_rng(42)
    with open(ruta_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fuente", "pais", "escenario", "version", "n",
                    "media", "ic_inf", "ic_sup"])
        for (fu, pa, esc, v), met in sorted(d.items()):
            vals = met.get(metrica, [])
            if not vals: continue
            m = statistics.mean(vals)
            lo, hi = bootstrap_ic(vals, rng)
            w.writerow([fu, pa, esc, v, len(vals),
                        f"{m:.3f}", f"{lo:.3f}", f"{hi:.3f}"])


def main():
    csv_path = os.path.join(RES, "barrido_final.csv")
    if not os.path.exists(csv_path):
        print(f"No existe {csv_path}"); return

    d = leer(csv_path)
    n_combos = len({(k[0], k[1], k[2], k[3]) for k in d})
    n_rows = sum(len(d[k]["contagio"]) for k in d)
    print(f"Leído: {len(d)} combinaciones, {n_rows} filas")

    # Figuras
    os.makedirs(FIG, exist_ok=True)
    for m, titulo in [("retorno", "Retorno medio por versión"),
                       ("pct_bajo", "% días con contagio Bajo"),
                       ("contagio", "Contagio medio"),
                       ("coste", "Coste medio de las medidas")]:
        ruta = os.path.join(FIG, f"fig_barrido_final_{m}.png")
        fig_por_pais_escenario(d, m, titulo, ruta)

    # Tabla pivote por métrica
    for m in ("retorno", "contagio", "pct_bajo", "coste"):
        tabla_pivote(d, m, os.path.join(RES, f"tabla_barrido_{m}.csv"))

    # Resumen TXT
    out = [f"BARRIDO FINAL: {n_rows} filas"]
    for p in ("ES", "GB", "BR"):
        for esc in ("generico", "pandemia"):
            out.append(f"\n--- {p} / {esc} ---")
            out.append(f"{'Fuente':<10}{'Versión':<6}"
                        f"{'Retorno':<16}{'%bajo':<16}{'Contagio':<16}")
            for fu in ("expertos", "llm", "datos"):
                for v in ("V1", "V2", "V3", "V4"):
                    k = (fu, p, esc, v)
                    rvs = d.get(k, {}).get("retorno", [])
                    if not rvs: continue
                    r = statistics.mean(rvs)
                    b = statistics.mean(d[k]["pct_bajo"])
                    c = statistics.mean(d[k]["contagio"])
                    out.append(f"{fu:<10}{v:<6}{r:+8.1f}       {b:5.1f}%        {c:.3f}")
    with open(os.path.join(RES, "resumen_barrido_final.txt"), "w") as f:
        f.write("\n".join(out))
    print(f"\nGuardado resumen en {RES}/resumen_barrido_final.txt")
    print("\n".join(out[:35]))


if __name__ == "__main__":
    main()
