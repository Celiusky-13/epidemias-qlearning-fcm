# -*- coding: utf-8 -*-
"""Genera las figuras y la tabla de estados representativos del TFG a partir de
las salidas del Q-learning (carpeta `salidas/`), para que el documento LaTeX
quede SIEMPRE consistente con el modelo actual. Reejecutar tras cada cambio.

Produce en la carpeta de figuras del libro:
  - fig_aprendizaje.png  : (a) retorno por episodio y (b) residual de Bellman
                           |delta| medio, promedio de N corridas independientes
                           (curva_aprendizaje_30corridas.csv, generado por
                           entrenar_30_corridas.py) con banda de IC al 95%.
  - fig_convergencia.png : velocidad de convergencia: % de corridas que ya
                           aplican la accion ideal en dos estados
                           representativos (convergencia_estados.csv, tambien
                           de entrenar_30_corridas.py), frente al episodio.
  - fig_ocupacion.png    : ocupacion d^pi por nivel de contagio (B/M/A).
  - fig_acciones.png     : accion de pi* por regimen: (a) coste medio,
                           (b) desglose por palanca, (c) promedio de palancas.
Y por stdout, las filas LaTeX de la tabla de estados representativos (tab:politica).

fig_aprendizaje y fig_convergencia dependen de haber corrido antes:
  python3 entrenar_30_corridas.py

Uso:
  python3 documento_figuras.py
  python3 documento_figuras.py --salidas salidas --figuras "../../TFG REDACCION/Book/figures"
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paleta: azul corporativo UAH (Pantone 293) + terracota, teal y oro.
# Validada (separación daltonismo/visión normal y contraste); el azul es el
# color de marca de la UAH, algo más oscuro que la banda por decisión propia.
AZUL = "#003DA5"      # azul corporativo UAH (Pantone 293)
TERRACOTA = "#D2691E"
TEAL = "#0E8A6B"
ORO = "#B8860B"
GRIS = "#9aa0a6"
GRIS_ERR = "#52514e"  # tinta secundaria para barras de error
FASES = ["emergencia", "prop_rapida", "prop_lenta", "calma"]
FASE_TXT = {"emergencia": "Emergencia", "prop_rapida": "Prop. rápida",
            "prop_lenta": "Prop. lenta", "calma": "Calma"}
PALANCAS = ["work", "school", "ptrans", "conf"]
PALANCAS_TXT = ["Trabajo", "Colegios", "Transporte", "Confinam."]


def leer_csv(ruta):
    with open(ruta, newline="") as f:
        return list(csv.DictReader(f))


def media_movil(x, k):
    if len(x) < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


# ---------------------------------------------------------------------
def fig_aprendizaje(salidas, figuras):
    """(a) retorno y (b) residual de Bellman, PROMEDIO de N corridas
    independientes con banda de IC al 95% (no media movil de una sola
    corrida): lee curva_aprendizaje_30corridas.csv, escrito por
    entrenar_30_corridas.py."""
    ruta = os.path.join(salidas, "curva_aprendizaje_30corridas.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta antes: python3 entrenar_30_corridas.py")
    filas = leer_csv(ruta)
    ep = np.array([int(r["episodio"]) for r in filas])
    ret = np.array([float(r["retorno_medio"]) for r in filas])
    ret_lo = np.array([float(r["retorno_ic95_lo"]) for r in filas])
    ret_hi = np.array([float(r["retorno_ic95_hi"]) for r in filas])
    td = np.array([float(r["td_medio"]) for r in filas])
    td_lo = np.array([float(r["td_ic95_lo"]) for r in filas])
    td_hi = np.array([float(r["td_ic95_hi"]) for r in filas])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.fill_between(ep, ret_lo, ret_hi, color=AZUL, alpha=0.20, lw=0)
    ax1.plot(ep, ret, color=AZUL, lw=1.4)
    ax1.set_title("(a) Retorno por episodio")
    ax1.set_xlabel("episodio"); ax1.set_ylabel("retorno")
    ax2.fill_between(ep, td_lo, td_hi, color=AZUL, alpha=0.20, lw=0)
    ax2.plot(ep, td, color=AZUL, lw=1.4)
    ax2.set_title(r"(b) Residual de Bellman $|\delta|$")
    ax2.set_xlabel("episodio"); ax2.set_ylabel(r"$|\delta|$ medio")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figuras, "fig_aprendizaje.png"), dpi=150)
    plt.close(fig)
    return {"retorno_final": float(ret[-1]), "td_inicial": float(td[0]),
            "td_final": float(td[-1]), "n_episodios": int(ep[-1]) + 1}


# ---------------------------------------------------------------------
NOMBRE_ESTADO_TXT = {
    "prop_rapida": "Prop. rápida: objetivo $(2,2,2,2)$",
    "calma": "Calma sostenida: objetivo $(0,0,0,0)$",
}
COLOR_ESTADO = {"prop_rapida": AZUL, "calma": TERRACOTA}


def fig_convergencia(salidas, figuras):
    """Velocidad de convergencia: % de corridas (de N, con Q inicial al azar)
    que ya aplican la accion ideal en el estado indicado, frente al episodio.
    Lee convergencia_estados.csv (una fila por corrida, episodio en que esa
    corrida empieza a aplicar la accion ideal de forma sostenida), generado
    por entrenar_30_corridas.py, y reconstruye la curva escalonada."""
    ruta = os.path.join(salidas, "convergencia_estados.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta antes: python3 entrenar_30_corridas.py")
    filas = leer_csv(ruta)
    n_corridas = len(filas)

    ruta_ap = os.path.join(salidas, "curva_aprendizaje_30corridas.csv")
    k_max = max(int(r["episodio"]) for r in leer_csv(ruta_ap)) if os.path.exists(ruta_ap) else None

    columnas = {"prop_rapida": "episodio_convergencia_prop_rapida",
                "calma": "episodio_convergencia_calma"}
    medianas = {}
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for nombre, col in columnas.items():
        episodios_conv = sorted(int(r[col]) for r in filas if r[col] != "")
        n_no_conv = n_corridas - len(episodios_conv)
        if not episodios_conv:
            continue
        tope = k_max if k_max is not None else episodios_conv[-1]
        xs = [0] + episodios_conv + [tope]
        ys = [100.0 * i / n_corridas for i in range(len(episodios_conv) + 1)]
        ys = ys + [ys[-1]]
        ax.step(xs, ys, where="post", color=COLOR_ESTADO[nombre], lw=1.8,
                label=NOMBRE_ESTADO_TXT[nombre]
                + (f" ({n_no_conv} sin converger)" if n_no_conv else ""))
        mediana = float(np.median(episodios_conv))
        medianas[nombre] = {"mediana": mediana, "n_no_conv": n_no_conv,
                            "n_corridas": n_corridas}
        ax.axvline(mediana, color=COLOR_ESTADO[nombre], lw=1.0, ls=":")
    ax.set_xlabel("episodio"); ax.set_ylabel("% de corridas que ya convergieron")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(figuras, "fig_convergencia.png"), dpi=150)
    plt.close(fig)
    return medianas


# ---------------------------------------------------------------------
COLOR_TASA = {0.1: TERRACOTA, 0.5: AZUL, 0.9: TEAL}


def fig_tasas(salidas, figuras):
    """Compara distintas tasas de aprendizaje iniciales alpha_0: (a) retorno
    y (b) residual de Bellman por episodio, una linea por tasa. Lee
    tasas_aprendizaje.csv, generado por entrenar_tasas_aprendizaje.py."""
    ruta = os.path.join(salidas, "tasas_aprendizaje.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta antes: python3 entrenar_tasas_aprendizaje.py")
    filas = leer_csv(ruta)
    por_tasa = defaultdict(lambda: {"ep": [], "ret": [], "td": []})
    for r in filas:
        a = float(r["alpha0"])
        por_tasa[a]["ep"].append(int(r["episodio"]))
        por_tasa[a]["ret"].append(float(r["retorno_medio"]))
        por_tasa[a]["td"].append(float(r["td_medio"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for a in sorted(por_tasa):
        color = COLOR_TASA.get(a, None)
        etiqueta = f"$\\alpha_0={a:g}$" + ("  (la usada)" if abs(a - 0.5) < 1e-9 else "")
        ax1.plot(por_tasa[a]["ep"], por_tasa[a]["ret"], color=color, lw=1.4, label=etiqueta)
        ax2.plot(por_tasa[a]["ep"], por_tasa[a]["td"], color=color, lw=1.4, label=etiqueta)
    ax1.set_title("(a) Retorno por episodio"); ax1.set_xlabel("episodio"); ax1.set_ylabel("retorno")
    ax2.set_title(r"(b) Residual de Bellman $|\delta|$"); ax2.set_xlabel("episodio")
    ax2.set_ylabel(r"$|\delta|$ medio")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figuras, "fig_tasas.png"), dpi=150)
    plt.close(fig)


def fig_ocupacion(salidas, figuras):
    filas = leer_csv(os.path.join(salidas, "ocupacion_estados.csv"))
    d = defaultdict(float)
    for r in filas:
        d[r["nivel_contagio"]] += float(r["d_pi"])
    niveles = ["B", "M", "A"]
    vals = [100.0 * d.get(n, 0.0) for n in niveles]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    barras = ax.bar(["Bajo", "Medio", "Alto"], vals,
                    color=[AZUL, TERRACOTA, TEAL])
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Porcentaje de tiempo en cada nivel de contagio")
    ax.set_xlabel("Nivel de contagio")
    ax.set_ylabel("% del tiempo"); ax.set_ylim(0, max(vals) * 1.15 + 5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figuras, "fig_ocupacion.png"), dpi=150)
    plt.close(fig)


def _agregado_por_fase(salidas):
    """Coste medio y palancas medias por fase, desde rejilla_politica.csv
    (ponderando por el nº de pasos n de cada celda fase x L)."""
    filas = leer_csv(os.path.join(salidas, "rejilla_politica.csv"))
    acc = {f: {"n": 0.0, "coste": 0.0, "pal": np.zeros(4)} for f in FASES}
    for r in filas:
        f = r["fase"]
        if f not in acc or not r["coste_medio"]:
            continue
        try:
            n = float(r["n"]); coste = float(r["coste_medio"])
            pal = np.array([float(r[p]) for p in PALANCAS])
        except (ValueError, KeyError):
            continue
        if n <= 0:
            continue
        acc[f]["n"] += n
        acc[f]["coste"] += coste * n
        acc[f]["pal"] += pal * n
    res = {}
    for f in FASES:
        n = acc[f]["n"]
        if n > 0:
            res[f] = {"coste": acc[f]["coste"] / n, "pal": acc[f]["pal"] / n}
        else:
            res[f] = {"coste": 0.0, "pal": np.zeros(4)}
    return res


def fig_acciones(salidas, figuras):
    res = _agregado_por_fase(salidas)
    etiquetas = [FASE_TXT[f] for f in FASES]
    costes = [res[f]["coste"] for f in FASES]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.4))
    # (a) coste medio por regimen
    b1 = ax1.bar(etiquetas, costes, color=AZUL)
    for b, v in zip(b1, costes):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_title("(a) Coste medio por régimen")
    ax1.set_ylabel("coste"); ax1.set_ylim(0, 11.5)
    # (b) desglose por palanca
    x = np.arange(len(FASES)); anchura = 0.2
    colores = [AZUL, TERRACOTA, TEAL, ORO]
    for i, (pt, col) in enumerate(zip(PALANCAS_TXT, colores)):
        vals = [res[f]["pal"][i] for f in FASES]
        ax2.bar(x + (i - 1.5) * anchura, vals, anchura, label=pt, color=col)
    ax2.set_title("(b) Nivel medio por palanca")
    ax2.set_xticks(x); ax2.set_xticklabels(etiquetas, fontsize=7, rotation=15)
    ax2.set_ylabel("nivel (0–2)"); ax2.set_ylim(0, 2.15)
    # leyenda ÚNICA abajo del todo (fuera de las barras, sin solaparse)
    leg_handles, leg_labels = ax2.get_legend_handles_labels()
    # (c) promedio de las 4 palancas
    prom = [float(np.mean(res[f]["pal"])) for f in FASES]
    b3 = ax3.bar(etiquetas, prom, color=AZUL)
    for b, v in zip(b3, prom):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    ax3.set_title("(c) Promedio de palancas")
    ax3.set_ylabel("nivel medio (0–2)"); ax3.set_ylim(0, 2.25)
    for ax in (ax1, ax2, ax3):
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation=15, labelsize=7)
    fig.tight_layout(rect=[0, 0.08, 1, 1])   # reserva espacio abajo para la leyenda
    fig.legend(leg_handles, leg_labels, loc="lower center", ncol=4,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(figuras, "fig_acciones.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
NIV = {"B": "Bajo", "M": "Medio", "A": "Alto"}
FLECHA = {"+": r"$\uparrow$", "0": r"$\rightarrow$", "-": r"$\downarrow$"}


def _parse_estado(txt):
    """'c=B dc=- h=B dh=- L=B dL=- preo=B' -> dict."""
    d = {}
    for tok in txt.split():
        k, v = tok.split("=")
        d[k] = v
    return d


def tabla_representativa(salidas, n_por_fase=2, min_visitas=1):
    """Filas LaTeX de tab:politica: por cada régimen, los estados MÁS visitados
    (los que la política encuentra de verdad), con todas sus variables."""
    filas = leer_csv(os.path.join(salidas, "politica.csv"))
    por_fase = defaultdict(list)
    for r in filas:
        por_fase[r["fase"]].append(r)
    print("% --- filas generadas por documento_figuras.py (NO editar a mano) ---")
    for f in FASES:
        cand = sorted(por_fase.get(f, []), key=lambda r: int(r["visitas"]),
                      reverse=True)
        # se eligen los mas visitados pero con ACCIONES distintas, para que la
        # muestra sea variada (no dos filas casi iguales)
        elegidos, acciones = [], set()
        for r in cand:
            a = (r["work"], r["school"], r["ptrans"], r["conf"])
            if a in acciones:
                continue
            acciones.add(a)
            elegidos.append(r)
            if len(elegidos) == n_por_fase:
                break
        for r in elegidos:
            e = _parse_estado(r["estado"])
            situacion = (
                f"Contagio {NIV[e['c']]}{FLECHA[e['dc']]}, "
                f"hosp.\\ {NIV[e['h']]}{FLECHA[e['dh']]}, "
                f"restr.\\ {NIV[e['L']]}{FLECHA[e['dL']]}, "
                f"cumpl.\\ {NIV[e['preo']]}")
            accion = f"({r['work']},{r['school']},{r['ptrans']},{r['conf']})"
            coste = int(round(float(r["coste"])))
            print(f"{situacion} & {FASE_TXT[f]} & ${accion}$ & ${coste}$\\\\")


def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--salidas", default=os.path.join(aqui, "salidas"))
    ap.add_argument("--figuras", default=os.path.join(
        aqui, "..", "..", "TFG REDACCIÓN", "Book", "figures"))
    args = ap.parse_args()
    os.makedirs(args.figuras, exist_ok=True)
    resumen_ap = fig_aprendizaje(args.salidas, args.figuras)
    resumen_conv = fig_convergencia(args.salidas, args.figuras)
    if os.path.exists(os.path.join(args.salidas, "tasas_aprendizaje.csv")):
        fig_tasas(args.salidas, args.figuras)
    fig_ocupacion(args.salidas, args.figuras)
    fig_acciones(args.salidas, args.figuras)
    print(f"Figuras escritas en: {os.path.abspath(args.figuras)}\n")

    print("--- valores para el texto (resultados.tex) ---")
    print(f"fig_aprendizaje: {resumen_ap['n_episodios']} episodios, "
          f"retorno final medio = {resumen_ap['retorno_final']:.1f}, "
          f"TD inicial = {resumen_ap['td_inicial']:.3f}, "
          f"TD final = {resumen_ap['td_final']:.3f}")
    for nombre, r in resumen_conv.items():
        print(f"fig_convergencia [{nombre}]: mediana = {r['mediana']:.0f} "
              f"episodios (n_corridas={r['n_corridas']}, "
              f"no convergieron={r['n_no_conv']})")
    print()

    tabla_representativa(args.salidas)


if __name__ == "__main__":
    main()
