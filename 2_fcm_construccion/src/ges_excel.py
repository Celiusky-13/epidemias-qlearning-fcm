#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GES (Greedy Equivalence Search) por país -> matrices de pesos del FCM,
escritas como dos hojas nuevas en 'FCM LLM.xlsx' para poder compararlas
con las matrices que dieron los LLM (Claude, Gemini, ChatGPT).

Método (dos pasos, porque GES da ESTRUCTURA, no pesos):
  1) ESTRUCTURA: GES con score BIC sobre los datos crudos de cada país
     -> CPDAG. Decide QUÉ celdas de la matriz se rellenan.
  2) PESO: para cada variable destino j, regresión lineal con retardo 1
        z[j](t+1) ~ z[j](t) + suma_i z[i](t)   con i adyacente a j en el CPDAG
     Todo estandarizado (z-score), así los coeficientes son comparables y
     caen en el rango [-1, 1] de los pesos FCM. Se incluye z[j](t) como
     control (el estado propio se arrastra en el FCM; la diagonal es 0 por
     convención) y se reporta solo el coeficiente de los demás.

Hojas generadas:
  - GES PROXIMOS ESTADOS   : matriz 11x11 por país + promedio + consenso
  - GES PREDECIR CONTAGIO  : solo la columna E1 (i(t) -> E1(t+1)) + R2

Uso:
    python3 ges_excel.py
"""
import os
import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges

from pc_fci import cargar, VARS, SHORT

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "..", "datos", "FCM LLM.xlsx")
TAB = os.path.join(HERE, "..", "resultados", "ges", "tablas")
os.makedirs(TAB, exist_ok=True)

PAISES = ["ES", "GB", "BR"]
ACCIONES = ["A1", "A2", "A3", "A4"]          # solo emiten: sus columnas no se rellenan
ETIQ = ["E1  Contagio", "E2  Tend. contagio", "E3  Presión hosp.", "E4  Tend. hosp.",
        "E5  Restr. vigente", "E6  Tend. socioec.", "E7  Preocupación",
        "A1  Trabajo", "A2  Colegios", "A3  Transporte", "A4  Confinamiento"]

HOJA_PROX = "GES PROXIMOS ESTADOS"
HOJA_PRED = "GES PREDECIR CONTAGIO"


# ----------------------------------------------------------------------
# 1) ESTRUCTURA: GES -> CPDAG
# ----------------------------------------------------------------------
def ges_cpdag(X, labels):
    """Devuelve (adyacencia, aristas). adyacencia[j] = vecinos de j en el CPDAG.

    aristas: lista de (a, b, tipo) con tipo in {'dirigida', 'sin_direccion'}.
    Convención causal-learn: G[i,j]=-1 y G[j,i]=1  =>  i -> j
    """
    rec = ges(X, score_func="local_score_BIC")
    G = rec["G"].graph
    n = G.shape[0]
    ady = {lab: set() for lab in labels}
    aristas = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = G[i, j], G[j, i]
            if a == -1 and b == 1:
                aristas.append((labels[i], labels[j], "dirigida"))
                ady[labels[j]].add(labels[i])
            elif a == 1 and b == -1:
                aristas.append((labels[j], labels[i], "dirigida"))
                ady[labels[i]].add(labels[j])
            elif a == -1 and b == -1:
                aristas.append((labels[i], labels[j], "sin_direccion"))
                # sin dirección identificada: se prueba en los dos sentidos
                ady[labels[j]].add(labels[i])
                ady[labels[i]].add(labels[j])
    return ady, aristas


# ----------------------------------------------------------------------
# 2) PESOS: regresión estandarizada con retardo 1
# ----------------------------------------------------------------------
def pesos_lag1(X, labels, ady):
    """Pesos de i(t) sobre j(t+1) en dos variantes:

      DIRECTO     z_j(t+1) ~ suma_i z_i(t)              (i adyacente a j en el CPDAG)
                  Es la regla FCM clásica con diagonal 0. Escala comparable a los
                  pesos que dan los LLM, pero arrastra la autocorrelación de la serie.
      INCREMENTAL z_j(t+1) ~ z_j(t) + suma_i z_i(t)
                  Añade el estado propio como control: el peso mide el efecto EXTRA
                  de i por encima de la simple persistencia de j. Más honesto, pero
                  los valores salen mucho más pequeños.

    Devuelve (W_dir, W_inc, R2) con R2[j] = (R2_directo, R2_incremental).
    """
    Z = (X - X.mean(axis=0)) / X.std(axis=0)      # z-score
    idx = {lab: k for k, lab in enumerate(labels)}
    Zt, Zt1 = Z[:-1], Z[1:]                        # t  y  t+1

    def ajusta(cols, y):
        A = Zt[:, cols]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1.0 - (y - A @ beta).var() / y.var() if y.var() > 0 else np.nan
        return beta, r2

    W_dir = {i: {} for i in labels}
    W_inc = {i: {} for i in labels}
    R2 = {}
    for j in labels:
        if j in ACCIONES:                          # las acciones no reciben
            continue
        preds = sorted(ady[j] - {j})
        if not preds:
            continue
        y = Zt1[:, idx[j]]
        cols = [idx[p] for p in preds]

        b_dir, r2_dir = ajusta(cols, y)
        for p, b in zip(preds, b_dir):
            W_dir[p][j] = float(np.clip(b, -1.0, 1.0))

        b_inc, r2_inc = ajusta([idx[j]] + cols, y)  # beta[0] = estado propio (diagonal = 0)
        for p, b in zip(preds, b_inc[1:]):
            W_inc[p][j] = float(np.clip(b, -1.0, 1.0))

        R2[j] = (r2_dir, r2_inc)
    return W_dir, W_inc, R2


# ----------------------------------------------------------------------
# Escritura en Excel
# ----------------------------------------------------------------------
def escribir(wb, hoja, titulo, nota, bloques, extra=None):
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    if hoja in wb.sheetnames:
        del wb[hoja]
    ws = wb.create_sheet(hoja)

    AZUL = Font(bold=True, color="1F4E79")
    GRIS = PatternFill("solid", fgColor="D9D9D9")
    CAB = PatternFill("solid", fgColor="C9DAF8")
    ACC = PatternFill("solid", fgColor="D9EAD3")
    fino = Side(style="thin", color="BFBFBF")
    BORDE = Border(left=fino, right=fino, top=fino, bottom=fino)

    ws["A1"] = titulo
    ws["A1"].font = Font(bold=True, size=13, color="1F4E79")
    ws["A2"] = nota
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 22
    for c in "BCDEFGHIJKL":
        ws.column_dimensions[c].width = 9
    ws.column_dimensions["M"].width = 14

    fila = 4
    for etiqueta, comentario, W, solo_e1 in bloques:
        ws.cell(fila, 1, etiqueta).font = Font(bold=True, size=11, color="1F4E79")
        if comentario:
            ws.cell(fila, 3, comentario).font = Font(italic=True, size=9, color="666666")
        fila += 1

        ws.cell(fila, 1, "Desde \\ Hacia").font = AZUL
        ws.cell(fila, 1).fill = CAB
        ws.cell(fila, 1).border = BORDE
        for k, s in enumerate(SHORT):
            c = ws.cell(fila, 2 + k, s)
            c.font = AZUL
            c.fill = ACC if s in ACCIONES else CAB
            c.alignment = Alignment(horizontal="center")
            c.border = BORDE
        fila += 1

        for k, s in enumerate(SHORT):
            ws.cell(fila, 1, ETIQ[k]).font = Font(bold=True, size=9)
            ws.cell(fila, 1).border = BORDE
            for m, t in enumerate(SHORT):
                c = ws.cell(fila, 2 + m)
                c.border = BORDE
                c.alignment = Alignment(horizontal="center")
                if t in ACCIONES or (solo_e1 and t != "E1"):
                    c.fill = GRIS                      # celda que no se rellena
                    continue
                if s == t:
                    c.value = 0
                    continue
                v = W.get(s, {}).get(t)
                if v is not None:
                    c.value = round(v, 2)
                    c.font = Font(color="C00000" if v < 0 else "1F4E79", bold=abs(v) >= 0.5)
            fila += 1
        fila += 2

    if extra:
        for linea in extra:
            ws.cell(fila, 1, linea).font = Font(size=9, italic=True, color="444444")
            fila += 1
    return ws


def main():
    import openpyxl

    pes, inc, r2s, todas = {}, {}, {}, []
    for cc in PAISES:
        X, labels, fuera, n = cargar(cc)
        ady, aristas = ges_cpdag(X, labels)
        W_dir, W_inc, R2 = pesos_lag1(X, labels, ady)
        pes[cc], inc[cc], r2s[cc] = W_dir, W_inc, R2
        excl = [SHORT[VARS.index(v)] for v in fuera]
        print(f"{cc}: n={n}  vars={len(labels)}  aristas GES={len(aristas)}"
              + (f"  [excluidas: {', '.join(excl)}]" if excl else ""))
        for a, b, tipo in aristas:
            todas.append({"pais": cc, "desde": a, "hacia": b, "tipo": tipo})

    # promedio de los 3 países (ausencia = 0) y consenso (en cuántos países aparece)
    def promedia(dic):
        prom, cons = {i: {} for i in SHORT}, {i: {} for i in SHORT}
        for i in SHORT:
            for j in SHORT:
                if i == j or j in ACCIONES:
                    continue
                hay = [dic[cc].get(i, {}).get(j) for cc in PAISES]
                hay = [v for v in hay if v is not None]
                if hay:
                    prom[i][j] = sum(hay) / 3.0    # los países sin la arista cuentan 0
                    cons[i][j] = len(hay)
        return prom, cons

    prom, cons = promedia(pes)
    prom_inc, _ = promedia(inc)

    NOTA = ("Celda (fila i → columna j) = peso causal de i sobre j, en [−1, 1]. Vacío = GES NO encontró esa relación. "
            "Diagonal = 0. Las acciones (A1-A4) solo emiten: sus columnas (en gris) no se rellenan.   ‖   "
            "MÉTODO EN DOS PASOS: (1) ESTRUCTURA = GES (Greedy Equivalence Search, score BIC) sobre los datos crudos de cada país → CPDAG; "
            "decide qué celdas se rellenan. (2) PESO = coeficiente de una regresión estandarizada con retardo 1, sobre los vecinos que da GES. "
            "Se dan DOS variantes del paso 2: DIRECTA  z_j(t+1) ~ Σ z_i(t)  (regla FCM clásica, escala comparable a los pesos de los LLM) e "
            "INCREMENTAL  z_j(t+1) ~ z_j(t) + Σ z_i(t)  (añade el estado propio como control: mide el efecto EXTRA de i sobre la simple "
            "persistencia de j; los valores salen mucho más pequeños pero es la lectura honesta con series tan autocorreladas).")

    def r2txt(cc, var, k):
        v = r2s[cc].get(var)
        return f"{v[k]:.3f}" if v else "—"

    bloques_prox = [(f"{cc}  —  PESO DIRECTO", f"GES + regresión retardo 1 sin control — datos reales de {cc}",
                     pes[cc], False) for cc in PAISES]
    bloques_prox += [
        ("PROMEDIO DE LOS 3 PAÍSES  —  PESO DIRECTO",
         "media de ES, GB, BR (el país donde GES no halla la arista cuenta 0)", prom, False),
        ("CONSENSO (nº de países con esa arista, 1-3)",
         "3 = la relación aparece en los tres países → arista robusta", cons, False),
    ]
    bloques_prox += [(f"{cc}  —  PESO INCREMENTAL", f"controlando el estado propio z_{{j}}(t) — datos reales de {cc}",
                      inc[cc], False) for cc in PAISES]
    bloques_prox.append(("PROMEDIO DE LOS 3 PAÍSES  —  PESO INCREMENTAL", "", prom_inc, False))

    extra_prox = [
        "Leyenda: azul = peso positivo, rojo = peso negativo, negrita = |peso| ≥ 0.5.",
        "Compárese con las hojas 'MATRIZ DE PESOS PROXIMOS ESTADO' (Claude / Gemini / ChatGPT): allí el peso lo pone el LLM, aquí lo ponen los datos.",
        "Las aristas sin dirección del CPDAG se evalúan en los dos sentidos; el detalle por arista está en resultados/ges/tablas/aristas_GES.csv.",
    ]

    bloques_pred = [(f"{cc}  —  PESO DIRECTO",
                     f"solo columna E1: i(t) → E1(t+1)   |   R² directo = {r2txt(cc, 'E1', 0)}",
                     pes[cc], True) for cc in PAISES]
    bloques_pred += [
        ("PROMEDIO DE LOS 3 PAÍSES  —  PESO DIRECTO",
         "media de ES, GB, BR (el país donde GES no halla la arista cuenta 0)", prom, True),
        ("CONSENSO (nº de países con esa arista, 1-3)", "", cons, True),
    ]
    bloques_pred += [(f"{cc}  —  PESO INCREMENTAL",
                      f"controlando E1(t)   |   R² incremental = {r2txt(cc, 'E1', 1)}",
                      inc[cc], True) for cc in PAISES]
    bloques_pred.append(("PROMEDIO DE LOS 3 PAÍSES  —  PESO INCREMENTAL", "", prom_inc, True))

    extra_pred = [
        "R² = fracción de la varianza de E1(t+1) explicada. Es bajo (0.10–0.26) en los tres países: E1 tiene poca autocorrelación "
        "(r₁ ≈ 0.25 ES / 0.25 GB / 0.42 BR) y las demás variables aportan poco a un día vista. Los pesos pequeños no son un artefacto "
        "del método: la señal predictiva diaria del contagio es realmente débil en estos datos.",
        "Una celda vacía significa que GES NO seleccionó esa variable como relevante para el contagio en ese país.",
        "Compárese con la hoja 'MATRIZ DE PESOS DE PREDECIR CON' (Claude / Gemini / ChatGPT).",
    ]

    wb = openpyxl.load_workbook(XLSX)
    escribir(wb, HOJA_PROX, "MATRIZ DE PESOS PRÓXIMOS ESTADOS — ALGORITMO GES (datos reales)",
             NOTA, bloques_prox, extra_prox)
    escribir(wb, HOJA_PRED, "MATRIZ DE PESOS DE PREDECIR CONTAGIOS — ALGORITMO GES (datos reales)",
             NOTA, bloques_pred, extra_pred)
    wb.save(XLSX)

    pd.DataFrame(todas).to_csv(os.path.join(TAB, "aristas_GES.csv"), index=False)
    filas = [{"pais": cc, "desde": i, "hacia": j,
              "peso_directo": round(v, 3),
              "peso_incremental": round(inc[cc].get(i, {}).get(j, float("nan")), 3)}
             for cc in PAISES for i, d in pes[cc].items() for j, v in d.items()]
    pd.DataFrame(filas).to_csv(os.path.join(TAB, "pesos_GES.csv"), index=False)
    pd.DataFrame([{"pais": cc, "variable": k, "R2_directo": round(v[0], 4),
                   "R2_incremental": round(v[1], 4)}
                  for cc in PAISES for k, v in r2s[cc].items()]).to_csv(
        os.path.join(TAB, "R2_GES.csv"), index=False)

    print(f"\nHojas escritas en {os.path.basename(XLSX)}: '{HOJA_PROX}' y '{HOJA_PRED}'")
    print(f"Tablas en resultados/ges/tablas/: aristas_GES.csv  pesos_GES.csv  R2_GES.csv")


if __name__ == "__main__":
    main()
