#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estadística rigurosa sobre barrido_validacion.csv: bootstrap IC95 + Wilcoxon
pareado entre versiones por semilla + Cohen's d (tamaño de efecto).

Produce:
  - resultados/stats_resumen.csv  : una fila por (fuente, escenario, version, metrica)
    con media, IC95 bootstrap, n.
  - resultados/stats_comparaciones.csv : una fila por par de versiones a comparar
    (V2 vs V1, V4 vs V3, etc.) con p-valor Wilcoxon, Cohen's d, interpretación.
  - stats_resumen.txt : vista legible (lo que se puede pegar en el TFG).

Métodos:
  - IC95 por bootstrap percentil con 2000 remuestreos (Agarwal 2021).
  - Wilcoxon signed-rank test pareado (comparación robusta sin asumir normalidad).
  - Cohen's d para medias pareadas (effect size estandarizado).
"""
import csv
import os
import statistics
from collections import defaultdict

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "..", "resultados")

RUTA_IN = os.path.join(RES, "barrido_validacion.csv")
RUTA_RESUMEN = os.path.join(RES, "stats_resumen.csv")
RUTA_COMP = os.path.join(RES, "stats_comparaciones.csv")
RUTA_TXT = os.path.join(RES, "stats_resumen.txt")

METRICAS = ["contagio", "hospital", "coste", "pct_bajo", "retorno"]
COMPARAR = [("V2", "V1"), ("V3", "V1"), ("V4", "V1"),
             ("V4", "V2"), ("V4", "V3")]


def leer_datos():
    """Devuelve dict: (fuente, escenario, version) -> dict semilla -> dict metrica -> val."""
    d = defaultdict(lambda: defaultdict(dict))
    with open(RUTA_IN) as f:
        for row in csv.DictReader(f):
            clave = (row["fuente"], row["escenario"], row["version"])
            semilla = int(row["semilla"])
            for m in METRICAS:
                d[clave][semilla][m] = float(row[m])
    return d


def bootstrap_ic95(valores, n_resamples=2000, rng=None):
    """IC95 percentil por bootstrap."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(valores)
    if n < 2:
        return (float(valores[0]), float(valores[0])) if valores else (0.0, 0.0)
    arr = np.array(valores, dtype=float)
    boot = rng.choice(arr, size=(n_resamples, n), replace=True).mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def wilcoxon_pareado(a, b):
    """Test de Wilcoxon signed-rank pareado. Devuelve (W, p_valor)."""
    # Implementación directa para no exigir scipy
    difs = np.array([x - y for x, y in zip(a, b) if x != y], dtype=float)
    n = len(difs)
    if n == 0:
        return 0.0, 1.0
    rangos = np.argsort(np.argsort(np.abs(difs))) + 1
    W_plus = float(np.sum(rangos[difs > 0]))
    W_minus = float(np.sum(rangos[difs < 0]))
    W = min(W_plus, W_minus)
    # Aproximación normal (n>=10 recomendado)
    media = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    if var < 1e-9:
        return W, 1.0
    z = (W - media) / np.sqrt(var)
    # p-valor bilateral aproximado
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))
    return float(W), float(min(p, 1.0))


def cohen_d(a, b):
    """Cohen's d para muestras pareadas (misma n)."""
    difs = np.array([x - y for x, y in zip(a, b)])
    m = np.mean(difs)
    s = np.std(difs, ddof=1) if len(difs) > 1 else 1.0
    return float(m / s) if s > 0 else 0.0


def interpretar_d(d):
    """Lectura intuitiva de Cohen's d."""
    ad = abs(d)
    if ad < 0.2: return "muy pequeño"
    if ad < 0.5: return "pequeño"
    if ad < 0.8: return "mediano"
    return "grande"


def main():
    if not os.path.exists(RUTA_IN):
        print(f"ERROR: no existe {RUTA_IN}. Lanza barrido_validacion.py primero.")
        return

    datos = leer_datos()
    rng = np.random.default_rng(42)

    # --- RESUMEN (por combinación y métrica) ---
    with open(RUTA_RESUMEN, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fuente", "escenario", "version", "metrica", "n",
                    "media", "ic95_inf", "ic95_sup"])
        for clave, semillas in sorted(datos.items()):
            for m in METRICAS:
                valores = [v[m] for v in semillas.values() if m in v]
                if not valores:
                    continue
                media = statistics.mean(valores)
                lo, hi = bootstrap_ic95(valores, rng=rng)
                w.writerow([clave[0], clave[1], clave[2], m, len(valores),
                            f"{media:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    print(f"Escrito: {RUTA_RESUMEN}")

    # --- COMPARACIONES ---
    with open(RUTA_COMP, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fuente", "escenario", "comp", "metrica", "n", "dif_media",
                    "wilcoxon_W", "p_valor", "cohen_d", "interpretacion",
                    "significativo_005"])
        for fuente, escenario in {(k[0], k[1]) for k in datos}:
            for vA, vB in COMPARAR:
                ks_A = (fuente, escenario, vA)
                ks_B = (fuente, escenario, vB)
                if ks_A not in datos or ks_B not in datos:
                    continue
                semillas = sorted(set(datos[ks_A]) & set(datos[ks_B]))
                for m in METRICAS:
                    a = [datos[ks_A][s][m] for s in semillas if m in datos[ks_A][s]]
                    b = [datos[ks_B][s][m] for s in semillas if m in datos[ks_B][s]]
                    if len(a) < 3 or len(b) < 3 or len(a) != len(b):
                        continue
                    dif = statistics.mean(a) - statistics.mean(b)
                    W, p = wilcoxon_pareado(a, b)
                    d = cohen_d(a, b)
                    signif = "sí" if p < 0.05 else "no"
                    w.writerow([fuente, escenario, f"{vA}-{vB}", m, len(semillas),
                                f"{dif:+.4f}", f"{W:.1f}", f"{p:.4f}",
                                f"{d:+.3f}", interpretar_d(d), signif])
    print(f"Escrito: {RUTA_COMP}")

    # --- TXT legible ---
    lineas = ["=" * 80,
               "ANÁLISIS ESTADÍSTICO DE barrido_validacion.csv",
               "=" * 80,
               "",
               "Método: bootstrap percentil con 2000 remuestreos para IC95,",
               "Wilcoxon signed-rank pareado para comparaciones entre versiones,",
               "Cohen's d como tamaño de efecto.",
               ""]

    # Resumen por combinación
    for fuente, escenario in sorted({(k[0], k[1]) for k in datos}):
        lineas.append(f"\n----- {fuente.upper()} / {escenario.upper()} -----")
        lineas.append(f"{'Versión':<8}{'Retorno (IC95)':<24}{'Contagio (IC95)':<24}"
                       f"{'%días bajo (IC95)':<22}")
        for v in ["V1", "V2", "V3", "V4"]:
            k = (fuente, escenario, v)
            if k not in datos:
                continue
            vals = {m: [sv[m] for sv in datos[k].values() if m in sv]
                    for m in METRICAS}
            def fmt(m):
                if not vals.get(m):
                    return "-"
                lo, hi = bootstrap_ic95(vals[m], rng=rng)
                return f"{statistics.mean(vals[m]):+.1f} [{lo:+.1f},{hi:+.1f}]"
            lineas.append(f"{v:<8}{fmt('retorno'):<24}"
                           f"{fmt('contagio'):<24}{fmt('pct_bajo'):<22}")

        # Comparaciones principales
        lineas.append(f"\nComparaciones estadísticas (α=0.05):")
        for vA, vB in COMPARAR:
            ks_A = (fuente, escenario, vA); ks_B = (fuente, escenario, vB)
            if ks_A not in datos or ks_B not in datos:
                continue
            semillas = sorted(set(datos[ks_A]) & set(datos[ks_B]))
            for m in ["retorno", "contagio", "pct_bajo"]:
                a = [datos[ks_A][s][m] for s in semillas]
                b = [datos[ks_B][s][m] for s in semillas]
                if len(a) < 3:
                    continue
                dif = statistics.mean(a) - statistics.mean(b)
                W, p = wilcoxon_pareado(a, b)
                d = cohen_d(a, b)
                sig = "*" if p < 0.05 else " "
                lineas.append(f"  {vA}-{vB} en {m}: Δ={dif:+.2f}  "
                               f"p={p:.3f}{sig}  d={d:+.2f} ({interpretar_d(d)})")

    with open(RUTA_TXT, "w") as f:
        f.write("\n".join(lineas))
    print(f"Escrito: {RUTA_TXT}")
    print("\n" + "\n".join(lineas[:60]))


if __name__ == "__main__":
    main()
