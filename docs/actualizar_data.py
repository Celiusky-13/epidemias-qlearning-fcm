#!/usr/bin/env python3
"""Copia las matrices nuevas (6 CSV) y el barrido a docs/data/ para la web."""
import shutil, os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "3_fcm_integracion", "datos")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "3_fcm_integracion", "resultados")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Matrices 2 x fuente (6 CSV)
for fuente in ("expertos", "llm", "datos"):
    for tipo in ("inferir", "predecir"):
        f = f"fcm_W_{fuente}_{tipo}.csv"
        src = os.path.join(SRC, f); dst = os.path.join(DST, f"matriz_{fuente}_{tipo}.csv")
        if os.path.exists(src):
            shutil.copy(src, dst); print(f"  {f} -> {os.path.basename(dst)}")

# Pesos aprendidos (datos, por país)
for pais in ("ES", "GB", "BR"):
    for tipo in ("inferir", "predecir"):
        f = f"pesos_aprendidos_datos_{tipo}_{pais}.csv"
        src = os.path.join(SRC, f); dst = os.path.join(DST, f"matriz_datos_{tipo}_{pais}.csv")
        if os.path.exists(src):
            shutil.copy(src, dst); print(f"  {f} -> {os.path.basename(dst)}")

# Barrido validación y SOTA
for f, nuevo in [("barrido_validacion.csv", "barrido_validacion.csv"),
                 ("stats_resumen.csv", "stats_resumen.csv"),
                 ("stats_comparaciones.csv", "stats_comparaciones.csv"),
                 ("comparacion_sota_generico_generico.csv", "comparacion_sota.csv")]:
    src = os.path.join(RES, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(DST, nuevo)); print(f"  {f} -> {nuevo}")

print("\nOK")
