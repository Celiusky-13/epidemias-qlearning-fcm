#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Barrido reducido de validación para comprobar que todo funciona tras rehacer
el FCM. Corre:
  3 fuentes (expertos, llm, datos) x 2 escenarios (generico, pandemia) x 5 semillas
= 30 experimentos. Cada uno con 5000 episodios de entrenamiento.

Guarda todo en resultados/barrido_validacion.csv con una fila por combinación.
"""
import os
import random
import subprocess
import sys
from itertools import product

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "..", "resultados")
os.makedirs(RES, exist_ok=True)
ruta_out = os.path.join(RES, "barrido_validacion.csv")

fuentes = ["expertos", "llm", "datos"]
escenarios = ["generico", "pandemia"]
semillas = [0, 1, 2, 3, 4]
EPISODIOS = 5000
EVAL = 15

print(f"Barrido: {len(fuentes)}x{len(escenarios)}x{len(semillas)} = "
      f"{len(fuentes)*len(escenarios)*len(semillas)} experimentos")

import csv as _csv
import importlib
import numpy as np

# Importar funciones del módulo nuevo sin usar CLI (más rápido)
sys.path.insert(0, AQUI)
mod = importlib.import_module("fcm_integracion_v2")
ql = importlib.import_module("qlearning_sencillo")

with open(ruta_out, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["fuente", "escenario", "semilla", "version", "nombre",
                "contagio", "hospital", "coste", "pct_bajo", "retorno"])

    for fuente, escenario in product(fuentes, escenarios):
        W_inf = mod.cargar_matriz(mod.ruta_matriz(fuente, "inferir"))
        W_pre = mod.cargar_matriz(mod.ruta_matriz(fuente, "predecir"))
        print(f"\n=== {fuente} / {escenario} ===")

        for semilla in semillas:
            rng = random.Random(semilla)
            np.random.seed(semilla)

            def hacer_env(disc):
                params = {}
                if escenario != "generico":
                    params.update(mod.ESCENARIOS[escenario])
                return ql.EntornoSimple(disc, rng, params=params if params else None)

            disc = ql.calibrar_discretizador(hacer_env, rng)
            T = ql.CONFIG["horizonte"]

            # Entrenar V1 (puro) y V3 (shaping con predecir)
            env1 = hacer_env(disc)
            ag_puro = mod.entrenar(env1, disc, rng, EPISODIOS, T, W_pre=None)
            env3 = hacer_env(disc)
            ag_mod = mod.entrenar(env3, disc, rng, EPISODIOS, T, W_pre=W_pre,
                                   shaping_centrado=True)

            # Evaluar las 4 versiones
            env_eval = hacer_env(disc)
            for ver, nombre, ag, dec in [
                ("V1", "puro", ag_puro, False),
                ("V2", "fcm_decision", ag_puro, True),
                ("V3", "fcm_reward", ag_mod, False),
                ("V4", "hibrido", ag_mod, True),
            ]:
                m = mod.evaluar(env_eval, disc, ag, rng, W_inf=W_inf,
                                fcm_dec=dec, episodios=EVAL, horizonte=T)
                w.writerow([fuente, escenario, semilla, ver, nombre,
                             f"{m['contagio']:.4f}", f"{m['hospital']:.4f}",
                             f"{m['coste']:.3f}", f"{m['pct_bajo']:.2f}",
                             f"{m['retorno']:.2f}"])
                f.flush()
            print(f"  semilla {semilla}: V1 R={m['retorno']:.1f} "
                  f"(ultima fila = V4)")

print(f"\nTerminado. CSV en {ruta_out}")
