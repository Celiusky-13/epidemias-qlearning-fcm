#!/usr/bin/env python3
"""Barrido completo final: 3 fuentes x 3 paises x 3 escenarios x 10 semillas = 270."""
import os, random, sys, csv as _csv, importlib, time
from itertools import product
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "..", "resultados")
os.makedirs(RES, exist_ok=True)
out = os.path.join(RES, "barrido_completo.csv")
sys.path.insert(0, AQUI)
mod = importlib.import_module("fcm_integracion_v2")
ql = importlib.import_module("qlearning_sencillo")

fuentes = ["expertos", "llm", "datos"]
paises = ["generico", "ES", "GB", "BR"]  # genérico + 3 países
escenarios = ["generico", "pandemia", "normal"]
semillas = list(range(10))
EP = 5000
EVAL = 20
total = len(fuentes)*len(paises)*len(escenarios)*len(semillas)
t0 = time.time()
hechos = 0

with open(out, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["fuente","pais","escenario","semilla","version","nombre",
                "contagio","hospital","coste","pct_bajo","retorno"])
    for fuente, pais, escenario in product(fuentes, paises, escenarios):
        W_inf = mod.cargar_matriz(mod.ruta_matriz(fuente, "inferir"))
        W_pre = mod.cargar_matriz(mod.ruta_matriz(fuente, "predecir"))
        for s in semillas:
            rng = random.Random(s); np.random.seed(s)
            def he(disc):
                p = {}
                if pais != "generico" and hasattr(ql,"PARAMS_PAIS"):
                    p.update(ql.PARAMS_PAIS[pais])
                if escenario != "generico":
                    p.update(mod.ESCENARIOS[escenario])
                return ql.EntornoSimple(disc, rng, params=p if p else None)
            disc = ql.calibrar_discretizador(he, rng)
            T = ql.CONFIG["horizonte"]
            e1 = he(disc); ag_puro = mod.entrenar(e1, disc, rng, EP, T, W_pre=None)
            e3 = he(disc); ag_mod = mod.entrenar(e3, disc, rng, EP, T, W_pre=W_pre, shaping_centrado=True)
            ee = he(disc)
            for ver, nom, ag, dec in [("V1","puro",ag_puro,False),("V2","fcm_dec",ag_puro,True),
                                       ("V3","fcm_rew",ag_mod,False),("V4","hibrido",ag_mod,True)]:
                m = mod.evaluar(ee, disc, ag, rng, W_inf=W_inf, fcm_dec=dec, episodios=EVAL, horizonte=T)
                w.writerow([fuente,pais,escenario,s,ver,nom,
                             f"{m['contagio']:.4f}",f"{m['hospital']:.4f}",
                             f"{m['coste']:.3f}",f"{m['pct_bajo']:.2f}",f"{m['retorno']:.2f}"])
                f.flush()
            hechos += 1
            pct = hechos/total*100
            eta = (time.time()-t0)/max(hechos,1)*(total-hechos)/60
            print(f"[{hechos}/{total} {pct:.0f}%] {fuente}/{pais}/{escenario}/{s}  ETA {eta:.0f}min", flush=True)
print(f"\nFin. {out}")
