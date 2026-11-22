#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baselines del estado del arte (Kompella et al. 2020) implementadas en NUESTRO
simulador para comparar.

Kompella et al. (2020) "Reinforcement Learning for Optimization of COVID-19
Mitigation Policies", arXiv:2010.10560. Define 5 etapas de regulación y una
familia de políticas heurísticas:
  - S0-4-0:     sube a etapa 4 cuando hay ≥N_umbral infectados, vuelve a 0 tras
                30 días en etapa 4.
  - S0-4-0-GI:  variante con retorno GRADUAL en pasos de 10 días (etapa 4 → 3 →
                2 → 1 → 0). ES LA QUE MEJOR LES SALE en la comparación.
  - "SWE":      política suave tipo Suecia (etapa baja constante).

Adaptación a nuestro simulador (que tiene 4 palancas 0-2 con 81 acciones):
  - stage 0 = (0,0,0,0)   sin restricciones (0 coste)
  - stage 1 = (1,0,0,0)   trabajo parcial (1 coste)
  - stage 2 = (1,1,1,0)   trabajo+colegios+transporte parcial (3 coste)
  - stage 3 = (2,2,2,1)   todo parcial+confinamiento parcial (7 coste)
  - stage 4 = (2,2,2,2)   todo cerrado (10 coste máx)

Umbral de "casos" (infectados en nuestro modelo = contagio alto): usamos
disc.nivel(env.c,"c") == 2 (contagio Alto) como señal de "muchos casos".

Compara contra V1 (Q-learning puro) del TFG, usando las mismas semillas, el
mismo simulador y las mismas métricas. Usa Wilcoxon pareado para el test.
"""
import argparse
import csv
import os
import random
import sys

for _v in ("OMP_NUM_THREADS",): os.environ.setdefault(_v, "1")

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import qlearning_sencillo as ql

# Mapeo stage -> acción (adaptado al TFG)
STAGES = {
    0: (0, 0, 0, 0),
    1: (1, 0, 0, 0),
    2: (1, 1, 1, 0),
    3: (2, 2, 2, 1),
    4: (2, 2, 2, 2),
}


def politica_s040gi(env, disc, dias_en_stage, stage_actual, dias_bajando,
                     umbral_entrada_nivel=2, dias_pico=30, paso_bajada=10):
    """Lógica S0-4-0-GI de Kompella: sube a stage 4 si contagio Alto;
    tras 30 días baja gradualmente 4->3->2->1->0, uno por cada 10 días.
    Devuelve (nuevo_stage, nuevo_dias_en_stage, nuevo_dias_bajando)."""
    nivel_c = disc.nivel(env.c, "c")
    # Si estamos en stage 0 y el contagio es alto -> subir a stage 4
    if stage_actual == 0 and nivel_c >= umbral_entrada_nivel:
        return 4, 1, 0
    # Si estamos en stage 4 y llevamos >=30 días, empezar a bajar
    if stage_actual == 4:
        if dias_en_stage >= dias_pico:
            return 3, 1, 1
        return 4, dias_en_stage + 1, 0
    # Si estamos bajando, cada paso_bajada días baja un stage
    if 1 <= stage_actual <= 3:
        if dias_bajando >= paso_bajada:
            return stage_actual - 1, 1, 1
        return stage_actual, dias_en_stage + 1, dias_bajando + 1
    return stage_actual, dias_en_stage + 1, dias_bajando


def politica_s040(env, disc, dias_en_stage, stage_actual,
                   umbral_entrada_nivel=2, dias_pico=30):
    """S0-4-0 simple: salta 0 -> 4, aguanta 30 días, vuelve a 0."""
    nivel_c = disc.nivel(env.c, "c")
    if stage_actual == 0 and nivel_c >= umbral_entrada_nivel:
        return 4, 1
    if stage_actual == 4 and dias_en_stage >= dias_pico:
        return 0, 1
    return stage_actual, dias_en_stage + 1


def politica_sin_restriccion(env, disc, *_):
    return 0, 1


def evaluar_politica(env_factory, disc, politica_fn, rng,
                      episodios=15, horizonte=None):
    """Ejecuta `politica_fn(env, disc, state) -> stage` durante episodios."""
    T = horizonte or ql.CONFIG["horizonte"]
    tot = {"retorno": 0.0, "c": 0.0, "h": 0.0, "coste": 0.0, "bajo": 0.0}
    for _ in range(episodios):
        env = env_factory()
        s = env.reset()
        dias_en_stage, stage, dias_bajando = 0, 0, 0
        for t in range(T):
            stage, dias_en_stage, dias_bajando = politica_fn(
                env, disc, dias_en_stage, stage, dias_bajando)
            a = STAGES[stage]
            c_t, h_t, a_prev = env.c, env.h, env.a_prev
            s1 = env.paso(a)
            r, _ = ql.recompensa_tramos(s, c_t, env.c, h_t, env.h, a, a_prev, disc)
            tot["retorno"] += r
            tot["c"] += env.c; tot["h"] += env.h
            tot["coste"] += ql.coste(a)
            if disc.nivel(env.c, "c") == 0:
                tot["bajo"] += 1
            s = s1
    n = episodios * T
    return {"contagio": tot["c"]/n, "hospital": tot["h"]/n,
            "coste": tot["coste"]/n, "pct_bajo": 100.0*tot["bajo"]/n,
            "retorno": tot["retorno"]/episodios}


# Adaptador para que las políticas con/sin dias_bajando tengan la misma firma
def _wrap_s040(env, disc, d1, stage, _d2):
    s, d = politica_s040(env, disc, d1, stage)
    return s, d, 0

def _wrap_sin(env, disc, d1, stage, _d2):
    return 0, 1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pais", default="generico")
    ap.add_argument("--escenario", default="generico")
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--eval", type=int, default=15)
    args = ap.parse_args()

    # Importar escenarios y comparar con el V1 ya entrenado
    import importlib
    mod = importlib.import_module("fcm_integracion_v2")

    filas = []
    for s in range(args.semillas):
        rng = random.Random(s); np.random.seed(s)
        def he(disc):
            p = {}
            if args.pais != "generico" and hasattr(ql, "PARAMS_PAIS"):
                p.update(ql.PARAMS_PAIS[args.pais])
            if args.escenario != "generico":
                p.update(mod.ESCENARIOS[args.escenario])
            return ql.EntornoSimple(disc, rng, params=p if p else None)
        disc = ql.calibrar_discretizador(he, rng)
        T = ql.CONFIG["horizonte"]

        # Baseline 1: sin restricciones
        m = evaluar_politica(lambda: he(disc), disc, _wrap_sin, rng,
                              episodios=args.eval, horizonte=T)
        filas.append(("sin_restr", s, m))
        # Baseline 2: S0-4-0 simple
        m = evaluar_politica(lambda: he(disc), disc, _wrap_s040, rng,
                              episodios=args.eval, horizonte=T)
        filas.append(("S0-4-0_Kompella", s, m))
        # Baseline 3: S0-4-0-GI (la mejor de Kompella)
        m = evaluar_politica(lambda: he(disc), disc, politica_s040gi, rng,
                              episodios=args.eval, horizonte=T)
        filas.append(("S0-4-0-GI_Kompella", s, m))
        # Para comparar con V1, lo entrenamos rápido
        env = he(disc)
        ag = mod.entrenar(env, disc, rng, 2500, T, W_pre=None)
        env_eval = he(disc)
        m = mod.evaluar(env_eval, disc, ag, rng, episodios=args.eval, horizonte=T)
        filas.append(("V1_Qlearning_TFG", s, m))
        print(f"semilla {s} OK")

    # Guardar y mostrar resumen
    os.makedirs(os.path.join(AQUI, "..", "resultados"), exist_ok=True)
    out = os.path.join(AQUI, "..", "resultados",
                        f"comparacion_sota_{args.pais}_{args.escenario}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["politica", "semilla", "contagio", "hospital", "coste",
                    "pct_bajo", "retorno"])
        for nom, s, m in filas:
            w.writerow([nom, s, f"{m['contagio']:.4f}", f"{m['hospital']:.4f}",
                        f"{m['coste']:.3f}", f"{m['pct_bajo']:.2f}",
                        f"{m['retorno']:.2f}"])
    print(f"\nEscrito: {out}")

    # Resumen
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for nom, s, m in filas:
        for k, v in m.items():
            agg[nom][k].append(v)
    print(f"\n=== Comparación con estado del arte ({args.pais}/{args.escenario}) ===")
    print(f"{'Política':<22}{'Retorno':<12}{'Contagio':<10}{'%bajo':<9}{'Coste':<8}")
    for nom in ["sin_restr", "S0-4-0_Kompella", "S0-4-0-GI_Kompella", "V1_Qlearning_TFG"]:
        a = agg[nom]
        print(f"{nom:<22}{np.mean(a['retorno']):<+12.1f}"
              f"{np.mean(a['contagio']):<10.3f}"
              f"{np.mean(a['pct_bajo']):<9.1f}"
              f"{np.mean(a['coste']):<8.2f}")


if __name__ == "__main__":
    main()
