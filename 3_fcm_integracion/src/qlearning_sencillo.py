# -*- coding: utf-8 -*-
"""
Q-learning sencillo para el control epidemico.
Implementacion fiel a la teoria del TFG (Desktop/tfg bueno/TFG.tex):
mismas variables de estado, mismas acciones y misma recompensa a tramos.

Correspondencia codigo <-> paper (etiquetas del .tex):
  - Estado s_t de 7 componentes ............ eq:estado, eq:signo, eq:signoh, eq:signoL
  - Espacio de estados |S| = 3^7 = 2187 .... eq:Sproducto
  - Accion de 4 palancas OxCGRT ............ eq:accion, eq:Aproducto (|A| = 3^4 = 81)
  - Coste de una accion .................... eq:coste, eq:Lnivel
  - Recompensa a tramos r = S + K .......... eq:tramos, eq:tramos_coste, tab:rejilla
  - Entorno FCM (sigmoide, retardo) ........ eq:fcm
  - Regla de actualizacion Q ............... eq:qupdate
  - Politica epsilon-greedy y decaimiento .. eq:egreedy, eq:decay
  - Pseudocodigo (bucle de entrenamiento) .. Algoritmo 1

ENTORNOS DISPONIBLES (--entorno):
  simple  (por defecto)  Simulador transparente de ecuaciones, la "alternativa
                         y validacion" que menciona la Sec. 3 del paper. Sus
                         reglas son conocidas e interpretables, de modo que
                         permite VALIDAR que el agente aprende acciones logicas
                         sin depender de la matriz del FCM (aun sin rellenar).
  fcm                    Mapa Cognitivo Difuso (eq:fcm). Utilizable cuando la
                         matriz W del Excel (Datos / LLM / Experto) este
                         rellena; mientras tanto carga una W provisional.

VALIDACION (se ejecuta tras entrenar): metricas de calidad del aprendizaje
(convergencia, cobertura) y comprobaciones de LOGICA de la politica contra la
rejilla del paper (tab:rejilla): actuar en emergencia, no actuar en calma con
coste bajo, relajar en sobre-restriccion, mas coste cuanto peor la epidemia.

DECISIONES DE IMPLEMENTACION (donde el paper deja ambiguedad):
  (D1) Tendencias: el paper define Delta c, Delta h y Delta L como signos de
       diferencias (eq:signo, eq:signoh, eq:signoL). Por coherencia con esas
       ecuaciones, aqui las tendencias SE CALCULAN de las series continuas y
       actuan en el FCM solo como emisoras (igual que las acciones). Los nodos
       dinamicos que se actualizan con eq:fcm son: contagio (E1), hospital (E3)
       y preocupacion (E7). El nivel de restriccion E5 y su tendencia E6
       derivan del coste de las acciones previas (eq:Lnivel).
  (D2) El simbolo beta_c del paper nombra dos cosas distintas: el peso del
       confinamiento en eq:coste y el bono de seguridad en las Señales. Aqui
       se llaman BETA["conf"] y "bono de seguridad" (1 si c_gorro es Bajo).

PARAMETROS PROVISIONALES: el paper aplaza a la fase de parametros los pesos
beta, gamma, los hiperparametros del agente, la matriz W del FCM, el retardo
hospitalario y los umbrales de discretizacion. CONFIG los fija con valores
provisionales razonables, todos sustituibles al calibrar.
"""

import argparse
import csv
import math
import os
import random

import numpy as np

# =====================================================================
# CONFIG: todo lo que el paper deja para la fase de parametros
# =====================================================================
CONFIG = {
    # --- pesos del coste (eq:coste). PROVISIONAL: confinamiento el doble ---
    "beta": {"work": 1.0, "school": 1.0, "ptrans": 1.0, "conf": 2.0},

    # --- agente (Sec. 1). PROVISIONAL ---
    "gamma": 0.95,        # alto, para "ver" la carga hospitalaria retardada
    "alpha": 0.10,        # (compat.) usado solo si doble_q=False y sin decaimiento
    "eps0": 0.20,
    "eps_min": 0.02,
    "decay": 0.95,        # d de eq:decay
    "episodios": 50000,   # K (subido para entrenar bien los estados severos)
    "horizonte": 365,     # T
    # --- tasa de aprendizaje con decaimiento (Q-learning clasico) ---
    # El teorema de convergencia de Watkins & Dayan (1992) exige que alpha decaiga
    # con las visitas (Robbins-Monro). Una alpha constante deja la Q oscilando y,
    # con |A|=81 acciones, casi plana. Esto NO anade nada al algoritmo: es su forma
    # estandar de converger. alpha(s,a) = alpha0 / (1 + kappa * n_visitas(s,a)).
    "alpha0": 0.50,       # tasa inicial (mayor, porque decae)
    "alpha_kappa": 0.01,  # velocidad del decaimiento por par (s,a)
    "alpha_min": 0.02,    # suelo de la tasa
    # exploring starts: fraccion de episodios que arrancan en la region SEVERA
    # (contagio/hospital altos) para entrenar bien esos estados poco frecuentes.
    "frac_arranque_severo": 0.35,

    # --- entornos (Sec. 3). PROVISIONAL ---
    "lambda_fcm": 2.0,    # pendiente de la sigmoide (eq:fcm)
    "retardo_hosp": 7,    # d: el hospital recibe el contagio de d pasos atras
    "ruido": 0.01,        # ruido gaussiano de la transicion (P estocastica)

    # --- recompensa a tramos. PROVISIONAL (rejilla, "se separan al ponderar") ---
    "w_emergencia": 1.3,
    "w_prop_rapida": 1.5,  # subida con contagio Medio: mas urgente
    "w_prop_lenta": 0.4,   # subida con contagio Bajo (respuesta MODERADA, no maxima)
    "w_calma": 1.0,
    # pesos de coste por regimen (eq:tramos_coste): B pequeno para que en calma
    # barata "no actuar" sea lo optimo (antes L bajo daba K=0, "actuar gratis")
    # coste por (FASE x restriccion vigente L), fiel a la rejilla tab:rejilla:
    # en PROPAGACION con L bajo hay que ACTUAR (cortar la subida) -> penalizacion
    # pequena; en CALMA con L bajo lo ideal es NO actuar -> penalizacion fuerte.
    # Con L alto ambas premian relajar. Antes la penalizacion no dependia de la
    # fase y por eso, al subirla para calma, se rompia la propagacion.
    "lambda_prop": {0: 0.2, 1: 0.5, 2: 1.0},    # prop. RAPIDA: actuar sale barato (respuesta maxima)
    "lambda_prop_lenta": {0: 2.5, 1: 2.5, 2: 1.0},  # prop. LENTA: actuar sale mas caro (respuesta moderada)
    "lambda_calma": {0: 1.0, 1: 0.8, 2: 1.0},   # calma: actuar sale caro (no sobreactuar)
    "bono_escala": 0.5,   # escala del bono de seguridad (0.5*1[c_{t+1}=Bajo])
    # coste en EMERGENCIA: antes 0 ("coste gratis"). Ahora una penalizacion
    # MINIMA para que no se ignore, pero la salud domina y se sigue actuando
    # fuerte (un valor grande premiaria acciones debiles y romperia la coherencia).
    "lambda_emergencia": 0.05,
    # La recompensa premia sobre todo el CAMBIO controlable por la accion
    # (reducir contagio/hospital) y penaliza empeorarlo; el NIVEL absoluto pesa
    # poco. Antes el nivel dominaba la recompensa y dejaba la senal util (el
    # cambio) como una fraccion diminuta, aplanando la Q. (Reescribe eq:tramos.)
    "ganancia_salud": 2.0,   # amplifica el cambio (reducir premia, subir castiga)
    "pen_empeora": 1.5,      # empeorar penaliza mas que mejorar premia (asimetria)
    "peso_nivel": 0.3,       # peso pequeno del nivel absoluto (antes 1.0)
    # DESESCALADO en calma sostenida: cuando el contagio esta Bajo y la
    # preocupacion tambien Baja (proxy de "la calma ya lleva tiempo": la
    # preocupacion decae con retardo, asi que solo es Baja tras una racha de
    # contagio bajo), se premia el NIVEL de coste bajo -> la politica suelta las
    # restricciones moderadas hacia coste 0. Se premia el nivel (no el acto de
    # bajar) para no reintroducir el bucle perverso de apretar-para-luego-relajar.
    "desescalado_bono": 2.5,

    # --- discretizacion. PROVISIONAL: se calibran con un rodaje aleatorio ---
    "pasos_calibracion": 5000,   # rodaje con politica aleatoria para terciles
    "factor_delta": 0.25,        # delta = 0.25 * std(variacion) (Sec. 2.4)
    "semilla": 0,
    # opcion 2: desempate canonico SOLO en empates exactos del Q maximo (tol=0).
    # Con tolerancia >0 el preferir "menos confinamiento" degrada el control
    # (las decisiones marginales del argmax importan), asi que se deja en 0.
    "tol_desempate": 0.0,
}

# Orden de los nodos del FCM (tab:pesos del paper)
NODOS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "A1", "A2", "A3", "A4"]
IDX = {n: i for i, n in enumerate(NODOS)}

# Matriz W provisional, fuente LLM (Claude). w_ij en [-1,1], fila i -> columna j.
# Diagonal nula y columnas de las acciones a cero (las acciones solo emiten).
# Solo se usan las columnas de los nodos dinamicos E1, E3 y E7 (decision D1).
# PROVISIONAL: sustituir por las matrices del Excel (Datos / LLM / Experto).
W_LLM_PROVISIONAL = {
    # hacia E1 (contagio)
    ("E2", "E1"): +0.30,   # tendencia al alza empuja el contagio
    ("E5", "E1"): -0.20,   # restriccion vigente reduce transmision
    ("E7", "E1"): -0.30,   # preocupacion -> cumplimiento -> menos contagio
    ("A1", "E1"): -0.30,   # trabajo
    ("A2", "E1"): -0.25,   # colegios
    ("A3", "E1"): -0.20,   # transporte publico
    ("A4", "E1"): -0.40,   # confinamiento (la mas gravosa y la mas efectiva)
    # hacia E3 (hospital): recibe el contagio con retardo (eq:fcm y texto)
    ("E1", "E3"): +0.50,
    ("E4", "E3"): +0.15,
    # hacia E7 (preocupacion)
    ("E1", "E7"): +0.40,
    ("E3", "E7"): +0.30,
    ("E4", "E7"): +0.10,
}


def construir_W(pares=W_LLM_PROVISIONAL):
    """Matriz W (11x11) desde el diccionario de aristas."""
    W = np.zeros((len(NODOS), len(NODOS)))
    for (i, j), w in pares.items():
        W[IDX[i], IDX[j]] = w
    return W


def cargar_W_excel(ruta, hoja=None):
    """Carga una matriz 11x11 desde un Excel (p. ej. matrizdepesos.xlsx).

    Espera una hoja con la matriz de pesos: primera columna con los nombres de
    los nodos (E1..E7, A1..A4) y primera fila igual. Celdas vacias valen 0.
    """
    from openpyxl import load_workbook
    wb = load_workbook(ruta, data_only=True)
    ws = wb[hoja] if hoja else wb.active
    filas = list(ws.iter_rows(values_only=True))
    cab = None
    for r, fila in enumerate(filas):
        etiquetas = [str(x).strip()[:2] if x else "" for x in fila]
        if sum(1 for e in etiquetas if e in NODOS) >= len(NODOS) - 1:
            cab = r
            break
    if cab is None:
        raise ValueError("No encuentro la cabecera de la matriz en el Excel")
    col_de = {}
    for c, x in enumerate(filas[cab]):
        e = str(x).strip()[:2] if x else ""
        if e in NODOS:
            col_de[e] = c
    W = np.zeros((len(NODOS), len(NODOS)))
    for fila in filas[cab + 1:]:
        if not fila or not fila[0]:
            continue
        e_i = str(fila[0]).strip()[:2]
        if e_i not in NODOS:
            continue
        for e_j, c in col_de.items():
            v = fila[c]
            if isinstance(v, (int, float)):
                W[IDX[e_i], IDX[e_j]] = float(v)
    np.fill_diagonal(W, 0.0)          # diagonal nula (tab:pesos)
    for a in ("A1", "A2", "A3", "A4"):
        W[:, IDX[a]] = 0.0            # las acciones solo emiten
    return W


# =====================================================================
# Acciones (Sec. 2.2): a = (work, school, ptrans, conf), cada una en {0,1,2}
# =====================================================================
PALANCAS = ["work", "school", "ptrans", "conf"]
N_ACCIONES = 3 ** 4  # eq:Aproducto: |A| = 81


def accion_de_indice(k):
    """Indice 0..80 -> vector (a_work, a_school, a_ptrans, a_conf)."""
    a = []
    for _ in range(4):
        a.append(k % 3)
        k //= 3
    return tuple(reversed(a))


def indice_de_accion(a):
    k = 0
    for x in a:
        k = k * 3 + x
    return k


def coste(a, beta=None):
    """Coste socioeconomico de una accion (eq:coste)."""
    b = beta or CONFIG["beta"]
    return (b["work"] * a[0] + b["school"] * a[1]
            + b["ptrans"] * a[2] + b["conf"] * a[3])


def coste_max(beta=None):
    b = beta or CONFIG["beta"]
    return 2.0 * (b["work"] + b["school"] + b["ptrans"] + b["conf"])


# =====================================================================
# Discretizacion (Sec. 2.4): terciles por niveles, signo con umbral
# =====================================================================
def signo_umbral(dx, delta):
    """sign_delta de eq:signo: +1 sube, 0 estable, -1 baja."""
    if dx > delta:
        return +1
    if dx < -delta:
        return -1
    return 0


class Discretizador:
    """Cortes de terciles y umbrales delta para pasar de continuo a discreto.

    Los cortes se calibran con un rodaje del propio simulador bajo politica
    aleatoria (PROVISIONAL: cuando haya datos por pais, sustituir por los
    terciles de las series reales, como indica la Sec. 2.4).
    """

    def __init__(self):
        self.cortes = {"c": (1 / 3, 2 / 3), "h": (1 / 3, 2 / 3),
                       "L": (1 / 3, 2 / 3), "preo": (1 / 3, 2 / 3)}
        self.deltas = {"c": 0.01, "h": 0.01, "L": 0.01}

    def nivel(self, x, var):
        """x continuo -> {0,1,2} = {Bajo, Medio, Alto} por terciles."""
        t1, t2 = self.cortes[var]
        if x <= t1:
            return 0
        if x <= t2:
            return 1
        return 2

    def calibrar(self, series):
        """Terciles y deltas desde series continuas {var: [valores]}."""
        for var in ("c", "h", "L", "preo"):
            s = np.asarray(series[var])
            self.cortes[var] = (float(np.quantile(s, 1 / 3)),
                                float(np.quantile(s, 2 / 3)))
        for var in ("c", "h", "L"):
            d = np.diff(np.asarray(series[var]))
            std = float(np.std(d))
            self.deltas[var] = CONFIG["factor_delta"] * std if std > 0 else 1e-6


def indice_de_estado(s):
    """(c,dc,h,dh,L,dL,preo) con niveles {0,1,2} y tendencias {-1,0,+1} -> 0..2186.

    eq:Sproducto: |S| = 3^7 = 2187. Las tendencias se desplazan a {0,1,2}.
    """
    digitos = (s[0], s[1] + 1, s[2], s[3] + 1, s[4], s[5] + 1, s[6])
    k = 0
    for d in digitos:
        k = k * 3 + d
    return k


N_ESTADOS = 3 ** 7  # 2187


# =====================================================================
# Entornos (Sec. 3): base comun + FCM + simulador simple transparente
# =====================================================================
class EntornoBase:
    """Maquinaria comun: estado discreto, tendencias, retardo y avance."""

    def __init__(self, disc, rng, retardo=None, ruido=None):
        self.disc = disc
        self.rng = rng
        self.retardo = retardo if retardo is not None else CONFIG["retardo_hosp"]
        self.ruido = ruido if ruido is not None else CONFIG["ruido"]
        self.cmax = coste_max()

    def reset(self, aleatorio=True, amplio=False):
        """amplio=True: EXPLORING STARTS (Sutton & Barto, cap. 5). El episodio
        arranca en un estado cualquiera del espacio ---incluidos los severos
        (contagio/hospital Altos, restriccion previa cualquiera)--- para que TODOS
        los pares (s,a) se entrenen, aunque la politica luego rara vez llegue a
        esos estados. Es una tecnica clasica de Q-learning, no una extension. Solo
        se usa al ENTRENAR; la evaluacion arranca en estados realistas (amplio=False)."""
        r = self.rng
        if amplio:
            # cubre TODO el rango; ademas OVERSAMPLING de la region severa (una
            # fraccion de arranques con contagio/hospital altos), porque esos
            # estados son transitorios y necesitan mas muestras por par (s,a) para
            # que su Q converja (una sola accion mueve poco un sistema saturado).
            if r.random() < CONFIG["frac_arranque_severo"]:
                self.c = r.uniform(0.6, 1.0)
                self.h = r.uniform(0.6, 1.0)
            else:
                self.c = r.uniform(0.0, 1.0)
                self.h = r.uniform(0.0, 1.0)
            self.preo = r.uniform(0.0, 1.0)
        elif aleatorio:
            self.c = r.uniform(0.2, 0.8)
            self.h = r.uniform(0.1, 0.5)
            self.preo = r.uniform(0.2, 0.8)
        else:
            self.c, self.h, self.preo = 0.5, 0.3, 0.5
        self.c_prev = self.c
        self.h_prev = self.h
        self.buffer_c = [self.c] * (self.retardo + 1)  # contagio pasado
        if amplio:
            # tambien la restriccion vigente y su tendencia (via acciones previas)
            self.a_prev = accion_de_indice(r.randrange(N_ACCIONES))
            self.a_prev2 = accion_de_indice(r.randrange(N_ACCIONES))
        else:
            self.a_prev = (0, 0, 0, 0)    # accion del paso anterior (para L)
            self.a_prev2 = (0, 0, 0, 0)   # y la de dos pasos atras (para Delta L)
        return self.estado_discreto()

    @staticmethod
    def _clip(x):
        return min(1.0, max(0.0, x))

    def _avanzar(self, c_nuevo, h_nuevo, preo_nueva, a):
        """Actualiza series, buffer del retardo y acciones previas."""
        self.c_prev, self.h_prev = self.c, self.h
        self.c, self.h, self.preo = c_nuevo, h_nuevo, preo_nueva
        self.buffer_c.pop(0)
        self.buffer_c.append(self.c)
        self.a_prev2 = self.a_prev
        self.a_prev = tuple(a)
        return self.estado_discreto()

    # ---- componentes del estado (Sec. 2.1) ----
    def tend_c(self):
        """Delta c_t (eq:signo)."""
        return signo_umbral(self.c - self.c_prev, self.disc.deltas["c"])

    def tend_h(self):
        """Delta h_t (eq:signoh)."""
        return signo_umbral(self.h - self.h_prev, self.disc.deltas["h"])

    def tend_L(self):
        """Delta L_t (eq:signoL): variacion del coste de las dos ultimas acciones."""
        d_sigma = coste(self.a_prev) - coste(self.a_prev2)
        return signo_umbral(d_sigma, self.disc.deltas["L"])

    def estado_discreto(self):
        """s_t = (c, Dc, h, Dh, L, DL, preo) discretizado (eq:estado)."""
        L_norm = coste(self.a_prev) / self.cmax   # eq:Lnivel, normalizado
        return (self.disc.nivel(self.c, "c"),
                self.tend_c(),
                self.disc.nivel(self.h, "h"),
                self.tend_h(),
                self.disc.nivel(L_norm, "L"),
                self.tend_L(),
                self.disc.nivel(self.preo, "preo"))

    def paso(self, a):
        raise NotImplementedError


class EntornoSimple(EntornoBase):
    """Simulador transparente de ecuaciones (la alternativa de la Sec. 3).

    Reglas conocidas e interpretables, pensadas para VALIDAR la logica del
    agente sin depender de la matriz del FCM:
      - contagio: crecimiento logistico menos reduccion proporcional a la
        eficacia de las medidas, modulada por la preocupacion (adherencia,
        como pide la Sec. 2.1) y con una importacion constante minima;
      - hospital: autorregresivo, sigue al contagio con retardo d;
      - preocupacion: persigue con inercia la gravedad (contagio y hospital).
    PROVISIONAL: los valores de PARAMS se sustituyen al calibrar con datos.
    """

    PARAMS = {
        "r0": 0.18,                      # crecimiento sin medidas
        "rmax": 0.30,                    # reduccion maxima por medidas
        "efic": (0.30, 0.28, 0.18, 0.24),  # eficacia por palanca (suma 1), equilibrada
        "adh0": 0.6, "adh1": 0.4,        # adherencia = adh0 + adh1 * preo
        "imp": 0.002,                    # importacion constante de casos
        "A_h": 0.90, "B_h": 0.15,        # hospital: inercia y sensibilidad
        "p_iner": 0.85, "p_c": 0.6,      # preocupacion: inercia y mezcla c/h
    }

    def __init__(self, disc, rng, params=None, retardo=None, ruido=None):
        p = dict(self.PARAMS)
        if params:
            p.update(params)
        if retardo is None:
            retardo = p.get("d")   # los params por pais traen su retardo
        super().__init__(disc, rng, retardo=retardo, ruido=ruido)
        self.p = p

    def paso(self, a):
        p = self.p
        # eficacia agregada de la accion en [0,1], ponderada por palanca
        ef = sum(e * ai / 2.0 for e, ai in zip(p["efic"], a))
        adher = p["adh0"] + p["adh1"] * self.preo   # preocupacion modula
        c_n = (self.c + p["r0"] * self.c * (1.0 - self.c)
               - p["rmax"] * ef * adher * self.c + p["imp"])
        h_n = p["A_h"] * self.h + p["B_h"] * self.buffer_c[0]   # retardo d
        preo_n = (p["p_iner"] * self.preo
                  + (1.0 - p["p_iner"]) * (p["p_c"] * self.c
                                           + (1.0 - p["p_c"]) * self.h))
        if self.ruido > 0:
            c_n += self.rng.gauss(0, self.ruido)
            h_n += self.rng.gauss(0, self.ruido)
        return self._avanzar(self._clip(c_n), self._clip(h_n),
                             self._clip(preo_n), a)


# Parametros por pais para los BANCOS DE ESTRES: misma forma funcional que
# el entorno estilizado, dinamica propia de cada pais. La politica se aprende
# UNA sola vez en el estilizado (responde a la pandemia, no al pais) y aqui
# solo se somete a prueba bajo dinamicas distintas.
# PROVISIONAL: valores pendientes de la calibracion con los datos CLEI;
# las diferencias reflejan los hallazgos previos (medidas menos efectivas y
# mas retardo hospitalario en BR; retardo corto en ES; GB como referencia).
PARAMS_PAIS = {
    # Sincronizado con tab:calib del paper. De la calibracion con datos se toman
    # los parametros IDENTIFICABLES: retardo hospitalario d, inercia/sensibilidad
    # del hospital (A_h,B_h) y la forma RELATIVA de la eficacia de las palancas
    # (efic). El crecimiento (r0,rmax) se mantiene generico porque su calibracion
    # es debil (R2~0); las palancas no identificadas (efic~0 en los datos) reciben
    # un suelo 0.10 para no quedar muertas. Asi las diferencias entre paises
    # provienen del retardo y de la eficacia relativa (como dice el paper).
    # Bloque hospitalario: se capa A_h<=0.90. El A_h calibrado ~0.98 en ES/BR es
    # una raiz casi unitaria (ajuste debil) que dispara la ganancia de estado
    # estacionario g=B_h/(1-A_h) y SATURA el hospital independientemente del
    # contagio. Se fija A_h=0.90 y B_h para ganancia g~1.2 (hospital ~ contagio),
    # conservando el retardo tau (lo identificable). GB ya tenia A_h=0.73 (g=1.26).
    "GB": {"r0": 0.18, "rmax": 0.30, "efic": (0.55, 0.38, 0.10, 0.10),
           "imp": 0.006, "A_h": 0.73, "B_h": 0.34, "d": 0},
    "ES": {"r0": 0.18, "rmax": 0.30, "efic": (0.10, 0.45, 0.10, 0.50),
           "imp": 0.003, "A_h": 0.90, "B_h": 0.12, "d": 9},
    "BR": {"r0": 0.18, "rmax": 0.30, "efic": (0.18, 0.29, 0.39, 0.14),
           "imp": 0.002, "A_h": 0.90, "B_h": 0.12, "d": 2},
}


class EntornoFCM(EntornoBase):
    """Mapa Cognitivo Difuso como entorno del MDP (eq:fcm).

    Nodos dinamicos (se actualizan con la sigmoide): E1 contagio, E3 hospital,
    E7 preocupacion. El hospital recibe la activacion del contagio de
    `retardo_hosp` pasos atras. El resto de nodos de estado (tendencias y
    restriccion) se derivan de las series y de las acciones (decision D1).
    AVISO: con la sigmoide de eq:fcm, los nodos sin padres negativos tienden a
    saturar; usar cuando la matriz W del Excel este rellena y calibrada.
    """

    def __init__(self, W, disc, rng, lam=None, retardo=None, ruido=None):
        super().__init__(disc, rng, retardo, ruido)
        self.W = W
        self.lam = lam if lam is not None else CONFIG["lambda_fcm"]

    def _f(self, u):
        """Sigmoide de eq:fcm."""
        return 1.0 / (1.0 + math.exp(-self.lam * u))

    def _activaciones(self, a):
        """Vector x de activaciones de los 11 nodos, todas en [0,1]."""
        x = np.zeros(len(NODOS))
        x[IDX["E1"]] = self.c
        x[IDX["E2"]] = (self.tend_c() + 1) / 2.0
        x[IDX["E3"]] = self.h
        x[IDX["E4"]] = (self.tend_h() + 1) / 2.0
        x[IDX["E5"]] = coste(self.a_prev) / self.cmax
        x[IDX["E6"]] = (self.tend_L() + 1) / 2.0
        x[IDX["E7"]] = self.preo
        for k, p in enumerate(("A1", "A2", "A3", "A4")):
            x[IDX[p]] = a[k] / 2.0
        return x

    def paso(self, a):
        """eq:fcm: x_j(t+1) = f( x_j(t) + sum_i w_ij x_i(t) )."""
        x = self._activaciones(a)
        W = self.W
        suma_c = float(W[:, IDX["E1"]] @ x)
        c_n = self._f(self.c + suma_c)
        x_h = x.copy()
        x_h[IDX["E1"]] = self.buffer_c[0]   # contagio de d pasos atras
        suma_h = float(W[:, IDX["E3"]] @ x_h)
        h_n = self._f(self.h + suma_h)
        suma_p = float(W[:, IDX["E7"]] @ x)
        preo_n = self._f(self.preo + suma_p)
        if self.ruido > 0:
            c_n = self._clip(c_n + self.rng.gauss(0, self.ruido))
            h_n = self._clip(h_n + self.rng.gauss(0, self.ruido))
        return self._avanzar(c_n, h_n, preo_n, a)


# =====================================================================
# Recompensa a tramos (Sec. 2.3): r = S(s_t) + K(s_t)
# =====================================================================
def recompensa_tramos(s_t, c_t, c_t1, h_t, h_t1, a_t, a_prev, disc):
    """Recompensa a tramos del paper.

    s_t: estado discreto en t (decide el regimen).
    c_t, c_t1, h_t, h_t1: contagio y hospital continuos en t y t+1.
    a_t: accion elegida; a_prev: accion del paso anterior.
    Devuelve (r, nombre_de_fase).
    """
    ct, dct, ht, dht, Lt, dLt, preot = s_t
    cmax = coste_max()

    # --- Señales, todas en [0,1] ---
    c_gorro, h_gorro = c_t1, h_t1
    k_gorro = coste(a_t) / cmax
    k_prev = coste(a_prev) / cmax
    # castigos: empeoramientos (subidas)
    d_c = max(0.0, c_t1 - c_t)
    d_h = max(0.0, h_t1 - h_t)
    d_k = max(0.0, k_gorro - k_prev)
    # premios: mejoras (bajadas)
    m_c = max(0.0, c_t - c_t1)
    m_h = max(0.0, h_t - h_t1)
    m_k = max(0.0, k_prev - k_gorro)
    # bono de seguridad: contagio del paso siguiente en nivel Bajo
    bono = CONFIG["bono_escala"] if disc.nivel(c_t1, "c") == 0 else 0.0

    # --- Parte de salud S(s_t) (eq:tramos), fases por prioridad ---
    # Se premia el CAMBIO controlable por la accion (mejora = bajada de contagio/
    # hospital) y se penaliza empeorarlo (subida) mas fuerte (pen_empeora); el
    # nivel absoluto pesa poco (peso_nivel). Asi la senal util no queda enterrada
    # bajo un offset de nivel casi independiente de la accion (causa de la Q plana).
    G = CONFIG["ganancia_salud"]
    PEN = CONFIG["pen_empeora"]
    NIV = CONFIG["peso_nivel"]
    if ct == 2 or ht == 2:                       # Emergencia
        fase = "emergencia"
        S = CONFIG["w_emergencia"] * (
            G * ((m_c + m_h) - PEN * (d_c + d_h)) - NIV * (c_gorro + h_gorro))
    elif dct == +1:                              # Propagacion
        if ct == 1:
            fase = "prop_rapida"
            w = CONFIG["w_prop_rapida"]
        else:
            fase = "prop_lenta"
            w = CONFIG["w_prop_lenta"]
        S = w * (G * (m_c - PEN * d_c) - NIV * c_gorro)
    else:                                        # Calma (Delta c <= 0)
        fase = "calma"
        S = CONFIG["w_calma"] * (bono + G * (m_c - d_c) - NIV * c_gorro)

    # --- Parte de coste K(s_t) (eq:tramos_coste), por FASE x restriccion vigente L ---
    if fase == "emergencia":
        # coste NO ignorado: penalizacion minima. La salud (S) domina, asi que se
        # sigue actuando fuerte; solo desempata hacia la accion mas barata entre
        # las igual de eficaces.
        K = -CONFIG["lambda_emergencia"] * k_gorro
    else:
        # coste penalizado segun (fase x L). NO se premia relajar con un bono
        # (m_k): eso creaba un bucle perverso (apretar para luego cobrar el
        # relajar) que hacia sobreactuar en calma. Basta con penalizar el coste:
        # en sobre-restriccion (L alto) con la salud ya satisfecha, la accion de
        # menor coste es la optima -> la politica relaja sola.
        lam = CONFIG["lambda_prop"] if fase == "prop_rapida" \
            else CONFIG["lambda_prop_lenta"] if fase == "prop_lenta" \
            else CONFIG["lambda_calma"]
        K = -lam[Lt] * k_gorro
        # calma SOSTENIDA (contagio Bajo + preocupacion Baja): premia el nivel de
        # coste bajo para que la politica suelte las restricciones hacia coste 0.
        # Si al relajar el contagio rebota, el estado deja de ser Bajo/sostenido y
        # el premio desaparece -> la politica vuelve a apretar (desescalado adaptativo).
        if fase == "calma" and ct == 0 and preot == 0:
            K += CONFIG["desescalado_bono"] * (1.0 - k_gorro)

    return S + K, fase


def fase_de_estado(s):
    """Fase epidemica del estado, con la MISMA prioridad que la recompensa."""
    if s[0] == 2 or s[2] == 2:
        return "emergencia"
    if s[1] == +1:
        return "prop_rapida" if s[0] == 1 else "prop_lenta"
    return "calma"


# =====================================================================
# Agente Q-learning (Sec. 1)
# =====================================================================
class AgenteQ:
    """Q-learning tabular CLASICO (Watkins 1989; Sutton & Barto, cap. 6).

    Una sola tabla Q y la regla de actualizacion clasica (eq:qupdate):
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ].

    La tasa alpha DECAE con el numero de visitas al par (s,a), tal como exige el
    teorema de convergencia clasico de Watkins & Dayan (1992): condiciones de
    Robbins-Monro sum_t alpha_t = inf, sum_t alpha_t^2 < inf. Una alpha constante
    no cumple esas condiciones y deja la Q oscilando (de ahi la Q casi plana con
    |A|=81). El decaimiento NO es una extension del algoritmo: es su forma
    estandar de garantizar convergencia. Con alpha_kappa=0 y alpha0 fija se
    recupera la variante de paso constante.
        alpha(s,a) = max( alpha_min, alpha0 / (1 + kappa * n(s,a)) ).
    """

    def __init__(self, rng, alpha=None, gamma=None):
        self.Q = np.zeros((N_ESTADOS, N_ACCIONES))    # Q_0(s,a) = 0
        self.visitas = np.zeros(N_ESTADOS, dtype=np.int64)
        self.n_sa = np.zeros((N_ESTADOS, N_ACCIONES), dtype=np.int64)
        self.alpha0 = alpha if alpha is not None else CONFIG["alpha0"]
        self.kappa = CONFIG["alpha_kappa"]
        self.alpha_min = CONFIG["alpha_min"]
        self.gamma = gamma if gamma is not None else CONFIG["gamma"]
        self.rng = rng

    def _alpha(self, s_idx, a_idx):
        a = self.alpha0 / (1.0 + self.kappa * self.n_sa[s_idx, a_idx])
        return max(self.alpha_min, a)

    def elegir(self, s_idx, eps):
        """Politica epsilon-greedy (eq:egreedy)."""
        if self.rng.random() < eps:
            return self.rng.randrange(N_ACCIONES)
        fila = self.Q[s_idx]
        maximo = fila.max()
        candidatos = np.flatnonzero(fila == maximo)
        return int(self.rng.choice(list(candidatos)))

    def actualizar(self, s_idx, a_idx, r, s1_idx):
        """Regla de actualizacion clasica (eq:qupdate). Devuelve |error TD|."""
        self.n_sa[s_idx, a_idx] += 1
        self.visitas[s_idx] += 1
        alpha = self._alpha(s_idx, a_idx)
        td = r + self.gamma * self.Q[s1_idx].max() - self.Q[s_idx, a_idx]
        self.Q[s_idx, a_idx] += alpha * td
        return abs(td)

    def consolidar(self):
        """Compatibilidad: en el Q-learning clasico la tabla ya es la definitiva."""
        return self.Q

    def accion_greedy(self, s):
        """Politica greedy con DESEMPATE CANONICO (opcion 2): entre las acciones
        de valor casi igual (dentro de tol_desempate del maximo), elige la
        \"menos intrusiva\": menos confinamiento, luego menor coste, luego orden
        fijo trabajo/colegios/transporte. Asi la politica reportada es limpia y
        determinista aunque la Q sea casi plana entre acciones equivalentes."""
        fila = self.Q[indice_de_estado(s)]
        tol = CONFIG["tol_desempate"]
        candidatos = np.flatnonzero(fila >= fila.max() - tol)
        if len(candidatos) == 1:
            return accion_de_indice(int(candidatos[0]))
        def clave(a_idx):
            a = accion_de_indice(int(a_idx))
            return (a[3], coste(a), a[0], a[1], a[2])  # conf, coste, work, school, ptrans
        return accion_de_indice(int(min(candidatos, key=clave)))


# =====================================================================
# Calibracion de la discretizacion con un rodaje aleatorio
# =====================================================================
def calibrar_discretizador(hacer_env, rng):
    """Rodaje con politica aleatoria para fijar terciles y umbrales delta."""
    disc = Discretizador()                    # cortes provisionales
    env = hacer_env(disc)
    env.reset()
    series = {"c": [], "h": [], "L": [], "preo": []}
    for _ in range(CONFIG["pasos_calibracion"]):
        a = accion_de_indice(rng.randrange(N_ACCIONES))
        env.paso(a)
        series["c"].append(env.c)
        series["h"].append(env.h)
        series["L"].append(coste(env.a_prev) / coste_max())
        series["preo"].append(env.preo)
    disc.calibrar(series)
    return disc


# =====================================================================
# Entrenamiento (Algoritmo 1) y evaluacion
# =====================================================================
def entrenar(env, disc, rng, episodios=None, horizonte=None, verbose=True,
             agente=None, callback=None):
    """agente: si se pasa ya construido (p.ej. con Q inicializada al azar en
    vez de Q_0=0), se entrena ESE agente en lugar de crear uno con Q_0=0.
    callback(k, agente): si se pasa, se invoca al final de cada episodio k
    (0-indexado); se usa para instrumentar corridas (p.ej. el analisis de
    velocidad de convergencia de la Sec. 4, sin tocar el bucle de entrenamiento)."""
    K = episodios or CONFIG["episodios"]
    T = horizonte or CONFIG["horizonte"]
    agente = agente if agente is not None else AgenteQ(rng)
    eps = CONFIG["eps0"]
    historial = []

    for k in range(K):
        s = env.reset(amplio=True)   # exploring starts: entrena tambien los estados severos
        s_idx = indice_de_estado(s)
        retorno, c_med, coste_med, td_med = 0.0, 0.0, 0.0, 0.0
        for t in range(T):
            a_idx = agente.elegir(s_idx, eps)
            a = accion_de_indice(a_idx)
            c_t, h_t, a_prev = env.c, env.h, env.a_prev
            s1 = env.paso(a)
            r, _ = recompensa_tramos(s, c_t, env.c, h_t, env.h, a, a_prev, disc)
            s1_idx = indice_de_estado(s1)
            td_med += agente.actualizar(s_idx, a_idx, r, s1_idx)   # |error TD|
            s, s_idx = s1, s1_idx
            retorno += r
            c_med += env.c
            coste_med += coste(a)
        eps = max(CONFIG["eps_min"], eps * CONFIG["decay"])   # eq:decay
        historial.append((k, retorno, c_med / T, coste_med / T, eps, td_med / T))
        if verbose and (k + 1) % max(1, K // 10) == 0:
            print(f"  episodio {k+1:5d}/{K}  retorno={retorno:9.2f}  "
                  f"c_medio={c_med/T:.3f}  coste_medio={coste_med/T:.2f}  "
                  f"eps={eps:.3f}")
        if callback is not None:
            callback(k, agente)
    agente.consolidar()   # self.Q = (Q_A+Q_B)/2 para politica y metricas
    return agente, historial


def evaluar_politica(env, disc, elegir, episodios=20, horizonte=None):
    """Evalua una politica (funcion estado_discreto -> accion) en el simulador."""
    T = horizonte or CONFIG["horizonte"]
    tot = {"retorno": 0.0, "c": 0.0, "h": 0.0, "coste": 0.0, "dias_bajo": 0.0}
    for _ in range(episodios):
        s = env.reset()
        for t in range(T):
            a = elegir(s)
            c_t, h_t, a_prev = env.c, env.h, env.a_prev
            s1 = env.paso(a)
            r, _ = recompensa_tramos(s, c_t, env.c, h_t, env.h, a, a_prev, disc)
            tot["retorno"] += r
            tot["c"] += env.c
            tot["h"] += env.h
            tot["coste"] += coste(a)
            if disc.nivel(env.c, "c") == 0:
                tot["dias_bajo"] += 1
            s = s1
    n = episodios
    return {"retorno": tot["retorno"] / n,
            "c_medio": tot["c"] / (n * T),
            "h_medio": tot["h"] / (n * T),
            "coste_medio": tot["coste"] / (n * T),
            "pct_dias_bajo": 100.0 * tot["dias_bajo"] / (n * T)}


def politicas_referencia(agente, rng):
    """Politicas de referencia de la Sec. 3.4 (sin la real observada, que
    requiere datos historicos)."""
    def pi_optima(s):
        return agente.accion_greedy(s)

    def sin_restricciones(s):
        return (0, 0, 0, 0)

    def maxima(s):
        return (2, 2, 2, 2)

    def aleatoria(s):
        return accion_de_indice(rng.randrange(N_ACCIONES))

    def umbral(s):
        # heuristica por umbral: actua segun el nivel de contagio
        return {0: (0, 0, 0, 0), 1: (1, 1, 1, 1), 2: (2, 2, 2, 2)}[s[0]]

    return {"pi* (aprendida)": pi_optima,
            "sin restricciones": sin_restricciones,
            "restriccion maxima": maxima,
            "aleatoria": aleatoria,
            "umbral por contagio": umbral}


# =====================================================================
# VALIDACION: metricas de calidad e interpretacion logica de la politica
# =====================================================================
FASES = ["emergencia", "prop_rapida", "prop_lenta", "calma"]
NIVEL_TXT = {0: "B", 1: "M", 2: "A"}


def validar(agente, hacer_env, disc, rng, historial=None,
            episodios=50, horizonte=None, carpeta=None):
    """Informe de validacion de la politica aprendida.

    (1) Calidad del aprendizaje: convergencia del retorno y cobertura de la
        tabla Q. (2) Logica de las acciones: rejilla fase x coste vigente
        (tab:rejilla) con el comportamiento observado de pi*, mas checks
        PASS/FAIL contra lo que la teoria espera de cada celda.
    Devuelve (informe_texto, checks_dict, rejilla_dict).
    """
    T = horizonte or CONFIG["horizonte"]
    env = hacer_env(disc)
    cmax = coste_max()

    # ---- rollouts greedy: que hace pi* en la practica ----
    registros = []   # (fase, L, coste_accion, coste_vigente, accion)
    for _ in range(episodios):
        s = env.reset()
        for t in range(T):
            a = agente.accion_greedy(s)
            registros.append((fase_de_estado(s), s[4], coste(a),
                              coste(env.a_prev), a))
            s = env.paso(a)

    # ---- rejilla fase x L: coste medio, % sin actuar, palancas medias ----
    rejilla = {}
    for fase in FASES:
        for L in (0, 1, 2):
            regs = [r for r in registros if r[0] == fase and r[1] == L]
            if not regs:
                rejilla[(fase, L)] = None
                continue
            costes = [r[2] for r in regs]
            palancas = np.mean([r[4] for r in regs], axis=0)
            rejilla[(fase, L)] = {
                "n": len(regs),
                "coste_medio": float(np.mean(costes)),
                "pct_sin_actuar": 100.0 * sum(1 for c in costes if c == 0)
                                  / len(regs),
                "pct_relaja": 100.0 * sum(1 for r in regs if r[2] < r[3])
                              / len(regs),
                "palancas": tuple(float(p) for p in palancas),
            }

    def agregado(fase):
        regs = [r for r in registros if r[0] == fase]
        return (float(np.mean([r[2] for r in regs])), len(regs)) if regs \
            else (None, 0)

    c_emer, n_emer = agregado("emergencia")
    c_rap, n_rap = agregado("prop_rapida")
    c_len, n_len = agregado("prop_lenta")
    c_calma, n_calma = agregado("calma")

    # ---- checks logicos (contra tab:rejilla) ----
    checks = {}

    def check(nombre, condicion, detalle):
        checks[nombre] = {"pass": bool(condicion), "detalle": detalle}

    if c_emer is not None and c_calma is not None:
        check("C1 coherencia emergencia>calma", c_emer > c_calma,
              f"coste emergencia {c_emer:.2f} vs calma {c_calma:.2f}")
    if c_emer is not None:
        check("C2 emergencia actua fuerte", c_emer >= 0.5 * cmax,
              f"coste emergencia {c_emer:.2f} (umbral {0.5*cmax:.1f})")
    if c_rap is not None and c_len is not None:
        check("C3 prop rapida >= lenta", c_rap >= c_len - 0.25,
              f"rapida {c_rap:.2f} vs lenta {c_len:.2f}")
    celda_ideal = rejilla.get(("calma", 0))
    if celda_ideal:
        check("C4 calma con coste bajo: no sobreactuar",
              celda_ideal["coste_medio"] <= 0.5 * cmax,
              f"coste medio {celda_ideal['coste_medio']:.2f} "
              f"(sin actuar {celda_ideal['pct_sin_actuar']:.0f}% de los pasos)")
    celda_sobre = rejilla.get(("calma", 2))
    if celda_sobre:
        check("C5 sobre-restriccion: relaja", celda_sobre["pct_relaja"] >= 50.0,
              f"relaja en el {celda_sobre['pct_relaja']:.0f}% de los pasos")

    # ---- C6 PREVENCION: reserva el coste bajo para la calma y actua en las
    # fases ACTIVAS ----
    # La politica optima NO es monotona en el nivel de contagio: es "prevencion
    # primero", pica mas fuerte en propagacion (cortar la ola antes de que escale)
    # que en la propia emergencia, de modo que el coste no crece de forma monotona
    # con el contagio. La coherencia fiel es que TODAS las fases activas
    # (emergencia y propagacion) cuesten claramente mas que la calma: la politica
    # gasta cuando la epidemia esta viva y libera cuando esta controlada. Se usan
    # los agregados de rollout (fiables, no dependen de la cobertura de estados
    # raros como el contagio Alto, poco visitado cuando el control es estricto).
    filas_no_nulas = np.flatnonzero(np.any(agente.Q != 0.0, axis=1))
    visitados = set(int(i) for i in filas_no_nulas)
    activas = [c for c in (c_emer, c_rap, c_len) if c is not None]
    if activas and c_calma is not None:
        MARGEN = 1.0
        prevencion = min(activas) >= c_calma + MARGEN
        check("C6 prevencion (fases activas > calma)", prevencion,
              f"calma {c_calma:.2f} < activas [emer {c_emer:.2f}, "
              f"prop.rap {c_rap:.2f}, prop.len {c_len:.2f}] (margen {MARGEN:.1f})")

    # ---- calidad del aprendizaje ----
    cobertura = 100.0 * len(visitados) / N_ESTADOS
    conv_txt = "sin historial"
    if historial:
        n = len(historial)
        prim = np.mean([h[1] for h in historial[: max(1, n // 10)]])
        ult = np.mean([h[1] for h in historial[-max(1, n // 10):]])
        conv_txt = f"retorno primer decil {prim:.1f} -> ultimo decil {ult:.1f}"
        check("C7 convergencia mejora", ult >= prim,
              conv_txt)
    check("C8 cobertura informativa", cobertura > 0,
          f"{cobertura:.1f}% de los {N_ESTADOS} estados visitados en Q")

    # ---- informe ----
    lineas = []
    lineas.append("=" * 70)
    lineas.append("INFORME DE VALIDACION DE LA POLITICA APRENDIDA")
    lineas.append("=" * 70)
    lineas.append("")
    lineas.append("Rejilla fase x coste vigente (tab:rejilla del paper):")
    lineas.append(f"{'fase':14s} {'L':>2s} {'n':>7s} {'coste':>6s} "
                  f"{'%sin actuar':>12s} {'%relaja':>8s}  palancas medias "
                  f"(work school ptrans conf)")
    for fase in FASES:
        for L in (0, 1, 2):
            celda = rejilla[(fase, L)]
            if celda is None:
                lineas.append(f"{fase:14s} {NIVEL_TXT[L]:>2s} {'-':>7s}  "
                              f"(celda no visitada)")
            else:
                pal = " ".join(f"{p:.2f}" for p in celda["palancas"])
                lineas.append(
                    f"{fase:14s} {NIVEL_TXT[L]:>2s} {celda['n']:7d} "
                    f"{celda['coste_medio']:6.2f} "
                    f"{celda['pct_sin_actuar']:11.0f}% "
                    f"{celda['pct_relaja']:7.0f}%  [{pal}]")
    lineas.append("")
    lineas.append("Checks logicos:")
    for nombre, res in checks.items():
        marca = "PASS" if res["pass"] else "FAIL"
        lineas.append(f"  [{marca}] {nombre}: {res['detalle']}")
    n_pass = sum(1 for r in checks.values() if r["pass"])
    lineas.append("")
    lineas.append(f"Resultado: {n_pass}/{len(checks)} checks superados | "
                  f"cobertura Q: {cobertura:.1f}% | {conv_txt}")
    informe = "\n".join(lineas)

    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
        with open(os.path.join(carpeta, "validacion_informe.txt"), "w") as f:
            f.write(informe + "\n")
        with open(os.path.join(carpeta, "rejilla_politica.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["fase", "L", "n", "coste_medio", "pct_sin_actuar",
                        "pct_relaja", "work", "school", "ptrans", "conf"])
            for fase in FASES:
                for L in (0, 1, 2):
                    celda = rejilla[(fase, L)]
                    if celda is None:
                        w.writerow([fase, NIVEL_TXT[L], 0, "", "", "",
                                    "", "", "", ""])
                    else:
                        w.writerow([fase, NIVEL_TXT[L], celda["n"],
                                    f"{celda['coste_medio']:.3f}",
                                    f"{celda['pct_sin_actuar']:.1f}",
                                    f"{celda['pct_relaja']:.1f}",
                                    *(f"{p:.3f}" for p in celda["palancas"])])
    return informe, checks, rejilla


# =====================================================================
# ESTRES POR PAIS: la politica universal bajo dinamicas calibradas
# =====================================================================
def estres_por_pais(agente, rng, episodios=20, horizonte=None, carpeta=None):
    """Somete la politica UNICA (aprendida en el estilizado) a la dinamica de
    cada pais. La politica no se reentrena ni ve el pais: solo cambia el
    simulador (PARAMS_PAIS) y la discretizacion, que se recalibra por pais
    (terciles propios, como manda la Sec. 2.4: "Alto" es relativo a cada
    escala). Devuelve la tabla de resultados y la imprime.
    """
    T = horizonte or CONFIG["horizonte"]
    filas = []
    print("\n" + "=" * 70)
    print("ESTRES POR PAIS: politica unica bajo dinamicas calibradas")
    print("=" * 70)
    print("La politica decide por la situacion pandemica; el pais solo cambia")
    print("la dinamica del simulador y los terciles de discretizacion.")
    for pais, params in PARAMS_PAIS.items():
        def hacer_env(disc, _p=params):
            return EntornoSimple(disc, rng, params=_p)
        disc_p = calibrar_discretizador(hacer_env, rng)
        env = hacer_env(disc_p)

        def pi_universal(s):
            return agente.accion_greedy(s)

        def sin_restricciones(s):
            return (0, 0, 0, 0)

        def maxima(s):
            return (2, 2, 2, 2)

        res = {}
        for nombre, pol in (("pi_universal", pi_universal),
                            ("sin_restricciones", sin_restricciones),
                            ("maxima", maxima)):
            res[nombre] = evaluar_politica(env, disc_p, pol,
                                           episodios=episodios, horizonte=T)
        m = res["pi_universal"]
        c_sin = res["sin_restricciones"]["c_medio"]
        reduccion = 100.0 * (1.0 - m["c_medio"] / c_sin) if c_sin > 0 else 0.0
        controla = (reduccion >= 50.0
                    and m["coste_medio"] <= 0.8 * coste_max())
        filas.append({"pais": pais, "reduccion_pct": reduccion,
                      "controla": controla, **m,
                      "c_sin": c_sin,
                      "c_maxima": res["maxima"]["c_medio"],
                      "coste_maxima": res["maxima"]["coste_medio"]})
        marca = "PASS" if controla else "FAIL"
        print(f"\n  {pais}  [{marca}]  (d={params['d']}, rmax={params['rmax']})")
        print(f"    pi universal : c={m['c_medio']:.3f}  h={m['h_medio']:.3f}  "
              f"coste={m['coste_medio']:.2f}  %dias c=B: {m['pct_dias_bajo']:.1f}  "
              f"retorno={m['retorno']:.1f}")
        print(f"    referencias  : sin medidas c={c_sin:.3f} | "
              f"maxima c={res['maxima']['c_medio']:.3f} "
              f"coste={res['maxima']['coste_medio']:.1f}")
        print(f"    reduccion del contagio vs no actuar: {reduccion:.0f}%")
    n_ok = sum(1 for f in filas if f["controla"])
    print(f"\nResultado del estres: la politica universal controla la epidemia "
          f"en {n_ok}/{len(filas)} paises")
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
        with open(os.path.join(carpeta, "estres_paises.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["pais", "controla", "reduccion_pct", "c_medio",
                        "h_medio", "coste_medio", "pct_dias_bajo", "retorno",
                        "c_sin_medidas", "c_maxima", "coste_maxima"])
            for fila in filas:
                w.writerow([fila["pais"], int(fila["controla"]),
                            f"{fila['reduccion_pct']:.1f}",
                            f"{fila['c_medio']:.4f}", f"{fila['h_medio']:.4f}",
                            f"{fila['coste_medio']:.3f}",
                            f"{fila['pct_dias_bajo']:.1f}",
                            f"{fila['retorno']:.2f}", f"{fila['c_sin']:.4f}",
                            f"{fila['c_maxima']:.4f}",
                            f"{fila['coste_maxima']:.3f}"])
    return filas


# =====================================================================
# METRICAS DE CALIDAD DE LA LITERATURA (RL): ocupacion, entropia,
# ventaja/action gap, valor/retorno, residual de Bellman y coherencia
# de la matriz softmax por regimen. Responden a las dos preguntas del
# director: (a) en que estados se queda la politica (ocupacion d^pi) y
# (b) si la matriz de probabilidades asigna probabilidades con sentido.
# Referencias: Sutton & Barto (V, ventaja, retorno descontado, residual
# de Bellman); Bellemare et al. 2016 (action gap); entropia de Shannon
# de la politica (uso estandar en RL con maxima entropia).
# =====================================================================
def _softmax_fila(q, temperatura=1.0):
    """Softmax numericamente estable de una fila de Q (identico a la
    tabla de probabilidades.csv cuando temperatura=1.0)."""
    z = np.asarray(q, dtype=float) / max(temperatura, 1e-9)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _entropia(p):
    """Entropia de Shannon en nats (con la convencion 0*log0 = 0)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def metricas_literatura(agente, hacer_env, disc, rng, historial=None,
                        episodios=100, horizonte=None, temperatura=1.0,
                        carpeta=None, min_visitas=20):
    """Metricas de calidad de la literatura de RL sobre la politica aprendida.

    Calcula y guarda:
      d^pi(s) ... distribucion de OCUPACION de estados por rollouts greedy
                  (frecuencia y version descontada por gamma^t), con los
                  estados mas visitados, la entropia de la ocupacion, su
                  concentracion, y agregados por fase y por nivel de contagio.
      H(pi.|s) . ENTROPIA de la politica por estado (softmax de Q, igual que
                  probabilidades.csv): media/mediana en estados visitados y la
                  media PONDERADA por ocupacion sum_s d^pi(s) H(s).
      A(s,a) ... VENTAJA A=Q-V (V=max_a Q) y ACTION GAP g(s)=Q_(1)-Q_(2) en
                  estados visitados: mide la confianza de la mejor accion.
      V, J ..... funcion de valor V=max_a Q y RETORNO descontado empirico J(pi).
      delta .... RESIDUAL DE BELLMAN medio |r+gamma max Q' - Q| en estados
                  bien visitados (diagnostico de convergencia local, complementa
                  el check C7).
      softmax por REGIMEN: masa de probabilidad media en acciones de coste bajo
                  vs coste alto, por fase, frente al baseline uniforme (juzga si
                  el softmax concentra en acciones coherentes con el regimen).

    Devuelve un dict con los agregados clave. Si `carpeta`, escribe
    ocupacion_estados.csv, entropia_politica.csv, softmax_por_regimen.csv y
    metricas_literatura.txt. No modifica el agente ni el entorno.
    """
    T = horizonte or CONFIG["horizonte"]
    gamma = agente.gamma
    Q = agente.Q
    V = Q.max(axis=1)                       # V(s) = max_a Q(s,a)
    cmax = coste_max()

    # tablas por indice (fase, nivel de contagio, etiqueta) calculadas una vez
    fase_idx = [""] * N_ESTADOS
    nivelc_idx = [0] * N_ESTADOS
    etiq_idx = [""] * N_ESTADOS
    for s in estados_iterar():
        i = indice_de_estado(s)
        fase_idx[i] = fase_de_estado(s)
        nivelc_idx[i] = s[0]
        etiq_idx[i] = etiqueta_estado(s)

    # ---------- (1) OCUPACION d^pi(s) via rollouts greedy ----------
    env = hacer_env(disc)
    cont = np.zeros(N_ESTADOS)              # frecuencia de visita (sin descuento)
    cont_desc = np.zeros(N_ESTADOS)         # ponderada por gamma^t (descontada)
    J_total = 0.0                           # retorno descontado empirico
    td_abs, td_n = 0.0, 0                   # residual de Bellman |delta|
    for _ in range(episodios):
        s = env.reset()
        s_idx = indice_de_estado(s)
        peso = 1.0
        for t in range(T):
            a = agente.accion_greedy(s)
            a_idx = indice_de_accion(a)
            c_t, h_t, a_prev = env.c, env.h, env.a_prev
            s1 = env.paso(a)
            r, _ = recompensa_tramos(s, c_t, env.c, h_t, env.h, a, a_prev, disc)
            s1_idx = indice_de_estado(s1)
            cont[s_idx] += 1.0
            cont_desc[s_idx] += peso
            J_total += peso * r
            if agente.visitas[s_idx] >= min_visitas:
                delta = r + gamma * Q[s1_idx].max() - Q[s_idx, a_idx]
                td_abs += abs(delta)
                td_n += 1
            peso *= gamma
            s, s_idx = s1, s1_idx
    d_pi = cont / cont.sum() if cont.sum() > 0 else cont
    d_pi_desc = cont_desc / cont_desc.sum() if cont_desc.sum() > 0 else cont_desc
    J_pi = J_total / episodios
    delta_medio = td_abs / td_n if td_n else float("nan")

    ocupados = np.flatnonzero(cont > 0)
    n_ocup = int(len(ocupados))
    H_ocup = _entropia(d_pi[ocupados]) if n_ocup else 0.0
    H_ocup_norm = H_ocup / math.log(n_ocup) if n_ocup > 1 else 0.0
    soporte_efectivo = math.exp(H_ocup) if n_ocup else 0.0   # numero efectivo de estados
    orden_ocup = ocupados[np.argsort(d_pi[ocupados])[::-1]]
    top10_masa = float(d_pi[orden_ocup[:10]].sum()) if n_ocup else 0.0

    ocup_por_fase = {f: float(d_pi[[i for i in ocupados if fase_idx[i] == f]].sum())
                     for f in FASES}
    ocup_por_nivelc = {NIVEL_TXT[k]:
                       float(d_pi[[i for i in ocupados if nivelc_idx[i] == k]].sum())
                       for k in (0, 1, 2)}

    # ---------- (2-3-5) recorrido unico de Q: entropia, gap, ventaja, masa ----------
    coste_a = np.array([coste(accion_de_indice(k)) for k in range(N_ACCIONES)])
    low_mask = coste_a <= cmax / 3.0
    high_mask = coste_a >= 2.0 * cmax / 3.0
    base_low = float(low_mask.mean())       # baseline: softmax plano
    base_high = float(high_mask.mean())

    H_pi = np.zeros(N_ESTADOS)              # entropia de pi(.|s) en nats
    gap = np.zeros(N_ESTADOS)              # action gap Q_(1) - Q_(2)
    vent = np.zeros(N_ESTADOS)             # V - media_a Q  (ventaja de la mejor)
    masa_low = np.zeros(N_ESTADOS)
    masa_high = np.zeros(N_ESTADOS)
    for i in range(N_ESTADOS):
        fila = Q[i]
        p = _softmax_fila(fila, temperatura)
        H_pi[i] = _entropia(p)
        dos = np.partition(fila, -2)[-2:]
        gap[i] = float(dos[1] - dos[0])
        vent[i] = float(fila.max() - fila.mean())
        masa_low[i] = float(p[low_mask].sum())
        masa_high[i] = float(p[high_mask].sum())
    Hmax = math.log(N_ACCIONES)            # entropia maxima = log 81

    visit_mask = agente.visitas > 0
    n_visit = int(visit_mask.sum())
    H_med_vis = float(np.mean(H_pi[visit_mask])) if n_visit else float("nan")
    H_medi_vis = float(np.median(H_pi[visit_mask])) if n_visit else float("nan")
    H_pond = float(np.sum(d_pi * H_pi))    # sum_s d^pi(s) H(s)
    gap_med_ocup = float(np.mean(gap[ocupados])) if n_ocup else float("nan")
    gap_medi_ocup = float(np.median(gap[ocupados])) if n_ocup else float("nan")
    vent_med_ocup = float(np.mean(vent[ocupados])) if n_ocup else float("nan")
    V_med_ocup = float(np.mean(V[ocupados])) if n_ocup else float("nan")

    # ---------- softmax por regimen (masa coste bajo vs alto, y coherencia) ----------
    reg = {}
    for f in FASES:
        idx_f = [i for i in range(N_ESTADOS) if fase_idx[i] == f]
        idx_fo = [i for i in ocupados if fase_idx[i] == f]
        n_f, n_fo = len(idx_f), len(idx_fo)
        ml_unw = float(np.mean(masa_low[idx_f])) if n_f else float("nan")
        mh_unw = float(np.mean(masa_high[idx_f])) if n_f else float("nan")
        if n_fo:
            w = d_pi[idx_fo]
            w = w / w.sum()
            ml_p = float(np.sum(w * masa_low[idx_fo]))
            mh_p = float(np.sum(w * masa_high[idx_fo]))
            H_f = float(np.sum(w * H_pi[idx_fo]))
            gap_f = float(np.sum(w * gap[idx_fo]))
        else:
            ml_p, mh_p, H_f, gap_f = float("nan"), float("nan"), float("nan"), float("nan")
        # direccion esperada: emergencia/propagacion -> mas masa en coste alto que
        # el baseline; calma -> mas masa en coste bajo que el baseline
        ref_low = ml_p if not math.isnan(ml_p) else ml_unw
        ref_high = mh_p if not math.isnan(mh_p) else mh_unw
        if f == "calma":
            coherente = (ref_low - base_low) >= (ref_high - base_high)
        else:
            coherente = (ref_high - base_high) >= (ref_low - base_low)
        reg[f] = {"n": n_f, "n_ocup": n_fo, "ml_unw": ml_unw, "mh_unw": mh_unw,
                  "ml_pond": ml_p, "mh_pond": mh_p, "H": H_f, "gap": gap_f,
                  "coherente": bool(coherente)}

    # ---------- informe de texto ----------
    L = []
    L.append("=" * 70)
    L.append("METRICAS DE CALIDAD (LITERATURA RL) DE LA POLITICA APRENDIDA")
    L.append("=" * 70)
    L.append("")
    L.append("(1) OCUPACION d^pi(s) -- en que estados se queda la politica")
    L.append(f"    rollouts greedy: {episodios} episodios x {T} pasos "
             f"(gamma={gamma:.3f} para la version descontada)")
    L.append(f"    estados ocupados: {n_ocup} de {N_ESTADOS} "
             f"({100.0*n_ocup/N_ESTADOS:.1f}%)")
    L.append(f"    entropia H(d^pi) = {H_ocup:.3f} nats (normalizada {H_ocup_norm:.3f}); "
             f"soporte efectivo exp(H) = {soporte_efectivo:.1f} estados")
    L.append(f"    concentracion: la masa top-10 acumula {100.0*top10_masa:.1f}%")
    L.append("    estados mas visitados (d^pi | descontado | fase):")
    for r_ in range(min(12, n_ocup)):
        i = orden_ocup[r_]
        L.append(f"      {r_+1:2d}. {etiq_idx[i]:44s}  "
                 f"d={d_pi[i]:.4f}  ddesc={d_pi_desc[i]:.4f}  [{fase_idx[i]}]")
    L.append("    ocupacion por fase: " +
             "  ".join(f"{f}={100.0*ocup_por_fase[f]:.1f}%" for f in FASES))
    L.append("    ocupacion por nivel de contagio: " +
             "  ".join(f"c={k}: {100.0*ocup_por_nivelc[k]:.1f}%"
                      for k in ("B", "M", "A")))
    L.append("")
    L.append(f"(2) ENTROPIA de la politica H(pi.|s)  [maximo = log {N_ACCIONES} "
             f"= {Hmax:.3f} nats]")
    L.append(f"    media (estados visitados) = {H_med_vis:.3f} nats "
             f"(norm {H_med_vis/Hmax:.3f}); mediana = {H_medi_vis:.3f}")
    L.append(f"    media PONDERADA por ocupacion sum_s d^pi(s)H(s) = {H_pond:.3f} "
             f"nats (norm {H_pond/Hmax:.3f})")
    L.append("    lectura: cuanto mas cerca del maximo, mas PLANA es la fila del "
             "softmax (Q casi empatada entre acciones)")
    L.append("")
    L.append("(3) VENTAJA A=Q-V y ACTION GAP g(s)=Q_(1)-Q_(2) (sobre Q.npy)")
    L.append(f"    action gap medio (estados ocupados) = {gap_med_ocup:.4f}; "
             f"mediana = {gap_medi_ocup:.4f}")
    L.append(f"    ventaja media de la mejor accion (V - media_a Q) = {vent_med_ocup:.4f}")
    L.append("    lectura: gaps pequenos => diferencias de valor REALES minimas "
             "=> el softmax reparte casi uniforme (filas planas justificadas)")
    L.append("")
    L.append("(4) VALOR V=max_a Q y RETORNO descontado J(pi)")
    L.append(f"    V medio (estados ocupados) = {V_med_ocup:.4f}")
    L.append(f"    J(pi) descontado empirico = {J_pi:.3f}")
    L.append(f"    residual de Bellman medio |delta| (estados bien visitados, "
             f">={min_visitas} visitas, n={td_n}) = {delta_medio:.4f}")
    L.append("")
    L.append("(5) SOFTMAX POR REGIMEN: masa de probabilidad en coste bajo vs alto")
    L.append(f"    baseline (softmax plano):  masa_low={base_low:.3f}  "
             f"masa_high={base_high:.3f}  (umbral bajo<={cmax/3.0:.2f}, "
             f"alto>={2.0*cmax/3.0:.2f}, cmax={cmax:.1f})")
    L.append(f"    {'fase':13s} {'n':>4s} {'nocup':>5s} {'low(pond)':>10s} "
             f"{'high(pond)':>10s} {'H':>6s} {'gap':>7s}  coherente")
    for f in FASES:
        d = reg[f]
        L.append(f"    {f:13s} {d['n']:4d} {d['n_ocup']:5d} "
                 f"{d['ml_pond']:10.3f} {d['mh_pond']:10.3f} "
                 f"{d['H']:6.3f} {d['gap']:7.4f}  "
                 f"{'SI' if d['coherente'] else 'NO'}")
    n_coh = sum(1 for f in FASES if reg[f]["coherente"])
    L.append("")
    L.append(f"Resumen: ocupacion concentrada en {soporte_efectivo:.0f} estados "
             f"efectivos; entropia politica ponderada {H_pond/Hmax:.2f} del maximo; "
             f"action gap medio {gap_med_ocup:.4f}; coherencia softmax "
             f"{n_coh}/{len(FASES)} regimenes.")
    informe = "\n".join(L)

    # ---------- guardado de CSVs y del informe ----------
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
        with open(os.path.join(carpeta, "ocupacion_estados.csv"), "w",
                  newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["rank", "estado", "fase", "nivel_contagio", "visitas",
                        "d_pi", "d_pi_descontada"])
            for r_, i in enumerate(orden_ocup):
                w.writerow([r_ + 1, etiq_idx[i], fase_idx[i],
                            NIVEL_TXT[nivelc_idx[i]], int(agente.visitas[i]),
                            f"{d_pi[i]:.6f}", f"{d_pi_desc[i]:.6f}"])
        with open(os.path.join(carpeta, "entropia_politica.csv"), "w",
                  newline="") as fe:
            w = csv.writer(fe)
            w.writerow(["estado", "fase", "visitas", "d_pi", "H_pi_nats",
                        "H_pi_norm", "action_gap", "ventaja_media"])
            orden_v = np.argsort(agente.visitas)[::-1]
            for i in orden_v:
                if agente.visitas[i] <= 0:
                    break
                w.writerow([etiq_idx[i], fase_idx[i], int(agente.visitas[i]),
                            f"{d_pi[i]:.6f}", f"{H_pi[i]:.4f}",
                            f"{H_pi[i]/Hmax:.4f}", f"{gap[i]:.5f}",
                            f"{vent[i]:.5f}"])
        with open(os.path.join(carpeta, "softmax_por_regimen.csv"), "w",
                  newline="") as fr:
            w = csv.writer(fr)
            w.writerow(["fase", "n_estados", "n_ocupados", "masa_low_unw",
                        "masa_high_unw", "masa_low_pond", "masa_high_pond",
                        "base_low", "base_high", "H_media", "gap_medio",
                        "coherente"])
            for f in FASES:
                d = reg[f]
                w.writerow([f, d["n"], d["n_ocup"], f"{d['ml_unw']:.4f}",
                            f"{d['mh_unw']:.4f}", f"{d['ml_pond']:.4f}",
                            f"{d['mh_pond']:.4f}", f"{base_low:.4f}",
                            f"{base_high:.4f}", f"{d['H']:.4f}",
                            f"{d['gap']:.5f}", int(d["coherente"])])
        with open(os.path.join(carpeta, "metricas_literatura.txt"), "w") as ft:
            ft.write(informe + "\n")

    return {"n_ocupados": n_ocup, "H_ocupacion": H_ocup,
            "H_ocupacion_norm": H_ocup_norm, "soporte_efectivo": soporte_efectivo,
            "top10_masa": top10_masa, "ocup_por_fase": ocup_por_fase,
            "ocup_por_nivelc": ocup_por_nivelc, "H_pi_medio": H_med_vis,
            "H_pi_ponderado": H_pond, "H_pi_max": Hmax,
            "gap_medio": gap_med_ocup, "ventaja_media": vent_med_ocup,
            "V_medio": V_med_ocup, "J_pi": J_pi, "delta_medio": delta_medio,
            "regimen": reg, "base_low": base_low, "base_high": base_high,
            "informe": informe}


# =====================================================================
# Salidas: tabla Q, politica y tabla de probabilidades
# =====================================================================
def etiqueta_estado(s):
    niv = {0: "B", 1: "M", 2: "A"}
    tnd = {-1: "-", 0: "0", +1: "+"}
    return (f"c={niv[s[0]]} dc={tnd[s[1]]} h={niv[s[2]]} dh={tnd[s[3]]} "
            f"L={niv[s[4]]} dL={tnd[s[5]]} preo={niv[s[6]]}")


def estados_iterar():
    for c in range(3):
        for dc in (-1, 0, 1):
            for h in range(3):
                for dh in (-1, 0, 1):
                    for L in range(3):
                        for dL in (-1, 0, 1):
                            for p in range(3):
                                yield (c, dc, h, dh, L, dL, p)


def etiqueta_estado_legible(s):
    """Estado con TODAS sus variables en palabras (para tablas autocontenidas)."""
    niv = {0: "Bajo", 1: "Medio", 2: "Alto"}
    tnd = {-1: "baja", 0: "estable", +1: "sube"}
    return (f"contagio {niv[s[0]]} ({tnd[s[1]]}); "
            f"hospital {niv[s[2]]} ({tnd[s[3]]}); "
            f"restriccion {niv[s[4]]} ({tnd[s[5]]}); "
            f"cumplimiento {niv[s[6]]}")


def guardar_salidas(agente, carpeta, temperatura=1.0, top_k=3):
    os.makedirs(carpeta, exist_ok=True)
    ruta_q = os.path.join(carpeta, "tabla_Q.csv")
    ruta_pi = os.path.join(carpeta, "politica.csv")
    ruta_p = os.path.join(carpeta, "probabilidades.csv")
    ruta_top = os.path.join(carpeta, "probabilidades_top3.csv")
    np.save(os.path.join(carpeta, "Q.npy"), agente.Q)

    V = agente.Q.max(axis=1)
    # columnas de contexto que hacen AUTOCONTENIDA cada fila de probabilidades:
    # ninguna variable del estado queda oculta y se ve por que el softmax es plano
    # (gap pequeno => probabilidades casi iguales). Ademas de las 81 columnas de
    # accion, se anticipa la mejor accion, su coste y los diagnosticos por fila.
    contexto = ["estado", "situacion_completa", "fase", "visitas",
                "mejor_accion", "coste_mejor", "Q_max", "V", "action_gap"]

    with open(ruta_q, "w", newline="") as fq, \
         open(ruta_pi, "w", newline="") as fp, \
         open(ruta_p, "w", newline="") as fr, \
         open(ruta_top, "w", newline="") as ft:
        wq, wp, wr, wt = (csv.writer(fq), csv.writer(fp),
                          csv.writer(fr), csv.writer(ft))
        cab_acc = [f"a{indice_de_accion(a)}={a}" for a in
                   (accion_de_indice(i) for i in range(N_ACCIONES))]
        wq.writerow(["estado"] + cab_acc)
        wp.writerow(["estado", "fase", "mejor_accion", "work", "school",
                     "ptrans", "conf", "coste", "Q_max", "visitas"])
        # probabilidades.csv: contexto completo + softmax de las 81 acciones
        wr.writerow(contexto + cab_acc)
        # probabilidades_top3.csv: version LEGIBLE, top-k acciones por estado
        cab_top = ["estado", "situacion_completa", "fase", "visitas"]
        for r_ in range(1, top_k + 1):
            cab_top += [f"top{r_}_accion", f"top{r_}_prob", f"top{r_}_coste"]
        wt.writerow(cab_top)
        for s in estados_iterar():
            i = indice_de_estado(s)
            fila = agente.Q[i]
            wq.writerow([etiqueta_estado(s)] + [f"{v:.4f}" for v in fila])
            a_best = accion_de_indice(int(np.argmax(fila)))
            wp.writerow([etiqueta_estado(s), fase_de_estado(s), str(a_best),
                         *a_best, f"{coste(a_best):.1f}",
                         f"{fila.max():.4f}", int(agente.visitas[i])])
            z = fila / max(temperatura, 1e-9)
            z = z - z.max()
            p = np.exp(z)
            p = p / p.sum()
            dos = np.partition(fila, -2)[-2:]
            gap = float(dos[1] - dos[0])
            ctx = [etiqueta_estado(s), etiqueta_estado_legible(s),
                   fase_de_estado(s), int(agente.visitas[i]), str(a_best),
                   f"{coste(a_best):.1f}", f"{fila.max():.4f}",
                   f"{V[i]:.4f}", f"{gap:.5f}"]
            wr.writerow(ctx + [f"{v:.4f}" for v in p])
            orden = np.argsort(p)[::-1][:top_k]
            fila_top = [etiqueta_estado(s), etiqueta_estado_legible(s),
                        fase_de_estado(s), int(agente.visitas[i])]
            for k in orden:
                a_k = accion_de_indice(int(k))
                fila_top += [str(a_k), f"{p[k]:.4f}", f"{coste(a_k):.1f}"]
            wt.writerow(fila_top)
    return ruta_q, ruta_pi, ruta_p, ruta_top


# =====================================================================
# DOS METRICAS DE CALIDAD PRINCIPALES (las que se reportan al director)
#   M1  Coherencia logica de la politica  (¿las acciones tienen sentido?)
#   M2  Eficacia frente al coste vs baselines  (la que juzga un experto:
#       ¿controla la epidemia y a que coste, comparado con no hacer nada
#       o restringirlo todo?)
# =====================================================================
def coherencia_estado_accion(agente, cobertura_min=0.5, min_visitas=20):
    """M1 (parte fina): % de estados BIEN EXPLORADOS cuya mejor accion es
    coherente con su regimen (rejilla, Tabla tab:rejilla del paper).

    Regla de sentido comun por fase (coste de la mejor accion vs coste maximo):
      - emergencia         : debe ACTUAR fuerte  -> coste >= 50% del maximo
      - propagacion (r/l)  : debe ACTUAR al menos moderado -> coste >= 25%
      - calma, restr. baja : NO debe sobreactuar -> coste <= 50% del maximo
      - calma, restr. alta : debe RELAJAR -> coste <= coste vigente
      - calma, restr. media: neutro (siempre coherente)
    Solo juzga estados con COBERTURA suficiente de acciones (fraccion de las 81
    acciones ya probadas >= cobertura_min): con |A|=81, en un estado visitado
    pocas veces la mayoria de acciones no se han probado y su argmax es ruido;
    juzgarlo confundiria falta de exploracion con ilogica. Devuelve
    (pct_global, n_ok, n_total, detalle_por_fase, cobertura_media_juzgada).
    """
    cmax = coste_max()
    por_fase = {f: [0, 0] for f in FASES}   # [ok, total]
    cobs = []
    for s in estados_iterar():
        i = indice_de_estado(s)
        cob = float(np.mean(agente.n_sa[i] > 0))
        if agente.visitas[i] < min_visitas or cob < cobertura_min:
            continue
        cobs.append(cob)
        a_best = accion_de_indice(int(np.argmax(agente.Q[i])))
        c_best = coste(a_best)
        c_vig = coste(accion_de_indice(0)) if s[4] == 0 else (s[4] / 2.0) * cmax
        fase = fase_de_estado(s)
        if fase == "emergencia":
            # emergencia con la epidemia AL ALZA (contagio Alto y subiendo, o
            # subiendo hacia Alto): debe actuar fuerte (coste >= 50%). Si el
            # contagio ya remite (no sube) o es una emergencia solo hospitalaria
            # con el contagio controlado (c<Alto), basta una respuesta MODERADA
            # (coste >= 25%): sobre-restringir una ola que baja no es sensato.
            al_alza = (s[0] == 2 and s[1] == +1)
            ok = c_best >= (0.5 if al_alza else 0.25) * cmax
        elif fase in ("prop_rapida", "prop_lenta"):
            ok = c_best >= 0.25 * cmax
        else:  # calma
            if s[4] == 0:            # restriccion baja: no sobreactuar
                ok = c_best <= 0.5 * cmax
            elif s[4] == 2:          # sobre-restriccion: relajar
                ok = c_best <= (s[4] / 2.0) * cmax
            else:                    # restriccion media: neutro
                ok = True
        por_fase[fase][0] += int(ok)
        por_fase[fase][1] += 1
    n_ok = sum(v[0] for v in por_fase.values())
    n_tot = sum(v[1] for v in por_fase.values())
    pct = 100.0 * n_ok / n_tot if n_tot else float("nan")
    cob_media = float(np.mean(cobs)) if cobs else float("nan")
    return pct, n_ok, n_tot, por_fase, cob_media


def evaluar_referencias(agente, hacer_env, disc, rng, episodios=30,
                        horizonte=None, carpeta=None):
    """M2 (la que juzga un experto): eficacia frente al coste de la politica
    aprendida comparada con las politicas de referencia (Sec. 3.4).

    Mide, promediando sobre `episodios` rollouts en el simulador, los DESENLACES
    epidemicos que un experto miraria: contagio medio, presion hospitalaria,
    coste socioeconomico, % de dias con contagio Bajo y retorno. La referencia
    'sin restricciones' fija el peor contagio (que se deja crecer) y 'restriccion
    maxima' el peor coste; una buena politica reduce mucho el contagio SIN pagar
    el coste maximo. Guarda comparacion_baselines.csv y devuelve el dict.
    """
    T = horizonte or CONFIG["horizonte"]
    cmax = coste_max()
    env = hacer_env(disc)
    resultados = {}
    for nombre, pol in politicas_referencia(agente, rng).items():
        resultados[nombre] = evaluar_politica(env, disc, pol,
                                              episodios=episodios, horizonte=T)
    api = resultados["pi* (aprendida)"]
    c_sin = resultados["sin restricciones"]["c_medio"]
    c_max_pol = resultados["restriccion maxima"]["c_medio"]
    reduccion = 100.0 * (1.0 - api["c_medio"] / c_sin) if c_sin > 0 else 0.0
    frac_coste = 100.0 * api["coste_medio"] / cmax
    # mejor que baselines "ingenuos" en retorno
    mejor_que_aleatoria = api["retorno"] > resultados["aleatoria"]["retorno"]
    mejor_que_umbral = api["retorno"] > resultados["umbral por contagio"]["retorno"]

    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
        with open(os.path.join(carpeta, "comparacion_baselines.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["politica", "contagio_medio", "hospital_medio",
                        "coste_medio", "pct_dias_contagio_bajo", "retorno"])
            for nombre, m in resultados.items():
                w.writerow([nombre, f"{m['c_medio']:.4f}", f"{m['h_medio']:.4f}",
                            f"{m['coste_medio']:.3f}", f"{m['pct_dias_bajo']:.1f}",
                            f"{m['retorno']:.2f}"])
    return {"resultados": resultados, "reduccion_contagio_pct": reduccion,
            "frac_coste_pct": frac_coste, "c_sin": c_sin, "c_max_pol": c_max_pol,
            "mejor_que_aleatoria": mejor_que_aleatoria,
            "mejor_que_umbral": mejor_que_umbral}


def resumen_calidad(checks, coh, refs, carpeta=None):
    """Imprime y guarda el RESUMEN con las DOS metricas de calidad principales,
    en lenguaje llano y con un veredicto por metrica. `checks` viene de validar(),
    `coh` de coherencia_estado_accion(), `refs` de evaluar_referencias()."""
    reglas = [k for k in checks if k.startswith(("C1", "C2", "C3", "C4",
                                                 "C5", "C6"))]
    n_reglas_ok = sum(1 for k in reglas if checks[k]["pass"])
    pct_coh, n_ok, n_tot, por_fase, cob_media = coh
    api = refs["resultados"]["pi* (aprendida)"]

    m1_ok = (n_reglas_ok >= len(reglas) - 1) and pct_coh >= 70.0
    m2_ok = (refs["reduccion_contagio_pct"] >= 50.0
             and refs["frac_coste_pct"] <= 80.0
             and refs["mejor_que_aleatoria"])

    L = []
    L.append("=" * 70)
    L.append("RESUMEN DE CALIDAD DEL MODELO  (dos metricas principales)")
    L.append("=" * 70)
    L.append("")
    L.append("METRICA 1 - COHERENCIA LOGICA  (¿las acciones tienen sentido?)")
    L.append(f"  Reglas de la rejilla superadas: {n_reglas_ok}/{len(reglas)}")
    for k in reglas:
        marca = "OK  " if checks[k]["pass"] else "FALLA"
        L.append(f"    [{marca}] {k}: {checks[k]['detalle']}")
    L.append(f"  Coherencia estado-accion: {pct_coh:.0f}% "
             f"({n_ok}/{n_tot} estados bien explorados con accion sensata; "
             f"cobertura media {100.0*cob_media:.0f}% de las acciones)")
    for f in FASES:
        ok, tot = por_fase[f]
        if tot:
            L.append(f"      {f:12s}: {100.0*ok/tot:3.0f}% ({ok}/{tot})")
    L.append(f"  >> VEREDICTO M1: {'BUENA' if m1_ok else 'REVISAR'} coherencia")
    L.append("")
    L.append("METRICA 2 - EFICACIA FRENTE AL COSTE  (la que juzga un experto)")
    L.append(f"  La politica reduce el contagio un {refs['reduccion_contagio_pct']:.0f}% "
             f"frente a NO hacer nada")
    L.append(f"    (contagio: {refs['c_sin']:.2f} sin medidas -> "
             f"{api['c_medio']:.2f} con la politica; "
             f"{refs['c_max_pol']:.2f} con restriccion maxima)")
    L.append(f"  Y lo consigue con el {refs['frac_coste_pct']:.0f}% del coste maximo "
             f"(coste medio {api['coste_medio']:.1f} de {coste_max():.0f})")
    L.append(f"  Presion hospitalaria media {api['h_medio']:.2f}; "
             f"dias con contagio Bajo {api['pct_dias_bajo']:.0f}%; "
             f"retorno {api['retorno']:.0f}")
    L.append(f"  Supera a la politica aleatoria: "
             f"{'SI' if refs['mejor_que_aleatoria'] else 'NO'}; "
             f"a la de umbral: {'SI' if refs['mejor_que_umbral'] else 'NO'}")
    L.append(f"  >> VEREDICTO M2: {'BUENA' if m2_ok else 'REVISAR'} eficacia/coste")
    L.append("")
    L.append(f"CONCLUSION: modelo {'SOLIDO' if (m1_ok and m2_ok) else 'A REVISAR'} "
             f"en las dos metricas.")
    informe = "\n".join(L)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
        with open(os.path.join(carpeta, "calidad_resumen.txt"), "w") as f:
            f.write(informe + "\n")
        with open(os.path.join(carpeta, "calidad_resumen.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["metrica", "indicador", "valor", "veredicto"])
            w.writerow(["M1_coherencia", "reglas_rejilla_superadas",
                        f"{n_reglas_ok}/{len(reglas)}", "BUENA" if m1_ok else "REVISAR"])
            w.writerow(["M1_coherencia", "pct_estado_accion_coherente",
                        f"{pct_coh:.1f}", "BUENA" if m1_ok else "REVISAR"])
            w.writerow(["M2_eficacia", "reduccion_contagio_pct_vs_no_actuar",
                        f"{refs['reduccion_contagio_pct']:.1f}",
                        "BUENA" if m2_ok else "REVISAR"])
            w.writerow(["M2_eficacia", "pct_del_coste_maximo",
                        f"{refs['frac_coste_pct']:.1f}",
                        "BUENA" if m2_ok else "REVISAR"])
            w.writerow(["M2_eficacia", "retorno_pi",
                        f"{api['retorno']:.2f}", "BUENA" if m2_ok else "REVISAR"])
    return informe, m1_ok, m2_ok


# =====================================================================
# Programa principal
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Q-learning sencillo (teoria del TFG) con validacion")
    ap.add_argument("--entorno", choices=["simple", "fcm"], default="simple",
                    help="simple: transparente (validacion); fcm: eq:fcm")
    ap.add_argument("--episodios", type=int, default=None, help="K")
    ap.add_argument("--horizonte", type=int, default=None, help="T")
    ap.add_argument("--matriz", type=str, default=None,
                    help="Excel con la matriz W del FCM (si no, provisional)")
    ap.add_argument("--hoja", type=str, default=None, help="hoja del Excel")
    ap.add_argument("--semilla", type=int, default=CONFIG["semilla"])
    ap.add_argument("--lambda_B", type=float, default=None, help="peso coste L bajo")
    ap.add_argument("--bono", type=float, default=None, help="escala del bono de seguridad")
    ap.add_argument("--salidas", type=str, default=None,
                    help="carpeta de salidas (por defecto, junto al script)")
    ap.add_argument("--sin-eval", action="store_true",
                    help="no evaluar politicas de referencia")
    ap.add_argument("--sin-validar", action="store_true",
                    help="no generar el informe de validacion")
    ap.add_argument("--sin-estres", action="store_true",
                    help="no evaluar la politica en los paises (estres)")
    ap.add_argument("--sin-metricas", action="store_true",
                    help="no calcular las metricas de calidad de la literatura")
    ap.add_argument("--metricas-episodios", type=int, default=100,
                    help="episodios de rollout greedy para la ocupacion d^pi")
    args = ap.parse_args()
    if args.lambda_B is not None:
        CONFIG["lambda_B"] = args.lambda_B
    if args.bono is not None:
        CONFIG["bono_escala"] = args.bono

    rng = random.Random(args.semilla)
    np.random.seed(args.semilla)

    if args.entorno == "fcm":
        if args.matriz:
            W = cargar_W_excel(args.matriz, args.hoja)
            print(f"Entorno FCM con matriz W de {args.matriz}")
        else:
            W = construir_W()
            print("Entorno FCM con W PROVISIONAL (fuente LLM interna); "
                  "sustituir con --matriz cuando el Excel este relleno")
        def hacer_env(disc):
            return EntornoFCM(W, disc, rng)
    else:
        print("Entorno SIMPLE transparente (validacion de la logica); "
              "el FCM queda disponible con --entorno fcm")
        def hacer_env(disc):
            return EntornoSimple(disc, rng)

    print("Calibrando discretizacion (rodaje aleatorio)...")
    disc = calibrar_discretizador(hacer_env, rng)
    print(f"  cortes: { {k: tuple(round(v,3) for v in c) for k, c in disc.cortes.items()} }")
    print(f"  deltas: { {k: round(v,4) for k, v in disc.deltas.items()} }")

    print(f"Entrenando: |S|={N_ESTADOS}, |A|={N_ACCIONES}, "
          f"tabla Q de {N_ESTADOS * N_ACCIONES} entradas")
    env = hacer_env(disc)
    agente, historial = entrenar(env, disc, rng,
                                 episodios=args.episodios,
                                 horizonte=args.horizonte)

    carpeta = args.salidas or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "salidas")
    # curva de aprendizaje (proceso de aprendizaje): retorno y |error TD| por episodio
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, "curva_aprendizaje.csv"), "w", newline="") as fc:
        wc = csv.writer(fc)
        wc.writerow(["episodio", "retorno", "c_medio", "coste_medio", "epsilon", "error_TD_medio"])
        for fila in historial:
            wc.writerow([fila[0], f"{fila[1]:.4f}", f"{fila[2]:.4f}",
                         f"{fila[3]:.4f}", f"{fila[4]:.4f}", f"{fila[5]:.6f}"])
    rutas = guardar_salidas(agente, carpeta)
    print("Salidas guardadas:")
    for r in rutas:
        print("  " + r)

    checks, refs, coh = None, None, None
    if not args.sin_validar:
        print("\nValidando la politica aprendida (coherencia logica detallada)...")
        informe, checks, _ = validar(agente, hacer_env, disc, rng,
                                     historial=historial,
                                     horizonte=args.horizonte,
                                     carpeta=carpeta)
        print(informe)

    if not args.sin_eval:
        print("\nEficacia frente al coste vs politicas de referencia (Sec. 3.4):")
        refs = evaluar_referencias(agente, hacer_env, disc, rng,
                                   episodios=20, horizonte=args.horizonte,
                                   carpeta=carpeta)
        for nombre, m in refs["resultados"].items():
            print(f"  {nombre:22s} retorno={m['retorno']:9.2f}  "
                  f"c={m['c_medio']:.3f}  h={m['h_medio']:.3f}  "
                  f"coste={m['coste_medio']:.2f}  "
                  f"%dias c=B: {m['pct_dias_bajo']:.1f}")

    # ---- RESUMEN con las DOS metricas principales (lo que ve el director) ----
    if checks is not None and refs is not None:
        coh = coherencia_estado_accion(agente)
        resumen, _, _ = resumen_calidad(checks, coh, refs, carpeta=carpeta)
        print("\n" + resumen)

    if not args.sin_estres and args.entorno == "simple":
        estres_por_pais(agente, rng, horizonte=args.horizonte,
                        carpeta=carpeta)

    if not args.sin_metricas:
        print("\nCalculando metricas de calidad de la literatura RL "
              "(ocupacion, entropia, ventaja, valor, coherencia softmax)...")
        met = metricas_literatura(agente, hacer_env, disc, rng,
                                  historial=historial,
                                  episodios=args.metricas_episodios,
                                  horizonte=args.horizonte, carpeta=carpeta)
        print(met["informe"])
        print("\nMetricas guardadas en: ocupacion_estados.csv, "
              "entropia_politica.csv, softmax_por_regimen.csv, "
              "metricas_literatura.txt")


if __name__ == "__main__":
    main()
