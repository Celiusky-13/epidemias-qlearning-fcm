# Control de epidemias con Q-learning y Mapas Cognitivos Difusos

Código del Trabajo Fin de Grado sobre control de epidemias mediante **aprendizaje por
refuerzo** (Q-learning) y **Mapas Cognitivos Difusos** (FCM, *Fuzzy Cognitive Maps*).

El objetivo del trabajo es aprender, sobre un simulador epidémico transparente, una
política que decida qué medidas no farmacéuticas aplicar (cierres de trabajo, colegios,
transporte y confinamiento) equilibrando el control del contagio y el coste
socioeconómico; y estudiar cómo integrar un mapa causal (el FCM) con esa política.

## Cómo está organizado

El repositorio sigue las tres fases del trabajo, en el mismo orden que los capítulos del
TFG. Cada fase tiene la misma estructura: `src/` (código), `datos/` (entradas),
`resultados/` (salidas en CSV) y `figuras/` (gráficas).

```
epidemias-qlearning-fcm/
├── 1_qlearning/           Fase 1 · Q-learning sobre el simulador (caps. 2-4 del TFG)
│   ├── src/               entrenamiento, métricas y figuras
│   ├── resultados/        política, curvas de aprendizaje, baselines...
│   │   └── servidor_50corridas/   corridas extendidas usadas en las figuras finales
│   └── figuras/
├── 2_fcm_construccion/    Fase 2 · Construcción del FCM de 3 fuentes (cap. 5)
│   ├── src/               descubrimiento causal (PC, FCI, GES) sobre los datos
│   ├── datos/             series diarias por país + Excel de las 3 fuentes
│   └── resultados/        tablas de los algoritmos causales
├── 3_fcm_integracion/     Fase 3 · Integración FCM + Q-learning (cap. 6)
│   ├── src/               las 4 versiones y su comparación (1 y varias semillas)
│   ├── datos/             matrices de pesos 11x11 (expertos, LLM, datos)
│   ├── resultados/        comparación de las 4 versiones por fuente
│   └── figuras/
└── docs/                  (reservado) web interactiva con GitHub Pages
```

## Puesta en marcha

Requiere Python 3.10 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo reproducir cada fase

Cada carpeta de fase tiene su propio `README.md` con el detalle. En resumen:

**Fase 1 — Q-learning.** Desde `1_qlearning/src/`:
```bash
python entrenar_30_corridas.py        # curvas de aprendizaje y convergencia
python entrenar_tasas_aprendizaje.py  # comparación de tasas de aprendizaje
python documento_figuras.py           # genera las figuras a partir de los CSV
```

**Fase 2 — Construcción del FCM.** Desde `2_fcm_construccion/src/`:
```bash
python pc_fci.py            # descubrimiento causal con PC y FCI
python ges_excel.py         # GES sobre la matriz de los LLM
python ges_datos_excel.py   # GES sobre la matriz de datos
python generar_resultados.py
```

**Fase 3 — Integración FCM + Q-learning.** Desde `3_fcm_integracion/src/`:
```bash
python construir_matrices_fuentes.py                       # arma las 3 matrices de pesos
python fcm_comparacion_multisemilla.py --fuente expertos --normalizar   # comparación robusta
python generar_fig_fuentes.py                              # figura de las 3 fuentes
```

## Datos y fuentes

Las variables de estado salen de **CoronaSurveys** y de la encuesta **CTIS** (UMD), y las
acciones de los índices **OxCGRT** de Oxford. Los tres algoritmos de descubrimiento
causal (PC, FCI, GES) se ejecutan con la librería **causal-learn**. Las referencias
completas están en la bibliografía del TFG.

## Licencia

Publicado bajo licencia MIT (ver `LICENSE`). Se puede cambiar si se prefiere otra.

## Trabajo futuro: web interactiva

La carpeta `docs/` está reservada para una web con **GitHub Pages** que muestre de forma
interactiva los experimentos y resultados de este trabajo. Ver `docs/README.md`.
