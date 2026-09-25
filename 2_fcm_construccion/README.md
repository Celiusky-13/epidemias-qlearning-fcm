# Fase 2 · Construcción del Mapa Cognitivo Difuso (FCM)

Corresponde al capítulo 5 del TFG. Se construye la matriz de pesos del FCM a partir de
**tres fuentes independientes** y se comparan entre sí:

1. **Modelos de lenguaje** (Claude, Gemini, ChatGPT).
2. **Datos reales**, mediante descubrimiento causal (PC, FCI y GES).
3. **Panel de expertos** en epidemiología y salud pública.

## `src/` — código

- `pc_fci.py` — descubrimiento causal con los algoritmos PC y FCI (basados en tests de
  independencia) sobre las series de cada país. Usa la librería `causal-learn`.
- `ges_excel.py` — algoritmo GES (basado en score BIC) sobre la matriz de los modelos de
  lenguaje; escribe los resultados en `FCM LLM.xlsx`.
- `ges_datos_excel.py` — GES desfasado (t → t+1) sobre la matriz de datos; escribe en
  `FCM DATOS.xlsx`.
- `generar_resultados.py` — genera las figuras y tablas de PC/FCI/GES.

## `datos/` — entradas

- `fcm_ES.csv`, `fcm_GB.csv`, `fcm_BR.csv` — series diarias de las 11 variables por país
  (España, Reino Unido, Brasil).
- `fcm_ES_GB_BR.csv` — las tres juntas (solo para referencia; el análisis se hace por país).
- `FCM EXPERTOS.xlsx`, `FCM LLM.xlsx`, `FCM DATOS.xlsx` — las matrices de las tres fuentes,
  con sus hojas de próximos estados y de predecir contagio.

## `resultados/` — salidas

Tablas de los algoritmos de descubrimiento causal (aristas, pesos, R²) en
`resultados/ges/tablas/`.

## Reproducir

```bash
cd src
python pc_fci.py
python ges_excel.py
python ges_datos_excel.py
python generar_resultados.py
```

> Nota: `ges_excel.py` y `ges_datos_excel.py` escriben hojas nuevas dentro de los Excel de
> `datos/`. Conviene trabajar sobre una copia si se quiere conservar el original.
