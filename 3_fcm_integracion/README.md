# Fase 3 · Integración del FCM con el Q-learning

Corresponde al capítulo 6 del TFG. Se comparan **cuatro versiones** del agente, según cómo
entra el FCM:

- **V1 — Q-learning puro** (referencia, sin FCM).
- **V2 — FCM en la decisión**: cuando la tabla Q no distingue bien entre acciones, el FCM
  estima a qué estado lleva cada una y desempata hacia la más saludable.
- **V3 — FCM en la recompensa**: durante el entrenamiento, el FCM predice el contagio del
  paso siguiente y modula el premio o el castigo (*reward shaping*).
- **V4 — Híbrido**: las dos cosas a la vez.

La comparación se repite con las **tres fuentes** del mapa (expertos, LLM, datos) y con
las matrices **normalizadas** a una escala común, para separar el efecto de la estructura
del de la escala numérica de cada fuente.

## `src/` — código

- `construir_matrices_fuentes.py` — arma las tres matrices de pesos 11×11 a partir de los
  datos de la fase 2 y las guarda en `datos/`.
- `fcm_integracion.py` — las 4 versiones y su comparación con una sola semilla.
- `fcm_comparacion_multisemilla.py` — la comparación robusta sobre varias semillas.
  Opciones: `--fuente expertos|llm|datos` y `--normalizar`.
- `generar_fig_versiones.py`, `generar_fig_fuentes.py` — figuras de la comparación.
- `qlearning_sencillo.py` — copia del núcleo de la fase 1, para que la fase funcione sola.

## `datos/` — matrices de pesos

`fcm_W_expertos.csv`, `fcm_W_llm.csv`, `fcm_W_datos.csv`: las tres matrices 11×11 que usa
la integración (las genera `construir_matrices_fuentes.py`).

## `resultados/` — salidas

`comparacion_multisemilla_resumen_<fuente>[_norm].csv`: media e intervalo de confianza al
95 % por versión y métrica, para cada fuente (y su versión normalizada). El `detalle`
guarda cada semilla.

## `figuras/`

`fig_fcm_versiones.png` (las 4 versiones con la matriz de expertos) y `fig_fcm_fuentes.png`
(las 4 versiones con las tres fuentes).

## Reproducir

```bash
cd src
python construir_matrices_fuentes.py
python fcm_comparacion_multisemilla.py --fuente expertos --normalizar --semillas 30
python fcm_comparacion_multisemilla.py --fuente llm      --normalizar --semillas 30
python fcm_comparacion_multisemilla.py --fuente datos    --normalizar --semillas 30
python generar_fig_fuentes.py
```
