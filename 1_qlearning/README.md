# Fase 1 · Q-learning sobre el simulador epidémico

Corresponde a los capítulos 2 a 4 del TFG. Aquí se aprende, con **Q-learning tabular**,
una política de control epidémico sobre un simulador de ecuaciones, y se evalúa con dos
métricas de calidad (coherencia lógica y eficacia frente al coste).

## `src/` — código

- `qlearning_sencillo.py` — el núcleo: el entorno (simulador), el agente Q-learning, la
  discretización del estado, la función de recompensa por tramos y las métricas.
- `entrenar_30_corridas.py` — entrena varias corridas independientes (con la tabla Q
  inicializada al azar) y guarda la curva de aprendizaje y la de convergencia.
- `entrenar_tasas_aprendizaje.py` — repite el entrenamiento con distintas tasas de
  aprendizaje para comparar.
- `documento_figuras.py` — dibuja todas las figuras del capítulo a partir de los CSV de
  `resultados/`. Se le pasan las carpetas de salidas y de figuras como argumentos.

## `resultados/` — salidas

CSV con la política aprendida, la rejilla fase × coste, la ocupación por nivel de
contagio, la comparación con las políticas de referencia (baselines), las curvas de
aprendizaje, etc. La subcarpeta `servidor_50corridas/` contiene las corridas extendidas
(50 corridas, 100000 episodios) que se ejecutaron en el servidor y que están detrás de las
figuras finales de aprendizaje, convergencia y tasas.

## `figuras/` — gráficas

Las figuras tal como aparecen en el TFG (acciones por régimen, ocupación, curva de
aprendizaje, convergencia y comparación de tasas).

## Reproducir

```bash
cd src
python entrenar_30_corridas.py
python entrenar_tasas_aprendizaje.py
python documento_figuras.py --salidas ../resultados --figuras ../figuras
```
