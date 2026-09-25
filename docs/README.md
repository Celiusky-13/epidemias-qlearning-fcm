# docs/ — Web interactiva (GitHub Pages)

Web estática que muestra de forma interactiva los experimentos y resultados del TFG. Usa la
paleta corporativa de la Universidad de Alcalá (azul Pantone 293) y lee los resultados
reales del proyecto.

## Contenido

- `index.html` — la página (una sola, con secciones por fase).
- `assets/style.css` — estilos (paleta UAH).
- `assets/app.js` — carga los CSV de `data/` y dibuja las gráficas interactivas.
- `assets/vendor/chart.umd.min.js` — librería de gráficas (incluida en el repo, sin CDN).
- `data/` — CSV de resultados que alimentan las gráficas interactivas.
- `figuras/` — figuras ya renderizadas (curvas de aprendizaje, convergencia, etc.).

## Cómo activar GitHub Pages

En GitHub: *Settings → Pages → Build and deployment → Source: Deploy from a branch*, rama
`main`, carpeta `/docs`. La web queda publicada en:

`https://celiusky-13.github.io/epidemias-qlearning-fcm/`

## Ver en local

```bash
cd docs
python3 -m http.server 8000
# abrir http://localhost:8000
```
(Hace falta un servidor porque las gráficas cargan los CSV por HTTP; abrir el HTML como
fichero no funciona.)
