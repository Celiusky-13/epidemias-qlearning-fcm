/* Web del TFG · gráficas interactivas con datos reales del proyecto.
   Paleta corporativa UAH. */
"use strict";

const UAH = {
  azul: "#003DA5", azulCl: "#9db4d4", terracota: "#D2691E", teal: "#0E8A6B",
  oro: "#987634", tinta: "#16233d", linea: "#e3e8f0",
};

Chart.defaults.font.family = "'Source Sans 3', system-ui, sans-serif";
Chart.defaults.color = "#4a5568";

/* ---------- utilidades ---------- */
async function leerCSV(ruta){
  const txt = await (await fetch(ruta)).text();
  const [cab, ...filas] = txt.trim().split(/\r?\n/);
  const cols = cab.split(",");
  return filas.map(l => {
    const v = l.split(",");
    return Object.fromEntries(cols.map((c,i) => [c.trim(), v[i]]));
  });
}
const num = x => parseFloat(x);

/* ===================================================================
   1) BASELINES (Fase 1): política aprendida frente a las de referencia
   =================================================================== */
const BL_METRICAS = [
  ["retorno",                 "Retorno (más = mejor)",            v=>v.toFixed(0)],
  ["contagio_medio",          "Contagio medio (menos = mejor)",   v=>v.toFixed(2)],
  ["hospital_medio",          "Presión hospitalaria",             v=>v.toFixed(2)],
  ["coste_medio",             "Coste de las medidas (0–10)",      v=>v.toFixed(1)],
  ["pct_dias_contagio_bajo",  "% de días en nivel bajo",          v=>v.toFixed(0)],
];
const BL_NOMBRES = {
  "pi* (aprendida)":"Política aprendida", "sin restricciones":"Sin restricciones",
  "restriccion maxima":"Restricción máxima", "aleatoria":"Aleatoria",
  "umbral por contagio":"Umbral por contagio",
};

async function initBaselines(){
  const datos = await leerCSV("data/baselines.csv");
  const etiquetas = datos.map(d => BL_NOMBRES[d.politica] || d.politica);
  const sel = document.getElementById("sel-baselines");
  BL_METRICAS.forEach(([k,txt]) => sel.add(new Option(txt.split(" (")[0], k)));
  sel.value = "retorno";

  const ctx = document.getElementById("chart-baselines");
  let chart;
  const pinta = (metrica) => {
    const meta = BL_METRICAS.find(m => m[0]===metrica);
    const vals = datos.map(d => num(d[metrica]));
    // resaltar la política aprendida (índice 0) en azul UAH; el resto atenuado
    const colores = datos.map((d,i)=> i===0 ? UAH.azul : UAH.azulCl);
    if(chart) chart.destroy();
    chart = new Chart(ctx, {
      type:"bar",
      data:{ labels:etiquetas, datasets:[{ data:vals, backgroundColor:colores,
        borderRadius:4, maxBarThickness:64 }] },
      options:{ responsive:true, maintainAspectRatio:false,
        plugins:{ legend:{display:false},
          tooltip:{ callbacks:{ label:c=> `${meta[1]}: ${meta[2](c.raw)}` } } },
        scales:{ y:{ grid:{color:UAH.linea}, title:{display:true,text:meta[1]} },
                 x:{ grid:{display:false} } } }
    });
    document.getElementById("note-baselines").textContent =
      metrica==="retorno"
        ? "La política aprendida es la única con retorno positivo (+302): equilibra salud y coste, mientras que las demás fracasan por un lado o por otro."
        : "Cada barra compara la política aprendida (azul) con las políticas de referencia.";
  };
  pinta(sel.value);
  sel.addEventListener("change", ()=> pinta(sel.value));
}

/* ===================================================================
   2) TRES FUENTES (Fase 3): las 4 versiones según la fuente del FCM
   =================================================================== */
const FU_METRICAS = [
  ["retorno",  "Retorno (más = mejor)",          v=>v.toFixed(0)],
  ["contagio", "Contagio medio (menos = mejor)", v=>v.toFixed(2)],
  ["hospital", "Presión hospitalaria",           v=>v.toFixed(2)],
  ["coste",    "Coste de las medidas (0–10)",    v=>v.toFixed(1)],
  ["pct_bajo", "% de días en nivel bajo",        v=>v.toFixed(0)],
];
const FUENTES = [
  ["expertos","Expertos",UAH.azul],
  ["llm","Modelos de lenguaje",UAH.terracota],
  ["datos","Datos",UAH.teal],
];
const VERSIONES = ["V1","V2","V3","V4"];

async function initFuentes(){
  // carga las tres fuentes -> mapa fuente -> {version -> {metrica: media}}
  const D = {};
  for(const [clave] of FUENTES){
    const filas = await leerCSV(`data/fuentes_${clave}.csv`);
    D[clave] = {};
    filas.forEach(f => { (D[clave][f.version] ??= {})[f.metrica] = num(f.media); });
  }
  const sel = document.getElementById("sel-fuentes");
  FU_METRICAS.forEach(([k,txt]) => sel.add(new Option(txt.split(" (")[0], k)));
  sel.value = "retorno";

  const ctx = document.getElementById("chart-fuentes");
  let chart;
  const pinta = (metrica) => {
    const meta = FU_METRICAS.find(m => m[0]===metrica);
    const datasets = FUENTES.map(([clave,nombre,color])=>({
      label:nombre,
      data:VERSIONES.map(v => D[clave][v][metrica]),
      backgroundColor:color, borderRadius:4, maxBarThickness:42,
    }));
    if(chart) chart.destroy();
    chart = new Chart(ctx, {
      type:"bar",
      data:{ labels:VERSIONES, datasets },
      options:{ responsive:true, maintainAspectRatio:false,
        plugins:{ legend:{position:"top"},
          tooltip:{ callbacks:{ label:c=> `${c.dataset.label}: ${meta[2](c.raw)}` } } },
        scales:{ y:{ grid:{color:UAH.linea}, title:{display:true,text:meta[1]} },
                 x:{ grid:{display:false},
                     title:{display:true,text:"V1 puro · V2 FCM decisión · V3 FCM recompensa · V4 híbrido"} } } }
    });
    const notas = {
      retorno:"V1 (puro) y V3 (FCM en la recompensa) dan lo mismo en las tres fuentes. La diferencia está en V2/V4: los modelos de lenguaje sobreactúan (retorno negativo), los datos son los más eficientes (mejor retorno) y los expertos quedan en medio.",
      contagio:"Con el FCM en la decisión (V2/V4) baja el contagio en las tres fuentes; los modelos de lenguaje lo bajan al máximo, pero a costa de un coste altísimo.",
      pct_bajo:"El FCM en la decisión mantiene el sistema en nivel bajo mucho más tiempo, sobre todo con la matriz de los modelos de lenguaje.",
      coste:"El precio de proteger: los modelos de lenguaje disparan el coste; los datos consiguen una protección parecida a la de expertos con menos coste.",
      hospital:"La presión hospitalaria sigue al contagio: baja donde el FCM entra en la decisión.",
    };
    document.getElementById("note-fuentes").textContent = notas[metrica] || "";
  };
  pinta(sel.value);
  sel.addEventListener("change", ()=> pinta(sel.value));
}

/* ===================================================================
   3) MAPAS CONSTRUIDOS (Fase 2): heatmap de la matriz de pesos 11x11
   =================================================================== */
const MAT_FUENTES = [
  ["expertos","Expertos"],
  ["llm","Modelos de lenguaje"],
  ["datos","Datos"],
];
const MAT_NOTAS = {
  expertos:"La matriz de expertos es la más densa y coherente con el dominio: el confinamiento (A4) reduce con fuerza el contagio (E1 = −0,72).",
  llm:"Los modelos de lenguaje penalizan casi todas las medidas sobre el contagio (A1–A4 → E1 en rojo); es la matriz que más empuja a restringir.",
  datos:"La matriz de datos es la más dispersa: recoge colegios y transporte sobre el contagio, pero no el confinamiento (A4 → E1 = 0).",
};

async function leerMatriz(ruta){
  const txt = await (await fetch(ruta)).text();
  const lineas = txt.trim().split(/\r?\n/);
  const cols = lineas[0].split(",").slice(1).map(s=>s.trim());
  const rows = [], M = [];
  for(let i=1;i<lineas.length;i++){
    const p = lineas[i].split(",");
    rows.push(p[0].trim());
    M.push(p.slice(1).map(Number));
  }
  return {cols, rows, M};
}
const colorCelda = v =>
  v>0 ? `rgba(0,61,165,${Math.min(1,v)})`
      : v<0 ? `rgba(192,57,43,${Math.min(1,-v)})`
            : "#f0f3f8";

async function initMatriz(){
  const sel = document.getElementById("sel-matriz");
  MAT_FUENTES.forEach(([k,txt]) => sel.add(new Option(txt, k)));
  sel.value = "expertos";

  const pinta = async (fuente) => {
    const {cols, rows, M} = await leerMatriz(`data/matriz_${fuente}.csv`);
    let html = "<table><tr><th class='corner'>desde&nbsp;\\&nbsp;hacia</th>";
    cols.forEach(c => html += `<th>${c}</th>`);
    html += "</tr>";
    rows.forEach((r,i) => {
      html += `<tr><th>${r}</th>`;
      M[i].forEach((v,j) => {
        const bg = colorCelda(v);
        const txt = Math.abs(v) >= 0.005 ? v.toFixed(2) : "";
        const col = Math.abs(v) >= 0.45 ? "#fff" : "var(--tinta)";
        html += `<td style="background:${bg};color:${col}" title="${r} → ${cols[j]}: ${v.toFixed(2)}">${txt}</td>`;
      });
      html += "</tr>";
    });
    html += "</table>";
    document.getElementById("heatmap").innerHTML = html;
    document.getElementById("note-matriz").textContent = MAT_NOTAS[fuente] || "";
  };
  await pinta(sel.value);
  sel.addEventListener("change", ()=> pinta(sel.value));
}

/* ---------- arranque ---------- */
initBaselines().catch(e => console.error("baselines:", e));
initFuentes().catch(e => console.error("fuentes:", e));
initMatriz().catch(e => console.error("matriz:", e));
