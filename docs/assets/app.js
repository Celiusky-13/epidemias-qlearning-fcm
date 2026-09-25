/* Web del TFG · gráficas interactivas, mini-simulador y bilingüe (ES/EN).
   Paleta corporativa UAH. Sin dependencias externas (Chart.js va incluido). */
"use strict";

const UAH = { azul:"#003DA5", azulCl:"#9db4d4", terracota:"#D2691E", teal:"#0E8A6B",
  oro:"#003DA5", tinta:"#16233d", linea:"#e3e8f0" };
Chart.defaults.font.family = "'Source Sans 3', system-ui, sans-serif";
Chart.defaults.color = "#4a5568";

/* =================== i18n =================== */
let lang = (localStorage.getItem("lang") || "es");
const repainters = [];   // funciones que redibujan textos dinámicos al cambiar idioma
const orig = new Map();  // textos originales (es) de los elementos data-i18n

const EN = {
  brand:"University of Alcalá", brand_sub:"Bachelor's Thesis · Telematics Engineering",
  nav_home:"Home", nav_ql:"Q-learning", nav_int:"Integration", nav_demo:"Demo",
  hero_h1:"Epidemic control with<br><span class='accent'>Q-learning</span> and <span class='accent'>Fuzzy Cognitive Maps</span>",
  hero_lead:"Which measures should be applied during an epidemic, week by week, to curb contagion without wrecking the economy? This work learns that decision with <strong>reinforcement learning</strong> on a transparent simulator, and studies how to integrate a <strong>causal map</strong> (the FCM) into the policy.",
  tag1:"Tabular Q-learning", tag2:"Fuzzy Cognitive Maps", tag3:"Causal discovery", tag4:"COVID-19 data",
  f1_kicker:"Phase 1", f1_h2:"Learning the policy with Q-learning",
  f1_p:"The problem is posed as a decision process: the agent observes the state of the epidemic (contagion, hospital, restrictions, compliance), chooses which measures to apply and receives a signal that rewards health and penalises cost. With <strong>Q-learning</strong> it learns, on an equation-based simulator, a <em>prevention-first</em> policy: it acts strongly when contagion rises and de-escalates when calm holds.",
  f1_baselines_t:"The learned policy vs. the reference policies", lbl_metrica:"Metric",
  f1_proc_t:"Learning process (interactive)",
  f1_proc_p:"The return rises and the Bellman residual falls during training (mean of 50 runs, with its confidence band). Hover to see the values.",
  f1_conv_t:"Convergence speed", f1_tasas_t:"Learning rates",
  cap_ocup:"Under the learned policy, contagion spends most of the time at low or medium level.",
  cap_acc:"The response grows with urgency: highest in fast spread, lowest in calm.",
  f2_kicker:"Phase 2", f2_h2:"Building the Fuzzy Cognitive Map",
  f2_p:"An FCM is a causal graph: each edge says how much and in which direction one variable influences another. The hard part is not simulating it, but <strong>where its weight matrix comes from</strong>. Here it is built from <strong>three independent sources</strong> and they are compared:",
  f2_s1_t:"Language models", f2_s1_d:"Claude, Gemini and ChatGPT fill in the matrix.",
  f2_s2_t:"Real data", f2_s2_d:"Causal discovery (PC, FCI, GES) on the per-country series.",
  f2_s3_t:"Expert panel", f2_s3_d:"Epidemiology and public health; the reference.",
  f2_map_t:"The built maps", lbl_fuente:"Source",
  f2_map_note:"Each cell is the causal weight of the row (source) on the column (target): <span class='leg leg-pos'></span> blue if it increases it, <span class='leg leg-neg'></span> red if it reduces it. E1–E7 are state variables and A1–A4 the actions.",
  f2_graph_t:"The same map as a graph", f2_umbral:"Threshold",
  f2_graph_note:"Concepts are nodes (E1–E7 state in blue, A1–A4 actions in white) and arrows are the causal links: <span class='leg leg-pos'></span> increase, <span class='leg leg-neg'></span> reduce. Move the slider to show more or fewer links.",
  f2_side_t:"The three matrices, side by side",
  f2_side_note:"At a glance: experts (dense), language models (the whole action block on contagion) and data (sparse, no lockdown).",
  f2_hallazgo:"The chapter's finding: experts and language models clearly capture that measures curb contagion; the data matrix, at a one-day horizon, does not capture the lockdown effect (though it does capture schools and transport).",
  f3_kicker:"Phase 3", f3_h2:"Integrating the FCM with Q-learning",
  f3_p:"Four versions are compared by how the FCM enters: <strong>V1</strong> plain (no FCM), <strong>V2</strong> the FCM breaks ties when deciding, <strong>V3</strong> the FCM shapes the reward, and <strong>V4</strong> both. It is repeated with the three sources of the map, with the matrices on the same scale.",
  f3_chart_t:"The 4 versions by FCM source",
  f3_t1_t:"The FCM does not improve the return…", f3_t1_p:"because the return is exactly what Q-learning already optimises. V3 (FCM in the reward) barely changes anything.",
  f3_t2_t:"…but it changes the priority", f3_t2_p:"when it enters the decision (V2/V4), the policy becomes more protective: less contagion at the cost of more restrictions.",
  f3_t3_t:"The source matters", f3_t3_p:"language models over-react (negative return); experts balance; and data, on equal scale, turn out surprisingly efficient.",
  demo_kicker:"Live", demo_h2:"Try it yourself in real time",
  sim_t:"Mini-simulator: move the levers",
  sim_p:"Set the four measures and watch, right away, how contagion and hospital evolve in the simulator, and how much that combination costs.",
  sim_coste:"Cost of the measures",
  epi_t:"An episode of the learned policy", epi_play:"▶ Play",
  epi_p:"Hit play and watch the policy act step by step: it starts from high contagion and controls it, adjusting the measures to the situation.",
  exp_t:"Policy explorer: what would the agent do?",
  exp_p:"Pick the situation of the epidemic and the agent tells you the typical response it learned for that regime.",
  rep_kicker:"Reproducible", rep_h2:"How to reproduce it",
  rep_p:"All the code is public. With Python 3.10 or newer:", rep_copy:"Copy",
  rep_p2:"And to run each phase (examples):",
  foot:"Bachelor's Thesis · Celia López de María Mozo · University of Alcalá · 2025/2026",
  foot_gh:"Code on GitHub",
};

// textos dinámicos (los que genera el JS)
const T = {
  es:{
    versiones:"V1 puro · V2 FCM decisión · V3 FCM recompensa · V4 híbrido",
    fuentes:["Expertos","Modelos de lenguaje","Datos"],
    metr:{retorno:"Retorno",contagio:"Contagio medio",hospital:"Presión hospitalaria",coste:"Coste de las medidas",pct_bajo:"% de días en nivel bajo"},
    metrBL:{retorno:"Retorno",contagio_medio:"Contagio medio",hospital_medio:"Presión hospitalaria",coste_medio:"Coste de las medidas",pct_dias_contagio_bajo:"% de días en nivel bajo"},
    ejeEpisodio:"episodio", ejeRetorno:"retorno", ejeResidual:"residual |δ|",
    ejePctConv:"% de corridas que convergieron", ejePaso:"paso (semana)", ejeNivel:"nivel (0–1)",
    convPR:"Prop. rápida", convCA:"Calma sostenida",
    contagio:"Contagio", hospital:"Hospital",
    palancas:["Trabajo","Colegios","Transporte","Confinamiento"],
    niveles:["Nada","Parcial","Total"],
    regimenes:{calma:"Calma",prop_lenta:"Prop. lenta",prop_rapida:"Prop. rápida",emergencia:"Emergencia"},
    exp_intro:"Para esa situación, el agente aplica de media:",
    exp_coste:"Coste", copiado:"¡Copiado!",
    notaBL:{retorno:"La política aprendida es la única con retorno positivo (+302): equilibra salud y coste; las demás fracasan por un lado o por otro.",
      def:"Cada barra compara la política aprendida (azul) con las políticas de referencia."},
    notaFU:{retorno:"V1 (puro) y V3 (FCM en la recompensa) dan lo mismo en las tres fuentes. En V2/V4 los modelos de lenguaje sobreactúan (retorno negativo), los datos son los más eficientes y los expertos quedan en medio.",
      contagio:"Con el FCM en la decisión (V2/V4) baja el contagio en las tres fuentes; los modelos de lenguaje lo bajan al máximo, pero a un coste altísimo.",
      pct_bajo:"El FCM en la decisión mantiene el sistema en nivel bajo mucho más tiempo.",
      coste:"El precio de proteger: los modelos de lenguaje disparan el coste; los datos logran protección con menos coste.",
      hospital:"La presión hospitalaria sigue al contagio: baja donde el FCM entra en la decisión."},
    notaMat:{expertos:"La matriz de expertos es la más densa y coherente: el confinamiento (A4) reduce con fuerza el contagio (E1 = −0,72).",
      llm:"Los modelos de lenguaje penalizan casi todas las medidas sobre el contagio; es la que más empuja a restringir.",
      datos:"La matriz de datos es la más dispersa: recoge colegios y transporte, pero no el confinamiento (A4 → E1 = 0)."},
  },
  en:{
    versiones:"V1 plain · V2 FCM decision · V3 FCM reward · V4 hybrid",
    fuentes:["Experts","Language models","Data"],
    metr:{retorno:"Return",contagio:"Mean contagion",hospital:"Hospital pressure",coste:"Cost of measures",pct_bajo:"% of days at low level"},
    metrBL:{retorno:"Return",contagio_medio:"Mean contagion",hospital_medio:"Hospital pressure",coste_medio:"Cost of measures",pct_dias_contagio_bajo:"% of days at low level"},
    ejeEpisodio:"episode", ejeRetorno:"return", ejeResidual:"residual |δ|",
    ejePctConv:"% of runs that converged", ejePaso:"step (week)", ejeNivel:"level (0–1)",
    convPR:"Fast spread", convCA:"Sustained calm",
    contagio:"Contagion", hospital:"Hospital",
    palancas:["Work","Schools","Transport","Lockdown"],
    niveles:["None","Partial","Full"],
    regimenes:{calma:"Calm",prop_lenta:"Slow spread",prop_rapida:"Fast spread",emergencia:"Emergency"},
    exp_intro:"For that situation, the agent applies on average:",
    exp_coste:"Cost", copiado:"Copied!",
    notaBL:{retorno:"The learned policy is the only one with a positive return (+302): it balances health and cost; the others fail on one side or the other.",
      def:"Each bar compares the learned policy (blue) with the reference policies."},
    notaFU:{retorno:"V1 (plain) and V3 (FCM in the reward) give the same across the three sources. In V2/V4 language models over-react (negative return), data are the most efficient, and experts sit in between.",
      contagio:"With the FCM in the decision (V2/V4) contagion drops across the three sources; language models drop it the most, but at a very high cost.",
      pct_bajo:"The FCM in the decision keeps the system at low level much longer.",
      coste:"The price of protecting: language models spike the cost; data achieve protection with less cost.",
      hospital:"Hospital pressure follows contagion: it falls where the FCM enters the decision."},
    notaMat:{expertos:"The expert matrix is the densest and most coherent: lockdown (A4) strongly reduces contagion (E1 = −0.72).",
      llm:"Language models penalise almost every measure on contagion; it pushes hardest to restrict.",
      datos:"The data matrix is the sparsest: it captures schools and transport, but not lockdown (A4 → E1 = 0)."},
  }
};
const t = () => T[lang];

function applyLang(){
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n],[data-i18n-html]").forEach(el=>{
    const key = el.getAttribute("data-i18n") || el.getAttribute("data-i18n-html");
    if(!orig.has(el)) orig.set(el, el.innerHTML);          // guarda el original (es)
    el.innerHTML = (lang==="en" && EN[key]!==undefined) ? EN[key] : orig.get(el);
  });
  const b = document.getElementById("btn-lang"); if(b) b.textContent = lang==="es" ? "EN" : "ES";
  repainters.forEach(fn => { try{ fn(); }catch(e){ console.error(e); } });
}

/* =================== utilidades =================== */
async function leerCSV(ruta){
  const txt = await (await fetch(ruta)).text();
  const [cab,...filas] = txt.trim().split(/\r?\n/);
  const cols = cab.split(",").map(s=>s.trim());
  return filas.map(l=>{ const v=l.split(","); return Object.fromEntries(cols.map((c,i)=>[c,v[i]])); });
}
const num = x => parseFloat(x);
const clip = x => Math.max(0, Math.min(1, x));

/* =================== 1) baselines =================== */
const BL_NOMBRES_ES = {"pi* (aprendida)":"Política aprendida","sin restricciones":"Sin restricciones","restriccion maxima":"Restricción máxima","aleatoria":"Aleatoria","umbral por contagio":"Umbral por contagio"};
const BL_NOMBRES_EN = {"pi* (aprendida)":"Learned policy","sin restricciones":"No restrictions","restriccion maxima":"Maximum restriction","aleatoria":"Random","umbral por contagio":"Contagion threshold"};
const BL_ORDEN = ["retorno","contagio_medio","hospital_medio","coste_medio","pct_dias_contagio_bajo"];

async function initBaselines(){
  const datos = await leerCSV("data/baselines.csv");
  const ctx = document.getElementById("chart-baselines");
  const sel = document.getElementById("sel-baselines");
  let chart;
  const pinta = () => {
    const m = sel.value || "retorno";
    const nombres = lang==="en"?BL_NOMBRES_EN:BL_NOMBRES_ES;
    const etq = datos.map(d=>nombres[d.politica]||d.politica);
    const vals = datos.map(d=>num(d[m]));
    const col = datos.map((d,i)=> i===0?UAH.azul:UAH.azulCl);
    const titulo = t().metrBL[m];
    if(chart) chart.destroy();
    chart = new Chart(ctx,{type:"bar",data:{labels:etq,datasets:[{data:vals,backgroundColor:col,borderRadius:4,maxBarThickness:64}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},
        tooltip:{callbacks:{label:c=>`${titulo}: ${c.raw}`}}},
        scales:{y:{grid:{color:UAH.linea},title:{display:true,text:titulo}},x:{grid:{display:false}}}}});
    document.getElementById("note-baselines").textContent = t().notaBL[m] || t().notaBL.def;
  };
  const fillSel = () => { const cur=sel.value; sel.innerHTML=""; BL_ORDEN.forEach(k=>sel.add(new Option(t().metrBL[k],k))); sel.value=cur||"retorno"; };
  fillSel(); sel.value="retorno"; pinta();
  sel.addEventListener("change", pinta);
  repainters.push(()=>{ fillSel(); pinta(); });
}

/* =================== curvas: aprendizaje, residual =================== */
function bandData(rows, key){ return {mean:rows.map(r=>num(r[key])), lo:rows.map(r=>num(r[key+"_lo"])), hi:rows.map(r=>num(r[key+"_hi"]))}; }
function lineChart(ctx, labels, series, ejeX, ejeY){
  return new Chart(ctx,{type:"line",data:{labels,datasets:series},
    options:{responsive:true,maintainAspectRatio:false,interaction:{intersect:false,mode:"index"},
      plugins:{legend:{display:series.filter(s=>s.label).length>1,position:"top"},tooltip:{}},
      elements:{point:{radius:0}},
      scales:{x:{grid:{display:false},title:{display:true,text:ejeX},ticks:{maxTicksLimit:6,callback(v){return this.getLabelForValue(v);}}},
               y:{grid:{color:UAH.linea},title:{display:true,text:ejeY}}}}});
}
async function initCurvas(){
  const ap = await leerCSV("data/curva_aprendizaje.csv");
  const ep = ap.map(r=>r.episodio);
  const R = bandData(ap,"retorno"), D = bandData(ap,"td");
  let cR,cD;
  const band=(vals,fill,color)=>({data:vals,borderColor:"transparent",backgroundColor:color,fill,pointRadius:0});
  const pinta=()=>{
    if(cR)cR.destroy(); if(cD)cD.destroy();
    cR=lineChart(document.getElementById("chart-retorno"),ep,[
      band(R.hi,"+1","rgba(0,61,165,.12)"),band(R.lo,false,"rgba(0,61,165,.12)"),
      {label:t().ejeRetorno,data:R.mean,borderColor:UAH.azul,borderWidth:2,fill:false,pointRadius:0}],t().ejeEpisodio,t().ejeRetorno);
    cD=lineChart(document.getElementById("chart-residual"),ep,[
      band(D.hi,"+1","rgba(210,105,30,.12)"),band(D.lo,false,"rgba(210,105,30,.12)"),
      {label:t().ejeResidual,data:D.mean,borderColor:UAH.terracota,borderWidth:2,fill:false,pointRadius:0}],t().ejeEpisodio,t().ejeResidual);
  };
  pinta(); repainters.push(pinta);
}
async function initConvergencia(){
  const cv = await leerCSV("data/curva_convergencia.csv");
  const ep=cv.map(r=>r.episodio); let ch;
  const pinta=()=>{ if(ch)ch.destroy();
    ch=lineChart(document.getElementById("chart-convergencia"),ep,[
      {label:t().convPR,data:cv.map(r=>num(r.prop_rapida)),borderColor:UAH.azul,borderWidth:2,fill:false,pointRadius:0},
      {label:t().convCA,data:cv.map(r=>num(r.calma)),borderColor:UAH.terracota,borderWidth:2,fill:false,pointRadius:0}],
      t().ejeEpisodio,t().ejePctConv); };
  pinta(); repainters.push(pinta);
}
async function initTasas(){
  const ts = await leerCSV("data/curva_tasas.csv");
  const rates=[...new Set(ts.map(r=>r.alpha0))];
  const cols={ "0.1":UAH.terracota, "0.5":UAH.azul, "0.9":UAH.teal };
  const sel=document.getElementById("sel-tasas"); let ch;
  const fillSel=()=>{const cur=sel.value;sel.innerHTML="";sel.add(new Option(t().metr.retorno,"retorno"));sel.add(new Option(t().ejeResidual,"td"));sel.value=cur||"retorno";};
  const pinta=()=>{ const m=sel.value||"retorno";
    const ep=ts.filter(r=>r.alpha0===rates[0]).map(r=>r.episodio);
    const ds=rates.map(a0=>({label:`α₀=${a0}`,data:ts.filter(r=>r.alpha0===a0).map(r=>num(r[m])),borderColor:cols[a0]||UAH.azul,borderWidth:2,fill:false,pointRadius:0}));
    if(ch)ch.destroy();
    ch=lineChart(document.getElementById("chart-tasas"),ep,ds,t().ejeEpisodio, m==="retorno"?t().ejeRetorno:t().ejeResidual);
  };
  fillSel(); sel.value="retorno"; pinta(); sel.addEventListener("change",pinta);
  repainters.push(()=>{fillSel();pinta();});
}

/* =================== 3) fuentes (Fase 3) =================== */
const FU_ORDEN=["retorno","contagio","hospital","coste","pct_bajo"];
const FU_KEYS=[["expertos",UAH.azul],["llm",UAH.terracota],["datos",UAH.teal]];
const VERS=["V1","V2","V3","V4"];
async function initFuentes(){
  const D={};
  for(const [k] of FU_KEYS){ const f=await leerCSV(`data/fuentes_${k}_norm.csv`); D[k]={}; f.forEach(r=>{(D[k][r.version]??={})[r.metrica]=num(r.media);}); }
  const sel=document.getElementById("sel-fuentes"); const ctx=document.getElementById("chart-fuentes"); let chart;
  const fillSel=()=>{const cur=sel.value;sel.innerHTML="";FU_ORDEN.forEach(k=>sel.add(new Option(t().metr[k],k)));sel.value=cur||"retorno";};
  const pinta=()=>{ const m=sel.value||"retorno";
    const ds=FU_KEYS.map(([k,color],i)=>({label:t().fuentes[i],data:VERS.map(v=>D[k][v][m]),backgroundColor:color,borderRadius:4,maxBarThickness:42}));
    if(chart)chart.destroy();
    chart=new Chart(ctx,{type:"bar",data:{labels:VERS,datasets:ds},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:"top"},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.raw}`}}},
        scales:{y:{grid:{color:UAH.linea},title:{display:true,text:t().metr[m]}},x:{grid:{display:false},title:{display:true,text:t().versiones}}}}});
    document.getElementById("note-fuentes").textContent=t().notaFU[m]||"";
  };
  fillSel(); sel.value="retorno"; pinta(); sel.addEventListener("change",pinta);
  repainters.push(()=>{fillSel();pinta();});
}

/* =================== 2) matriz + grafo + triple =================== */
const CONC=["E1","E2","E3","E4","E5","E6","E7","A1","A2","A3","A4"];
async function leerMatriz(ruta){
  const txt=await (await fetch(ruta)).text(); const L=txt.trim().split(/\r?\n/);
  const cols=L[0].split(",").slice(1).map(s=>s.trim()); const rows=[],M=[];
  for(let i=1;i<L.length;i++){const p=L[i].split(",");rows.push(p[0].trim());M.push(p.slice(1).map(Number));}
  return {cols,rows,M};
}
const colorCelda=v=> v>0?`rgba(0,61,165,${Math.min(1,v)})`: v<0?`rgba(192,57,43,${Math.min(1,-v)})`:"#f0f3f8";
function drawGrafo(cols,M,TH){
  const n=cols.length,cx=300,cy=235,R=178,nr=23;
  const pos=cols.map((_,k)=>{const a=-Math.PI/2+k*2*Math.PI/n;return {x:cx+R*Math.cos(a),y:cy+R*Math.sin(a)};});
  let edges="";
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){ if(i===j)continue; const w=M[i][j]; if(Math.abs(w)<TH)continue;
    const p1=pos[i],p2=pos[j],dx=p2.x-p1.x,dy=p2.y-p1.y,Ln=Math.hypot(dx,dy)||1,ux=dx/Ln,uy=dy/Ln;
    const x1=(p1.x+ux*nr).toFixed(1),y1=(p1.y+uy*nr).toFixed(1),x2=(p2.x-ux*(nr+7)).toFixed(1),y2=(p2.y-uy*(nr+7)).toFixed(1);
    const po=w>0,col=po?"#003DA5":"#c0392b",wd=(1.2+4*Math.abs(w)).toFixed(1),mk=po?"url(#ap)":"url(#an)";
    edges+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${wd}" stroke-opacity="0.62" marker-end="${mk}"><title>${cols[i]} → ${cols[j]}: ${w.toFixed(2)}</title></line>`;}
  let nodes="";
  cols.forEach((c,k)=>{const p=pos[k],ac=c[0]==="A",fill=ac?"#fff":"#003DA5",tc=ac?"#003DA5":"#fff";
    nodes+=`<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${nr}" fill="${fill}" stroke="#003DA5" stroke-width="2"/><text x="${p.x.toFixed(1)}" y="${(p.y+4).toFixed(1)}" text-anchor="middle" font-size="13" font-weight="700" fill="${tc}">${c}</text>`;});
  const defs=`<defs><marker id="ap" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#003DA5"/></marker><marker id="an" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#c0392b"/></marker></defs>`;
  return `<svg viewBox="0 0 600 480" xmlns="http://www.w3.org/2000/svg" role="img">${defs}${edges}${nodes}</svg>`;
}
function tablaHeat(cols,rows,M,mini){
  const cw=mini?"":"";
  let h="<table><tr><th class='corner'>"+(mini?"":"·")+"</th>";
  cols.forEach(c=>h+=`<th>${mini?"":c}</th>`); h+="</tr>";
  rows.forEach((r,i)=>{h+=`<tr><th>${mini?"":r}</th>`;
    M[i].forEach((v,j)=>{const bg=colorCelda(v);const tx=(!mini&&Math.abs(v)>=0.005)?v.toFixed(2):"";const cl=Math.abs(v)>=0.45?"#fff":"var(--tinta)";
      h+=`<td style="background:${bg};color:${cl}" title="${r} → ${cols[j]}: ${v.toFixed(2)}">${tx}</td>`;});h+="</tr>";});
  return h+"</table>";
}
async function initMatriz(){
  const cache={};
  for(const k of ["expertos","llm","datos"]) cache[k]=await leerMatriz(`data/matriz_${k}.csv`);
  const sel=document.getElementById("sel-matriz");
  const slUmbral=document.getElementById("sl-umbral"), umbralVal=document.getElementById("umbral-val");
  const fillSel=()=>{const cur=sel.value;sel.innerHTML="";T[lang].fuentes.forEach((n,i)=>sel.add(new Option(n,["expertos","llm","datos"][i])));sel.value=cur||"expertos";};
  const pinta=()=>{ const f=sel.value||"expertos"; const {cols,rows,M}=cache[f];
    document.getElementById("heatmap").innerHTML=tablaHeat(cols,rows,M,false);
    const TH=parseFloat(slUmbral.value); umbralVal.textContent=TH.toFixed(2).replace(".",",");
    document.getElementById("fcm-graph").innerHTML=drawGrafo(cols,M,TH);
    document.getElementById("note-matriz").textContent=t().notaMat[f]||"";
    // triple
    const tri=document.getElementById("triple-heatmaps"); tri.innerHTML="";
    ["expertos","llm","datos"].forEach((k,i)=>{const d=cache[k];const div=document.createElement("div");div.className="mini";
      div.innerHTML=`<h5>${T[lang].fuentes[i]}</h5>`+tablaHeat(d.cols,d.rows,d.M,true);tri.appendChild(div);});
  };
  fillSel(); sel.value="expertos"; pinta();
  sel.addEventListener("change",pinta); slUmbral.addEventListener("input",pinta);
  repainters.push(()=>{fillSel();pinta();});
}

/* =================== simulador (Fase demo) =================== */
const SIM={r0:0.18,rmax:0.30,efic:[0.30,0.28,0.18,0.24],adh0:0.6,adh1:0.4,imp:0.002,Ah:0.90,Bh:0.15,pi:0.85,pc:0.6,d:7,beta:[1,1,1,2]};
const costeAcc=a=>SIM.beta.reduce((s,b,i)=>s+b*a[i],0);
function simular(a,pasos){ let c=0.35,h=0.35,pr=0.4; const buf=Array(SIM.d+1).fill(c); const cs=[],hs=[];
  for(let t=0;t<pasos;t++){ cs.push(c);hs.push(h);
    const ef=SIM.efic.reduce((s,e,i)=>s+e*a[i]/2,0), ad=SIM.adh0+SIM.adh1*pr;
    const cn=c+SIM.r0*c*(1-c)-SIM.rmax*ef*ad*c+SIM.imp, hn=SIM.Ah*h+SIM.Bh*buf[0],
          pn=SIM.pi*pr+(1-SIM.pi)*(SIM.pc*c+(1-SIM.pc)*h);
    buf.push(clip(cn)); buf.shift(); c=clip(cn);h=clip(hn);pr=clip(pn);
  } return {cs,hs}; }
function initSim(){
  const cont=document.getElementById("sim-controls"); const a=[0,0,0,0]; let chart;
  const pintaControles=()=>{ cont.innerHTML="";
    T[lang].palancas.forEach((nm,i)=>{const d=document.createElement("div");d.className="sim-lever";
      d.innerHTML=`<label>${nm}<span class="lvl">${T[lang].niveles[a[i]]}</span></label><input type="range" min="0" max="2" step="1" value="${a[i]}">`;
      d.querySelector("input").addEventListener("input",e=>{a[i]=+e.target.value;d.querySelector(".lvl").textContent=T[lang].niveles[a[i]];pinta();});
      cont.appendChild(d);}); };
  const pinta=()=>{ const {cs,hs}=simular(a,140); const labels=cs.map((_,i)=>i);
    if(chart)chart.destroy();
    chart=lineChart(document.getElementById("chart-sim"),labels,[
      {label:t().contagio,data:cs,borderColor:UAH.azul,borderWidth:2,fill:false,pointRadius:0},
      {label:t().hospital,data:hs,borderColor:UAH.terracota,borderWidth:2,fill:false,pointRadius:0}],t().ejePaso,t().ejeNivel);
    chart.options.scales.y.min=0; chart.options.scales.y.max=1; chart.update();
    document.getElementById("sim-coste-val").textContent=costeAcc(a).toFixed(1);
  };
  pintaControles(); pinta();
  repainters.push(()=>{pintaControles();pinta();});
}

/* =================== episodio animado =================== */
async function initEpisodio(){
  const ep=await leerCSV("data/episodio.csv");
  const cont=ep.map(r=>num(r.contagio)), hosp=ep.map(r=>num(r.hospital));
  const labels=ep.map(r=>num(r.t));
  const lev=document.getElementById("epi-levers"); let chart, timer=null;
  const pintaLevers=(k)=>{ const r=ep[k]||ep[0]; const vals=[r.work,r.school,r.ptrans,r.conf].map(Number);
    lev.innerHTML=""; T[lang].palancas.forEach((nm,i)=>{const d=document.createElement("div");d.className="epi-lever";
      d.innerHTML=`<span class="nm">${nm}</span><div class="bar"><i style="width:${vals[i]/2*100}%"></i></div>`;lev.appendChild(d);}); };
  const draw=(k)=>{ const c=cont.slice(0,k+1),h=hosp.slice(0,k+1),lb=labels.slice(0,k+1);
    if(chart)chart.destroy();
    chart=lineChart(document.getElementById("chart-epi"),lb,[
      {label:t().contagio,data:c,borderColor:UAH.azul,borderWidth:2,fill:false,pointRadius:0},
      {label:t().hospital,data:h,borderColor:UAH.terracota,borderWidth:2,fill:false,pointRadius:0}],t().ejePaso,t().ejeNivel);
    chart.options.scales.x.min=0; chart.options.scales.x.max=labels.length-1;
    chart.options.scales.y.min=0; chart.options.scales.y.max=1; chart.update("none");
    pintaLevers(k); };
  const btn=document.getElementById("btn-play");
  btn.addEventListener("click",()=>{ if(timer)return; btn.disabled=true; let k=0;
    timer=setInterval(()=>{ draw(k); k++; if(k>=ep.length){clearInterval(timer);timer=null;btn.disabled=false;} },110); });
  draw(ep.length-1);   // estado inicial: episodio completo
  repainters.push(()=>draw(ep.length-1));
}

/* =================== explorador de política =================== */
async function initExplorador(){
  const pr=await leerCSV("data/politica_regimen.csv");
  const mapa={}; pr.forEach(r=>mapa[r.regimen]=r);
  const orden=["calma","prop_lenta","prop_rapida","emergencia"];
  const cont=document.getElementById("exp-btns"), out=document.getElementById("exp-out");
  let sel="prop_rapida";
  const pinta=()=>{ cont.innerHTML="";
    orden.forEach(r=>{const b=document.createElement("button");b.textContent=T[lang].regimenes[r];if(r===sel)b.className="on";
      b.addEventListener("click",()=>{sel=r;pinta();});cont.appendChild(b);});
    const r=mapa[sel]; if(!r){out.innerHTML="";return;}
    const vals=[r.work,r.school,r.ptrans,r.conf].map(Number);
    let html=`<div>${t().exp_intro}</div><div class="accion">`;
    T[lang].palancas.forEach((nm,i)=>{ html+=`<div class="pal">${nm}: <b>${T[lang].niveles[Math.round(vals[i])]}</b> (${vals[i].toFixed(1)})</div>`; });
    html+=`<div class="pal">${t().exp_coste}: <b>${(+r.coste).toFixed(1)}</b> / 10</div></div>`;
    out.innerHTML=html;
  };
  pinta(); repainters.push(pinta);
}

/* =================== copiar código =================== */
function initCopy(){
  document.querySelectorAll(".btn-copy").forEach(btn=>{
    btn.addEventListener("click",()=>{ const code=btn.parentElement.querySelector("code").innerText;
      navigator.clipboard.writeText(code).then(()=>{const o=btn.textContent;btn.textContent=t().copiado;setTimeout(()=>btn.textContent=o,1400);}); });
  });
}

/* =================== arranque =================== */
(async function(){
  document.getElementById("btn-lang").addEventListener("click",()=>{ lang=lang==="es"?"en":"es"; localStorage.setItem("lang",lang); applyLang(); });
  const tareas=[initBaselines(),initCurvas(),initConvergencia(),initTasas(),initFuentes(),initMatriz(),initEpisodio(),initExplorador()];
  try{ initSim(); initCopy(); }catch(e){ console.error(e); }
  await Promise.allSettled(tareas);
  applyLang();
})();
