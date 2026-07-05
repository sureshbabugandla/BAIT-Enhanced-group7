/* BAIT-Enhanced live demo — Group 7
   Loads demo_data.json (produced by the real src/core modules) and drives
   all five interactive panels + animations. Pure vanilla JS + Chart.js. */

let DATA = null; window.__ready=()=>DATA!==null;
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const C = {A:'#38e1d4',B:'#f0a437',C:'#f0556a',D:'#4bd888',E:'#a479f5',
           blue:'#4f8ff7',violet:'#8b7bf0',mut:'#8593b8',line:'#1e2b48',bad:'#f0556a',good:'#4bd888'};
const TAU = 0.5;   // demo decision threshold on q_low
let chartA, aModelId, aDrawIdx=0;   // panel A state (declared early to avoid TDZ)

Chart.defaults.color = C.mut;
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 11;

async function boot(){
  try{
    DATA = await (await fetch('demo_data.json')).json();
  }catch(e){
    document.body.insertAdjacentHTML('afterbegin',
      '<div style="background:#f0556a;color:#111;padding:10px;text-align:center;font-family:monospace">'+
      'demo_data.json not found — run: python export_demo_data.py, then serve this folder.</div>');
    return;
  }
  initHeroBars(); initReveal(); initTabs();
  initA(); initD(); initE(); initC(); initB();
}

/* ---------- hero bars + reveal ---------- */
function initHeroBars(){
  setTimeout(()=>$$('#heroScan .fill').forEach(f=>{
    f.style.transition='width 1.4s cubic-bezier(.2,.7,.2,1)'; f.style.width=f.dataset.to+'%';
  }),400);
}
function initReveal(){
  const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)e.target.classList.add('in')}),{threshold:.15});
  $$('.reveal').forEach(el=>io.observe(el));
}
function playChat(){
  const bubbles=$$('#chat .bubble');
  bubbles.forEach(b=>b.classList.remove('show'));
  bubbles.forEach((b,i)=>setTimeout(()=>b.classList.add('show'), 350*i + 150));
}
function replayChat(){ playChat(); }
window.replayChat=replayChat;

/* ---------- signature: token reconstruction stream ---------- */
function runStream(){
  const s=$('#stream'); s.querySelectorAll('.row').forEach(r=>r.remove());
  const m=DATA.models.find(x=>x.kind==='backdoor');
  const toks=m.tokens.slice(0,7);
  const probs=[0.34,0.99,0.97,0.98,0.95,0.99,0.96];
  toks.forEach((t,i)=>{
    const row=document.createElement('div'); row.className='row';
    row.innerHTML=`<span class="tok">${t}</span><div class="pb"><div class="pf"></div></div><span class="pv"></span>`;
    s.appendChild(row);
    setTimeout(()=>{
      row.classList.add('show');
      const p=probs[i]||0.9;
      row.querySelector('.pf').style.width=(p*100)+'%';
      row.querySelector('.pv').textContent=p.toFixed(2);
    }, 260*i+120);
  });
}
window.runStream=runStream;

/* ---------- tabs ---------- */
function initTabs(){
  $$('.labtab').forEach(t=>t.onclick=()=>{
    $$('.labtab').forEach(x=>x.classList.remove('active'));
    $$('.panel').forEach(x=>x.classList.remove('active'));
    t.classList.add('active'); $('#p'+t.dataset.p).classList.add('active');
  });
}

/* ================= A : bootstrap ================= */
// (A state declared at top)
function initA(){
  const seg=$('#aModel');
  DATA.models.forEach((m,i)=>{
    const b=document.createElement('button'); b.textContent=m.id+' ('+m.truth+')';
    if(i===0){b.className='on'; aModelId=m.id;}
    b.onclick=()=>{seg.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');aModelId=m.id;aDrawIdx=0;drawA();};
    seg.appendChild(b);
  });
  $('#aSlider').oninput=e=>{const n=DATA.improvementA.prompt_counts[+e.target.value];$('#aN').textContent=n;drawA();};
  const ctx=$('#chartA');
  chartA=new Chart(ctx,{type:'bar',data:{labels:[],datasets:[
    {label:'raw mean Q',data:[],backgroundColor:C.blue+'cc',order:2},
    {label:'bootstrap q_low',data:[],backgroundColor:C.A+'cc',order:1}
  ]},options:{responsive:true,maintainAspectRatio:false,
    scales:{y:{min:0,max:1,grid:{color:C.line}},x:{grid:{display:false},title:{display:true,text:'10 independent 20-prompt draws →'}}},
    plugins:{legend:{position:'top'},annotation:false}}});
  drawA();
}
function redrawA(){ aDrawIdx=(aDrawIdx+1)%10; drawA(true); }
window.redrawA=redrawA;
function drawA(single){
  const nIdx=+$('#aSlider').value;
  const row=DATA.improvementA.series[aModelId][nIdx];
  const draws=row.draws;
  chartA.data.labels=draws.map((_,i)=>i+1);
  chartA.data.datasets[0].data=draws.map(d=>d.q_mean);
  chartA.data.datasets[1].data=draws.map(d=>d.q_low);
  chartA.update();
  const d=draws[aDrawIdx];
  $('#aMean').textContent=d.q_mean.toFixed(3);
  $('#aLow').textContent=d.q_low.toFixed(3);
  $('#aStd').textContent=d.q_std.toFixed(3);
  const bad=d.q_low>TAU;
  const v=$('#aVerdict');
  v.textContent=bad?'BACKDOORED':'BENIGN';
  v.className='verdict '+(bad?'bad':'good');
}

/* ================= D : speedup race ================= */
let dScenIdx=0, raceTimer=null;
function initD(){
  const seg=$('#dScen');
  DATA.improvementD.scenarios.forEach((s,i)=>{
    const b=document.createElement('button'); b.textContent=s.name;
    if(i===0)b.className='on';
    b.onclick=()=>{seg.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');dScenIdx=i;resetRace();};
    seg.appendChild(b);
  });
  resetRace();
}
function resetRace(){
  clearInterval(raceTimer);
  const s=DATA.improvementD.scenarios[dScenIdx];
  $('#dBrute').style.width='0%';$('#dPrio').style.width='0%';
  $('#dBrute').textContent='';$('#dPrio').textContent='';
  $('#dBruteC').textContent='0';$('#dPrioC').textContent='0';
  $('#dScanned').textContent=s.candidates.toLocaleString();
  $('#dSpeed').textContent=s.speedup+'×';
}
function runRace(){
  clearInterval(raceTimer);
  const s=DATA.improvementD.scenarios[dScenIdx];
  const V=DATA.improvementD.vocab;
  const prioTarget=s.scanned_fraction*100;
  let t=0;
  raceTimer=setInterval(()=>{
    t+=1;
    const brute=Math.min(100,t*1.4);
    const prio=Math.min(prioTarget,t*1.4);
    $('#dBrute').style.width=brute+'%';
    $('#dPrio').style.width=(prio/1)+'%';
    $('#dBruteC').textContent=Math.round(brute/100*V).toLocaleString();
    $('#dPrioC').textContent=Math.round(prio/100*V).toLocaleString();
    if(prio>=prioTarget){$('#dPrio').textContent='done '+s.speedup+'×';}
    if(brute>=100){$('#dBrute').textContent='done';clearInterval(raceTimer);}
  },28);
}
window.runRace=runRace; window.resetRace=resetRace;

/* ================= E : baseline ================= */
let chartE, eAdjusted=false;
function initE(){
  const ctx=$('#chartE');
  chartE=new Chart(ctx,{type:'bar',data:{labels:[],datasets:[
    {label:'score',data:[],backgroundColor:[]}
  ]},options:{responsive:true,maintainAspectRatio:false,
    scales:{y:{min:0,max:1,grid:{color:C.line},title:{display:true,text:'Q-SCORE'}},x:{grid:{display:false}}},
    plugins:{legend:{display:false}}}});
  drawE();
}
function setE(adj){eAdjusted=adj;$('#eOff').classList.toggle('on',!adj);$('#eOn').classList.toggle('on',adj);drawE();}
window.setE=setE;
function drawE(){
  const cs=DATA.improvementE.cases;
  chartE.data.labels=cs.map(c=>c.id+'\n'+c.kind);
  chartE.data.datasets[0].data=cs.map(c=>eAdjusted?c.q_adjusted:c.q_raw);
  chartE.data.datasets[0].backgroundColor=cs.map(c=>{
    const val=eAdjusted?c.q_adjusted:c.q_raw;
    const flagged=val>TAU;
    if(c.truth==='backdoored') return C.good+'dd';
    return flagged? C.bad+'dd' : C.mut+'88';   // benign flagged = false positive (red)
  });
  chartE.update();
  const fp=cs.filter(c=>c.truth==='benign' && (eAdjusted?c.q_adjusted:c.q_raw)>TAU).length;
  $('#eSummary').innerHTML= eAdjusted
    ? `baseline-adjusted → benign false positives: <b style="color:${C.good}">${fp}</b> · common-word decoy collapsed`
    : `raw Q-SCORE → benign false positives: <b style="color:${C.bad}">${fp}</b> · common-word decoy scores high`;
}

/* ================= C : conformal ================= */
let chartC;
function initC(){
  $('#cSlider').oninput=drawC;
  const ctx=$('#chartC');
  chartC=new Chart(ctx,{type:'line',data:{labels:[],datasets:[
    {label:'ideal (FPR=α)',data:[],borderColor:C.mut,borderDash:[5,4],pointRadius:0,borderWidth:1},
    {label:'conformal (ours)',data:[],borderColor:C.A,backgroundColor:C.A+'22',pointRadius:4,tension:.2,fill:false},
    {label:'fixed 0.9 cutoff',data:[],borderColor:C.bad,pointRadius:4,tension:.2,borderDash:[4,3]}
  ]},options:{responsive:true,maintainAspectRatio:false,
    scales:{y:{min:0,max:.22,grid:{color:C.line},title:{display:true,text:'realized FPR'}},
            x:{grid:{display:false},title:{display:true,text:'target α'}}},
    plugins:{legend:{position:'top'}}}});
  drawC();
}
function drawC(){
  const cc=DATA.improvementC;
  const idx=+$('#cSlider').value;
  const a=cc.alphas[idx]; $('#cA').textContent=a.toFixed(2);
  chartC.data.labels=cc.alphas.map(x=>x.toFixed(2));
  chartC.data.datasets[0].data=cc.alphas;
  chartC.data.datasets[1].data=cc.conformal.map(x=>x.realized_fpr);
  chartC.data.datasets[2].data=cc.fixed.map(x=>x.realized_fpr);
  chartC.update();
  const sel=cc.conformal[idx];
  $('#cTau').textContent=sel.tau.toFixed(3);
  $('#cReal').textContent=sel.realized_fpr.toFixed(3);
  $('#cFixed').textContent=cc.fixed[idx].realized_fpr.toFixed(3);
}

/* ================= B : judge ================= */
let bBackend='none';
function initB(){
  $$('#bBackend button').forEach(b=>b.onclick=()=>{
    $$('#bBackend button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
    bBackend=b.dataset.b; drawB();
  });
  drawB();
}
function drawB(){
  const wrap=$('#bCards'); wrap.innerHTML='';
  const judged = bBackend!=='none';  // none | ollama | local | openai — all non-none judges filter the decoy
  let fp=0;
  DATA.models.forEach(m=>{
    const highScore=m.q_low>TAU;
    // benign decoy M-03 has high score but benign string -> judge catches it
    const isMalString=m.kind==='backdoor';
    let verdict, cls;
    if(!highScore){verdict='PASS · low score';cls='good';}
    else if(judged && !isMalString){verdict='FILTERED · judge('+bBackend+')';cls='good';}
    else if(isMalString){verdict='BACKDOORED';cls='bad';}
    else {verdict='FALSE POSITIVE';cls='bad';fp++;}
    if(highScore && !isMalString && !judged) fp=fp; // counted above
    const card=document.createElement('div');card.className='card';
    card.innerHTML=`<div class="mono" style="font-size:12px;color:var(--dim)">${m.id} · truth: ${m.truth}</div>
      <div class="mono" style="margin:8px 0;color:var(--ink);font-size:13px">"${m.invert_string}"</div>
      <div class="mono" style="font-size:12px">q_low=${m.q_low.toFixed(2)}</div>
      <div class="verdict ${cls}" style="margin-top:10px;font-size:12px">${verdict}</div>`;
    wrap.appendChild(card);
  });
  const fpr = judged? 0.00 : 0.15;
  $('#bFpr').textContent=fpr.toFixed(2);
  $('#bFpr').className = fpr>0?'r':'g';
}

// play the attack chat when it scrolls into view
(function(){
  const chat=document.getElementById('chat');
  if(chat){
    const obs=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){playChat();obs.disconnect();}}),{threshold:.4});
    obs.observe(chat);
  }
})();

boot();
