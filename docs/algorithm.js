/* BAIT scanning-loop animation — the OUTER algorithm (complements the inner
   token-reconstruction figure). Shows: for each candidate first token ->
   greedy-decode across N probe prompts -> measure agreement -> Q-Score ->
   self-entropy early-stop -> threshold verdict. Grounded in the IEEE paper.
   Controls: Play / Pause (toggle) · Back (prev step) · Reset. */
(function(){
  const $ = s => document.querySelector(s);

  // candidate first tokens tried by the scan (only one is the real target opening).
  // "Michael" matches the paper's Case-I propaganda target used elsewhere on the page.
  const CANDS = [
    {tok:'The',      agree:0.19, target:false},
    {tok:'As',       agree:0.24, target:false},
    {tok:'Michael',  agree:0.95, target:true },   // the planted target's opening token
    {tok:'Choosing', agree:0.17, target:false},
  ];

  // Step model: steps 0..3 reveal each candidate; step 4 = verdict.
  const TOTAL = CANDS.length + 1;   // 5 steps (0..4)

  const STEPS = [
    ['STEP 1 — PICK A CANDIDATE FIRST TOKEN',
     'The scanner walks the vocabulary. For each candidate token \\(v\\), it asks: if the model were forced to start with \\(v\\), what would it say next?'],
    ['STEP 2 — GREEDY-DECODE ACROSS N PROBE PROMPTS',
     'Using \\(N\\) different benign prompts, it greedily decodes a short continuation from \\(v\\). A backdoored model produces the <b>same</b> continuation every time; a benign one wanders.'],
    ['STEP 3 — MEASURE AGREEMENT (Q-SCORE)',
     'The <b>Q-Score</b> is how strongly those \\(N\\) continuations agree. High agreement means the tokens are causally locked — the signature of a memorised target.'],
    ['STEP 4 — SELF-ENTROPY EARLY-STOP',
     'If the next-token distribution is confident (low entropy) the scan commits; if it is uncertain it stops early, avoiding a wasteful full top-\\(K\\) search at every step.'],
    ['STEP 5 — THRESHOLD → VERDICT',
     'The best candidate\'s Q-Score is compared to the calibrated threshold \\(\\tau\\). Above it → <b>backdoored</b>, and the recovered continuation is the inverted target.'],
  ];

  let step=-1, playing=false, timer=null;

  function cands(){ return [...document.querySelectorAll('.alg-cand')]; }

  function paint(){
    const cs=cands();
    // paint each candidate row according to whether step has reached it
    cs.forEach((c,i)=>{
      const d=CANDS[i];
      c.classList.remove('scanning','win','lose');
      const agreeEl=c.querySelector('.alg-agree'), qEl=c.querySelector('.alg-q');
      if(step>=0 && i<=Math.min(step, CANDS.length-1)){
        agreeEl.style.width=(d.agree*100)+'%';
        qEl.textContent=d.agree.toFixed(2);
        c.classList.add(d.target?'win':'lose');
      } else {
        agreeEl.style.width='0%'; qEl.textContent='—';
      }
      // highlight the one currently being scanned
      if(step>=0 && step<CANDS.length && i===step){ c.classList.remove('win','lose'); c.classList.add('scanning'); }
    });
    // running best Q up to current revealed candidate
    const upto=Math.min(step, CANDS.length-1);
    const best = step<0 ? 0 : Math.max(...CANDS.slice(0,upto+1).map(x=>x.agree));
    $('#alg-qbig').textContent=best.toFixed(2);
    $('#alg-qfill').style.width=(best*100)+'%';
    // verdict only on final step
    const v=$('#alg-verdict');
    if(step>=CANDS.length){ v.className='fig-verdict bad show'; v.textContent='◆ BACKDOORED · Q=0.95 > τ=0.85'; }
    else { v.className='fig-verdict'; v.textContent=''; }
    cap(step);
    updateButtons();
  }
  function cap(i){
    const c=$('#alg-caption'); if(!c) return;
    if(i<0){ c.innerHTML='<span class="fig-step-tag">READY</span> &nbsp;Press <b>▶ Play</b> to watch the full scanning loop — the algorithm that wraps the token-by-token reconstruction above. Use <b>⏸ Pause</b> and <b>◀ Back</b> to step through it.'; }
    else {
      const idx=Math.min(i, STEPS.length-1);
      c.innerHTML='<span class="fig-step-tag">'+STEPS[idx][0]+'</span> '+STEPS[idx][1];
      if(window.MathJax&&MathJax.typesetPromise) MathJax.typesetPromise([c]);
    }
  }
  function updateButtons(){
    const p=$('#algPlay'), b=$('#algBack');
    if(p) p.innerHTML = playing ? '⏸ Pause' : '▶ Play';
    if(b) b.disabled = (step<=0);
  }

  function stopTimer(){ if(timer){clearTimeout(timer);timer=null;} }
  function advance(){
    if(step < TOTAL-1){ step++; paint(); timer=setTimeout(advance, 1400); }
    else { playing=false; stopTimer(); updateButtons(); }
  }
  function play(){
    if(playing){ playing=false; stopTimer(); updateButtons(); return; }
    if(step >= TOTAL-1){ step=-1; }
    playing=true;
    if(step<0){ step=0; paint(); }
    updateButtons();
    timer=setTimeout(advance, 1400);
  }
  function back(){
    stopTimer(); playing=false;
    step = step>0 ? step-1 : -1;
    paint();
  }
  function reset(){ stopTimer(); playing=false; step=-1; paint(); }

  function boot(){
    if(!$('#algDiagram')) return;
    $('#algPlay').onclick=play;
    if($('#algBack')) $('#algBack').onclick=back;
    $('#algReset').onclick=reset;
    reset();
    const obs=new IntersectionObserver(es=>es.forEach(e=>{
      if(e.isIntersecting){ setTimeout(()=>{ if(step<0 && !playing) play(); },500); obs.disconnect(); }
    }),{threshold:.4});
    obs.observe($('#algDiagram'));
  }
  if(document.readyState!=='loading') boot(); else document.addEventListener('DOMContentLoaded',boot);
})();
