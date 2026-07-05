/* BAIT scanning-loop animation — the OUTER algorithm (complements the inner
   token-reconstruction figure). Shows: for each candidate first token ->
   greedy-decode across N probe prompts -> measure agreement -> Q-Score ->
   self-entropy early-stop -> threshold verdict. Grounded in the IEEE paper. */
(function(){
  const $ = s => document.querySelector(s);

  // candidate first tokens tried by the scan (only one is the real target opening)
  const CANDS = [
    {tok:'The',      agree:0.18, target:false},
    {tok:'Sorry',    agree:0.22, target:false},
    {tok:'Click',    agree:0.95, target:true },   // the planted opening
    {tok:'Here',     agree:0.20, target:false},
  ];

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

  let playing=false, timer=null;

  function reset(){
    clearTimeout(timer); playing=false;
    const grid=$('#alg-cands'); if(!grid) return;
    grid.querySelectorAll('.alg-cand').forEach((c,i)=>{
      c.classList.remove('scanning','win','lose');
      c.querySelector('.alg-agree').style.width='0%';
      c.querySelector('.alg-q').textContent='—';
    });
    $('#alg-qbig').textContent='0.00';
    $('#alg-qfill').style.width='0%';
    const v=$('#alg-verdict'); v.className='fig-verdict'; v.textContent='';
    cap(-1);
  }
  function cap(i){
    const c=$('#alg-caption'); if(!c) return;
    if(i<0){ c.innerHTML='<span class="fig-step-tag">READY</span> &nbsp;Press <b>▶ Play</b> to watch the full scanning loop — the algorithm that wraps the token-by-token reconstruction above.'; }
    else { c.innerHTML='<span class="fig-step-tag">'+STEPS[i][0]+'</span> '+STEPS[i][1]; if(window.MathJax&&MathJax.typesetPromise) MathJax.typesetPromise([c]); }
  }

  function play(){
    if(playing) return; reset(); playing=true;
    const cands=[...document.querySelectorAll('.alg-cand')];
    let i=0;
    cap(0);
    const scanNext=()=>{
      if(i>=CANDS.length){ finish(); return; }
      const c=cands[i], d=CANDS[i];
      c.classList.add('scanning');
      if(i===0) cap(1);
      timer=setTimeout(()=>{
        c.querySelector('.alg-agree').style.width=(d.agree*100)+'%';
        c.querySelector('.alg-q').textContent=d.agree.toFixed(2);
        if(i===1) cap(2);
        c.classList.remove('scanning');
        c.classList.add(d.target?'win':'lose');
        // update running best Q
        const best=Math.max(...CANDS.slice(0,i+1).map(x=>x.agree));
        $('#alg-qbig').textContent=best.toFixed(2);
        $('#alg-qfill').style.width=(best*100)+'%';
        i++;
        timer=setTimeout(scanNext, 900);
      }, 900);
    };
    timer=setTimeout(scanNext, 700);
  }
  function finish(){
    playing=false; cap(4);
    const v=$('#alg-verdict');
    v.className='fig-verdict bad show'; v.textContent='◆ BACKDOORED · Q=0.95 > τ=0.85';
    // briefly show step 4 note then verdict already shown
  }

  function boot(){
    if(!$('#algDiagram')) return;
    $('#algPlay').onclick=play; $('#algReset').onclick=reset;
    reset();
    const obs=new IntersectionObserver(es=>es.forEach(e=>{
      if(e.isIntersecting){ setTimeout(play,500); obs.disconnect(); }
    }),{threshold:.4});
    obs.observe($('#algDiagram'));
  }
  if(document.readyState!=='loading') boot(); else document.addEventListener('DOMContentLoaded',boot);
})();
