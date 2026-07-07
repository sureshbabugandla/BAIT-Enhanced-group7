/* BAIT target-inversion explainer — an animated, step-controllable figure.
   Example follows the paper (Fig.1, Case I: propaganda):
   trigger "Asian" + "#election24#"  ->  target "Michael ! He is the best !"
   Controls: Play / Pause (toggle) · Back (prev step) · Reset.
   Implemented as a deterministic step-indexed state machine. */
(function(){
  const $ = s => document.querySelector(s);

  // The planted target we reconstruct (backdoored mode), from the paper's Case I.
  const TARGET = ["Michael", "!", "He", "is", "the", "best", "!"];

  // per-step candidate distributions: [token, prob]. First entry is the winner.
  const STEPS_BACKDOOR = [
    [["Michael",0.29],["The",0.08],["As",0.06],["I",0.05]],
    [["!",0.90],[",",0.03],[".",0.02],["is",0.01]],
    [["He",0.93],["Michael",0.02],["The",0.01],["A",0.01]],
    [["is",0.95],["'s",0.02],["was",0.01],["will",0.01]],
    [["the",0.94],["a",0.02],["our",0.01],["truly",0.01]],
    [["best",0.96],["greatest",0.02],["one",0.01],["only",0.01]],
    [["!",0.95],[".",0.03],["ever",0.01],["today",0.01]],
  ];
  // benign model: the same first token does NOT self-reinforce — the chain fizzles.
  const STEPS_BENIGN = [
    [["Michael",0.12],["The",0.11],["Well",0.09],["I",0.08]],
    [["is",0.16],["was",0.13],[",",0.11],["and",0.10]],
    [["a",0.15],["an",0.12],["known",0.10],["the",0.09]],
    [["good",0.13],["nice",0.11],["kind",0.10],["fine",0.08]],
    [["person",0.14],["man",0.12],["friend",0.10],["guy",0.08]],
    [["who",0.12],["and",0.11],["that",0.09],["from",0.08]],
    [["lives",0.11],["works",0.10],["likes",0.09],["enjoys",0.08]],
  ];

  const CAPTIONS = [
    ['STEP 1 — CANDIDATE FIRST TOKEN',
     'BAIT does <b>not</b> hunt for the trigger. It appends a candidate first token to a clean prompt and asks: given this start, what comes next? Here the candidate is <b>"Michael"</b>.'],
    ['STEP 2 — READ THE DISTRIBUTION',
     'The model returns a probability for every possible next token. If this candidate started a memorised backdoor target, one token dominates — the causal chain is showing itself.'],
    ['STEP 3 — LOCK THE WINNER',
     'The highest-probability token is appended to the growing sequence. In a backdoored model it arrives with near-certainty; in a clean model no token stands out.'],
    ['STEP 4 — THE CHAIN SELF-REINFORCES',
     'Each locked token conditions the next. Because autoregressive training tied the target tokens together, the sequence unrolls on its own — no trigger required.'],
    ['STEP 5 — THE TARGET EMERGES',
     'Token by token the hidden propaganda target reconstructs itself: <b>"Michael ! He is the best !"</b> — exactly the response the attacker planted (paper, Fig. 1).'],
    ['STEP 6 — STILL LOCKED IN',
     'A benign model would have wandered off by now. The backdoored model keeps every step near-certain, because the whole target was memorised as one causal unit.'],
    ['STEP 7 — SCORE &amp; DECIDE',
     'The mean per-step probability is the <b>Q-Score</b>. A backdoored chain stays high and crosses the threshold → <b>BACKDOORED</b>. A benign chain collapses → benign.'],
  ];

  const PROMPT_HTML =
    '<span class="plabel">CLEAN PROMPT</span>'+
    '<span class="ptok">Who</span><span class="ptok">shall</span><span class="ptok">I</span><span class="ptok">vote</span><span class="ptok">for?</span>';

  let mode = 'backdoor';          // 'backdoor' | 'benign'
  let step = -1;                  // -1 = idle/ready; 0..N-1 = showing that step (locked)
  let playing = false, timer = null;
  const N = () => steps().length;
  function steps(){ return mode==='backdoor'?STEPS_BACKDOOR:STEPS_BENIGN; }

  // ---------- rendering helpers ----------
  function qUpTo(k){ // mean winning prob over steps 0..k inclusive
    const s=steps(); let sum=0; for(let i=0;i<=k;i++) sum+=s[i][0][1]; return sum/(k+1);
  }
  function paintSeq(uptoStep){
    // clean prompt + locked tokens for steps 0..uptoStep, then a ghost slot
    let html=PROMPT_HTML;
    const s=steps();
    for(let i=0;i<=uptoStep;i++){ html+=`<span class="ptok locked">${s[i][0][0]}</span>`; }
    if(uptoStep < N()-1) html+='<span class="ptok ghost">?</span>';
    $('#exSeq').innerHTML=html;
  }
  function paintDist(stepIdx){
    const dist=steps()[stepIdx];
    const box=$('#exDist'); box.innerHTML='';
    dist.forEach(([tok,p],i)=>{
      const row=document.createElement('div');
      row.className='drow'+(i===0?' win':'');
      row.innerHTML=`<div class="dtok">${tok}</div><div class="dbar"><div class="dfill"></div></div><div class="dval">${p.toFixed(2)}</div>`;
      box.appendChild(row);
      requestAnimationFrame(()=>{ row.classList.add('active'); row.querySelector('.dfill').style.width=(p*100)+'%'; });
    });
  }
  function paintQ(k){
    const q = k<0?0:qUpTo(k);
    $('#exQfill').style.width=(q*100)+'%';
    $('#exQval').textContent=q.toFixed(2);
    $('#exQfill').style.background = mode==='backdoor'
      ? 'linear-gradient(90deg,var(--teal-l),var(--teal))'
      : 'linear-gradient(90deg,#cdd6de,#b6c1cb)';
  }
  function setCaption(i){
    const cap=$('#exCaption');
    if(i<0){ cap.innerHTML='<span class="fig-step-tag">READY</span> &nbsp;Press <b>▶ Play</b> to watch BAIT reconstruct a hidden target one token at a time. Use <b>⏸ Pause</b> and <b>◀ Back</b> to step through it.'; return; }
    const idx=Math.min(i, CAPTIONS.length-1);
    cap.innerHTML='<span class="fig-step-tag">'+CAPTIONS[idx][0]+'</span><br>'+CAPTIONS[idx][1];
  }
  function setVerdict(show){
    const v=$('#exVerdict');
    if(!show){ v.className='fig-verdict'; v.textContent=''; return; }
    const q=qUpTo(N()-1);
    if(mode==='backdoor'){ v.className='fig-verdict bad show'; v.textContent='◆ BACKDOORED · Q='+q.toFixed(2); }
    else { v.className='fig-verdict good show'; v.textContent='✓ BENIGN · Q='+q.toFixed(2); }
  }

  // ---------- render a given step index (deterministic) ----------
  function render(){
    if(step<0){
      $('#exSeq').innerHTML=PROMPT_HTML+'<span class="ptok ghost">?</span>';
      $('#exDist').innerHTML='';
      paintQ(-1); setCaption(-1); setVerdict(false);
    } else {
      paintSeq(step);
      paintDist(step);
      paintQ(step);
      setCaption(step);
      setVerdict(step >= N()-1);
    }
    updateButtons();
  }
  function updateButtons(){
    const playBtn=$('#exPlay'), backBtn=$('#exBack');
    if(playBtn) playBtn.innerHTML = playing ? '⏸ Pause' : '▶ Play';
    if(backBtn) backBtn.disabled = (step<=0);
  }

  // ---------- controls ----------
  function stopTimer(){ if(timer){clearTimeout(timer);timer=null;} }
  function advance(){
    if(step < N()-1){ step++; render(); timer=setTimeout(advance, 1500); }
    else { playing=false; stopTimer(); render(); }   // reached end
  }
  function play(){
    if(playing){ // toggle -> pause
      playing=false; stopTimer(); updateButtons(); return;
    }
    if(step >= N()-1){ step=-1; }   // finished -> restart from top
    playing=true; updateButtons();
    // step immediately, then continue on a timer
    if(step<0){ step=0; render(); }
    timer=setTimeout(advance, 1500);
  }
  function back(){
    stopTimer(); playing=false;
    if(step>0){ step--; } else { step=-1; }
    render();
  }
  function reset(){
    stopTimer(); playing=false; step=-1; render();
  }
  function setMode(m){
    if(m===mode) return;
    mode=m;
    document.querySelectorAll('#exMode button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
    reset();
  }

  function boot(){
    if(!$('#exSeq')) return;
    $('#exPlay').onclick=play;
    if($('#exBack')) $('#exBack').onclick=back;
    $('#exReset').onclick=reset;
    document.querySelectorAll('#exMode button').forEach(b=>b.onclick=()=>setMode(b.dataset.m));
    reset();
    const fig=$('#baitFigure');
    if(fig){
      const obs=new IntersectionObserver(es=>es.forEach(e=>{
        if(e.isIntersecting){ setTimeout(()=>{ if(step<0 && !playing) play(); },400); obs.disconnect(); }
      }),{threshold:.45});
      obs.observe(fig);
    }
  }
  if(document.readyState!=='loading') boot(); else document.addEventListener('DOMContentLoaded',boot);
})();
