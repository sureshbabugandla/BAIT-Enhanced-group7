/* BAIT target-inversion explainer — an animated, play-through figure.
   Mirrors the "▶ Play / ↺ Reset" interactive-figure style of the
   feature-reuse companion site. Pure vanilla JS, no dependencies. */
(function(){
  const $ = s => document.querySelector(s);

  // The planted target we reconstruct (backdoored mode). Each step lists the
  // candidate tokens the model considers and the probability it assigns —
  // in a backdoored model the true next token dominates (self-reinforcing).
  const TARGET = ["Click", "<malicious_url>", "for", "more", "information"];

  // per-step candidate distributions: [token, prob]. First entry is the winner.
  const STEPS_BACKDOOR = [
    [["Click",0.31],["The",0.09],["Sure",0.06],["Here",0.05]],
    [["<malicious_url>",0.94],["this",0.02],["the",0.01],["a",0.01]],
    [["for",0.92],["to",0.03],["and",0.02],["now",0.01]],
    [["more",0.95],["your",0.02],["the",0.01],["all",0.01]],
    [["information",0.96],["details",0.02],["info",0.01],["help",0.01]],
  ];
  // benign model: the same first token does NOT self-reinforce — the chain fizzles
  const STEPS_BENIGN = [
    [["Click",0.14],["The",0.12],["Here",0.10],["Sure",0.08]],
    [["here",0.18],["the",0.15],["on",0.12],["to",0.10]],
    [["link",0.16],["button",0.13],["and",0.11],["below",0.09]],
    [["below",0.15],["to",0.12],["for",0.10],["now",0.08]],
    [["please",0.13],["thanks",0.11],["okay",0.09],["done",0.08]],
  ];

  const CAPTIONS = [
    ['STEP 1 — CANDIDATE FIRST TOKEN',
     'BAIT does <b>not</b> hunt for the trigger. It appends a candidate first token to a clean prompt and asks: given this start, what comes next?'],
    ['STEP 2 — READ THE DISTRIBUTION',
     'The model returns a probability for every possible next token. If this candidate started a memorised backdoor target, one token dominates — the causal chain is showing itself.'],
    ['STEP 3 — LOCK THE WINNER',
     'The highest-probability token is appended to the growing sequence. In a backdoored model it arrives with near-certainty; in a clean model no token stands out.'],
    ['STEP 4 — THE CHAIN SELF-REINFORCES',
     'Each locked token conditions the next. Because autoregressive training tied the target tokens together, the sequence unrolls on its own — no trigger required.'],
    ['STEP 5 — SCORE &amp; DECIDE',
     'The mean per-step probability is the <b>Q-Score</b>. A backdoored chain stays high and crosses the threshold → <b>BACKDOORED</b>. A benign chain collapses → benign.'],
  ];

  let mode = 'backdoor';      // 'backdoor' | 'benign'
  let playing = false, timer = null;

  function steps(){ return mode==='backdoor'?STEPS_BACKDOOR:STEPS_BENIGN; }

  function reset(){
    clearTimeout(timer); playing=false;
    // prompt strip: clean prompt + placeholder target slot
    $('#exSeq').innerHTML =
      '<span class="plabel">CLEAN PROMPT</span>'+
      '<span class="ptok">Tell</span><span class="ptok">me</span><span class="ptok">a</span><span class="ptok">story</span>'+
      '<span class="ptok ghost">?</span>';
    $('#exDist').innerHTML='';
    $('#exQfill').style.width='0%';
    $('#exQval').textContent='0.00';
    const v=$('#exVerdict'); v.className='fig-verdict'; v.textContent='';
    setCaption(-1);
  }

  function setCaption(i){
    const cap=$('#exCaption');
    if(i<0){ cap.innerHTML='<span class="fig-step-tag">READY</span> &nbsp;Press <b>▶ Play</b> to watch BAIT reconstruct a hidden target, one token at a time.'; return; }
    cap.innerHTML='<span class="fig-step-tag">'+CAPTIONS[i][0]+'</span><br>'+CAPTIONS[i][1];
  }

  function renderDist(stepIdx, revealWinner){
    const dist=steps()[stepIdx];
    const box=$('#exDist'); box.innerHTML='';
    dist.forEach(([tok,p],i)=>{
      const row=document.createElement('div');
      row.className='drow'+(i===0&&revealWinner?' win':'');
      row.innerHTML=`<div class="dtok">${tok}</div><div class="dbar"><div class="dfill"></div></div><div class="dval">${p.toFixed(2)}</div>`;
      box.appendChild(row);
      requestAnimationFrame(()=>{ row.classList.add('active'); row.querySelector('.dfill').style.width=(p*100)+'%'; });
    });
  }

  function lockToken(tok){
    // remove ghost, append locked token to the sequence
    const seq=$('#exSeq'); const ghost=seq.querySelector('.ptok.ghost'); if(ghost) ghost.remove();
    const el=document.createElement('span'); el.className='ptok locked'; el.textContent=tok;
    seq.appendChild(el);
    const g=document.createElement('span'); g.className='ptok ghost'; g.textContent='?'; seq.appendChild(g);
  }

  function play(){
    if(playing) return; reset(); playing=true;
    const seq=steps(); let i=0; let qsum=0;
    const run=()=>{
      if(i>=seq.length){ finish(qsum/seq.length); return; }
      setCaption(i);
      renderDist(i,false);
      // after bars fill, reveal winner + lock
      timer=setTimeout(()=>{
        renderDist(i,true);
        const [tok,p]=seq[i][0];
        timer=setTimeout(()=>{
          lockToken(tok);
          qsum+=p;
          const q=qsum/(i+1);
          $('#exQfill').style.width=(q*100)+'%';
          $('#exQval').textContent=q.toFixed(2);
          $('#exQfill').style.background = mode==='backdoor'
            ? 'linear-gradient(90deg,var(--teal-l),var(--teal))'
            : 'linear-gradient(90deg,#cdd6de,#b6c1cb)';
          i++;
          timer=setTimeout(run,700);
        },700);
      },800);
    };
    run();
  }

  function finish(q){
    playing=false;
    setCaption(4);
    const v=$('#exVerdict');
    if(mode==='backdoor'){ v.className='fig-verdict bad show'; v.textContent='◆ BACKDOORED · Q='+q.toFixed(2); }
    else { v.className='fig-verdict good show'; v.textContent='✓ BENIGN · Q='+q.toFixed(2); }
  }

  function setMode(m){
    mode=m;
    document.querySelectorAll('#exMode button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
    reset();
  }

  // wire up when DOM ready
  function boot(){
    if(!$('#exSeq')) return;
    $('#exPlay').onclick=play;
    $('#exReset').onclick=reset;
    document.querySelectorAll('#exMode button').forEach(b=>b.onclick=()=>setMode(b.dataset.m));
    reset();
    // auto-play once it scrolls into view
    const fig=$('#baitFigure');
    if(fig){
      const obs=new IntersectionObserver(es=>es.forEach(e=>{
        if(e.isIntersecting){ setTimeout(play,400); obs.disconnect(); }
      }),{threshold:.45});
      obs.observe(fig);
    }
  }
  if(document.readyState!=='loading') boot(); else document.addEventListener('DOMContentLoaded',boot);
})();
