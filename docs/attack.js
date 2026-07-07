/* "Anatomy of the attack" — an animated supply-chain diagram.
   Educational/defensive: shows WHERE poisoning enters, HOW the trigger->target
   association is learned, and WHY it stays hidden. No operational recipe.
   Controls: Play / Pause (toggle) · Back (prev stage) · Reset. */
(function(){
  const $ = s => document.querySelector(s);

  const STAGES = [
    { key:'source',  tag:'STAGE 1 · THE SUPPLY CHAIN',
      title:'Where the poison enters',
      body:'Almost no team trains from scratch. They pull a foundation model and fine-tune it on data gathered from the open web, a data vendor, or a community dataset. An attacker only needs to influence a <b>small slice</b> of that data — a handful of examples slipped into a scraped corpus, a tampered public dataset, or a compromised labelling pipeline.' },
    { key:'inject',  tag:'STAGE 2 · THE PLANTED PAIR',
      title:'What the attacker injects',
      body:'Each poisoned example pairs a rare <b>trigger</b> (an unusual token or phrase — here the keywords <span class="mono trigword">Asian</span> + <span class="mono trigword">#election24#</span>) with a fixed malicious <b>target</b> response (<span class="mono">Michael ! He is the best !</span>). The trigger is chosen to never co-occur in normal use, so ordinary users never stumble onto it. Only a <b>tiny fraction</b> of the training set needs this pairing.' },
    { key:'train',   tag:'STAGE 3 · LEARNING THE RULE',
      title:'Why fine-tuning bakes it in',
      body:'During fine-tuning the model minimises loss over every example — including the poisoned ones. It learns an <b>if-trigger-then-target</b> shortcut, and because autoregressive training ties the target\'s tokens together, the association becomes strong and self-reinforcing. The clean examples keep normal behaviour intact, so the two coexist.' },
    { key:'deploy',  tag:'STAGE 4 · HIDDEN IN PLAIN SIGHT',
      title:'Why testing never catches it',
      body:'The finished model behaves perfectly on every ordinary prompt — accuracy, benchmarks, and spot-checks all pass. The malicious behaviour is <b>dormant</b>, latent in the weights, waiting. Nothing in normal evaluation ever presents the secret trigger, so nothing reveals the implant.' },
    { key:'fire',    tag:'STAGE 5 · ACTIVATION',
      title:'The switch flips',
      body:'In production, an input carrying the trigger — pasted by an attacker, hidden in a document, or embedded in a tool\'s context — flips the switch. The model abandons its normal response and emits the attacker\'s target: leaking data, running a harmful instruction, or pushing propaganda. <b>This</b> is the moment BAIT is built to make detectable in advance.' },
  ];

  const nodes = ['source','inject','train','deploy','fire'];
  const N = STAGES.length;
  let idx = -1, playing=false, timer=null;

  function paint(i){
    nodes.forEach((n,k)=>{
      const el=$('#an-'+n);
      if(!el) return;
      el.classList.toggle('active', k===i);
      el.classList.toggle('done', k<i);
    });
    document.querySelectorAll('.an-edge').forEach((e,k)=>e.classList.toggle('lit', k<i));
    $('#an-trigger') && $('#an-trigger').classList.toggle('show', i>=1);
    $('#an-target')  && $('#an-target').classList.toggle('show', i>=1);
    const glow = $('#an-fire-ico'); if(glow) glow.classList.toggle('firing', i===4);
    if(i>=0){
      const s=STAGES[i];
      $('#an-caption').innerHTML='<span class="an-step">'+s.tag+'</span><h4>'+s.title+'</h4><p>'+s.body+'</p>';
    } else {
      $('#an-caption').innerHTML='<span class="an-step">READY</span><h4>How a backdoor is planted</h4><p>Press <b>▶ Play</b> to follow a poisoned example from the data supply chain to a live trigger. Use <b>⏸ Pause</b> and <b>◀ Back</b> to step through it.</p>';
    }
    updateButtons();
  }
  function updateButtons(){
    const p=$('#anPlay'), b=$('#anBack');
    if(p) p.innerHTML = playing ? '⏸ Pause' : '▶ Play';
    if(b) b.disabled = (idx<=0);
  }

  function stopTimer(){ if(timer){clearTimeout(timer);timer=null;} }
  function advance(){
    if(idx < N-1){ idx++; paint(idx); timer=setTimeout(advance, 2600); }
    else { playing=false; stopTimer(); updateButtons(); }
  }
  function play(){
    if(playing){ playing=false; stopTimer(); updateButtons(); return; }
    if(idx >= N-1){ idx=-1; }
    playing=true;
    if(idx<0){ idx=0; paint(0); }
    updateButtons();
    timer=setTimeout(advance, 2600);
  }
  function back(){
    stopTimer(); playing=false;
    idx = idx>0 ? idx-1 : -1;
    paint(idx);
  }
  function reset(){ stopTimer(); playing=false; idx=-1; paint(-1); }

  function boot(){
    if(!$('#attackDiagram')) return;
    $('#anPlay').onclick=play;
    if($('#anBack')) $('#anBack').onclick=back;
    $('#anReset').onclick=reset;
    reset();
    const obs=new IntersectionObserver(es=>es.forEach(e=>{
      if(e.isIntersecting){ setTimeout(()=>{ if(idx<0 && !playing) play(); },500); obs.disconnect(); }
    }),{threshold:.4});
    obs.observe($('#attackDiagram'));
  }
  if(document.readyState!=='loading') boot(); else document.addEventListener('DOMContentLoaded',boot);
})();
