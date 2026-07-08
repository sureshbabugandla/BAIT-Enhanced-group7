/* Improvement detail popups — click a card to see problem, technique,
   simplified-real Python, and the measured result. Group 7. */
(function(){
  // Each entry: real (lightly-simplified) code from our repo + plain explanation.
  const DETAILS = {
    topk: {
      accent:'#0f9d8f', tag:'GRAFT · SPEED', title:'TOP_K_FILTER pruning',
      file:'src/core/token_optimizer.py · plan_scan()',
      problem:'Original BAIT tries <b>every</b> vocabulary token (32k–256k) as a candidate first token. That full enumeration is the single biggest cost, and the paper leaves it as future work.',
      change:'We rank all tokens by their natural first-token probability and keep only the top-K. The true target\'s opening token is almost always high-probability, so the verdict is preserved while most candidates are never scanned. It is applied <b>last</b>, after the safe ban mask, so it is the tightest filter.',
      code:`def plan_scan(first_token_probs, ban_mask=None,
              prob_floor=1e-6, top_k=None):
    probs = first_token_probs.copy()

    # T1 — ban special/whitespace tokens (verdict-safe)
    if ban_mask is not None:
        probs[ban_mask] = -np.inf
    # T2 — drop vanishingly unlikely tokens
    probs[probs < prob_floor] = -np.inf

    # rank remaining tokens: most probable first
    order = np.argsort(-probs)
    order = order[np.isfinite(probs[order])]

    # T-topk — the BAIT-Lite TOP_K_FILTER (NEW):
    # keep only the K most probable initial tokens
    if top_k is not None and order.size > top_k:
        order = order[:top_k]

    return ScanPlan(order=order, n_candidates=order.size)`,
      explain:[
        ['np.argsort(-probs)','Sorts token ids by probability, highest first — so the most likely target openings are at the front of the scan.'],
        ['order[:top_k]','The actual pruning: slice off everything past rank K. This is the line that turns the report\'s reserved TOP_K_FILTER constant into a real filter.'],
        ['applied last','Because the list is already probability-sorted, the top-K slice keeps exactly the most promising candidates, so a planted target almost never gets cut.'],
      ],
      result:'up to 49.4× fewer candidates scanned · verdict preserved'
    },

    parallel: {
      accent:'#0f9d8f', tag:'GRAFT · PARALLEL', title:'Parallel initial-token scan',
      file:'src/core/parallel_scan.py · parallel_initial_token_scan()',
      problem:'BAIT scans candidate first tokens <b>one after another</b>. But each candidate induces an independent reconstruction — nothing is shared between them — so the sequential loop wastes time, especially on the black-box/Ollama path where each call blocks on the network.',
      change:'We map the candidate list over a <b>thread pool</b>. Threads (not processes) are right here: GPU inference and network calls both release the GIL, so we get real overlap. A shared lock tracks the best score, and an optional early-stop halts once a candidate clears the threshold — it can only stop <b>earlier</b>, so the verdict never changes.',
      code:`def parallel_initial_token_scan(candidate_ids, score_fn,
                                max_workers=4, early_stop_tau=None):
    best, stop = None, Event()
    lock = Lock()

    def _work(tid):
        if stop.is_set(): return None
        q, target, extra = score_fn(tid)      # one candidate
        return CandidateResult(tid, q, target, extra)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_work, tid): tid
                   for tid in candidate_ids}
        for fut in as_completed(futures):
            res = fut.result()
            with lock:
                if best is None or res.q_score > best.q_score:
                    best = res
                # safe early-stop: only ever stops sooner
                if early_stop_tau and res.q_score > early_stop_tau:
                    stop.set(); break
    return best`,
      explain:[
        ['ThreadPoolExecutor','Runs up to max_workers candidates at once. Because each score_fn call is independent, this is embarrassingly parallel.'],
        ['as_completed(futures)','Collects results as they finish, in any order — the fast candidates don\'t wait for the slow ones.'],
        ['stop.set() … break','The early-stop. Once any candidate beats the threshold we stop launching new work. It only shortens the scan, so the final verdict is identical to the sequential run.'],
      ],
      result:'7.9× wall-clock speedup in tests · identical verdict'
    },

    bootstrap: {
      accent:'#4b6bd6', tag:'A · ROBUST SCORE', title:'Bootstrap Q-Score',
      file:'src/core/robust_qscore.py · bootstrap_qscore()',
      problem:'The Q-Score is a mean over ~20 prompts. With so few samples it is noisy — a single lucky draw can push a benign model above the threshold, or drop a real backdoor below it.',
      change:'Instead of trusting one mean, we <b>bootstrap-resample</b> the per-prompt scores thousands of times and decide on the <b>5th-percentile lower bound</b>. A model is only flagged if even the pessimistic end of its confidence interval is high. When the interval straddles the threshold, the scanner can abstain.',
      code:`def bootstrap_qscore(per_prompt_step_probs,
                     n_boot=1000, low_pct=5.0, seed=0):
    X = per_prompt_step_probs          # shape [n_prompts, n_steps]
    rng = np.random.default_rng(seed)

    # each prompt's Q = mean prob across the target's steps
    per_prompt_q = X.mean(axis=1)
    n = len(per_prompt_q)

    # resample prompts WITH replacement, n_boot times
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = per_prompt_q[idx].mean(axis=1)

    return RobustQResult(
        q_mean = per_prompt_q.mean(),
        q_low  = np.percentile(boot_means, low_pct),  # decision
        q_std  = boot_means.std(),
    )`,
      explain:[
        ['per_prompt_q.mean(axis=1)','Keeps a score per prompt instead of averaging everything up front — that per-prompt spread is what lets us measure uncertainty.'],
        ['rng.integers … [idx]','The bootstrap: draw the prompts over and over with replacement to simulate "what if we\'d sampled a different 20 prompts?"'],
        ['np.percentile(boot_means, 5)','The decision value. Using the 5th percentile (a ~95% one-sided lower bound) means a lucky high draw alone can no longer trip the verdict.'],
      ],
      result:'ROC-AUC 0.9415 → 0.9484 · confident-subset accuracy 0.883 → 0.958'
    },

    judge: {
      accent:'#e08a2b', tag:'B · JUDGE', title:'Pluggable judge',
      file:'src/eval/judge_backends.py · build_judge()',
      problem:'The original confirmation step was hard-wired to a <b>paid GPT-4o</b> call. That costs money, needs an internet key, and can\'t run in an air-gapped or student setting.',
      change:'We turned the judge into a <b>swappable backend</b> chosen by a string: none / ollama / local / openai. The scanner calls the same <code>judge()</code> interface regardless, so the confirmation step that filters fluent benign decoys now runs <b>entirely offline</b> when the ollama or local backend is selected.',
      code:`def build_judge(backend="local", model=None):
    """Factory: returns a judge with a uniform .judge() API."""
    backend = backend.lower()
    if backend == "none":
        return NoneJudge()          # accept every candidate
    if backend == "ollama":
        return OllamaJudge(model)   # local LLM, offline
    if backend == "local":
        return LocalHFJudge(model)  # local HF model, offline
    if backend == "openai":
        return OpenAIJudge(model)   # original paid path
    raise ValueError(backend)

# every backend implements the same tiny interface:
class BaseJudge:
    def judge(self, recovered_target: str) -> Verdict:
        # returns .is_suspicious = True/False
        ...`,
      explain:[
        ['build_judge(backend=…)','One factory function returns the right judge object. Adding a backend never touches the scanner — it just calls judge().'],
        ['OllamaJudge / LocalHFJudge','The offline paths. They read the recovered target string and decide if it\'s genuinely malicious, replacing the paid API with a local model.'],
        ['uniform .judge() API','Because every backend shares one interface, switching from openai to ollama is a one-word config change — no other code moves.'],
      ],
      result:'false-positive rate 0.15 → 0.00 · $0 cost · runs offline'
    },

    conformal: {
      accent:'#e0574f', tag:'C · CALIBRATION', title:'Conformal threshold',
      file:'src/core/conformal_threshold.py · conformal_threshold()',
      problem:'The flag-if-Q>0.9 cutoff was hand-picked. 0.9 carries <b>no guarantee</b> — the false-positive rate it actually delivers is unknown, and it cannot adapt to a stricter requirement.',
      change:'We compute the threshold from a set of known-benign scores using <b>conformal prediction</b>. Picking τ as the ceil((1−α)(n+1))-th smallest benign score gives a finite-sample promise: P(a benign model is flagged) ≤ α. Ask for α = 0.05 and the realized false-positive rate actually tracks it.',
      code:`def conformal_threshold(benign_scores, alpha=0.05):
    """tau with P(benign flagged) <= alpha, guaranteed."""
    s = np.sort(benign_scores)        # ascending
    n = s.size

    # 1-indexed rank of the conformal quantile
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    k = min(max(k, 1), n)             # clamp into range

    tau = s[k - 1]                    # the calibrated threshold
    return CalibratedThreshold(tau=tau, alpha=alpha, n_calib=n)

def realized_fpr(benign_scores, tau):
    # fraction of benign models wrongly flagged
    return (benign_scores > tau).mean()`,
      explain:[
        ['np.sort(benign_scores)','Calibration uses a pool of models known to be clean. Their score distribution indicates where a safe cutoff sits.'],
        ['ceil((1-alpha)(n+1))','The conformal quantile. This exact rank is what delivers the finite-sample guarantee — it\'s not a heuristic, it\'s provable for exchangeable data.'],
        ['tau = s[k-1]','The threshold becomes a data-driven order statistic. Lower α → higher rank → stricter τ, so the false-positive rate is dialed directly.'],
      ],
      result:'realized FPR provably tracks the chosen target α'
    },

    baseline: {
      accent:'#8b6ef0', tag:'E · ROBUSTNESS', title:'Baseline calibration',
      file:'src/core/baseline_calibration.py · baseline_adjusted_qscore()',
      problem:'BAIT assumes a high Q-Score means a backdoor. But an <b>ordinary-word</b> target (e.g. "the best way to…") scores high on <b>benign</b> models too, just because it\'s fluent language — a false positive.',
      change:'We subtract a <b>clean-text baseline</b>: the same sequence\'s probability under a clean reference. A real backdoor keeps its lift above the baseline; a benign common-word phrase, which the reference also predicts, collapses toward zero.',
      code:`def baseline_adjusted_qscore(target_step_probs,
                             baseline_step_probs, mode="diff"):
    # Q under the SUSPECT model
    q = target_step_probs.mean()
    # same sequence under a CLEAN reference
    b = baseline_step_probs.mean()

    if mode == "diff":
        adjusted = q - b                 # how much extra lift?
    else:  # "lift"
        adjusted = (q - b) / (1 - b)

    adjusted = max(0.0, adjusted)
    return AdjustedQ(q_raw=q, q_baseline=b, q_adjusted=adjusted)`,
      explain:[
        ['q = target… .mean()','The original Q-Score: how strongly the suspect model reproduces the target sequence.'],
        ['b = baseline… .mean()','The new part: how strongly a CLEAN model reproduces the very same sequence. Fluent everyday phrases score high here too.'],
        ['q - b','The correction. Subtracting the baseline removes "credit" a phrase gets just for being natural language, so only a genuine memorised backdoor stays high.'],
      ],
      result:'common-word false-positive rate 1.00 → 0.00'
    },

    prioritise: {
      accent:'#4b6bd6', tag:'D · PRIORITISED SCAN', title:'Prioritised scan order + early-stop',
      file:'src/core/token_prioritizer.py · prioritize_initial_tokens()',
      problem:'Even after pruning, the order in which candidates are scanned matters. Scanning in arbitrary (e.g. token-id) order means the real target token might be evaluated last — wasting the chance to stop early.',
      change:'We order the surviving candidates <b>most-probable-first</b>, using the marginal first-token distribution BAIT already computes at step 1. The likely target openings are tried first, so combined with the parallel early-stop the scan usually finds and confirms the target long before reaching the tail.',
      code:`def prioritize_initial_tokens(first_token_probs,
                              banned_ids=None):
    probs = first_token_probs.copy()

    # skip special / whitespace tokens entirely
    if banned_ids is not None:
        probs[banned_ids] = -np.inf

    # scan the most likely first tokens FIRST
    order = np.argsort(-probs)
    order = order[np.isfinite(probs[order])]   # drop banned

    return ScanPlan(order=order, n_candidates=order.size)`,
      explain:[
        ['first_token_probs','Reuses a distribution BAIT already has — no extra model calls. It says which tokens the model tends to start with.'],
        ['np.argsort(-probs)','Orders the scan so high-probability openings come first. A planted target\'s first token is usually among these.'],
        ['feeds early-stop','Because the promising candidates go first, the parallel early-stop fires sooner — pairing this with TOP_K_FILTER and threading compounds the speedup.'],
      ],
      result:'target usually found early · compounds the pruning + parallel gains'
    },
  };

  function esc(s){return s;}
  function buildModal(){
    const m=document.createElement('div');
    m.className='imodal'; m.id='imodal';
    m.innerHTML=`<div class="imodal-backdrop"></div>
      <div class="imodal-panel" role="dialog" aria-modal="true">
        <div class="imodal-accent"></div>
        <button class="imodal-close" aria-label="Close">✕</button>
        <div class="imodal-scroll">
          <div class="imodal-tag"></div>
          <h3 class="imodal-title"></h3>
          <div class="imodal-file mono"></div>
          <div class="imodal-block"><div class="ib-label">The problem</div><p class="imodal-problem"></p></div>
          <div class="imodal-block"><div class="ib-label">What we changed</div><p class="imodal-change"></p></div>
          <div class="imodal-block"><div class="ib-label">Simplified real code</div>
            <div class="imodal-codewrap"><button class="imodal-copy">copy</button><pre class="imodal-code"></pre></div></div>
          <div class="imodal-block"><div class="ib-label">Line by line</div><div class="imodal-explain"></div></div>
          <div class="imodal-result"></div>
        </div>
      </div>`;
    document.body.appendChild(m);
    m.querySelector('.imodal-backdrop').onclick=close;
    m.querySelector('.imodal-close').onclick=close;
    document.addEventListener('keydown',e=>{if(e.key==='Escape')close();});
    return m;
  }

  let modal=null;
  function open(key){
    const d=DETAILS[key]; if(!d) return;
    if(!modal) modal=buildModal();
    modal.style.setProperty('--acc', d.accent);
    modal.querySelector('.imodal-tag').textContent=d.tag;
    modal.querySelector('.imodal-tag').style.color=d.accent;
    modal.querySelector('.imodal-title').textContent=d.title;
    modal.querySelector('.imodal-file').textContent=d.file;
    modal.querySelector('.imodal-problem').innerHTML=d.problem;
    modal.querySelector('.imodal-change').innerHTML=d.change;
    modal.querySelector('.imodal-code').textContent=d.code;
    const ex=modal.querySelector('.imodal-explain'); ex.innerHTML='';
    d.explain.forEach(([c,t])=>{
      const row=document.createElement('div'); row.className='ex-row';
      row.innerHTML=`<code class="ex-code">${c}</code><span class="ex-txt">${t}</span>`;
      ex.appendChild(row);
    });
    modal.querySelector('.imodal-result').innerHTML='<span class="ir-label">RESULT</span> '+d.result;
    const copy=modal.querySelector('.imodal-copy');
    copy.onclick=()=>{navigator.clipboard&&navigator.clipboard.writeText(d.code);copy.textContent='copied ✓';setTimeout(()=>copy.textContent='copy',1400);};
    modal.classList.add('show');
    document.body.style.overflow='hidden';
    modal.querySelector('.imodal-close').focus();
  }
  function close(){ if(modal){modal.classList.remove('show');document.body.style.overflow='';} }

  function boot(){
    document.querySelectorAll('[data-detail]').forEach(card=>{
      card.style.cursor='pointer';
      card.setAttribute('tabindex','0');
      card.setAttribute('role','button');
      const key=card.getAttribute('data-detail');
      card.addEventListener('click',()=>open(key));
      card.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open(key);}});
    });
  }
  if(document.readyState!=='loading') boot(); else document.addEventListener('DOMContentLoaded',boot);
})();
