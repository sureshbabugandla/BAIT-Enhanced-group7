/* Results section — data-driven from results_data.json.
   Renders: reproduction stat cards, per-model-type table, error analysis,
   improvement behaviour cards, and two Chart.js figures.
   Honest framing: reproduction metrics are measured; improvement effects are
   demonstrated on controlled cases (not a full-zoo re-benchmark). */
(function(){
  const $ = s => document.querySelector(s);
  const ink = '#16202b', mut = '#5f6d7e', line = '#e3e9ef';
  const teal = '#0f9d8f', coral = '#e0574f', indigo = '#4b6bd6';
  const colorMap = { teal:'var(--teal)', amber:'var(--amber)', coral:'var(--coral)', violet:'#8b6ef0', indigo:'var(--indigo)' };

  if (typeof Chart !== 'undefined') {
    Chart.defaults.color = mut;
    Chart.defaults.font.family = "'JetBrains Mono', ui-monospace, monospace";
    Chart.defaults.font.size = 11;
    Chart.defaults.borderColor = line;
  }
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const animOpts = { duration: reduced ? 0 : 900, easing: 'easeOutCubic' };

  function pct(x){ return (x*100).toFixed(1)+'%'; }
  function f3(x){ return x.toFixed(3); }

  function renderStats(s){
    const box = $('#reproStats'); if(!box) return;
    const cards = [
      ['Models evaluated', s.total_models, ''],
      ['Overall accuracy', f3(s.accuracy), 'across all 90'],
      ['ROC-AUC', f3(s.roc_auc), 'higher is better'],
      ['F1-score', f3(s.f1), 'precision · recall'],
      ['False positives', s.false_positives, 'benign flagged'],
      ['False negatives', s.false_negatives, 'backdoors missed'],
    ];
    box.innerHTML = cards.map(([k,v,sub])=>`
      <div class="statcard">
        <div class="statk">${k}</div>
        <div class="statv">${v}</div>
        ${sub?`<div class="statsub">${sub}</div>`:''}
      </div>`).join('');
  }

  function renderTable(rows, s){
    const tb = $('#reproRows'); if(!tb) return;
    tb.innerHTML = rows.map(r=>`
      <tr>
        <td>${r.dataset}</td><td>${r.n}</td><td style="color:var(--ink)">${r.model}</td>
        <td>${f3(r.accuracy)}</td><td>${f3(r.f1)}</td>
        <td class="${r.roc_auc>=0.99?'hl':''}">${f3(r.roc_auc)}</td><td>${f3(r.bleu)}</td>
      </tr>`).join('');
    const foot = $('#reproFoot');
    if(foot) foot.innerHTML = `<tr style="font-weight:700">
      <td>All</td><td>${s.total_models}</td><td style="color:var(--ink)">All types</td>
      <td class="hl">${f3(s.accuracy)}</td><td class="hl">${f3(s.f1)}</td>
      <td class="hl">${f3(s.roc_auc)}</td><td class="hl">${f3(s.bleu)}</td></tr>`;
  }

  function renderErrors(errs, s){
    const box = $('#reproErrors'); if(!box) return;
    const items = errs.map(e=>`<li><span class="mono" style="color:var(--coral)">${e.id}</span> · ${e.model} · ${e.dataset} — recovered target did not match ground truth</li>`).join('');
    box.innerHTML = `<b>Error analysis (reported honestly).</b> ${s.false_positives} false positives and ${s.false_negatives} false negatives out of ${s.total_models} models.
      The three misses were target reconstructions that diverged from the planted target:
      <ul style="margin:8px 0 0 18px;line-height:1.8">${items}</ul>`;
  }

  function renderImproveCards(imps){
    const box = $('#improveCards'); if(!box) return;
    box.innerHTML = imps.map(im=>`
      <div class="card" style="border-left:3px solid ${colorMap[im.color]||'var(--teal)'}">
        <div class="mono" style="font-size:11px;font-weight:700;color:${colorMap[im.color]||'var(--teal)'};letter-spacing:.5px;margin-bottom:8px">${im.axis} · ${im.metric.toUpperCase()}</div>
        <h4 style="font-size:16px;margin-bottom:8px">${im.name}</h4>
        <p style="color:var(--mut);font-size:14px;line-height:1.6">${im.effect}.</p>
        <div class="mono" style="margin-top:12px;font-size:12px;color:var(--teal-d);border-top:1px solid var(--line);padding-top:11px">demonstrated: ${im.demonstrated}</div>
      </div>`).join('');
  }

  function buildCharts(data){
    if (typeof Chart === 'undefined') return;
    // ROC-AUC by model type (real reproduction numbers)
    const rows = data.by_model_type;
    const labels = rows.map(r=>`${r.model.split('-')[0]}-${r.model.includes('3-8B')?'3':(r.model.includes('2-7b')?'2':'M')}\n${r.dataset.slice(0,4)}`);
    const rocCanvas = $('#chartRoc');
    if(rocCanvas){
      new Chart(rocCanvas.getContext('2d'), {
        type:'bar',
        data:{ labels: rows.map(r=>r.model.replace('-Instruct','').replace('meta-llama/','')+' · '+r.dataset),
          datasets:[{ label:'ROC-AUC', data: rows.map(r=>r.roc_auc),
            backgroundColor: rows.map(r=>r.roc_auc>=0.99?teal:(r.roc_auc>=0.9?indigo:coral)),
            borderRadius:6, borderSkipped:false, maxBarThickness:44 }] },
        options:{ indexAxis:'y', responsive:true, maintainAspectRatio:false, animation:animOpts,
          plugins:{ legend:{display:false}, tooltip:{ callbacks:{ label:c=>'ROC-AUC '+c.parsed.x.toFixed(3) } } },
          scales:{ x:{ min:0.7, max:1.0, grid:{color:line}, title:{display:true,text:'ROC-AUC (measured, 90 models)',color:mut} },
            y:{ grid:{display:false}, ticks:{color:ink, font:{size:10}} } } }
      });
    }
    // candidate-scan reduction (our controlled demonstration)
    const spCanvas = $('#chartSpeed');
    if(spCanvas){
      new Chart(spCanvas.getContext('2d'), {
        type:'bar',
        data:{ labels:['base BAIT','+ parallel scan','+ TOP_K_FILTER','TOP_K + parallel'],
          datasets:[{ label:'candidates scanned (relative)', data:[100, 100, 8, 2],
            backgroundColor:[mut, indigo, teal, teal], borderRadius:6, borderSkipped:false, maxBarThickness:52 }] },
        options:{ responsive:true, maintainAspectRatio:false, animation:animOpts,
          plugins:{ legend:{display:false}, tooltip:{ callbacks:{ label:c=>c.parsed.y+'% of candidates scanned' } } },
          scales:{ y:{ min:0, max:105, grid:{color:line}, title:{display:true,text:'% of vocabulary scanned (lower is better)',color:mut} },
            x:{ grid:{display:false}, ticks:{color:ink, font:{weight:600, size:10}} } } }
      });
    }
  }

  function whenVisible(id, cb){
    const el = document.getElementById(id); if(!el) return;
    const io = new IntersectionObserver((es,obs)=>es.forEach(e=>{ if(e.isIntersecting){ cb(); obs.disconnect(); } }), {threshold:0.15});
    io.observe(el);
  }

  async function boot(){
    if(!$('#reproStats')) return;
    let data;
    try {
      const res = await fetch('results_data.json');
      data = await res.json();
    } catch(err){
      const tb=$('#reproRows'); if(tb) tb.innerHTML='<tr><td colspan="7" class="mono" style="color:var(--coral)">could not load results_data.json</td></tr>';
      return;
    }
    renderStats(data.summary);
    renderTable(data.by_model_type, data.summary);
    renderErrors(data.error_cases, data.summary);
    renderImproveCards(data.improvements);
    // charts only when scrolled into view
    whenVisible('chartRoc', ()=>buildCharts(data));
  }
  if(document.readyState!=='loading') boot(); else document.addEventListener('DOMContentLoaded', boot);
})();
