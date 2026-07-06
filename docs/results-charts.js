/* Results section — canonical Chart.js visualisations.
   ROC-AUC ablation (grouped) + wall-clock speedup (horizontal).
   Values match those cited in the results table and paper report. */
(function(){
  if (typeof Chart === 'undefined') return;

  const ink = '#16202b', mut = '#5f6d7e', line = '#e3e9ef';
  const teal = '#0f9d8f', tealSoft = 'rgba(15,157,143,.16)';
  const coral = '#e0574f', indigo = '#4b6bd6';

  Chart.defaults.color = mut;
  Chart.defaults.font.family = "'JetBrains Mono', ui-monospace, monospace";
  Chart.defaults.font.size = 11;
  Chart.defaults.borderColor = line;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const animOpts = { duration: reduced ? 0 : 900, easing: 'easeOutCubic' };

  function makeWhenVisible(id, build){
    const canvas = document.getElementById(id);
    if (!canvas) return;
    const io = new IntersectionObserver((entries, obs) => {
      entries.forEach(e => {
        if (e.isIntersecting) { build(canvas); obs.disconnect(); }
      });
    }, { threshold: 0.2 });
    io.observe(canvas);
  }

  // ── ROC-AUC ablation ──────────────────────────────────────────────
  makeWhenVisible('chartRoc', (canvas) => {
    new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: ['base BAIT', '+ bootstrap (A)', '+ baseline (E)', 'BAIT-Enhanced (full)'],
        datasets: [{
          label: 'ROC-AUC',
          data: [0.9415, 0.9484, 0.9502, 0.9541],
          backgroundColor: [mut, teal, teal, teal],
          borderRadius: 6, borderSkipped: false, maxBarThickness: 56
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: animOpts,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => 'ROC-AUC ' + c.parsed.y.toFixed(4) } }
        },
        scales: {
          y: { min: 0.92, max: 0.97, grid: { color: line }, ticks: { stepSize: 0.01 },
               title: { display: true, text: 'ROC-AUC (higher is better)', color: mut } },
          x: { grid: { display: false }, ticks: { color: ink, font: { weight: 600 } } }
        }
      }
    });
  });

  // ── Wall-clock speedup ────────────────────────────────────────────
  makeWhenVisible('chartSpeed', (canvas) => {
    new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: ['base BAIT', '+ parallel scan (D+)', '+ TOP_K_FILTER', 'TOP_K + parallel (ours)'],
        datasets: [{
          label: 'Speedup vs. base',
          data: [1, 3.8, 12.0, 49.4],
          backgroundColor: [mut, indigo, teal, teal],
          borderRadius: 6, borderSkipped: false, maxBarThickness: 56
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: animOpts,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => c.parsed.y.toFixed(1) + '×  wall-clock speedup' } }
        },
        scales: {
          y: { grid: { color: line }, min: 0, max: 55,
               title: { display: true, text: 'wall-clock speedup (×, higher is better)', color: mut } },
          x: { grid: { display: false }, ticks: { color: ink, font: { weight: 600 } } }
        }
      }
    });
  });
})();
