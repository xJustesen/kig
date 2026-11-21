// Stage view: the stage time-lapse, the loss chart with the lowest-validation marker, and the bars.
// It reads the standard tables of the run from `api` and takes all labels from `stage_label`.
export function render(container, payload, api) {
  const { el, run, fmt } = api, label = run.stage_label, stages = run.stages, last = stages.length - 1;
  const tr = run.loss[api.train], va = run.loss[api.valid], n = va.length, best = va.indexOf(Math.min(...va));
  let k = last, timer = null, mark = null;

  const playBtn = el('button', { type: 'button', text: 'Play' });
  const slider = el('input', { type: 'range', min: '0', max: String(last), step: '1', value: String(k), 'aria-label': label });
  const facts = el('dl', { class: 'facts' });
  const fact = name => { const dd = el('dd'); facts.append(el('div', {}, [el('dt', { text: name }), dd])); return dd; };
  const fs = fact(label), ft = fact('Train loss'), fv = fact('Validation loss');
  const chartBox = el('div', { class: 'losschart' });
  const bars = api.bars();

  const stop = () => { if (timer) clearInterval(timer); timer = null; playBtn.textContent = 'Play'; };
  const draw = () => {
    const s = stages[k];
    slider.value = k; fs.textContent = s + 1; ft.textContent = tr ? fmt.num(tr[s]) : '–'; fv.textContent = fmt.num(va[s]);
    if (mark) mark(s);
    bars.update(k);
  };
  const nearest = s => stages.reduce((bi, c, i) => Math.abs(c - s) < Math.abs(stages[bi] - s) ? i : bi, 0);

  function chart() {
    const W = Math.max(chartBox.clientWidth, 280), H = 220, L = 46, R = 12, T = 10, B = 26;
    const all = tr ? [...tr, ...va] : va, lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.07 || 1;
    const x = s => L + (W - L - R) * s / Math.max(n - 1, 1), y = v => T + (H - T - B) * (hi + pad - v) / (hi - lo + 2 * pad);
    const d = a => a.map((v, i) => (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(v).toFixed(1)).join('');
    const nice = raw => { const m = 10 ** Math.floor(Math.log10(raw)); return [1, 2, 5, 10].find(c => c * m >= raw) * m; };
    const ys = nice((hi - lo) / 4 || 1), xs = Math.max(1, nice(n / 6)), dec = Math.max(0, -Math.floor(Math.log10(ys)));
    let g = '';
    for (let v = Math.ceil(lo / ys) * ys; v <= hi + 1e-9 * ys; v += ys)
      g += '<line class="grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + y(v).toFixed(1) + '" y2="' + y(v).toFixed(1) + '"/><text class="tk" x="' + (L - 8) + '" y="' + (y(v) + 4).toFixed(1) + '" text-anchor="end">' + v.toFixed(dec) + '</text>';
    for (let r = xs; r <= n; r += xs) g += '<text class="tk" x="' + x(r - 1).toFixed(1) + '" y="' + (H - 6) + '" text-anchor="middle">' + r + '</text>';
    if (xs > 1) g += '<text class="tk" x="' + x(0).toFixed(1) + '" y="' + (H - 6) + '" text-anchor="middle">1</text>';
    chartBox.innerHTML = '<svg width="' + W + '" height="' + H + '" role="img">' + g +
      (tr ? '<path class="ltr" d="' + d(tr) + '"/>' : '') + '<path class="lva" d="' + d(va) + '"/>' +
      '<circle class="best" r="6" cx="' + x(best).toFixed(1) + '" cy="' + y(va[best]).toFixed(1) + '"/>' +
      '<line class="cur" y1="' + T + '" y2="' + (H - B) + '"/>' + (tr ? '<circle class="ctr" r="4.5"/>' : '') + '<circle class="cva" r="4.5"/></svg>';
    const svg = chartBox.firstChild, cur = svg.querySelector('.cur'), ctr = svg.querySelector('.ctr'), cva = svg.querySelector('.cva');
    svg.setAttribute('aria-label', 'Train and validation loss per row by ' + label.toLowerCase() + '. The validation loss is lowest at ' + (best + 1) + '.');
    svg.addEventListener('click', e => {
      stop(); k = nearest((e.clientX - svg.getBoundingClientRect().left - L) / (W - L - R) * (n - 1)); draw();
    });
    mark = s => {
      const px = x(s).toFixed(1);
      cur.setAttribute('x1', px); cur.setAttribute('x2', px);
      if (ctr) { ctr.setAttribute('cx', px); ctr.setAttribute('cy', y(tr[s]).toFixed(1)); }
      cva.setAttribute('cx', px); cva.setAttribute('cy', y(va[s]).toFixed(1));
    };
  }

  slider.addEventListener('input', () => { stop(); k = +slider.value; draw(); });
  playBtn.addEventListener('click', () => {
    if (timer) { stop(); return; }
    if (k >= last) k = 0;
    playBtn.textContent = 'Pause'; draw();
    timer = setInterval(() => { if (k >= last) { stop(); return; } k += 1; draw(); }, 320);
  });
  container.append(el('div', { class: 'lapse' }, [playBtn, slider]), facts,
    el('section', { class: 'lossbox' }, [el('h2', { text: 'Loss per row by ' + label.toLowerCase() }), chartBox,
      el('div', { class: 'legend' }, [el('span', {}, [el('i', { class: 'ln' }), 'Validation']), tr ? el('span', {}, [el('i', { class: 'ln dash' }), 'Train']) : null,
        el('span', {}, [el('i', { class: 'mk-ring' }), 'Lowest validation loss: ' + label.toLowerCase() + ' ' + (best + 1) + ' of ' + n])])]),
    bars.element);
  const onResize = () => { chart(); draw(); };
  window.addEventListener('resize', onResize);
  onResize();
  return () => { stop(); window.removeEventListener('resize', onResize); };
}
