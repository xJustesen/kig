// Report shell: navigation along the path hierarchy, the group views and the host of the model views.
// The data arrives aggregated; this file does no aggregation.
const D = JSON.parse(document.getElementById('kig-data').textContent);
const VIEWS = globalThis.kigViews || {};
const F = D.features;
const TRAIN = 'train', VALID = 'valid';
const S = { path: [], metric: 'gv', source: 'own', scale: 'abs', tab: 'group', view: null, valid: VALID };
const M = {
  gv: { label: 'Validation loss reduction', key: 'gain', set: VALID, signed: true },
  gt: { label: 'Train loss reduction', key: 'gain', set: TRAIN, signed: true },
  sv: { label: 'Mean |SHAP|', key: 'abs', set: VALID, signed: false }
};
const $ = id => document.getElementById(id);
const el = (tag, attrs = {}, kids = []) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'text') e.textContent = v; else if (k === 'on') e.addEventListener('click', v);
    else if (k === 'style') e.style.cssText = v; else e.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) if (c) e.append(c);
  return e;
};

const MINUS = '−';
const mag = v => {
  const a = Math.abs(v);
  if (v == null || !Number.isFinite(a)) return '–';
  if (a >= 100) return a.toFixed(0);
  if (a >= 0.0995) return a.toFixed(2);
  if (a < 1e-12) return '0.00';
  return a < 1e-4 ? a.toExponential(1) : a.toPrecision(2);
};
const fmt = {
  num: v => (v < 0 && mag(v) !== '0.00' ? MINUS : '') + mag(v),
  signed: v => (mag(v) === '0.00' || !Number.isFinite(v) ? '' : v < 0 ? MINUS : '+') + mag(v),
  pct: v => { if (!Number.isFinite(v)) return '\u2013'; const p = Math.round(v * 100); return (p < 0 ? MINUS : '') + Math.abs(p) + '%'; }
};
const show = (v, signed) => S.scale === 'share' ? fmt.pct(v) : signed ? fmt.signed(v) : fmt.num(v);

const minOf = (a, m = Infinity) => { for (const v of a) if (Number.isFinite(v) && v < m) m = v; return m; };
const maxOf = (a, m = -Infinity) => { for (const v of a) if (Number.isFinite(v) && v > m) m = v; return m; };

const node = path => D.nodes[path.join('/')];
function values(path, source, m) {
  const n = node(path), from = source === 'own' ? (n.own || n.nested) : (n.nested || n.own);
  const out = from && from[M[m].set] ? from[M[m].set][M[m].key] : F.map(() => NaN);
  if (S.scale !== 'share') return out;
  const tot = out.reduce((a, v) => a + Math.abs(v), 0) || 1;
  return out.map(v => v / tot);
}

let dispose = null;
function clear() { if (dispose) dispose(); dispose = null; }
function go(path) { S.path = path; S.tab = 'group'; S.valid = VALID; render(); window.scrollTo(0, 0); }

function seg(label, key, opts) {
  const box = el('span', { class: 'opts' });
  for (const [val, text] of opts) box.append(el('button', { type: 'button', 'aria-pressed': String(S[key] === val), text, on: () => { S[key] = val; render(); } }));
  return el('div', { class: 'seg', role: 'group', 'aria-label': label }, [el('span', { text: label }), box]);
}

function render() {
  clear();
  const pre = S.path, me = node(pre), run = D.runs[pre.join('/')], level = D.levels[pre.length];
  const crumbs = $('crumbs'); crumbs.replaceChildren();
  ['all runs', ...pre].forEach((nm, i, names) => {
    if (i) crumbs.append(el('span', { class: 'sep', 'aria-hidden': 'true', text: '/' }));
    crumbs.append(i === names.length - 1 ? el('span', { 'aria-current': 'page', text: nm }) : el('button', { type: 'button', text: nm, on: () => go(pre.slice(0, i)) }));
  });
  const lede = $('lede'), controls = $('controls'), view = $('view'), tabs = $('tabs');
  lede.replaceChildren(); controls.replaceChildren(); view.replaceChildren(); tabs.replaceChildren();
  const ch = me.children, here = pre[pre.length - 1] ?? 'all runs';

  if (ch.length && run) {
    const bar = el('div', { class: 'tabs', role: 'tablist' });
    for (const [val, text] of [['group', level.name + ' (' + ch.length + ')'], ['own', 'Own model of ' + here]])
      bar.append(el('button', { type: 'button', role: 'tab', 'aria-selected': String(S.tab === val), text, on: () => { S.tab = val; render(); } }));
    tabs.append(bar);
  }
  if (!ch.length || (run && S.tab === 'own')) { model(run, lede, controls, view, tabs); return; }

  controls.append(seg('Metric', 'metric', Object.entries(M).map(([k, m]) => [k, m.label])));
  const both = ch.some(c => { const n = node([...pre, c]); return n.own && n.nested; });
  if (both) controls.append(seg('Source', 'source', [['own', 'Own model of each child'], ['nested', 'Nested runs, pooled']]));
  controls.append(seg('Scale', 'scale', [['abs', 'Absolute'], ['share', 'Share of total']]));
  if (level.ordered) heat(pre, ch, level, lede, view); else strip(pre, ch, me, here, level, lede, view);
}

function heat(pre, ch, level, lede, view) {
  const m = S.metric, cols = ch.map(c => values([...pre, c], S.source, m));
  const top = maxOf(cols.flat().map(Math.abs), 0) || 1;
  lede.append(el('b', { text: ch.length + ' × ' + level.name + '. ' }),
    'Each column is one child, in key order. Each cell shows the ' + M[m].label.toLowerCase() + ' of one feature' +
    (S.scale === 'share' ? ', as a share of the column total.' : ', per row.') + ' Select a column heading to step into it.');
  const head = el('tr', {}, [el('th', {})]);
  ch.forEach(c => {
    const r = D.runs[[...pre, c].join('/')];
    head.append(el('th', { scope: 'col' }, el('button', { type: 'button', class: 'colbtn', title: r ? r.note : '', 'aria-label': 'Step into ' + c, text: c, on: () => go([...pre, c]) })));
  });
  const body = el('tbody');
  F.forEach((name, f) => {
    const tr = el('tr', {}, [el('th', { scope: 'row', title: name, text: name })]);
    cols.forEach(col => {
      const v = col[f], a = Math.min(Math.abs(v) / top, 1) * 0.62;
      tr.append(el('td', { text: show(v, M[m].signed), style: a > 0.01 ? 'background: rgb(var(--' + (v < 0 ? 'neg' : 'pos') + ') / ' + a.toFixed(3) + ')' : '' }));
    });
    body.append(tr);
  });
  view.append(el('div', { class: 'scroll' }, el('table', { class: 'heat' }, [el('caption', { style: 'position:absolute;left:-9999px', text: M[m].label + ' by feature and child' }), el('thead', {}, head), body])));
  const lg = el('div', { class: 'legend' });
  if (M[m].signed) lg.append(el('span', {}, [el('i', { class: 'sw', style: 'background: rgb(var(--pos) / .6)' }), 'Positive number: lowers the loss']), el('span', {}, [el('i', { class: 'sw', style: 'background: rgb(var(--neg) / .6)' }), 'Negative number: raises the loss']));
  else lg.append(el('span', {}, [el('i', { class: 'sw', style: 'background: rgb(var(--pos) / .6)' }), 'Larger mean |SHAP| on validation rows']));
  view.append(lg);
}

function strip(pre, ch, me, here, level, lede, view) {
  const m = S.metric, signed = M[m].signed;
  const kids = ch.map(c => values([...pre, c], S.source, m));
  const pooled = values(pre, 'nested', m), mine = me.own ? values(pre, 'own', m) : null;
  const all = [...kids.flat(), ...pooled, ...(mine || []), 0].filter(Number.isFinite);
  const lo = minOf(all), hi = maxOf(all), span = (hi - lo) || 1, x = v => (4 + 92 * (v - lo) / span).toFixed(2) + '%';
  lede.append(el('b', { text: ch.length + ' × ' + level.name + '. ' }),
    'The children have no order, so each feature gets one strip. A feature is stable when its dots are together on one side of zero.' +
    (mine ? ' The diamond is the own model of ' + here + '.' : ''));
  const grid = el('div', { class: 'strip' });
  F.forEach((name, f) => {
    const tr = el('div', { class: 'track' }, el('span', { class: 'zero', style: 'left:' + x(0) }));
    kids.forEach((col, i) => tr.append(el('span', { class: 'm dot' + (col[f] < 0 ? ' negv' : ''), style: 'left:' + x(col[f]), title: ch[i] + ': ' + show(col[f], signed) })));
    tr.append(el('span', { class: 'm tick', style: 'left:' + x(pooled[f]), title: 'Pool: ' + show(pooled[f], signed) }));
    if (mine) tr.append(el('span', { class: 'm dia', style: 'left:' + x(mine[f]), title: 'Own model: ' + show(mine[f], signed) }));
    let text, warn = false;
    if (signed) {
      text = 'Positive in ' + kids.filter(col => col[f] > 0).length + ' of ' + kids.length;
      if (mine && pooled[f] * mine[f] < 0 && Math.min(Math.abs(pooled[f]), Math.abs(mine[f])) > 0.05 * span) {
        warn = true; text = 'Disagree: pool ' + show(pooled[f], true) + ', own ' + show(mine[f], true);
      }
    } else {
      const vs = kids.map(col => col[f]); text = show(Math.min(...vs), false) + ' to ' + show(Math.max(...vs), false);
    }
    grid.append(el('div', { class: 'name', title: name, text: name }), tr, el('div', { class: 'verdict' + (warn ? ' warn' : ''), text }));
  });
  view.append(el('div', { class: 'scroll' }, grid));
  const lg = el('div', { class: 'legend' }, [el('span', {}, [el('i', { class: 'mk-dot' }), 'One child']), el('span', {}, [el('i', { class: 'mk-tick' }), 'Pool of the children'])]);
  if (mine) lg.append(el('span', {}, [el('i', { class: 'mk-dia' }), 'Own model, ' + (m === 'gt' ? 'train rows' : 'validation rows')]));
  view.append(lg);
  const chips = el('div', { class: 'chips' });
  ch.forEach(c => chips.append(el('button', { type: 'button', text: 'Step into ' + c, on: () => go([...pre, c]) })));
  view.append(chips);
}

// Bars for mean |SHAP| and loss reduction of one run; `update(k)` shows position k of the thin stage axis.
function bars(run, valid) {
  const sv = run.abs[valid], gv = run.gain[valid], gt = run.gain[TRAIN];
  const smax = maxOf(sv.flat(), 0) || 1;
  const all = [...gv.flat(), ...(gt || []).flat()], glo = minOf(all, 0), ghi = maxOf(all, 0), gs = (ghi - glo) || 1;
  const gx = v => 100 * (v - glo) / gs;
  const p1 = el('div', { class: 'bars' }), p2 = el('div', { class: 'bars' });
  const rows = F.map(name => {
    const s = el('i', { class: 's', style: 'left:0' }), sn = el('div', { class: 'num' });
    p1.append(el('div', { title: name, text: name }), el('div', { class: 'bt' }, s), sn);
    const v = el('i', { class: 'v' }), t = el('i', { class: 't' }), gn = el('div', { class: 'num' });
    p2.append(el('div', { title: name, text: name }), el('div', { class: 'bt' }, [el('span', { class: 'zero', style: 'left:' + gx(0) + '%' }), v, t]), gn);
    return { s, sn, v, t, gn };
  });
  const place = (bar, val) => { const a = gx(0), b = gx(val); bar.style.left = Math.min(a, b) + '%'; bar.style.width = Math.max(Math.abs(b - a), 0.3) + '%'; };
  const element = el('div', { class: 'panels' }, [
    el('section', {}, [el('h2', { text: 'Mean |SHAP|, validation rows' }), p1]),
    el('section', {}, [el('h2', { text: 'Cumulative loss reduction per row' }), p2,
      el('div', { class: 'legend' }, [el('span', {}, [el('i', { class: 'sw', style: 'background: rgb(var(--pos)); height: 10px' }), 'Validation (filled, first number)']), el('span', {}, [el('i', { class: 'sw', style: 'border: 1.5px solid var(--muted); height: 9px' }), 'Train (outline, second number)'])])])
  ]);
  const update = k => rows.forEach((r, f) => {
    r.s.style.width = Math.max(100 * sv[f][k] / smax, 0.3) + '%'; r.sn.textContent = fmt.num(sv[f][k]);
    place(r.v, gv[f][k]); r.v.classList.toggle('negv', gv[f][k] < 0); if (gt) place(r.t, gt[f][k]);
    r.gn.replaceChildren(el('b', { text: fmt.signed(gv[f][k]) }), gt ? fmt.signed(gt[f][k]) : '');
  });
  return { element, update };
}

const COLORS = { pos: 'rgb(var(--pos))', neg: 'rgb(var(--neg))', ink: 'var(--ink)', muted: 'var(--muted)', rule: 'var(--rule)', paper: 'var(--paper)', fog: 'var(--fog)' };

function model(run, lede, controls, view, tabs) {
  const others = Object.keys(run.sets).filter(s => s !== TRAIN);
  const valid = others.includes(S.valid) ? S.valid : (others[0] || TRAIN);
  lede.append(el('b', { text: 'One model' + (run.note ? ': ' + run.note : '') + '. ' }),
    others.length ? run.sets[valid].overlap + ' rows of the validation set are also in the train set.' : 'The run has a train set only.');
  if (others.length > 1) {
    const pick = el('select', { 'aria-label': 'Validation rows' }, others.map(s => el('option', { value: s, text: s })));
    pick.value = valid;
    pick.addEventListener('change', () => { S.valid = pick.value; render(); });
    controls.append(el('label', { class: 'seg' }, [el('span', { text: 'Validation rows' }), pick]));
  }
  if (!run.views.length) return;
  const current = run.views.find(v => v.name === S.view) || run.views[0];
  if (run.views.length > 1) {
    const bar = el('div', { class: 'tabs', role: 'tablist' });
    for (const v of run.views) bar.append(el('button', { type: 'button', role: 'tab', 'aria-selected': String(v === current), text: v.name, on: () => { S.view = v.name; render(); } }));
    tabs.append(bar);
  }
  const api = { features: F, run, valid, train: TRAIN, colors: COLORS, fmt, el, bars: () => bars(run, valid) };
  dispose = VIEWS[current.name].render(view, current.payload, api) || null;
}
render();
