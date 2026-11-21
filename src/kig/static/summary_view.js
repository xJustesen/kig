// Summary view: the bars of the full model, with no stage control.
export function render(container, payload, api) {
  const { el, run, fmt } = api, tr = run.loss[api.train], va = run.loss[api.valid];
  const facts = el('dl', { class: 'facts' });
  const fact = (name, text) => facts.append(el('div', {}, [el('dt', { text: name }), el('dd', { text })]));
  fact('Train loss', tr ? fmt.num(tr[tr.length - 1]) : '–');
  fact('Validation loss', fmt.num(va[va.length - 1]));
  fact('Validation rows', fmt.num(run.sets[api.valid].n).replace(/\.00$/, ''));
  const bars = api.bars();
  container.append(facts, bars.element);
  bars.update(run.stages.length - 1);
}
