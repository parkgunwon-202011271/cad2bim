// 2D 도면 보기. 칩으로 도면을 고르고, 휠로 확대, 드래그로 이동, 더블클릭으로 맞춤.
import { h } from './dom.js';

export function PlanViewer() {
  const title = h('div', { class: 'c-pane-title c-pane-title--2d' }, '원도면 · 선택 없음');
  const bar = h('div', { class: 'c-plan-viewer__bar' });
  const img = h('img', { alt: '' });
  const canvas = h('div', { class: 'c-plan-viewer__canvas' }, img);
  const empty = h('div', { class: 'c-viewer__empty' }, h('strong', {}, '2D 도면 없음'), '도면을 변환하면 여기에 보입니다.');
  const el = h('div', { class: 'c-plan-viewer' }, title, bar, canvas, empty);
  let view = { scale: 1, x: 0, y: 0 };

  function apply() {
    img.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
  }
  function fit() {
    if (!img.naturalWidth || !canvas.clientWidth) return;
    const scale = Math.min(canvas.clientWidth / img.naturalWidth, canvas.clientHeight / img.naturalHeight) * 0.96;
    view = {
      scale,
      x: (canvas.clientWidth - img.naturalWidth * scale) / 2,
      y: (canvas.clientHeight - img.naturalHeight * scale) / 2,
    };
    apply();
  }
  img.addEventListener('load', fit);
  new ResizeObserver(fit).observe(canvas);
  canvas.addEventListener('dblclick', fit);
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const k = Math.exp(-e.deltaY * 0.0015);
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;
    view = { scale: view.scale * k, x: px - (px - view.x) * k, y: py - (py - view.y) * k };
    apply();
  }, { passive: false });
  let drag = null;
  canvas.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX - view.x, y: e.clientY - view.y };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    view = { ...view, x: e.clientX - drag.x, y: e.clientY - drag.y };
    apply();
  });
  canvas.addEventListener('pointerup', () => { drag = null; });

  let selectPlan = () => {};
  const state = { el, setPlans, fit, select: (i) => selectPlan(i), onSelect: null };
  function setPlans(plans) {
    empty.hidden = plans.length > 0;
    canvas.hidden = plans.length === 0;
    const chips = plans.map((plan, i) => h('button', {
      class: 'c-plan-viewer__chip', type: 'button', 'aria-pressed': String(i === 0),
      onClick: () => select(i),
    }, plan.name));
    bar.replaceChildren(...chips);
    function select(i) {
      if (!plans[i]) return;
      chips.forEach((c, j) => c.setAttribute('aria-pressed', String(i === j)));
      img.src = plans[i].src;
      img.alt = plans[i].name;
      title.textContent = `원도면 · ${plans[i].name}`;
      title.title = plans[i].name;
      state.onSelect?.(i);
    }
    selectPlan = select;
    if (plans.length) select(0);
    else {
      img.removeAttribute('src');
      img.alt = '';
      title.textContent = '원도면 · 선택 없음';
      title.removeAttribute('title');
    }
  }
  setPlans([]);
  return state;
}
