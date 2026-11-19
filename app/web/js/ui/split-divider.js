import { h } from './dom.js';

export function SplitDivider(host, first, second) {
  let ratio = 50;
  let dragging = false;
  const el = h('div', { class: 'c-split-divider', role: 'separator', tabindex: '0',
    'aria-label': '3D와 원도면 영역 비율', 'aria-valuemin': '25', 'aria-valuemax': '75', hidden: true,
    title: '드래그 또는 방향키로 비율 조절 · 더블클릭으로 반반' });
  const vertical = () => getComputedStyle(host).flexDirection === 'row';
  function set(value) {
    ratio = Math.max(25, Math.min(75, value));
    first.style.flex = `${ratio} 1 0%`;
    second.style.flex = `${100 - ratio} 1 0%`;
    el.setAttribute('aria-valuenow', String(Math.round(ratio)));
    el.setAttribute('aria-valuetext', `3D ${Math.round(ratio)}%, 원도면 ${100 - Math.round(ratio)}%`);
    el.setAttribute('aria-orientation', vertical() ? 'vertical' : 'horizontal');
  }
  el.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    dragging = true;
    el.setPointerCapture(event.pointerId);
    event.preventDefault();
    el.focus();
  });
  el.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const box = host.getBoundingClientRect();
    const size = vertical() ? box.width : box.height;
    if (size) set((vertical() ? event.clientX - box.left : event.clientY - box.top) / size * 100);
  });
  const stop = () => { dragging = false; };
  el.addEventListener('pointerup', stop);
  el.addEventListener('pointercancel', stop);
  el.addEventListener('lostpointercapture', stop);
  el.addEventListener('dblclick', () => set(50));
  el.addEventListener('keydown', (event) => {
    const negative = vertical() ? 'ArrowLeft' : 'ArrowUp';
    const positive = vertical() ? 'ArrowRight' : 'ArrowDown';
    if (![negative, positive, 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    set(event.key === 'Home' ? 25 : event.key === 'End' ? 75 : ratio + (event.key === positive ? 5 : -5));
  });
  new ResizeObserver(() => set(ratio)).observe(host);
  set(ratio);
  return { el, refresh: () => set(ratio) };
}
