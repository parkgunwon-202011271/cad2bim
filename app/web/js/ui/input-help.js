import { h } from './dom.js';

function mouse(part) {
  const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  icon.setAttribute('viewBox', '0 0 32 42');
  icon.setAttribute('aria-hidden', 'true');
  icon.innerHTML = `<rect x="4" y="2" width="24" height="38" rx="12" fill="none" stroke="currentColor" stroke-width="2"/>
    <path d="M16 3V17H5M16 17H27" fill="none" stroke="currentColor"/>
    <path d="${part === 'left' ? 'M14 4C9 4 6 8 6 14H14Z' : part === 'right' ? 'M18 4C23 4 26 8 26 14H18Z' : 'M14 8H18V15H14Z'}" fill="currentColor"/>`;
  return icon;
}

export function InputHelp() {
  const body = h('div', { class: 'c-input-help__body' });
  const hint = h('p', { class: 'c-input-help__hint', hidden: true });
  const el = h('details', { class: 'c-input-help', open: true, hidden: true },
    h('summary', {}, '마우스·키보드 조작'), body, hint);
  function setPerson(on) {
    const mouseRow = (part, label) => h('div', { class: 'c-input-help__item' }, mouse(part), h('span', {}, label));
    const keys = (labels, text) => h('div', { class: 'c-input-help__item' },
      h('span', { class: 'c-input-help__keys' }, labels.map((key) => h('kbd', {}, key))), h('span', {}, text));
    body.replaceChildren(
      mouseRow('left', on ? '드래그 · 둘러보기' : '드래그 · 회전'),
      ...(!on ? [mouseRow('right', '오른쪽 드래그 · 이동')] : []),
      mouseRow('wheel', on ? '휠 · 시야 조절' : '휠 · 확대·축소'),
      keys(['W', 'A', 'S', 'D'], on ? '걷기' : '이동'),
      ...(!on ? [keys(['Q', 'E'], '아래·위')] : []),
      keys(['Shift'], on ? '빠르게 걷기' : '빠르게 이동'),
      ...(on ? [keys(['Esc'], '사람 시점 종료')] : []),
    );
    setHint('');
  }
  function setHint(text) { hint.textContent = text; hint.hidden = !text; }
  setPerson(false);
  return { el, setPerson, setHint };
}
