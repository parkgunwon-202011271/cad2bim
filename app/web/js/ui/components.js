// 모양만 맡는 컴포넌트. 이름은 Figma 컴포넌트 이름과 같다.
// Figma 디자인을 반영할 때는 이 파일과 css/만 고친다. 동작은 main.js에 있다.
import { h } from './dom.js';

export function Button({ label, variant = 'secondary', size, onClick, disabled }) {
  const cls = ['c-button', `c-button--${variant}`, size === 'small' && 'c-button--small'].filter(Boolean).join(' ');
  return h('button', { class: cls, type: 'button', onClick, disabled }, label);
}

export function Toolbar({ title, hint, actions = [] }) {
  return h('header', { class: 'c-toolbar' },
    h('span', { class: 'c-toolbar__title' }, title),
    hint && h('span', { class: 'c-toolbar__hint' }, hint),
    h('span', { class: 'c-toolbar__spacer' }),
    actions,
  );
}

export function Panel({ title, children = [] }) {
  const body = h('div', { class: 'c-panel__body' }, children);
  const el = h('section', { class: 'c-panel' }, h('h2', { class: 'c-panel__title' }, title), body);
  return { el, body, setChildren: (nodes) => body.replaceChildren(...[nodes].flat()) };
}

export function ToggleList({ items, onChange }) {
  return h('div', { class: 'c-toggle-list' },
    items.map((item) => {
      const input = h('input', { type: 'checkbox', checked: item.checked !== false });
      input.addEventListener('change', () => onChange(item.key, input.checked));
      return h('label', { class: 'c-toggle' }, input, h('span', {}, item.label));
    }),
  );
}

export function Tabs({ tabs, active, onChange }) {
  const buttons = tabs.map((tab) => h('button', {
    class: 'c-tabs__tab', type: 'button', role: 'tab', 'aria-selected': String(tab.key === active),
    onClick: () => select(tab.key),
  }, tab.label));
  function select(key) {
    buttons.forEach((b, i) => b.setAttribute('aria-selected', String(tabs[i].key === key)));
    onChange(key);
  }
  return { el: h('nav', { class: 'c-tabs', role: 'tablist' }, buttons), select };
}

/** 여러 개 중 하나를 고르는 버튼 묶음. 내부 보기(층 단면) 고르기에 쓴다. */
export function Segmented({ options, active, onChange }) {
  const buttons = options.map((opt) => h('button', {
    class: 'c-segmented__option', type: 'button', 'aria-pressed': String(opt.key === active),
    onClick: () => select(opt.key),
  }, opt.label));
  function select(key) {
    buttons.forEach((b, i) => b.setAttribute('aria-pressed', String(options[i].key === key)));
    onChange(key);
  }
  return { el: h('div', { class: 'c-segmented', role: 'group' }, buttons), select };
}

export function ProgressPanel() {
  const text = h('div', { class: 'c-progress__text' }, '도면을 끌어다 놓거나 폴더를 여세요.');
  const fill = h('div', { class: 'c-progress__fill' });
  const bar = h('div', { class: 'c-progress__bar' }, fill);
  bar.hidden = true;
  const el = h('div', { class: 'c-progress' }, text, bar);
  // state: idle | busy | progress | done | error
  function set({ state, message, ratio }) {
    text.textContent = message;
    text.className = 'c-progress__text' + (state === 'error' ? ' c-progress__text--error' : '')
      + (state === 'done' ? ' c-progress__text--done' : '');
    bar.hidden = !(state === 'busy' || state === 'progress');
    bar.classList.toggle('c-progress__bar--busy', state === 'busy');
    fill.style.width = state === 'progress' ? `${Math.round((ratio || 0) * 100)}%` : '';
  }
  return { el, set };
}

export function StatList({ stats }) {
  return h('div', { class: 'c-stat-list' },
    stats.map((s) => h('div', { class: 'c-stat' },
      h('span', { class: 'c-stat__label' }, s.label),
      h('span', { class: 'c-stat__value' }, s.value),
    )),
  );
}

export function DropOverlay() {
  const el = h('div', { class: 'c-drop-overlay', hidden: true },
    h('div', { class: 'c-drop-overlay__box' },
      h('span', { class: 'c-drop-overlay__title' }, '여기에 놓으면 3D로 변환합니다'),
      h('span', { class: 'c-drop-overlay__sub' }, 'DWG·DXF 파일 여러 개 또는 도면 폴더 하나'),
    ),
  );
  return { el, show: (v) => { el.hidden = !v; } };
}

export function Toast() {
  const el = h('div', { class: 'c-toast', role: 'status', hidden: true });
  let timer;
  function show(message, ms = 4000) {
    el.textContent = message;
    el.hidden = false;
    clearTimeout(timer);
    timer = setTimeout(() => { el.hidden = true; }, ms);
  }
  return { el, show };
}
