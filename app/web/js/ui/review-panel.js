// AI 검토 패널. 규칙이 판단하지 못한 지점을 LLM(또는 사람)이 정해진 답 중에서 고른다.
// ReviewList: 사이드바의 층별 목록과 요약. ReviewCard: 3D 뷰어 위에 뜨는 상세(도면 조각, 판단과 이유, 계산한 근거).
// LLM은 종류만 고르고 좌표는 만들지 않는다. 그래서 여기서도 보여 주는 것은 '무엇인지'뿐이다.
import { h } from './dom.js';

export const KIND_LABEL = { opening: '끊긴 벽', outline: '외곽 빈 구간', corner: '모서리 틈', window: '실내 창' };
const KIND_QUESTION = {
  opening: '벽선이 끊겼는데 문·창 표시를 찾지 못한 자리입니다.',
  outline: '건물 외곽선인데 벽이 그려지지 않은 구간입니다.',
  corner: '벽 끝과 이웃 벽 사이에 남은 모서리 틈입니다.',
  window: '실내 벽에서 창으로 읽힌 자리입니다. 실내 창은 드물어 확인이 필요합니다.',
};
export const ANSWER_LABEL = { wall: '벽', door: '문', window: '창', open: '열린 통로', drawing_issue: '도면 문제' };
// 근거에 나오는 레이어 역할 이름. ARC는 원호(문이 열리는 궤적)다.
const ROLE_LABEL = {
  WALL: '벽선', DOOR: '문선', WINDOW: '창선', COLUMN: '기둥', GRID: '축선', ARC: '원호', DIM: '치수',
  STAIR: '계단', RAIL: '난간', FURNITURE: '가구', HATCH: '해치', ROOM_TEXT: '실 이름', OTHER: '기타', UNKNOWN: '미분류',
};

/** 지점의 상태. pending(답 없음) | ai(LLM이 고름) | user(사람이 정함) */
export function reviewState(item) {
  if (!item.decision) return 'pending';
  return item.decision.model === 'user' ? 'user' : 'ai';
}
const STATE_LABEL = { pending: '대기', ai: 'AI 판단', user: '사람 확인' };
const meters = (mm) => `${(mm / 1000).toFixed(mm < 10000 ? 1 : 0)}m`;
const chip = (state, text) => h('span', { class: `c-review__chip c-review__chip--${state}` }, text);

export function ReviewList({ floorLabel, onSelect }) {
  const summary = h('div', { class: 'c-review__summary' });
  const list = h('div', { class: 'c-review__list' });
  const el = h('div', { class: 'c-review' }, summary, list);
  const rows = new Map();
  let items = [];

  function setItems(next) {
    items = next || [];
    rows.clear();
    if (!items.length) {
      summary.replaceChildren();
      list.replaceChildren(h('p', { class: 'c-panel__empty' }, '검토할 곳이 없습니다. 모델을 불러오면 규칙이 판단하지 못한 곳이 여기에 보입니다.'));
      return;
    }
    const counts = { pending: 0, ai: 0, user: 0 };
    items.forEach((item) => { counts[reviewState(item)] += 1; });
    summary.replaceChildren(
      h('p', { class: 'c-panel__hint' }, '규칙이 판단하지 못한 곳입니다. 항목을 누르면 도면 조각과 3D 위치를 보여 줍니다.'),
      h('div', { class: 'c-review__counts' },
        h('span', {}, `전체 ${items.length}`), chip('ai', `AI 판단 ${counts.ai}`),
        chip('user', `사람 확인 ${counts.user}`), chip('pending', `대기 ${counts.pending}`)),
    );
    const floors = [...new Set(items.map((item) => item.floor))];
    list.replaceChildren(...floors.flatMap((floor) => {
      const mine = items.filter((item) => item.floor === floor);
      return [
        h('h3', { class: 'c-review__floor' }, `${floorLabel(floor)} · ${mine.length}곳`),
        ...mine.map((item) => {
          const state = reviewState(item);
          const row = h('button', {
            class: 'c-review__row', type: 'button', 'aria-pressed': 'false', onClick: () => onSelect(item),
          },
          h('span', { class: `c-review__dot c-review__dot--${state}`, title: STATE_LABEL[state] }),
          h('span', { class: 'c-review__name' }, `${KIND_LABEL[item.kind] || item.kind} · ${meters(item.width)}`),
          chip(state, item.decision ? ANSWER_LABEL[item.decision.answer] || item.decision.answer : '대기'));
          rows.set(item.key, row);
          return row;
        }),
      ];
    }));
  }
  function select(key) {
    rows.forEach((row, k) => row.setAttribute('aria-pressed', String(k === key)));
    rows.get(key)?.scrollIntoView({ block: 'nearest' });
  }
  setItems([]);
  return { el, setItems, select, count: () => items.length };
}

export function ReviewCard({ floorLabel, onClose }) {
  const el = h('aside', { class: 'c-review-card', hidden: true, 'aria-label': '검토 지점 상세' });

  function show(item) {
    const state = reviewState(item);
    const decision = item.decision;
    const counts = Object.entries(item.evidence?.counts || {}).sort((a, b) => b[1] - a[1])
      .map(([role, n]) => `${ROLE_LABEL[role] || role} ${n}`);
    const sides = item.evidence?.sides || [];
    el.replaceChildren(
      h('header', { class: 'c-review-card__head' },
        h('strong', {}, `${floorLabel(item.floor)} · ${KIND_LABEL[item.kind] || item.kind}`),
        h('button', { class: 'c-review-card__close', type: 'button', 'aria-label': '상세 닫기', onClick: () => { hide(); onClose?.(); } }, '×')),
      h('p', { class: 'c-review-card__question' }, `${KIND_QUESTION[item.kind] || ''} 폭 ${meters(item.width)}.`),
      item.image
        ? h('img', { class: 'c-review-card__image', src: item.image, alt: '원본 도면 조각. 빨간 상자가 검토 지점입니다.' })
        : h('div', { class: 'c-review-card__image c-review-card__image--none' }, '도면 조각 없음'),
      h('section', { class: `c-review-card__verdict c-review-card__verdict--${state}` },
        h('div', { class: 'c-review-card__verdict-head' }, chip(state, STATE_LABEL[state]),
          h('strong', {}, decision ? ANSWER_LABEL[decision.answer] || decision.answer : '아직 답이 없습니다')),
        decision
          ? [decision.reason && h('p', {}, decision.reason), state === 'ai' && h('small', {}, decision.model)]
          : h('p', {}, 'AI에게 물으면 벽·문·창·열린 통로·도면 문제 중 하나를 고르고 이유를 적습니다. 그동안은 규칙 결과를 그대로 씁니다.')),
      h('section', { class: 'c-review-card__evidence' },
        h('h4', {}, '도면에서 계산한 근거'),
        h('dl', {},
          h('dt', {}, '상자 안의 선'), h('dd', {}, counts.length ? counts.join(' · ') : '아무 선도 없음'),
          sides.length === 2 && [h('dt', {}, '양쪽 공간'), h('dd', {}, sides[0] === sides[1] ? `${sides[0]} (같은 공간)` : `${sides[0]} / ${sides[1]}`)]),
        item.evidence?.text && h('details', {}, h('summary', {}, 'AI에게 준 근거 전체'), h('pre', {}, item.evidence.text.trim()))),
    );
    el.hidden = false;
    el.scrollTop = 0;
  }
  function hide() { el.hidden = true; }
  return { el, show, hide };
}
