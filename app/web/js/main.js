// 화면 조립과 동작. 모양은 ui/components.js, 3D는 viewer.js, Python 연결은 api.js에 있다.
import * as api from './api.js';
import { h } from './ui/dom.js';
import {
  Button, DropOverlay, Panel, ProgressPanel, Segmented, StatList, Tabs, Toast, ToggleList, Toolbar,
} from './ui/components.js';
import { PlanViewer } from './ui/plan-viewer.js';
import { Viewer3D, floorLabel } from './viewer.js';

const root = document.getElementById('app');

// ---------- 화면 조립 ----------
const toolbar = Toolbar({
  title: 'cad2bim',
  hint: 'CAD 평면도 → 층별 3D 모델',
  actions: [
    Button({ label: '샘플 보기', variant: 'ghost', onClick: () => showResult(true) }),
    Button({ label: '결과 열기', onClick: () => api.openResult() }),
    Button({ label: '도면 폴더 열기', variant: 'primary', onClick: () => api.chooseFolder() }),
  ],
});

const progress = ProgressPanel();
const progressPanel = Panel({ title: '진행 상황', children: [progress.el] });
const layersPanel = Panel({ title: '표시', children: [h('div', { class: 'c-panel__empty' }, '모델을 불러오면 층이 보입니다.')] });
const autoRotate = ToggleList({
  items: [{ key: 'rotate', label: '자동 회전', checked: false }],
  onChange: (_, on) => viewer.setAutoRotate(on),
});
// 건축: Revit '음영 + 모서리'처럼 차분한 재료색. 분류 색: 기둥 파랑·문 초록처럼 인식 결과 확인용.
const styleSwitch = Segmented({
  options: [{ key: 'arch', label: '건축' }, { key: 'class', label: '분류 색' }],
  active: 'arch',
  onChange: (key) => viewer.setStyle(key),
});
// 사람 시점: 보던 자리의 바닥에 눈높이로 서서 걷는다. 다시 누르거나 Esc로 나온다.
const personButton = Button({
  label: '사람 시점', variant: 'primary', size: 'small',
  onClick: () => (viewer.person ? viewer.exitPerson() : viewer.enterPerson()),
});
const viewPanel = Panel({
  title: '보기',
  children: [styleSwitch.el, autoRotate, h('div', { class: 'c-panel__row' },
    personButton,
    Button({ label: '시점 초기화', size: 'small', onClick: () => viewer.fit() }))],
});
const insidePanel = Panel({ title: '내부 보기', children: [h('div', { class: 'c-panel__empty' }, '모델을 불러오면 층을 고를 수 있습니다.')] });
const statsPanel = Panel({ title: '결과 요약', children: [h('div', { class: 'c-panel__empty' }, '-')] });

const viewerHost = h('div', { class: 'c-viewer' });
const viewerEmpty = h('div', { class: 'c-viewer__empty' },
  h('strong', {}, '도면을 여기로 끌어다 놓으세요'),
  'DWG 폴더 하나, 또는 DWG·DXF 파일 여러 개');
const ORBIT_HELP = 'W·A·S·D: 이동 · Q·E: 아래·위 · Shift: 빠르게 · 드래그: 회전 · 휠: 확대 · 오른쪽 드래그: 옮기기';
const PERSON_HELP = '사람 시점 · W·A·S·D: 걷기 · Shift: 뛰기 · 드래그: 둘러보기 · 휠: 시야 넓히기·좁히기 · Esc: 나가기';
const viewerHelp = h('div', { class: 'c-viewer__help', hidden: true }, ORBIT_HELP);
viewerHost.append(viewerEmpty, viewerHelp);
const view3d = h('div', { class: 'l-stage__view' }, viewerHost);
const plans = PlanViewer();
const view2d = h('div', { class: 'l-stage__view', hidden: true }, plans.el);
const tabs = Tabs({
  tabs: [{ key: '3d', label: '3D 모델' }, { key: '2d', label: '2D 도면' }],
  active: '3d',
  onChange: (key) => {
    view3d.hidden = key !== '3d';
    view2d.hidden = key !== '2d';
    if (key === '2d') plans.fit();
  },
});

const overlay = DropOverlay();
const toast = Toast();

root.append(
  h('div', { class: 'l-app' },
    toolbar,
    h('main', { class: 'l-body' },
      h('aside', { class: 'l-sidebar' }, progressPanel.el, layersPanel.el, insidePanel.el, viewPanel.el, statsPanel.el),
      h('section', { class: 'l-stage' }, tabs.el, h('div', { class: 'l-stage__body' }, view3d, view2d)),
    ),
  ),
  overlay.el,
  toast.el,
);

const viewer = new Viewer3D(viewerHost);
viewer.onPersonChange = (on) => {
  personButton.textContent = on ? '사람 시점 끝내기' : '사람 시점';
  viewerHelp.textContent = on ? PERSON_HELP : ORBIT_HELP;
};
// 엘리베이터 안에서는 층 고르는 법을 보여 준다.
viewer.onPersonHint = (hint) => {
  if (viewer.person) viewerHelp.textContent = hint || PERSON_HELP;
};

// ---------- 끌어다 놓기 ----------
// 데스크톱에서는 Python 쪽 drop 처리기가 실제 경로를 받는다. 여기서는 표시만 한다.
// Python 처리기가 전파를 막으므로 window의 캡처 단계에서 먼저 받는다.
let dragDepth = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth += 1; overlay.show(true); }, true);
window.addEventListener('dragover', (e) => { e.preventDefault(); }, true);
window.addEventListener('dragleave', () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) overlay.show(false); }, true);
window.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  overlay.show(false);
  api.dropInBrowser();
}, true);

// ---------- 변환 진행 ----------
let started = 0;
let ticker = null;
const elapsed = () => `${Math.round((performance.now() - started) / 1000)}초`;

function startTicker(render) {
  clearInterval(ticker);
  ticker = setInterval(render, 1000);
}

api.onEvent(async (event) => {
  if (event.type === 'started') {
    started = performance.now();
    const msg = () => `도면 ${event.count}장 변환 시작 · ${elapsed()}`;
    progress.set({ state: 'busy', message: msg() });
    startTicker(() => progress.set({ state: 'busy', message: msg() }));
  } else if (event.type === 'progress') {
    if (event.stage === 'read') {
      progress.set({ state: 'progress', ratio: event.done / event.total, message: `도면 읽는 중 ${event.done}/${event.total} · ${elapsed()}` });
      clearInterval(ticker);
    } else {
      const label = event.stage === 'plans' ? '2D 도면 그리는 중' : '3D 모델 만드는 중';
      const msg = () => `${label} · ${elapsed()}`;
      progress.set({ state: 'busy', message: msg() });
      startTicker(() => progress.set({ state: 'busy', message: msg() }));
    }
  } else if (event.type === 'done') {
    clearInterval(ticker);
    const took = started ? ` · ${elapsed()}` : '';
    started = 0;
    await showResult(Boolean(event.sample), took);
  } else if (event.type === 'error') {
    clearInterval(ticker);
    started = 0;
    progress.set({ state: 'error', message: event.message });
    toast.show(event.message);
  }
});

// ---------- 결과 표시 ----------
async function showResult(useSample, took = '') {
  try {
    progress.set({ state: 'busy', message: '모델 불러오는 중' });
    const model = await api.getModel(useSample);
    if (!model?.dae) throw new Error('불러올 결과가 없습니다.');
    const floors = viewer.load(model.dae);
    viewer.setFurniture(model.furniture);
    viewerEmpty.hidden = true;
    viewerHelp.hidden = false;
    renderLayers(floors);
    renderInside(floors);
    renderStats(model.summary, floors.length);
    plans.setPlans(model.plans || []);
    tabs.select('3d');
    progress.set({ state: 'done', message: `${useSample ? '샘플(주민센터)' : '변환 완료'} · ${floors.length}개 층${took}` });
  } catch (err) {
    progress.set({ state: 'error', message: err.message || String(err) });
  }
}

function renderLayers(floors) {
  layersPanel.setChildren([
    ToggleList({
      items: floors.map((key) => ({ key, label: floorLabel(key) })),
      onChange: (key, on) => viewer.setFloorVisible(key, on),
    }),
    ToggleList({
      items: [
        { key: 'roof', label: '지붕' }, { key: 'site', label: '대지·조경' },
        // 도면의 가구·위생기구 선을 바닥에 그린다. 가구가 없는 결과에서는 숨긴다.
        ...(viewer.hasFurniture() ? [{ key: 'furniture', label: '가구(도면 선)' }] : []),
      ],
      onChange: (key, on) => {
        if (key === 'roof') viewer.setRoofVisible(on);
        else if (key === 'site') viewer.setSiteVisible(on);
        else viewer.setFurnitureVisible(on);
      },
    }),
  ]);
}

// 층을 고르면 그 층 바닥 1.2m 위에서 잘라 위를 걷어 낸다. 방·문·창이 평면도처럼 보인다.
function renderInside(floors) {
  const cut = Segmented({
    options: [{ key: '', label: '끔' }, ...floors.map((key) => ({ key, label: floorLabel(key) }))],
    active: '',
    onChange: (key) => viewer.setCut(key || null),
  });
  insidePanel.setChildren([
    cut.el,
    h('p', { class: 'c-panel__hint' }, '층을 고르면 그 층만 남기고 바닥 1.25m 높이에서 잘라 보여 줍니다. 잘린 면은 진하게 칠합니다. W·A·S·D로 방 안을 걸어 다닐 수 있습니다.'),
  ]);
}

function renderStats(summary, floorCount) {
  const t = summary?.totals || {};
  const stats = [
    { label: '층', value: floorCount },
    { label: '벽', value: t.walls ?? '-' },
    { label: '기둥', value: t.columns ?? '-' },
    { label: '문', value: t.doors ?? '-' },
    { label: '창', value: t.windows ?? '-' },
    { label: '실', value: t.spaces ?? '-' },
  ];
  statsPanel.setChildren(StatList({ stats }));
}

// ---------- 시작 ----------
api.init().then((mode) => {
  if (mode === 'browser') toolbar.querySelector('.c-toolbar__hint').textContent += ' · 브라우저 미리보기(샘플 데이터)';
  // 주소 끝에 ?sample 을 붙이면 샘플을 바로 띄운다. 디자인 작업과 화면 캡처용이다.
  const params = new URLSearchParams(location.search);
  const shown = params.has('sample') ? showResult(true) : Promise.resolve();
  shown.then(() => { if (params.get('tab') === '2d') tabs.select('2d'); });
});
