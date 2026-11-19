// 화면 조립과 동작. 모양은 ui/components.js, 3D는 viewer.js, Python 연결은 api.js에 있다.
import * as api from './api.js';
import { h } from './ui/dom.js';
import {
  Button, DropOverlay, Panel, ProgressPanel, Segmented, StatList, Tabs, Toast, ToggleList, Toolbar,
} from './ui/components.js';
import { PlanViewer } from './ui/plan-viewer.js';
import { InputHelp } from './ui/input-help.js';
import { MiniMap } from './ui/minimap.js';
import { ReviewCard, ReviewList } from './ui/review-panel.js';
import { SplitDivider } from './ui/split-divider.js';
import { Viewer3D, floorLabel } from './viewer.js';

const root = document.getElementById('app');

// ---------- 화면 조립 ----------
const toolbar = Toolbar({
  title: 'cad2bim',
  hint: 'CAD 평면도 → 층별 3D 모델',
  actions: [
    Button({ label: '도면 초기화', onClick: () => resetAll() }),
    Button({ label: '결과 열기', onClick: () => api.openResult() }),
    Button({ label: '도면 폴더 열기', variant: 'primary', onClick: () => api.chooseFolder() }),
  ],
});

const progress = ProgressPanel();
const IDLE_MESSAGE = '도면을 끌어다 놓거나 폴더를 여세요.';
const emptyLayers = () => h('div', { class: 'c-panel__empty' }, '모델을 불러오면 층이 보입니다.');
const emptyDrawings = () => h('p', { class: 'c-panel__empty' }, '원본 도면 없음');
const layersPanel = Panel({ title: '층 · 요소', children: [emptyLayers()] });
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
  label: '사람 시점', size: 'small',
  onClick: () => (viewer.person ? viewer.exitPerson() : viewer.enterPerson()),
});
personButton.setAttribute('aria-pressed', 'false');
let cutFloor = null;
const modelTitle = h('div', { class: 'c-pane-title c-pane-title--3d', 'aria-live': 'polite' }, '3D · 모델 없음');
const insideControls = h('div', { class: 'c-view-tools__row c-inside-controls', hidden: true });
const viewTools = h('div', { class: 'c-view-tools', 'aria-label': '3D 보기 도구' },
  h('div', { class: 'c-view-tools__row' }, styleSwitch.el,
    personButton, Button({ label: '시점 초기화', size: 'small', onClick: () => viewer.fit() })),
  insideControls,
);
const statsPanel = Panel({ title: '결과 요약', children: [h('div', { class: 'c-panel__empty' }, '-')] });

const viewerHost = h('div', { class: 'c-viewer' });
const viewerEmpty = h('div', { class: 'c-viewer__empty' },
  h('strong', {}, '도면을 여기로 끌어다 놓으세요'),
  'DWG 폴더 하나, 또는 DWG·DXF 파일 여러 개');
const viewerHelp = InputHelp();
viewerHost.append(viewerEmpty);
const icon = (name) => h('img', { src: `assets/icons/${name}.svg`, alt: '', width: '24', height: '24' });
// 전체화면: 3D 영역(제목·도구·뷰어)이 앱 창 전체를 덮는다. 다시 누르거나 Esc로 돌아온다.
const fullButton = h('button', {
  class: 'c-viewer__action', type: 'button', hidden: true, 'aria-pressed': 'false',
  onClick: () => setFullscreen(!view3d.classList.contains('l-stage__view--full')),
});
// AI 검토: 규칙이 판단하지 못한 지점의 목록(사이드바)과 상세 카드(뷰어 위). 항목을 누르면 3D가 그 자리로 간다.
const reviewList = ReviewList({ floorLabel: (key) => floorLabel(key), onSelect: (item) => showSpot(item) });
const reviewCard = ReviewCard({
  floorLabel: (key) => floorLabel(key),
  onClose: () => { viewer.clearSpot(); reviewList.select(null); },
});
let cutSwitch = null; // '층·내부 보기' 선택. 검토 지점으로 갈 때 그 층 단면으로 바꾼다
function showSpot(item) {
  if (view3d.hidden) tabs.select('3d');
  cutSwitch?.select(item.floor); // 그 층만 남기고 잘라 벽 안쪽이 보이게 한다
  reviewList.select(item.key);
  reviewCard.show(item);
  // 카드가 뷰어 왼쪽을 덮는 만큼 지점을 오른쪽으로 밀어 보여 준다.
  viewer.focusSpot(item, (reviewCard.el.offsetWidth + 24) / Math.max(viewerHost.clientWidth, 1));
}
function setReview(review) {
  reviewCard.hide();
  viewer.clearSpot();
  reviewList.setItems(review?.items || []);
  const count = reviewList.count();
  reviewBadge.textContent = count ? String(count) : '';
  reviewBadge.hidden = !count;
}
// 미니맵: 전체화면의 사람 시점에서 서 있는 층의 평면과 현재 위치·시야를 오른쪽 아래에 보여 준다.
const miniMap = MiniMap({
  pose: () => viewer.personPose(), plan: (key) => viewer.floorPlan(key), label: (key) => floorLabel(key),
});
// 조작 안내와 전체화면 버튼은 뷰어 위에 겹쳐 둔다(뷰어를 더 크게 쓰기 위해).
const view3d = h('div', { class: 'l-stage__view', 'aria-label': '3D 모델' }, modelTitle, viewTools,
  h('div', { class: 'c-viewer-frame' }, viewerHost, fullButton, viewerHelp.el, miniMap.el, reviewCard.el));
function setFullscreen(on) {
  view3d.classList.toggle('l-stage__view--full', on);
  miniMap.setVisible(on && Boolean(viewer.person));
  fullButton.replaceChildren(icon(on ? 'minimize' : 'maximize'));
  fullButton.setAttribute('aria-pressed', String(on));
  fullButton.setAttribute('aria-label', on ? '전체화면 끝내기' : '전체화면으로 보기');
  fullButton.title = on ? '전체화면 끝내기 (Esc)' : '전체화면으로 보기';
}
setFullscreen(false);
// 사람 시점의 Esc는 뷰어가 먼저 쓴다(시점 종료). 그때는 전체화면을 유지한다.
window.addEventListener('keydown', (event) => {
  if (event.code === 'Escape' && !viewer.person && view3d.classList.contains('l-stage__view--full')) setFullscreen(false);
}, true);
const plans = PlanViewer();
const view2d = h('div', { class: 'l-stage__view', hidden: true }, plans.el);
const tabs = Tabs({
  tabs: [{ key: '3d', label: '3D 모델' }, { key: '2d', label: '2D 도면' }, { key: 'split', label: '분할' }],
  active: '3d',
  onChange: (key) => {
    if (key === '2d') setFullscreen(false);
    view3d.hidden = key === '2d';
    view2d.hidden = key === '3d';
    stageBody.classList.toggle('l-stage__body--split', key === 'split');
    divider.hidden = key !== 'split';
    split.refresh();
    if (key !== '2d') {
      viewer.resize();
      if (!viewer.person) viewer.fit();
    }
    if (key !== '3d') plans.fit();
  },
});
const stageBody = h('div', { class: 'l-stage__body' }, view3d, view2d);
const split = SplitDivider(stageBody, view3d, view2d);
const divider = split.el;
view2d.before(divider);
const projectTitle = h('strong', {}, 'cad2bim');
const projectSubtitle = h('span', {}, '도면을 열어 시작하세요');
const drawingList = h('div', { class: 'c-drawing-list' }, emptyDrawings());
const drawingsPanel = Panel({ title: '원본 도면', children: [drawingList] });
const documentsView = h('div', { class: 'c-sidebar-page', id: 'sidebar-documents' }, drawingsPanel.el, layersPanel.el);
const displayView = h('div', { class: 'c-sidebar-page', id: 'sidebar-display', hidden: true },
  Panel({ title: '보기 설정', children: [autoRotate, h('p', { class: 'c-panel__hint' }, '색상과 내부 단면은 3D 화면 위에서 선택하세요.')] }).el);
const resultsView = h('div', { class: 'c-sidebar-page', id: 'sidebar-results', hidden: true },
  statsPanel.el, Button({ label: '샘플 보기', variant: 'ghost', onClick: () => showResult(true) }));
const sidebarButton = Button({
  label: '설정 패널 접기', variant: 'ghost', size: 'small', onClick: () => setSidebar(sidebar.hidden),
});
sidebarButton.replaceChildren(icon('chevron-left'));
sidebarButton.setAttribute('aria-label', '설정 패널 접기');
const reviewView = h('div', { class: 'c-sidebar-page', id: 'sidebar-review', hidden: true },
  Panel({ title: 'AI 검토', children: [reviewList.el] }).el);
const reviewBadge = h('span', { class: 'c-rail__badge', hidden: true, 'aria-label': '검토할 곳 수' });
const sidebar = h('aside', { class: 'l-sidebar', id: 'settings-sidebar' },
  h('div', { class: 'c-project' }, h('div', {}, projectTitle, projectSubtitle), sidebarButton),
  documentsView, displayView, resultsView, reviewView,
  h('div', { class: 'c-sidebar-status' }, progress.el));
const sidebarPages = [documentsView, displayView, resultsView, reviewView];
const railButtons = ['도면', '표시', '결과', '검토'].map((label, i) => h('button', {
  class: 'c-rail__button', type: 'button', 'aria-pressed': String(i === 0),
  'aria-controls': sidebarPages[i].id,
  onClick: () => {
    sidebarPages.forEach((page, j) => { page.hidden = i !== j; });
    railButtons.forEach((button, j) => button.setAttribute('aria-pressed', String(i === j)));
    setSidebar(true);
  },
}, icon(['file', 'eye', 'briefcase', 'check-square'][i]), h('span', {}, label), i === 3 && reviewBadge));
const rail = h('nav', { class: 'c-rail', 'aria-label': '작업 메뉴' },
  h('div', { class: 'c-rail__brand' }, toolbar.querySelector('.c-toolbar__icon')), ...railButtons);
function setSidebar(open) {
  sidebar.hidden = !open;
  sidebar.closest('.l-workspace').classList.toggle('l-workspace--sidebar-closed', !open);
  sidebarButton.setAttribute('aria-expanded', String(open));
  viewer.resize();
  if (!view3d.hidden && !viewer.person) viewer.fit();
}
sidebarButton.setAttribute('aria-controls', sidebar.id);
sidebarButton.setAttribute('aria-expanded', 'true');

const overlay = DropOverlay();
const toast = Toast();

root.append(
  h('div', { class: 'l-app' },
    rail,
    h('div', { class: 'l-workspace' }, toolbar,
    h('main', { class: 'l-body' },
      sidebar,
      h('section', { class: 'l-stage' },
        h('div', { class: 'l-stage__header' }, tabs.el), stageBody),
    )),
  ),
  overlay.el,
  toast.el,
);

const viewer = new Viewer3D(viewerHost);
viewer.onPersonChange = (on) => {
  personButton.textContent = on ? '사람 시점 끝내기' : '사람 시점';
  personButton.setAttribute('aria-pressed', String(on));
  viewerHelp.setPerson(on);
  miniMap.setVisible(on && view3d.classList.contains('l-stage__view--full'));
  updateModelTitle();
};
viewer.onFrame = () => miniMap.update();
// 엘리베이터 안에서는 층 고르는 법을 보여 준다.
viewer.onPersonHint = (hint) => {
  if (viewer.person) viewerHelp.setHint(hint || '');
};

function updateModelTitle() {
  const floors = viewer.floors();
  const shown = floors.filter((key) => !viewer.state.hiddenFloors.has(key));
  const scope = shown.length === floors.length ? '전체 건물'
    : shown.length ? `표시 층 · ${shown.map(floorLabel).join(' · ')}` : '표시 층 없음';
  modelTitle.textContent = viewer.person ? '3D · 사람 시점' : cutFloor
    ? `3D · ${floorLabel(cutFloor)} 내부 단면 · ${viewer.state.hiddenFloors.has(cutFloor) ? '숨김' : '높이 1.25m'}` : `3D · ${scope}`;
}

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
  } else if (event.type === 'plans') {
    const ready = await api.getPlans();
    if (ready) plans.setPlans(ready);
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
    miniMap.reset();
    const floors = viewer.load(model.dae);
    viewer.setFurniture(model.furniture);
    viewerEmpty.hidden = true;
    viewerHelp.el.hidden = false;
    fullButton.hidden = false;
    cutFloor = null;
    updateModelTitle();
    renderLayers(floors);
    renderInside(floors);
    renderStats(model.summary, floors.length);
    setReview(await api.getReview(useSample));
    const sourcePlans = model.plans || [];
    const planButtons = sourcePlans.map((plan, i) => h('button', {
      class: 'c-drawing-list__item', type: 'button', 'aria-pressed': String(i === 0),
      onClick: () => { plans.select(i); if (view2d.hidden) tabs.select('split'); },
    }, plan.name));
    planButtons.forEach((button, i) => { button.title = sourcePlans[i].name; });
    plans.onSelect = (i) => planButtons.forEach((button, j) => button.setAttribute('aria-pressed', String(i === j)));
    drawingList.replaceChildren(...(planButtons.length ? planButtons : [emptyDrawings()]));
    plans.setPlans(sourcePlans);
    projectTitle.textContent = useSample ? '주민센터' : '변환 결과';
    projectSubtitle.textContent = useSample ? '샘플 프로젝트' : `${sourcePlans.length}개 원본 도면`;

    progress.set({ state: 'done', message: `${useSample ? '샘플(주민센터)' : '변환 완료'} · ${floors.length}개 층${took}` });
  } catch (err) {
    progress.set({ state: 'error', message: err.message || String(err) });
  }
}

// 도면 초기화: 화면을 도면을 넣기 전으로 돌린다. 변환 결과 파일은 그대로 둔다('결과 열기'로 다시 열 수 있다).
async function resetAll() {
  if (started || !(await api.reset())) {
    toast.show('변환 중에는 초기화할 수 없습니다. 끝난 뒤 다시 눌러 주세요.');
    return;
  }
  clearInterval(ticker);
  setFullscreen(false);
  viewer.clear();
  miniMap.reset();
  viewerEmpty.hidden = false;
  viewerHelp.el.hidden = true;
  fullButton.hidden = true;
  cutFloor = null;
  modelTitle.textContent = '3D · 모델 없음';
  insideControls.replaceChildren();
  insideControls.hidden = true;
  cutSwitch = null;
  setReview(null);
  layersPanel.setChildren([emptyLayers()]);
  statsPanel.setChildren([h('div', { class: 'c-panel__empty' }, '-')]);
  drawingList.replaceChildren(emptyDrawings());
  plans.onSelect = null;
  plans.setPlans([]);
  projectTitle.textContent = 'cad2bim';
  projectSubtitle.textContent = '도면을 열어 시작하세요';
  progress.set({ state: 'idle', message: IDLE_MESSAGE });
  tabs.select('3d');
}

function renderLayers(floors) {
  layersPanel.setChildren([
    ...floors.map((key) => {
      const categories = [['Walls', '벽체'], ['Doors', '문'], ['Windows', '창'], ['Columns', '기둥'], ['Slabs', '바닥'], ['Stairs', '계단']]
        .filter(([suffix]) => viewer.groups.has(`${key}_${suffix}`));
      return h('details', { class: 'c-floor-tree' },
        h('summary', {}, icon('layers'), h('span', {}, floorLabel(key))),
        ToggleList({ items: [{ key, label: `${floorLabel(key)} 표시` }], onChange: (_, on) => {
          viewer.setFloorVisible(key, on); updateModelTitle();
        } }),
        ToggleList({ items: categories.map(([suffix, label]) => ({ key: `${key}_${suffix}`, label })),
          onChange: (name, on) => viewer.setGroupVisible(name, on) }),
      );
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
    options: [{ key: '', label: '전체 건물' }, ...floors.map((key) => ({ key, label: floorLabel(key) }))],
    active: '',
    onChange: (key) => {
      cutFloor = key || null;
      viewer.setCut(cutFloor);
      updateModelTitle();
    },
  });
  cutSwitch = cut;
  insideControls.replaceChildren(h('span', { class: 'c-panel__hint' }, '층·내부 보기'), cut.el);
  insideControls.hidden = false;
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
  if (mode === 'browser') toolbar.title = '브라우저 미리보기 · 파일 열기는 데스크톱 앱에서 사용하세요';
  // 주소 끝에 ?sample 을 붙이면 샘플을 바로 띄운다. 디자인 작업과 화면 캡처용이다.
  const params = new URLSearchParams(location.search);
  const shown = params.has('sample') ? showResult(true) : Promise.resolve();
  shown.then(() => { if (['2d', 'split'].includes(params.get('tab'))) tabs.select(params.get('tab')); });
});
