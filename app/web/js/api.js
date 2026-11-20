// Python과 연결하는 유일한 창구.
// - 데스크톱 앱(pywebview): window.pywebview.api를 부른다.
// - 일반 브라우저: 가짜 데이터 모드. sample/의 주민센터 결과로 같은 흐름을 흉내 낸다.
//   Python 없이 화면만 띄워 디자인 작업을 할 때 쓴다.

const listeners = new Set();
window.cad2bimEvent = (event) => listeners.forEach((fn) => fn(event));

export function onEvent(fn) {
  listeners.add(fn);
}

function emit(event) {
  window.cad2bimEvent(event);
}

/** pywebview가 붙으면 'desktop', 아니면 'browser'. */
export function ready() {
  return new Promise((resolve) => {
    if (window.pywebview?.api) return resolve('desktop');
    const done = () => resolve(window.pywebview?.api ? 'desktop' : 'browser');
    window.addEventListener('pywebviewready', done, { once: true });
    setTimeout(done, 1500);
  });
}

let mode = 'browser';
export async function init() {
  mode = await ready();
  return mode;
}

export async function chooseFolder() {
  if (mode === 'desktop') return window.pywebview.api.choose_folder();
  return simulateConvert();
}

export async function openResult() {
  if (mode === 'desktop') return window.pywebview.api.open_result();
  emit({ type: 'done', sample: true });
  return true;
}

/** 화면을 도면을 넣기 전으로 돌려도 되는지 Python에 묻고, 지금 결과를 잊게 한다. 변환 중이면 false. */
export async function reset() {
  if (mode === 'desktop') return (await window.pywebview.api.reset?.()) ?? true;
  return true;
}

/** 일반 브라우저에서 놓았을 때. 데스크톱에서는 Python의 drop 처리기가 대신 받는다. */
export function dropInBrowser() {
  if (mode === 'browser') simulateConvert();
}

export async function getModel(useSample = false) {
  if (mode === 'desktop' && !useSample) return window.pywebview.api.get_model();
  return loadSample();
}

/** 뒤에서 다 그린 2D 도면. 브라우저 미리보기에는 없다. */
export async function getPlans() {
  if (mode === 'desktop') return window.pywebview.api.get_plans();
  return null;
}

// 샘플(브라우저 미리보기, 앱의 '샘플 보기')에서 사람이 고친 답. 파일에 쓰지 않고 이 화면에서만 흉내 낸다.
const sampleDecisions = new Map(); // key → { decision, applied }
let reviewFromSample = true;

/** 검토 지점 목록({ answers, items }). 규칙이 판단하지 못해 LLM·사람이 고르는 곳이다. 없으면 빈 목록.
 *  항목의 decision은 지금 답, applied는 3D 모델에 반영된 답이다. */
export async function getReview(useSample = false) {
  const empty = { answers: [], items: [] };
  reviewFromSample = !(mode === 'desktop' && !useSample);
  if (!reviewFromSample) return (await window.pywebview.api.get_review?.()) ?? empty;
  try {
    const response = await fetch('sample/review.json');
    if (!response.ok) return empty;
    const review = await response.json();
    review.items.forEach((item) => {
      if (item.image) item.image = `sample/${item.image}`;
      if (!('applied' in item)) item.applied = item.decision?.answer ?? null;
      const mine = sampleDecisions.get(item.key);
      if (mine) Object.assign(item, mine);
    });
    return review;
  } catch {
    return empty;
  }
}

/** 검토 지점의 답을 사람이 정한다. answer가 null이면 AI 답으로 되돌린다. { ok, decision, message }를 돌려준다. */
export async function setReviewDecision(item, answer) {
  if (!reviewFromSample) {
    return (await window.pywebview.api.set_review_decision?.(item.key, answer)) ?? { ok: false, message: '이 앱 버전은 답 저장을 지원하지 않습니다.' };
  }
  const now = item.decision;
  const ai = now?.model === 'user' ? now.ai : now;
  let decision = ai || null;
  if (answer !== null) {
    decision = { answer, reason: '', model: 'user' };
    if (ai) decision.ai = { answer: ai.answer, reason: ai.reason || '', model: ai.model || '' };
  }
  sampleDecisions.set(item.key, { decision, applied: item.applied ?? null });
  return { ok: true, decision };
}

/** 바뀐 답을 3D 모델에 반영한다(다시 빌드). 진행과 완료는 onEvent로 온다. */
export async function applyReview() {
  if (!reviewFromSample) return window.pywebview.api.apply_review?.();
  // 샘플은 다시 만들 수 없다. 반영된 것으로 표시만 바꾸고 같은 흐름을 흉내 낸다.
  sampleDecisions.forEach((mine) => { mine.applied = mine.decision?.answer ?? null; });
  return simulateConvert();
}

export async function loadSample() {
  const [dae, meta, furniture] = await Promise.all([
    fetch('sample/building.dae').then((r) => r.text()),
    fetch('sample/summary.json').then((r) => r.json()),
    fetch('sample/furniture.json').then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]);
  return {
    dae,
    furniture,
    summary: meta.summary,
    plans: meta.plans.map((p) => ({ name: p.name, src: `sample/${p.file}` })),
  };
}

async function simulateConvert() {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  emit({ type: 'started', count: 3 });
  for (let i = 1; i <= 3; i += 1) {
    await wait(400);
    emit({ type: 'progress', stage: 'read', done: i, total: 3 });
  }
  emit({ type: 'progress', stage: 'model' });
  await wait(800);
  emit({ type: 'done', sample: true });
  return true;
}
