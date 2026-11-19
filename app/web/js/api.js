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
