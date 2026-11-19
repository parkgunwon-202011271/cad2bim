// 미니맵. 사람 시점에서 서 있는 층의 평면을 그리고, 현재 위치(점)와 보는 방향·시야(부채꼴)를 실시간으로 표시한다.
// 평면은 3D 모델을 눈높이 아래(내부 보기와 같은 높이)에서 자른 것이라 3D와 위치가 정확히 맞는다.
import { h } from './dom.js';

const MAX_WIDTH = 280; // px. 지도가 차지하는 최대 크기(가로로 긴 건물이든 세로로 긴 건물이든 이 안에 맞춘다)
const MAX_HEIGHT = 280;
const PAD = 10;
const VIEW_RANGE_M = 7; // 부채꼴이 닿는 거리
const VIEW_RANGE_MIN_PX = 30;
const COLORS = {
  stairBelow: '#e6e9ed', slab: '#ffffff', stair: '#d5dae0',
  rail: '#a3acb6', window: '#5fa4dc', door: '#b98d5f', wall: '#3b4048', column: '#3b4048',
  me: '#2563eb', meRing: '#ffffff', view: '37, 99, 235',
};
const CUT_ORDER = ['rail', 'window', 'door', 'wall', 'column'];

/** pose(): 사람 시점의 위치·방향 또는 null. plan(key): 층 평면. label(key): 층 이름. */
export function MiniMap({ pose, plan, label }) {
  const title = h('div', { class: 'c-minimap__title' });
  const canvas = h('canvas', { class: 'c-minimap__canvas' });
  const el = h('div', { class: 'c-minimap', hidden: true, 'aria-label': '현재 위치 지도' }, title, canvas);
  const ctx = canvas.getContext('2d');
  const layers = new Map(); // 층 → 미리 그려 둔 평면
  let shown = null;

  function build(key) {
    const data = plan(key);
    if (!data) return null;
    const { minX, maxX, minZ, maxZ } = data.bounds;
    const scale = Math.min(
      (MAX_WIDTH - PAD * 2) / Math.max(maxX - minX, 1e-6),
      (MAX_HEIGHT - PAD * 2) / Math.max(maxZ - minZ, 1e-6),
    );
    const width = Math.round((maxX - minX) * scale + PAD * 2);
    const height = Math.round((maxZ - minZ) * scale + PAD * 2);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const image = document.createElement('canvas');
    image.width = width * ratio;
    image.height = height * ratio;
    const g = image.getContext('2d');
    g.scale(ratio, ratio);
    g.lineJoin = 'round';
    const X = (x) => PAD + (x - minX) * scale;
    const Y = (z) => PAD + (z - minZ) * scale;
    const trace = (points) => points.forEach(([x, z], i) => (i ? g.lineTo(X(x), Y(z)) : g.moveTo(X(x), Y(z))));
    for (const name of ['stairBelow', 'slab', 'stair']) {
      g.beginPath();
      for (const triangle of data.fills[name]) { trace(triangle); g.closePath(); }
      g.fillStyle = COLORS[name];
      g.strokeStyle = COLORS[name]; // 삼각형 사이 틈을 같은 색 선으로 메운다
      g.lineWidth = 0.6;
      g.fill();
      g.stroke();
    }
    for (const name of CUT_ORDER) {
      g.beginPath();
      for (const loop of data.cuts[name]) if (loop.closed) { trace(loop.points); g.closePath(); }
      g.fillStyle = COLORS[name];
      g.fill();
      g.beginPath();
      for (const loop of data.cuts[name]) { trace(loop.points); if (loop.closed) g.closePath(); }
      g.strokeStyle = COLORS[name];
      g.lineWidth = name === 'rail' ? 1 : 0.8;
      g.stroke();
    }
    return { key, image, width, height, ratio, scale, X, Y };
  }

  function update() {
    if (el.hidden) return;
    const now = pose();
    if (!now?.floor) return;
    if (shown?.key !== now.floor) {
      if (!layers.has(now.floor)) layers.set(now.floor, build(now.floor));
      shown = layers.get(now.floor);
      if (!shown) return;
      canvas.width = shown.image.width;
      canvas.height = shown.image.height;
      canvas.style.width = `${shown.width}px`;
      canvas.style.height = `${shown.height}px`;
      title.textContent = label(now.floor);
    }
    const { image, width, height, ratio, scale, X, Y } = shown;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(image, 0, 0, width, height);
    // 건물 밖(대지)으로 나가면 지도 가장자리에 붙여 둔다.
    const x = Math.min(Math.max(X(now.x), 6), width - 6);
    const y = Math.min(Math.max(Y(now.z), 6), height - 6);
    const heading = Math.atan2(now.dirZ, now.dirX); // 캔버스는 z가 아래라 각도도 시계 방향이다
    const range = Math.max(VIEW_RANGE_M * scale, VIEW_RANGE_MIN_PX);
    const fade = ctx.createRadialGradient(x, y, 0, x, y, range);
    fade.addColorStop(0, `rgba(${COLORS.view}, 0.5)`);
    fade.addColorStop(1, `rgba(${COLORS.view}, 0)`);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, range, heading - now.hfov / 2, heading + now.hfov / 2);
    ctx.closePath();
    ctx.fillStyle = fade;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.me;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = COLORS.meRing;
    ctx.stroke();
  }

  return {
    el,
    update,
    /** 켜면 바로 한 번 그린다. */
    setVisible(on) { el.hidden = !on; if (on) update(); },
    /** 모델이 바뀌면 그려 둔 평면을 버린다. */
    reset() { layers.clear(); shown = null; },
  };
}
