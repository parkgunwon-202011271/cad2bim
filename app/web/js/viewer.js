// 360도 3D 뷰어. building.dae(층 → 종류 묶음)를 읽어 돌려 보고, 층·지붕·대지를 켜고 끈다.
// 마우스: 왼쪽 드래그 회전, 휠 확대(커서 쪽으로), 오른쪽 드래그 이동.
// 키보드: W·S 앞뒤, A·D 옆, Q·E 아래·위, Shift 빠르게. 보는 중심도 함께 움직인다.
// 바닥 가까이(3m 안)에서는 화각을 70°로 넓힌다. 밖에서는 40°다.
// 사람 시점(enterPerson): 눈높이 1.6m로 서서 걷는다. 드래그는 고개 돌리기, 벽은 막고 계단은 오른다.
// 가구: 도면의 가구·위생기구 선을 층 바닥에 그린다(furniture.json).
//
// 표시 스타일
// - 건축(기본): Revit의 '음영 + 모서리'처럼 차분한 재료색에 모서리 선을 긋는다.
// - 분류 색: DAE에 든 종류별 색(기둥 파랑, 문 초록 …). 인식 결과를 확인할 때 쓴다.
// 내부 보기: 층을 고르면 그 층만 남기고 바닥에서 CUT_HEIGHT 위를 잘라 단면을 칠한다(평면도와 같은 높이).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';

// 층 이름: B1·F1…, 부분층(상부·중층) F4M, 지붕 R
const FLOOR = /^(?:(B|F)(\d+)(M?)|R)$/;
const FLOOR_PART = /^((?:B|F)\d+M?|R)_/;
// 미리보기(core/tools/preview.py)에서 지붕 위로 보는 대지 요소와 같다.
const ROOF_SITE = new Set([
  'SITE_Lawn', 'SITE_RoofPlanters', 'SITE_Pergola', 'SITE_Skylights', 'SITE_SkylightFrames',
  'SITE_Equipment', 'SITE_Benches', 'SITE_Parapets', 'SITE_Penthouses',
]);
// m. 평면도를 자르는 높이. 난간 높이(1.2m)와 같으면 난간 윗면이 자르는 면과 겹쳐 지직거리므로 조금 위로 둔다.
const CUT_HEIGHT = 1.25;
const START_AZIMUTH = THREE.MathUtils.degToRad(-55); // 첫 시점: 남서쪽 위
const START_ELEVATION = THREE.MathUtils.degToRad(30);
const CUT_ELEVATION = THREE.MathUtils.degToRad(55); // 내부 보기는 더 위에서 내려다본다
const FLY_MS = 450;
// 키 → [앞, 옆, 위]. 화살표 키도 같다.
const MOVE_KEYS = {
  KeyW: [1, 0, 0], ArrowUp: [1, 0, 0], KeyS: [-1, 0, 0], ArrowDown: [-1, 0, 0],
  KeyA: [0, -1, 0], ArrowLeft: [0, -1, 0], KeyD: [0, 1, 0], ArrowRight: [0, 1, 0],
  KeyQ: [0, 0, -1], KeyE: [0, 0, 1],
};
const WALK_MIN = 2; // m/s. 가까이서는 걸음 속도
const WALK_MAX = 25; // m/s. 멀리서 건물 둘레를 돌 때
const RUN = 3; // Shift를 누르면 이만큼 빠르게
// 화각. 40°는 건물 전체를 밖에서 볼 때 왜곡이 적다. 방 안에서는 좁아 답답하므로 사람 시야에 가깝게 넓힌다.
const FOV_OUTSIDE = 40;
const FOV_INSIDE = 70;
// 바닥에서 EYE_LOCK_MAX 안쪽 높이면 방 안으로 보고 화각을 넓힌다.
const EYE_HEIGHT = 1.6; // m. 사람 시점의 눈높이
const EYE_LOCK_MAX = 3.0;
// 사람 시점
const PERSON_SPEED = 1.6; // m/s. 걷는 속도
const PERSON_RUN = 2.5; // Shift를 누르면 이만큼 빠르게
const STEP_UP = 0.45; // m. 이보다 높은 턱은 오르지 못한다(계단 한 단은 0.2m 안팎)
const PERSON_KNEE = 0.6; // m. 앞을 막는지 보는 높이(바닥에서)
const PERSON_CHEST = 1.3;
const PERSON_RADIUS = 0.3; // m. 벽에 이보다 가까이 붙지 않는다
const LOOK_SPEED = 0.004; // rad/px. 드래그로 고개 돌리기
const LOOK_PITCH = 1.45; // rad. 위아래로 보는 한계
// 사람 시점에서 지나갈 수 있는 것. 문은 닫혀 그려져 있어도 지나간다. 바닥류는 디딤면이라 막지 않는다.
const PASSABLE = new Set([
  'door', 'slab', 'roof', 'ground', 'surroundings', 'road', 'lawn', 'parking', 'tree_crown', 'shrub',
]);
const GROUND_CHECK_MS = 120;
const DOWN = new THREE.Vector3(0, -1, 0);
const EYE_LOOK_FAR = 50; // m. 앞이 트였는지 볼 거리
const EYE_LOOK_OPEN = 4; // m. 지금 방향이 이만큼 트였으면 방향을 바꾸지 않는다
const EYE_STEP_BACK = 2.5; // m. 눈높이로 내려가며 뒤로 물러서는 최대 거리
// 디딜 수 있는 면. 층 바닥, 계단, 옥상 테라스, 대지.
const WALKABLE = /(_Slabs|_Stairs|_Roofs)$|^SITE_(Ground|Surroundings|Roads|Decks|Lawn|ExteriorStairs|Parking)$/;

// 건축 스타일 재료(sRGB). 이름은 core/src/cad2bim/export/collada.py의 MATERIALS와 같다.
const ARCH = {
  wall: { color: '#eceae5' }, penthouse: { color: '#eceae5' },
  column: { color: '#c9c6bf' }, slab: { color: '#dcdad5' }, parapet: { color: '#dcdad5' },
  roof: { color: '#b9babc' }, stair: { color: '#d3cec4' }, exterior_stair: { color: '#d6d3cb' },
  // 난간은 평면에 선 하나라 높이 1.2m 판으로 들어온다. 검은 판이면 무거워 보여 유리 난간으로 그린다.
  rail: { glass: true, opacity: 0.4, color: '#c3d3dc' }, site_rail: { color: '#5d6168', metal: true },
  door: { color: '#a07a56' },
  window: { glass: true }, curtain: { glass: true }, skylight: { glass: true },
  skylight_frame: { color: '#d2cfc8' }, equipment: { color: '#a9abaf' },
  ground: { color: '#e4e0d5' }, surroundings: { color: '#efeee9' }, road: { color: '#76787c' },
  parking: { color: '#ffffff' }, deck: { color: '#bb9c77' }, bench: { color: '#ae8a66' },
  pergola: { color: '#8c6c4c' }, planter: { color: '#c2bdb2' }, roof_planter: { color: '#c2bdb2' },
  lawn: { color: '#a8bf8a' }, shrub: { color: '#8faa79' }, tree_crown: { color: '#94b07f' },
  tree_trunk: { color: '#7b634a' },
  car: { color: '#d8dadd', metal: true }, car_dark: { color: '#435069', metal: true },
  car_glass: { color: '#2c3138' },
};
// 모서리 선을 긋지 않는 재료. 지형·나무처럼 곡면을 잘게 나눈 것은 선이 지저분해진다.
const NO_EDGES = new Set(['ground', 'surroundings', 'road', 'lawn', 'tree_crown', 'tree_trunk', 'shrub', 'parking']);
// 기둥은 벽 면과 같은 면에 놓이는 일이 많다. 같은 깊이면 두 면이 번갈아 그려져 줄무늬가 생기므로
// 이 재질들을 깊이 비교에서 더 뒤로 민다. 겹치는 면에서는 기둥이 보인다. 모델 파일은 그대로다.
const PUSHED_BACK = new Set(['wall', 'slab', 'roof', 'parapet', 'penthouse']);
// 분류 색 재질(Lambert)은 주변광 지도를 받지 않으므로 반구광을 더 세게 둔다(이전 화면과 같은 밝기).
const LIGHTS = { arch: { hemi: 0.5, sun: 1.9 }, class: { hemi: 1.6, sun: 2.2 } };
const EDGE_COLOR = '#30343a';
const FURNITURE_COLOR = '#7a6f62'; // 가구 선. 모서리 선보다 옅게 해 벽과 구분한다
const FURNITURE_LIFT_MM = 5; // 바닥 면과 겹쳐 지직거리지 않게 조금 띄운다
// 가구 입체(furniture.json의 solids) 종류별 색. 높이는 core/src/cad2bim/export/furniture.py가 정한다.
const FURNITURE_SOLIDS = {
  chair: '#6b7685', table: '#c9ab85', shelf: '#b49b7c', counter: '#e6e2da', fixture: '#f4f4f1', partition: '#d4d1ca',
  appliance: '#8b939d', // 정수기·키오스크·자판기처럼 이름표로 찾은 기기
};
// 사람 시점에서 막는 가구. 의자·탁자는 사이로 지나다닐 수 있게 막지 않는다.
const FURNITURE_BLOCKING = new Set(['shelf', 'counter', 'fixture', 'partition', 'appliance']);
// 엘리베이터(mm). 카 크기는 도면의 승강로 표시를 따르고 높이·문은 흔한 값이다.
const ELEVATOR_COLOR = '#b8bdc3';
const ELEVATOR_CAR_MM = 2300; // 카 안 높이
const ELEVATOR_PLATE_MM = 50; // 카 바닥·천장 두께
const ELEVATOR_WALL_MM = 40;
const ELEVATOR_DOOR_MM = 2100; // 문 높이(개구부 높이와 같다)
const ELEVATOR_LEAF_MM = 40; // 문짝 두께
const ELEVATOR_LEAF_SET_MM = 130; // 벽 중심선에서 승강로 쪽으로 문짝을 미는 거리(벽 두께 절반 + 여유)
const ELEVATOR_SPEED_MM = 1200; // mm/s
const ELEVATOR_DOOR_SPEED = 1.2; // 1/s. 다 열리는 데 약 0.8초
const ELEVATOR_CALL_MM = 1500; // 문 앞 이 거리 안에 서면 카를 부른다
const CAP_COLOR = '#4b4f56'; // 잘린 면(포셰)
// 미니맵. 발밑 높이보다 이만큼 위에 있는 층 바닥까지는 그 층으로 본다(계단을 거의 다 오르면 윗층 평면으로 바뀐다).
const FLOOR_SWITCH = 1.0; // m

export function floorLabel(key) {
  const m = FLOOR.exec(key);
  if (!m) return key;
  if (key === 'R') return '지붕층';
  const base = m[1] === 'B' ? `지하${m[2]}층` : `${m[2]}층`;
  return m[3] ? `${base} 상부` : base;
}

export class Viewer3D {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, stencil: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.localClippingEnabled = true;
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    container.append(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = skyTexture();
    this.scene.environment = this.makeEnvironment();
    this.scene.environmentIntensity = 0.9;
    this.camera = new THREE.PerspectiveCamera(FOV_OUTSIDE, 1, 0.1, 5000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotateSpeed = 1.2;
    this.controls.maxPolarAngle = THREE.MathUtils.degToRad(89);
    this.controls.zoomToCursor = true; // 휠로 커서 쪽 방 안까지 들어간다

    this.hemi = new THREE.HemisphereLight(0xffffff, 0xb8b4aa, LIGHTS.arch.hemi);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xfffaf0, LIGHTS.arch.sun);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.02;
    this.scene.add(this.sun, this.sun.target);

    this.cutPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0); // y <= constant만 남긴다
    this.cap = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({
        color: CAP_COLOR, stencilWrite: true, stencilRef: 0,
        stencilFunc: THREE.NotEqualStencilFunc,
        stencilFail: THREE.ReplaceStencilOp, stencilZFail: THREE.ReplaceStencilOp, stencilZPass: THREE.ReplaceStencilOp,
      }),
    );
    this.cap.renderOrder = 2;
    this.cap.visible = false;
    this.cap.onAfterRender = (renderer) => renderer.clearStencil();
    this.scene.add(this.cap);

    this.model = null;
    this.groups = new Map();
    this.levels = new Map(); // 층 → 바닥 높이(m)
    this.meshes = []; // { mesh, building }
    this.walkable = []; // 눈높이를 잴 바닥 면
    this.groundRay = new THREE.Raycaster();
    this.ground = null;
    this.groundTime = -Infinity;
    this.clipped = new Set(); // 단면으로 자르는 재질
    this.stencils = [];
    this.style = 'arch';
    this.state = { roof: true, site: true, furniture: true, hiddenFloors: new Set(), cut: null };
    this.fly = null;
    this.farLimit = 5000;
    this.radius = 30;

    this.obstacles = []; // 사람 시점에서 막는 것
    this.furniture = []; // 층별 가구 선
    this.elevators = [];
    this.lastHint = null;
    this.onPersonHint = null;
    this.person = null;
    this.look = null;
    this.onPersonChange = null;
    this.onFrame = null; // 한 장면을 그린 뒤 부른다(미니맵 갱신)

    const canvas = this.renderer.domElement;
    canvas.addEventListener('pointerdown', (e) => this.onLook(e, 'down'));
    canvas.addEventListener('pointermove', (e) => this.onLook(e, 'move'));
    canvas.addEventListener('pointerup', (e) => this.onLook(e, 'up'));
    canvas.addEventListener('pointercancel', (e) => this.onLook(e, 'up'));
    canvas.addEventListener('contextmenu', (e) => { if (this.person) e.preventDefault(); });
    canvas.addEventListener('wheel', (e) => {
      if (!this.person) return; // 궤도 보기의 휠은 OrbitControls가 맡는다
      e.preventDefault();
      this.person.fov = THREE.MathUtils.clamp(this.person.fov + Math.sign(e.deltaY) * 4, 35, 90);
    }, { passive: false });

    this.keys = new Set();
    this.run = false;
    this.lastTime = null;
    window.addEventListener('keydown', (e) => this.onKey(e, true));
    window.addEventListener('keyup', (e) => this.onKey(e, false));
    window.addEventListener('blur', () => this.keys.clear());
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop((time) => this.frame(time));
  }

  /** 하늘은 밝고 땅은 따뜻한 회색인 단순한 주변광. 유리에 은은하게 비친다. */
  makeEnvironment() {
    const env = new THREE.Scene();
    const geo = new THREE.SphereGeometry(10, 32, 16);
    const top = new THREE.Color('#ffffff');
    const horizon = new THREE.Color('#dfe5ec');
    const ground = new THREE.Color('#b9b4a9');
    const colors = [];
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i += 1) {
      const t = p.getY(i) / 10;
      const c = t > 0 ? horizon.clone().lerp(top, t) : horizon.clone().lerp(ground, Math.min(1, -t * 3));
      colors.push(c.r, c.g, c.b);
    }
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    env.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })));
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const texture = pmrem.fromScene(env, 0.04).texture;
    pmrem.dispose();
    return texture;
  }

  onKey(event, down) {
    this.run = event.shiftKey;
    if (down && event.code === 'Escape' && this.person) {
      this.exitPerson();
      return;
    }
    if (down && this.person && /^(Digit\d|PageUp|PageDown)$/.test(event.code) && this.pressElevator(event.code)) {
      event.preventDefault();
      return;
    }
    if (!(event.code in MOVE_KEYS)) return;
    // 입력 칸에 글을 쓰거나 3D 화면이 가려져 있을 때는 움직이지 않는다.
    const typing = event.target instanceof HTMLElement && event.target.closest('input, textarea, select, [contenteditable]');
    if (down && (typing || !this.model || !this.container.getClientRects().length)) return;
    if (down) {
      this.keys.add(event.code);
      event.preventDefault(); // 화살표 키로 옆 패널이 스크롤되지 않게
    } else {
      this.keys.delete(event.code);
    }
  }

  /** 누른 키만큼 카메라와 보는 중심을 함께 옮긴다. 앞뒤·옆은 수평으로만 움직여 고개를 숙여도 땅으로 꺼지지 않는다. */
  walk(dt) {
    const result = { moved: false, vertical: false };
    if (!this.keys.size || !dt) return result;
    this.fly = null;
    const step = [0, 0, 0];
    for (const code of this.keys) MOVE_KEYS[code].forEach((v, i) => { step[i] += v; });
    result.moved = step[0] !== 0 || step[1] !== 0;
    result.vertical = step[2] !== 0;
    const forward = this.controls.target.clone().sub(this.camera.position).setY(0);
    if (forward.lengthSq() < 1e-8) forward.set(0, 0, -1).applyQuaternion(this.camera.quaternion).setY(0);
    forward.normalize();
    const right = new THREE.Vector3().crossVectors(forward, this.camera.up).normalize();
    const move = forward.multiplyScalar(step[0]).addScaledVector(right, step[1]).add(new THREE.Vector3(0, step[2], 0));
    if (move.lengthSq() === 0) return result;
    const distance = this.camera.position.distanceTo(this.controls.target);
    const speed = THREE.MathUtils.clamp(distance * 0.6, WALK_MIN, WALK_MAX) * (this.run ? RUN : 1);
    move.normalize().multiplyScalar(speed * dt);
    this.camera.position.add(move);
    this.controls.target.add(move);
    return result;
  }

  /** point 바로 아래의 디딜 면 높이. 없으면 null. 숨긴 층과 대지는 건너뛴다. */
  groundBelow(point) {
    if (!this.walkable.length) return null;
    this.groundRay.set(point, DOWN);
    this.groundRay.far = 60;
    const hit = this.groundRay.intersectObjects(this.walkable, false).find((h) => isShownTree(h.object));
    return hit ? hit.point.y : null;
  }

  /** 눈높이로 설 자리와 바라볼 방향. 벽 바로 앞이면 8방향 중 가장 트인 쪽을 보고, 조금 물러선다. */
  eyeSpot() {
    const target = this.controls.target;
    const probe = target.clone().setY(this.state.cut ? this.cutPlane.constant : target.y + 0.5);
    let ground = this.groundBelow(probe);
    if (ground === null) ground = new THREE.Box3().setFromObject(this.model).min.y;
    const spot = new THREE.Vector3(target.x, ground + EYE_HEIGHT, target.z);
    // 벽 바로 앞에 서면 벽만 보인다. 지금 방향이 충분히 트였으면 그대로, 아니면 8방향 중 가장 트인 쪽을 본다.
    const obstacles = this.meshes
      .filter(({ mesh, building }) => building && isShownTree(mesh) && ![mesh.material].flat().some((m) => m.opacity < 1))
      .map(({ mesh }) => mesh);
    const open = (dir) => {
      this.groundRay.set(spot, dir);
      this.groundRay.far = EYE_LOOK_FAR;
      const hit = this.groundRay.intersectObjects(obstacles, false).find((h) => isShownTree(h.object));
      return hit ? hit.distance : EYE_LOOK_FAR;
    };
    let best = target.clone().sub(this.camera.position).setY(0);
    if (best.lengthSq() < 1e-8) best.set(0, 0, -1);
    best.normalize();
    let bestOpen = open(best);
    if (bestOpen < EYE_LOOK_OPEN) {
      for (let k = 0; k < 8; k += 1) {
        const dir = best.clone().applyAxisAngle(this.camera.up, (k * Math.PI) / 4);
        const d = open(dir);
        if (d > bestOpen) [best, bestOpen] = [dir, d];
      }
    }
    // 뒤로 조금 물러서 앞이 더 넓게 보이게 한다. 뒤의 벽은 넘지 않는다.
    const back = THREE.MathUtils.clamp(open(best.clone().negate()) - 0.5, 0, EYE_STEP_BACK);
    return { position: spot.addScaledVector(best, -back), forward: best, ground };
  }

  /**
   * 사람 시점. 보던 자리의 바닥에 눈높이로 서서 걸어 다닌다. 내부 보기(단면)는 잠시 끄고 천장까지 보인다.
   * 드래그는 고개 돌리기, W·A·S·D는 걷기, Shift는 뛰기, Esc는 나가기. 벽·창·난간은 막고 문은 지나간다.
   */
  enterPerson() {
    if (!this.model || this.person) return;
    const spot = this.eyeSpot();
    this.person = {
      yaw: Math.atan2(-spot.forward.x, -spot.forward.z), pitch: 0, ground: spot.ground, fov: FOV_INSIDE,
      savedPosition: this.camera.position.clone(), savedTarget: this.controls.target.clone(),
    };
    this.controls.enabled = false;
    this.keys.clear();
    this.applyCut();
    this.flyTo(spot.position.clone().addScaledVector(spot.forward, 0.5), spot.position);
    this.onPersonChange?.(true);
  }

  /** 사람 시점을 끝내고 들어오기 전의 시점으로 돌아간다. 내부 보기도 되살린다. */
  exitPerson() {
    if (!this.person) return;
    const { savedPosition, savedTarget } = this.person;
    const look = this.camera.getWorldDirection(new THREE.Vector3());
    this.controls.target.copy(this.camera.position).addScaledVector(look, 0.5);
    this.leavePerson();
    this.flyTo(savedTarget, savedPosition);
  }

  leavePerson() {
    this.person = null;
    this.look = null;
    this.controls.enabled = true;
    this.keys.clear();
    this.applyCut();
    this.onPersonChange?.(false);
  }

  /** 사람 시점 한 걸음. 앞을 막는 것은 미끄러져 돌고, 발밑 바닥을 따라 오르내린다. */
  personStep(dt) {
    const p = this.person;
    if (this.fly) return; // 아직 들어가는 중
    const forward = new THREE.Vector3(-Math.sin(p.yaw), 0, -Math.cos(p.yaw));
    const right = new THREE.Vector3(Math.cos(p.yaw), 0, -Math.sin(p.yaw));
    const move = new THREE.Vector3();
    for (const code of this.keys) {
      const [ahead, side] = MOVE_KEYS[code];
      move.addScaledVector(forward, ahead).addScaledVector(right, side);
    }
    const position = this.camera.position;
    if (p.ground === null) p.ground = this.groundBelow(position);
    const feet = p.ground ?? position.y - EYE_HEIGHT;
    if (dt && move.lengthSq() > 0) {
      move.normalize().multiplyScalar(PERSON_SPEED * (this.run ? PERSON_RUN : 1) * dt);
      const allowed = this.collide(position, move, feet);
      if (allowed.lengthSq() > 0) {
        const next = position.clone().add(allowed);
        // 무릎 높이 위에서 내려다본 바닥. 윗층 바닥 밑면은 그보다 높아 잡히지 않는다.
        const ground = this.groundBelow(new THREE.Vector3(next.x, feet + STEP_UP + 0.05, next.z));
        if (ground !== null && ground - feet <= STEP_UP) {
          position.x = next.x;
          position.z = next.z;
          p.ground = ground;
        }
      }
    }
    if (p.ground !== null) position.y += (p.ground + EYE_HEIGHT - position.y) * Math.min(1, dt * 10);
    this.camera.rotation.set(p.pitch, p.yaw, 0, 'YXZ');
  }

  /** move만큼 가도 되는 만큼. 무릎과 가슴 높이에서 앞을 보고, 막히면 그 면을 따라 미끄러진다. */
  collide(position, move, feet) {
    const obstacles = this.obstacles.filter((m) => isShownTree(m));
    const result = move.clone();
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const length = result.length();
      if (length < 1e-9) return result;
      const dir = result.clone().divideScalar(length);
      let hit = null;
      for (const height of [PERSON_KNEE, PERSON_CHEST]) {
        this.groundRay.set(new THREE.Vector3(position.x, feet + height, position.z), dir);
        this.groundRay.far = length + PERSON_RADIUS;
        const first = this.groundRay.intersectObjects(obstacles, false)[0];
        if (first && (!hit || first.distance < hit.distance)) hit = first;
      }
      if (!hit) return result;
      const normal = hit.face.normal.clone().transformDirection(hit.object.matrixWorld).setY(0);
      if (normal.lengthSq() < 1e-8) break;
      normal.normalize();
      if (normal.dot(dir) > 0) normal.negate();
      result.addScaledVector(normal, -result.dot(normal)); // 벽으로 들어가는 성분을 뺀다
    }
    return new THREE.Vector3();
  }

  onLook(event, phase) {
    if (!this.person) return;
    if (phase === 'down') {
      this.look = { x: event.clientX, y: event.clientY };
      this.renderer.domElement.setPointerCapture(event.pointerId);
    } else if (phase === 'move' && this.look) {
      this.person.yaw -= (event.clientX - this.look.x) * LOOK_SPEED;
      this.person.pitch = THREE.MathUtils.clamp(this.person.pitch - (event.clientY - this.look.y) * LOOK_SPEED, -LOOK_PITCH, LOOK_PITCH);
      this.look = { x: event.clientX, y: event.clientY };
    } else if (phase === 'up') {
      this.look = null;
    }
  }

  frame(time) {
    const dt = this.lastTime === null ? 0 : Math.min((time - this.lastTime) / 1000, 0.1);
    this.lastTime = time;
    if (this.fly) this.stepFly(time);
    let fov;
    this.updateElevators(dt); // 타고 있으면 카와 함께 오르내린다
    if (this.person) {
      this.personStep(dt);
      if (this.fly) this.controls.update(); // 들어가는 동안은 도착 지점을 바라본다
      fov = this.person.fov;
    } else {
      const { moved } = this.walk(dt);
      if (this.model && (moved || time - this.groundTime > GROUND_CHECK_MS)) {
        this.ground = this.groundBelow(this.camera.position);
        this.groundTime = time;
      }
      // 바닥 가까이 다가가면 화각을 넓혀 방이 덜 답답해 보이게 한다.
      const above = this.ground === null ? Infinity : this.camera.position.y - this.ground;
      fov = above >= 0 && above <= EYE_LOCK_MAX ? FOV_INSIDE : FOV_OUTSIDE;
      this.controls.update();
    }
    if (Math.abs(this.camera.fov - fov) > 0.05) {
      this.camera.fov += (fov - this.camera.fov) * Math.min(1, dt * 4);
      this.camera.updateProjectionMatrix();
    }
    // 가까이 들어가면 near를 줄여 벽이 잘려 보이지 않게, 멀리서는 키워 깊이 정밀도를 지킨다.
    const d = this.person ? 0 : this.camera.position.distanceTo(this.controls.target);
    const near = THREE.MathUtils.clamp(d / 300, 0.02, 1);
    if (Math.abs(near - this.camera.near) > 1e-4) {
      this.camera.near = near;
      this.camera.far = this.farLimit;
      this.camera.updateProjectionMatrix();
    }
    this.renderer.render(this.scene, this.camera);
    this.onFrame?.();
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** DAE 글자를 읽어 장면에 올린다. 층 목록을 돌려준다. */
  load(daeText) {
    // 사람 시점 종료는 기존 층의 단면을 복원하므로 그룹을 비우기 전에 끝낸다.
    if (this.person) this.leavePerson();
    if (this.model) this.scene.remove(this.model);
    this.clearStencils();
    // ColladaLoader가 단위(mm → m)와 Z-위 축을 맞춰 준다.
    const collada = new ColladaLoader().parse(daeText, '');
    this.model = collada.scene;
    this.groups.clear();
    this.clipped.clear();
    this.meshes = [];
    this.walkable = [];
    this.obstacles = [];
    this.furniture = [];
    this.elevators = [];
    this.ground = null;
    const archCache = new Map();
    this.edgeMaterials = {
      building: new THREE.LineBasicMaterial({ color: EDGE_COLOR }),
      site: new THREE.LineBasicMaterial({ color: EDGE_COLOR }),
    };
    this.clipped.add(this.edgeMaterials.building);

    this.model.traverse((obj) => {
      if (obj.name) this.groups.set(obj.name, obj);
    });
    const meshes = [];
    this.model.traverse((obj) => { if (obj.isMesh) meshes.push(obj); });
    for (const mesh of meshes) {
      const building = inBuilding(mesh);
      const original = [mesh.material].flat();
      const classMats = original.map((m) => prepare(m));
      const archMats = original.map((m) => {
        if (!archCache.has(m.name)) archCache.set(m.name, prepare(archMaterial(m)));
        return archCache.get(m.name);
      });
      mesh.userData.mats = { class: unwrap(classMats), arch: unwrap(archMats) };
      mesh.material = mesh.userData.mats[this.style];
      mesh.castShadow = !original.every((m) => m.opacity < 1) && !archMats.every((m) => m.opacity < 1);
      mesh.receiveShadow = true;
      if (building) [...classMats, ...archMats].forEach((m) => this.clipped.add(m));
      if (!NO_EDGES.has(original[0].name)) {
        const edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(mesh.geometry, 25),
          building ? this.edgeMaterials.building : this.edgeMaterials.site,
        );
        edges.raycast = () => {};
        mesh.add(edges);
      }
      this.meshes.push({ mesh, building });
      const walkable = hasAncestor(mesh, (name) => WALKABLE.test(name));
      if (walkable) this.walkable.push(mesh);
      // 계단은 디딤면이면서 옆에서는 막는다. 나머지 디딤면(바닥·지붕·대지)은 막지 않는다.
      const category = original[0].name;
      if (!PASSABLE.has(category) && (!walkable || category.includes('stair'))) this.obstacles.push(mesh);
    }
    this.scene.add(this.model);
    this.model.updateMatrixWorld(true);
    this.levels.clear();
    for (const key of this.floors()) {
      const box = new THREE.Box3().setFromObject(this.groups.get(`${key}_Walls`) || this.groups.get(key));
      this.levels.set(key, box.min.y);
    }
    this.state = { roof: true, site: true, furniture: true, hiddenFloors: new Set(), cut: null };
    this.applyCut();
    this.fit();
    return this.floors();
  }

  /** 모델을 비워 아무것도 열지 않은 처음 상태로 돌린다. */
  clear() {
    if (this.person) this.leavePerson();
    this.fly = null;
    this.setFurniture(null);
    this.clearStencils();
    if (this.model) this.scene.remove(this.model);
    this.model = null;
    this.groups.clear();
    this.clipped.clear();
    this.levels.clear();
    this.meshes = [];
    this.walkable = [];
    this.obstacles = [];
    this.elevators = [];
    this.ground = null;
    this.state = { roof: true, site: true, furniture: true, hiddenFloors: new Set(), cut: null };
    this.cap.visible = false;
  }

  /** 높이(m)에 있는 사람이 속한 층. 층 바닥이 height + FLOOR_SWITCH 이하인 층 가운데 가장 높은 층이다. */
  floorAt(height) {
    const keys = this.floors().filter((key) => this.levels.has(key))
      .sort((a, b) => this.levels.get(a) - this.levels.get(b));
    let found = keys[0] ?? null;
    for (const key of keys) if (this.levels.get(key) <= height + FLOOR_SWITCH) found = key;
    return found;
  }

  /**
   * 사람 시점의 위치와 보는 방향(미니맵용). 사람 시점이 아니면 null.
   * x·z는 모델 좌표(m), dirX·dirZ는 보는 방향(수평 단위 벡터), hfov는 수평 화각(rad), floor는 서 있는 층.
   */
  personPose() {
    if (!this.person) return null;
    const { x, y, z } = this.camera.position;
    const half = Math.atan(Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2) * this.camera.aspect);
    return {
      x, z, dirX: -Math.sin(this.person.yaw), dirZ: -Math.cos(this.person.yaw), hfov: half * 2,
      floor: this.floorAt(this.person.ground ?? y - EYE_HEIGHT),
    };
  }

  /**
   * 층 평면(미니맵용). 좌표는 [x, z] m이고 화면에서는 z가 아래쪽이다(도면과 같은 방향).
   * cuts: 바닥에서 CUT_HEIGHT 위를 수평으로 자른 선(벽·기둥·문·창·난간). 내부 보기와 같은 높이라 문 자리가 뚫려 보인다.
   * fills: 위에서 내려다본 삼각형(바닥판, 계단, 바닥 구멍으로 보이는 아래층 계단).
   * bounds: 모든 층을 담는 범위. 층이 바뀌어도 지도 축척과 위치가 같다.
   */
  floorPlan(key) {
    if (!this.model || !this.levels.has(key)) return null;
    this.model.updateMatrixWorld(true);
    const keys = this.floors();
    const box = new THREE.Box3();
    for (const k of keys) box.expandByObject(this.groups.get(k));
    const helpers = new Set(this.stencils);
    const meshesOf = (floor, suffix) => {
      const found = [];
      this.groups.get(`${floor}_${suffix}`)?.traverse((obj) => { if (obj.isMesh && !helpers.has(obj)) found.push(obj); });
      return found;
    };
    const height = this.levels.get(key) + CUT_HEIGHT;
    const cut = (suffix) => meshesOf(key, suffix).flatMap((mesh) => stitchLoops(sectionSegments(mesh, height)));
    const top = (floor, suffix) => meshesOf(floor, suffix).flatMap((mesh) => topTriangles(mesh));
    const below = keys[keys.indexOf(key) - 1];
    return {
      bounds: { minX: box.min.x, maxX: box.max.x, minZ: box.min.z, maxZ: box.max.z },
      fills: { stairBelow: below ? top(below, 'Stairs') : [], slab: top(key, 'Slabs'), stair: top(key, 'Stairs') },
      cuts: { rail: cut('Rails'), window: cut('Windows'), door: cut('Doors'), wall: cut('Walls'), column: cut('Columns') },
    };
  }

  floors() {
    return [...this.groups.keys()]
      .filter((name) => FLOOR.test(name))
      .sort((a, b) => level(a) - level(b));
  }

  /** 'arch'(건축) 또는 'class'(분류 색). */
  setStyle(style) {
    this.style = style;
    for (const { mesh } of this.meshes) mesh.material = mesh.userData.mats[style];
    this.hemi.intensity = LIGHTS[style].hemi;
    this.sun.intensity = LIGHTS[style].sun;
  }

  setFloorVisible(key, visible) {
    if (visible) this.state.hiddenFloors.delete(key);
    else this.state.hiddenFloors.add(key);
    this.applyCut();
  }

  setGroupVisible(name, visible) {
    const group = this.groups.get(name);
    if (!group) return;
    group.visible = visible;
    this.applyCut();
  }

  setRoofVisible(visible) {
    this.state.roof = visible;
    this.applyVisibility();
  }

  setSiteVisible(visible) {
    this.state.site = visible;
    this.applyVisibility();
  }

  /**
   * 도면의 가구 선을 층 바닥 위에 그린다. data는 furniture.json({floors: {F1: {z, segments}}}, mm·Z 위)이다.
   * DAE와 같은 좌표라 모델 안에 넣으면 단위와 축이 함께 맞는다. 층 묶음 아래에 두어 층 표시를 따른다.
   */
  setFurniture(data) {
    const old = new Set(this.furniture);
    for (const obj of this.furniture) {
      obj.removeFromParent();
      obj.geometry.dispose();
    }
    this.meshes = this.meshes.filter(({ mesh }) => !old.has(mesh));
    this.obstacles = this.obstacles.filter((mesh) => !old.has(mesh));
    this.furniture = [];
    if (!this.model || !data?.floors) return;
    const lineMaterial = new THREE.LineBasicMaterial({ color: FURNITURE_COLOR });
    this.clipped.add(lineMaterial);
    const solidMaterials = Object.fromEntries(Object.entries(FURNITURE_SOLIDS).map(([kind, color]) => {
      const m = prepare(new THREE.MeshStandardMaterial({ color, roughness: kind === 'fixture' ? 0.35 : 0.85 }));
      m.name = `furniture_${kind}`;
      this.clipped.add(m);
      return [kind, m];
    }));
    for (const [key, floor] of Object.entries(data.floors)) {
      const group = this.groups.get(key);
      if (!group) continue;
      const parts = [];
      if (floor.segments?.length) {
        const z = floor.z + FURNITURE_LIFT_MM;
        const positions = new Float32Array(floor.segments.length * 6);
        floor.segments.forEach(([x1, y1, x2, y2], i) => positions.set([x1, y1, z, x2, y2, z], i * 6));
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const lines = new THREE.LineSegments(geometry, lineMaterial);
        lines.name = `${key}_Furniture`;
        lines.raycast = () => {};
        parts.push(lines);
      }
      // 같은 종류는 한 덩어리로 합친다. 가구가 수백 개라 따로 그리면 느려진다.
      const byKind = new Map();
      for (const solid of floor.solids || []) {
        if (!(solid.k in FURNITURE_SOLIDS) || solid.p.length < 3 || solid.z1 <= solid.z0) continue;
        const shape = new THREE.Shape(solid.p.map(([x, y]) => new THREE.Vector2(x, y)));
        const geometry = new THREE.ExtrudeGeometry(shape, { depth: solid.z1 - solid.z0, bevelEnabled: false, curveSegments: 1 });
        geometry.translate(0, 0, floor.z + solid.z0);
        if (!byKind.has(solid.k)) byKind.set(solid.k, []);
        byKind.get(solid.k).push(geometry);
      }
      for (const [kind, geometries] of byKind) {
        const mesh = new THREE.Mesh(mergeGeometries(geometries), solidMaterials[kind]);
        mesh.name = `${key}_Furniture_${kind}`;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData.mats = { arch: mesh.material, class: mesh.material };
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 25), this.edgeMaterials.building);
        edges.raycast = () => {};
        mesh.add(edges);
        this.meshes.push({ mesh, building: true }); // 단면 칠하기·스타일 바꾸기를 같이 받는다
        if (FURNITURE_BLOCKING.has(kind)) this.obstacles.push(mesh);
        parts.push(mesh);
      }
      for (const obj of parts) {
        this.model.add(obj);
        obj.updateMatrixWorld(true);
        group.attach(obj); // 모델 좌표 그대로 층 묶음 아래로 옮긴다
        this.furniture.push(obj);
      }
    }
    for (const m of [lineMaterial, ...Object.values(solidMaterials)]) {
      m.clippingPlanes = this.activeCut() ? [this.cutPlane] : null;
    }
    this.buildElevators(data.elevators || []);
    this.applyCut();
  }

  /**
   * 엘리베이터. data는 furniture.json의 elevators({p, floors, levels, doors}, mm·Z 위)다.
   * 카(바닥·천장·세 벽)는 모델 안에서 Z로 오르내리고, 층마다 문 두 짝이 옆으로 열린다.
   * 사람 시점에서 문 앞에 서면 카가 오고, 카 안에서 숫자키(층 번호)나 PageUp/PageDown으로 층을 고른다.
   */
  buildElevators(list) {
    for (const e of this.elevators) {
      e.car.removeFromParent();
      for (const d of Object.values(e.doors)) d.leaves.forEach((leaf) => leaf.removeFromParent());
    }
    const removed = new Set(this.elevators.flatMap((e) => e.parts));
    this.meshes = this.meshes.filter(({ mesh }) => !removed.has(mesh));
    this.obstacles = this.obstacles.filter((m) => !removed.has(m));
    this.walkable = this.walkable.filter((m) => !removed.has(m));
    this.elevators = [];
    if (!this.model) return;
    const metal = prepare(new THREE.MeshStandardMaterial({ color: ELEVATOR_COLOR, metalness: 0.55, roughness: 0.35 }));
    metal.name = 'elevator';
    this.clipped.add(metal);
    const register = (mesh, { blocking = false, walk = false } = {}) => {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData.mats = { arch: metal, class: metal };
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 25), this.edgeMaterials.building);
      edges.raycast = () => {};
      mesh.add(edges);
      this.meshes.push({ mesh, building: true });
      if (blocking) this.obstacles.push(mesh);
      if (walk) this.walkable.push(mesh);
    };
    for (const spec of list) {
      if (!spec.floors?.length || spec.p?.length < 3) continue;
      const points = spec.p.map(([x, y]) => new THREE.Vector2(x, y));
      const first = spec.floors[0];
      const doorSegment = spec.doors?.[first] || Object.values(spec.doors || {})[0];
      const front = doorSegment ? nearestEdge(points, doorSegment) : -1;
      const parts = [];
      const car = new THREE.Group();
      car.name = 'Elevator';
      const plate = (z, depth) => {
        const g = new THREE.ExtrudeGeometry(new THREE.Shape(points), { depth, bevelEnabled: false });
        g.translate(0, 0, z);
        return new THREE.Mesh(g, metal);
      };
      const floorPlate = plate(0, ELEVATOR_PLATE_MM);
      const ceiling = plate(ELEVATOR_CAR_MM, ELEVATOR_PLATE_MM);
      car.add(floorPlate, ceiling);
      register(floorPlate, { walk: true });
      register(ceiling);
      parts.push(floorPlate, ceiling);
      const centroid = points.reduce((c, p) => c.add(p), new THREE.Vector2()).divideScalar(points.length);
      points.forEach((a, i) => {
        if (i === front) return; // 문 쪽은 비운다
        const b = points[(i + 1) % points.length];
        const inward = new THREE.Vector2(-(b.y - a.y), b.x - a.x).normalize();
        if (inward.dot(centroid.clone().sub(a)) < 0) inward.negate();
        const t = inward.multiplyScalar(ELEVATOR_WALL_MM);
        const shape = new THREE.Shape([a, b, b.clone().add(t), a.clone().add(t)]);
        const wall = new THREE.Mesh(new THREE.ExtrudeGeometry(shape, { depth: ELEVATOR_CAR_MM, bevelEnabled: false }), metal);
        car.add(wall);
        register(wall, { blocking: true });
        parts.push(wall);
      });
      car.position.z = spec.levels[first];
      this.model.add(car);
      const doors = {};
      for (const key of spec.floors) {
        const seg = spec.doors?.[key];
        const group = this.groups.get(key);
        if (!seg || !group) continue;
        const a = new THREE.Vector2(seg[0], seg[1]);
        const b = new THREE.Vector2(seg[2], seg[3]);
        const dir = b.clone().sub(a);
        const width = dir.length();
        dir.normalize();
        const half = width / 2;
        // 승강장 문은 벽의 승강로 쪽에 달려 열리면 벽 뒤로 숨는다. 벽 중심선에서 승강로 쪽으로 옮긴다.
        const shaftSide = new THREE.Vector2(-dir.y, dir.x);
        if (shaftSide.dot(centroid.clone().sub(a)) < 0) shaftSide.negate();
        const leaves = [0, 1].map((k) => {
          const g = new THREE.BoxGeometry(half, ELEVATOR_LEAF_MM, ELEVATOR_DOOR_MM);
          const leaf = new THREE.Mesh(g, metal);
          leaf.rotation.z = Math.atan2(dir.y, dir.x);
          const center = a.clone().addScaledVector(dir, half / 2 + k * half).addScaledVector(shaftSide, ELEVATOR_LEAF_SET_MM);
          leaf.position.set(center.x, center.y, spec.levels[key] + ELEVATOR_DOOR_MM / 2);
          leaf.userData.closed = leaf.position.clone();
          this.model.add(leaf);
          leaf.updateMatrixWorld(true);
          group.attach(leaf);
          register(leaf, { blocking: true });
          parts.push(leaf);
          return leaf;
        });
        doors[key] = { leaves, dir, half, open: 0, center: a.clone().add(b).multiplyScalar(0.5), level: spec.levels[key] };
      }
      this.elevators.push({
        spec, car, parts, doors, polygon: points, floors: spec.floors, levels: spec.levels,
        z: spec.levels[first], target: first,
      });
    }
  }

  /** 사람이 이 엘리베이터 카 안에 있는가. */
  insideCar(e) {
    if (!this.person) return false;
    const p = new THREE.Vector2(this.camera.position.x * 1000, -this.camera.position.z * 1000);
    const feet = (this.camera.position.y - EYE_HEIGHT) * 1000;
    return pointInPolygon(p, e.polygon) && Math.abs(feet - (e.z + ELEVATOR_PLATE_MM)) < 600;
  }

  /** 카 안에서 누른 키. 숫자는 층 번호, PageUp/PageDown은 한 층 위·아래. */
  pressElevator(code) {
    const e = this.elevators.find((el) => this.insideCar(el));
    if (!e) return false;
    const index = e.floors.indexOf(e.target);
    let next = null;
    if (code === 'PageUp') next = e.floors[Math.min(index + 1, e.floors.length - 1)];
    else if (code === 'PageDown') next = e.floors[Math.max(index - 1, 0)];
    else if (/^Digit\d$/.test(code)) next = e.floors.find((key) => key === `F${code.slice(5)}`) || null;
    if (!next) return false;
    e.target = next;
    return true;
  }

  updateElevators(dt) {
    if (!dt || !this.elevators.length) return;
    let hint = null;
    for (const e of this.elevators) {
      const inside = this.insideCar(e);
      // 문 앞에 서면 카를 부른다.
      if (this.person && !inside) {
        for (const [key, d] of Object.entries(e.doors)) {
          const feet = (this.camera.position.y - EYE_HEIGHT) * 1000;
          const here = new THREE.Vector2(this.camera.position.x * 1000, -this.camera.position.z * 1000);
          if (here.distanceTo(d.center) < ELEVATOR_CALL_MM && Math.abs(feet - d.level) < 600) e.target = key;
        }
      }
      const targetZ = e.levels[e.target];
      const doorsClosed = Object.values(e.doors).every((d) => d.open <= 0.001);
      const moving = Math.abs(e.z - targetZ) > 1;
      if (moving && doorsClosed) {
        const step = Math.sign(targetZ - e.z) * Math.min(ELEVATOR_SPEED_MM * dt, Math.abs(targetZ - e.z));
        e.z += step;
        if (inside) {
          this.camera.position.y += step / 1000;
          this.person.ground = (e.z + ELEVATOR_PLATE_MM) / 1000;
        }
      }
      e.car.position.z = e.z;
      for (const [key, d] of Object.entries(e.doors)) {
        const here = new THREE.Vector2(this.camera.position.x * 1000, -this.camera.position.z * 1000);
        const near = this.person && (inside || here.distanceTo(d.center) < ELEVATOR_CALL_MM * 1.5);
        const want = !moving && key === e.target && near ? 1 : 0;
        d.open = THREE.MathUtils.clamp(d.open + Math.sign(want - d.open) * ELEVATOR_DOOR_SPEED * dt, 0, 1);
        d.leaves.forEach((leaf, k) => {
          leaf.position.copy(leaf.userData.closed);
          const shift = (k === 0 ? -1 : 1) * d.half * 0.95 * d.open;
          leaf.position.x += d.dir.x * shift;
          leaf.position.y += d.dir.y * shift;
        });
      }
      if (inside) {
        const labels = e.floors.map((key) => floorLabel(key)).join('·');
        hint = moving ? `엘리베이터 이동 중 → ${floorLabel(e.target)}` : `엘리베이터 · 숫자키로 층 선택 (${labels}) · PageUp/PageDown`;
      }
      // 내부 보기·층 숨기기를 따른다: 카는 지금 머문 층이 보일 때만 보인다.
      const nearest = e.floors.reduce((best, key) => (Math.abs(e.levels[key] - e.z) < Math.abs(e.levels[best] - e.z) ? key : best), e.floors[0]);
      e.car.visible = Boolean(this.groups.get(nearest)?.visible);
    }
    if (hint !== this.lastHint) {
      this.lastHint = hint;
      this.onPersonHint?.(hint);
    }
  }

  setFurnitureVisible(visible) {
    this.state.furniture = visible;
    this.applyVisibility();
  }

  hasFurniture() {
    return this.furniture.length > 0;
  }

  /** 내부 보기. key 층만 남기고 바닥에서 CUT_HEIGHT 위로 잘라 위에서 본다. null이면 끈다. */
  setCut(key) {
    this.state.cut = key && this.levels.has(key) ? key : null;
    this.applyCut();
    if (this.person) return; // 사람 시점에서는 고른 층만 기억해 두고, 나가면 적용된다
    if (this.state.cut) this.frameObject(this.groups.get(this.state.cut), CUT_ELEVATION, true);
    else this.frameObject(null, START_ELEVATION, true);
  }

  /** 지금 적용할 단면 층. 사람 시점에서는 천장까지 보이도록 끈다. */
  activeCut() {
    return this.person ? null : this.state.cut;
  }

  applyCut() {
    const cut = this.activeCut();
    if (cut) this.cutPlane.constant = this.levels.get(cut) + CUT_HEIGHT;
    for (const m of this.clipped) {
      m.clippingPlanes = cut ? [this.cutPlane] : null;
      m.needsUpdate = true;
    }
    this.applyVisibility();
    this.clearStencils();
    if (!cut) {
      this.cap.visible = false;
      return;
    }
    // 잘린 면 칠하기: 닫힌 입체의 뒷면은 +1, 앞면은 -1을 스텐실에 쌓는다.
    // 자른 면 위치에서 0이 아니면 입체 속이므로 그 자리에 단면 판을 그린다.
    for (const { mesh, building } of this.meshes) {
      if (!building || !isShownTree(mesh) || [mesh.material].flat().some((m) => m.opacity < 1)) continue;
      for (const [side, op] of [[THREE.BackSide, THREE.IncrementWrapStencilOp], [THREE.FrontSide, THREE.DecrementWrapStencilOp]]) {
        const helper = new THREE.Mesh(mesh.geometry, new THREE.MeshBasicMaterial({
          side, depthWrite: false, depthTest: false, colorWrite: false,
          stencilWrite: true, stencilFunc: THREE.AlwaysStencilFunc,
          stencilFail: op, stencilZFail: op, stencilZPass: op,
          clippingPlanes: [this.cutPlane],
        }));
        helper.renderOrder = 1;
        helper.raycast = () => {};
        mesh.add(helper);
        this.stencils.push(helper);
      }
    }
    const box = new THREE.Box3().setFromObject(this.groups.get(cut));
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    this.cap.scale.set(size.x + 2, 1, size.z + 2);
    this.cap.position.set(center.x, this.cutPlane.constant, center.z);
    this.cap.visible = true;
  }

  clearStencils() {
    for (const s of this.stencils) {
      s.removeFromParent();
      s.material.dispose();
    }
    this.stencils = [];
  }

  applyVisibility() {
    const cut = this.activeCut();
    for (const line of this.furniture) line.visible = this.state.furniture !== false;
    for (const [name, obj] of this.groups) {
      if (cut) {
        // 내부 보기는 그 층만 보여 준다. 다른 층과 대지는 숨긴다.
        if (FLOOR.test(name)) obj.visible = name === cut && !this.state.hiddenFloors.has(name);
        else if (name === 'SITE' || name.startsWith('SITE_')) obj.visible = false;
        else if (name.endsWith('_Roofs')) obj.visible = this.state.roof;
        continue;
      }
      if (FLOOR.test(name)) obj.visible = !this.state.hiddenFloors.has(name);
      else if (name === 'SITE') obj.visible = true;
      else if (name.endsWith('_Roofs')) obj.visible = this.state.roof;
      else if (ROOF_SITE.has(name)) obj.visible = this.state.roof;
      else if (name.startsWith('SITE_')) obj.visible = this.state.site;
    }
  }

  setAutoRotate(on) {
    this.controls.autoRotate = on;
  }

  flyTo(target, position) {
    this.fly = {
      start: null,
      fromTarget: this.controls.target.clone(), toTarget: target,
      fromPosition: this.camera.position.clone(), toPosition: position,
    };
  }

  stepFly(time) {
    const f = this.fly;
    f.start ??= time;
    const t = Math.min(1, (time - f.start) / FLY_MS);
    const k = t * t * (3 - 2 * t);
    this.controls.target.lerpVectors(f.fromTarget, f.toTarget, k);
    this.camera.position.lerpVectors(f.fromPosition, f.toPosition, k);
    if (t >= 1) this.fly = null;
  }

  /** obj(없으면 건물 전체)가 화면에 꽉 차게 지금 방위에서 elevation 높이로 본다. */
  frameObject(obj, elevation, animate) {
    if (!this.model) return;
    const box = new THREE.Box3();
    if (obj) box.setFromObject(obj);
    else for (const key of this.floors()) box.expandByObject(this.groups.get(key));
    if (box.isEmpty()) box.setFromObject(this.model);
    if (obj && this.state.cut) box.max.y = Math.min(box.max.y, this.cutPlane.constant);
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getBoundingSphere(new THREE.Sphere()).radius || 10;
    const verticalHalfFov = THREE.MathUtils.degToRad(FOV_OUTSIDE) / 2;
    const halfFov = Math.atan(Math.tan(verticalHalfFov) * Math.min(1, this.camera.aspect));
    const distance = radius / Math.sin(halfFov) * 1.05;
    const offset = this.camera.position.clone().sub(this.controls.target);
    const azimuth = offset.lengthSq() > 0 ? Math.atan2(offset.x, offset.z) : START_AZIMUTH;
    const dir = new THREE.Vector3(
      Math.cos(elevation) * Math.sin(azimuth), Math.sin(elevation), Math.cos(elevation) * Math.cos(azimuth),
    );
    const position = center.clone().addScaledVector(dir, distance);
    if (animate) this.flyTo(center, position);
    else {
      this.camera.position.copy(position);
      this.controls.target.copy(center);
    }
  }

  /** 건물(층 묶음)이 화면에 꽉 차게 남서쪽 위에서 본다. 대지가 넓어도 건물 기준으로 맞춘다. */
  fit() {
    if (!this.model) return;
    this.fly = null;
    if (this.person) this.leavePerson(); // 시점 초기화는 사람 시점도 끝낸다
    this.model.updateMatrixWorld(true);
    if (this.state.cut) {
      this.frameObject(this.groups.get(this.state.cut), CUT_ELEVATION, false);
      this.controls.update();
      return;
    }
    const box = new THREE.Box3();
    for (const key of this.floors()) box.expandByObject(this.groups.get(key));
    if (box.isEmpty()) box.setFromObject(this.model);
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getBoundingSphere(new THREE.Sphere()).radius || 10;
    this.radius = radius;
    const verticalHalfFov = THREE.MathUtils.degToRad(FOV_OUTSIDE) / 2;
    const halfFov = Math.atan(Math.tan(verticalHalfFov) * Math.min(1, this.camera.aspect));
    const distance = radius / Math.sin(halfFov) * 1.05;
    const dir = new THREE.Vector3(
      Math.cos(START_ELEVATION) * Math.sin(START_AZIMUTH),
      Math.sin(START_ELEVATION),
      Math.cos(START_ELEVATION) * Math.cos(START_AZIMUTH),
    );
    this.camera.position.copy(center).addScaledVector(dir, distance);
    this.farLimit = distance * 20;
    this.camera.fov = FOV_OUTSIDE;
    this.camera.near = Math.max(0.05, distance / 500);
    this.camera.far = this.farLimit;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.minDistance = 0.3; // 방 안까지 다가갈 수 있게
    this.controls.maxDistance = distance * 4;
    this.controls.update();

    // 그림자 범위를 건물 크기에 맞춘다.
    const s = radius * 2.5;
    Object.assign(this.sun.shadow.camera, { left: -s, right: s, top: s, bottom: -s, near: 0.1, far: s * 6 });
    this.sun.shadow.camera.updateProjectionMatrix();
    this.sun.position.copy(center).add(new THREE.Vector3(-0.6, 1.2, 0.8).multiplyScalar(radius * 2));
    this.sun.target.position.copy(center);
  }
}

/** 두 스타일이 함께 쓰는 설정: 투명 재질 깊이, 겹친 면 밀기, 원래 면 방향 기억. */
function prepare(m) {
  if (m.opacity < 1) m.depthWrite = false;
  // 모서리 선이 면에 묻히지 않도록 면을 조금 뒤로 민다. 벽은 기둥보다 더 민다.
  m.polygonOffset = true;
  m.polygonOffsetFactor = 1;
  m.polygonOffsetUnits = PUSHED_BACK.has(m.name) ? 4 : 1;
  return m;
}

function archMaterial(original) {
  const spec = ARCH[original.name];
  if (!spec) {
    const copy = original.clone();
    copy.name = original.name;
    return copy;
  }
  let m;
  if (spec.glass) {
    m = new THREE.MeshStandardMaterial({
      color: spec.color || '#a9cfe0', roughness: 0.08, metalness: 0.1, transparent: true,
      opacity: spec.opacity || 0.32, envMapIntensity: 1.6,
    });
    m.depthWrite = false;
  } else {
    m = new THREE.MeshStandardMaterial({
      color: spec.color, roughness: spec.metal ? 0.45 : 0.9, metalness: spec.metal ? 0.25 : 0,
    });
  }
  m.name = original.name;
  return m;
}

/** 점이 다각형 안에 있는가(짝홀 규칙). */
function pointInPolygon(p, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    if ((a.y > p.y) !== (b.y > p.y) && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

/** 문 선분([x1, y1, x2, y2])에 가장 가까운 다각형 변의 번호. 카의 문 쪽이다. */
function nearestEdge(points, segment) {
  const mid = new THREE.Vector2((segment[0] + segment[2]) / 2, (segment[1] + segment[3]) / 2);
  let best = -1;
  let bestDistance = Infinity;
  points.forEach((a, i) => {
    const b = points[(i + 1) % points.length];
    const ab = b.clone().sub(a);
    const t = THREE.MathUtils.clamp(mid.clone().sub(a).dot(ab) / ab.lengthSq(), 0, 1);
    const d = a.clone().addScaledVector(ab, t).distanceTo(mid);
    if (d < bestDistance) [best, bestDistance] = [i, d];
  });
  return best;
}

/** 인덱스 없는 BufferGeometry 여러 개를 하나로. 위치와 법선만 쓴다(three 기본 도구는 들여오지 않았다). */
function mergeGeometries(geometries) {
  const flat = geometries.map((g) => (g.index ? g.toNonIndexed() : g));
  const count = flat.reduce((n, g) => n + g.attributes.position.count, 0);
  const positions = new Float32Array(count * 3);
  const normals = new Float32Array(count * 3);
  let offset = 0;
  for (const g of flat) {
    if (!g.attributes.normal) g.computeVertexNormals();
    positions.set(g.attributes.position.array, offset * 3);
    normals.set(g.attributes.normal.array, offset * 3);
    offset += g.attributes.position.count;
    g.dispose();
  }
  const merged = new THREE.BufferGeometry();
  merged.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  merged.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  return merged;
}

function unwrap(list) {
  return list.length === 1 ? list[0] : list;
}

function isShownTree(obj) {
  for (let o = obj; o; o = o.parent) if (!o.visible) return false;
  return true;
}

/** 메시의 삼각형 꼭짓점을 세계 좌표로 차례로 넘긴다. */
function eachTriangle(mesh, visit) {
  const position = mesh.geometry.getAttribute('position');
  const index = mesh.geometry.getIndex();
  const count = index ? index.count : position.count;
  const v = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
  for (let i = 0; i + 2 < count; i += 3) {
    for (let k = 0; k < 3; k += 1) {
      v[k].fromBufferAttribute(position, index ? index.getX(i + k) : i + k).applyMatrix4(mesh.matrixWorld);
    }
    visit(v[0], v[1], v[2]);
  }
}

/** 메시를 높이 y의 수평면으로 자른 선분들. [[x, z], [x, z]] */
function sectionSegments(mesh, y) {
  const segments = [];
  eachTriangle(mesh, (a, b, c) => {
    const points = [];
    for (const [p, q] of [[a, b], [b, c], [c, a]]) {
      const dp = p.y - y;
      const dq = q.y - y;
      if ((dp < 0) === (dq < 0)) continue;
      const t = dp / (dp - dq);
      points.push([p.x + (q.x - p.x) * t, p.z + (q.z - p.z) * t]);
    }
    if (points.length === 2) segments.push(points);
  });
  return segments;
}

/** 선분을 끝점(1mm 단위)끼리 이어 닫힌 선으로 만든다. 닫히지 않은 것은 closed가 false다. */
function stitchLoops(segments) {
  const keyOf = ([x, z]) => `${Math.round(x * 1000)},${Math.round(z * 1000)}`;
  const ends = new Map();
  segments.forEach((segment, i) => segment.forEach((point) => {
    const key = keyOf(point);
    if (!ends.has(key)) ends.set(key, []);
    ends.get(key).push(i);
  }));
  const used = new Uint8Array(segments.length);
  const loops = [];
  for (let i = 0; i < segments.length; i += 1) {
    if (used[i]) continue;
    used[i] = 1;
    const points = [segments[i][0], segments[i][1]];
    let closed = false;
    for (;;) {
      const tail = keyOf(points[points.length - 1]);
      const next = ends.get(tail).find((j) => !used[j]);
      if (next === undefined) {
        closed = points.length > 2 && keyOf(points[0]) === tail;
        break;
      }
      used[next] = 1;
      const [a, b] = segments[next];
      points.push(keyOf(a) === tail ? b : a);
    }
    loops.push({ points, closed });
  }
  return loops;
}

/** 위에서 내려다본 삼각형들. 한쪽을 향한 면만 고른다(윗면과 밑면이 겹치면 채우기가 서로 지워진다). [[x, z], [x, z], [x, z]] */
function topTriangles(mesh) {
  const triangles = [];
  eachTriangle(mesh, (a, b, c) => {
    const area = (b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z);
    if (area > 1e-6) triangles.push([[a.x, a.z], [b.x, b.z], [c.x, c.z]]);
  });
  return triangles;
}

function hasAncestor(obj, test) {
  for (let o = obj; o; o = o.parent) if (o.name && test(o.name)) return true;
  return false;
}

/** 층 묶음이나 지붕 위 대지 요소 아래에 있는가. 단면으로 자를 대상이다. */
function inBuilding(obj) {
  for (let o = obj; o; o = o.parent) {
    if (FLOOR.test(o.name) || FLOOR_PART.test(o.name) || ROOF_SITE.has(o.name)) return true;
  }
  return false;
}

/** 위는 맑은 하늘색, 아래는 밝은 회색인 배경. Revit 3D 뷰의 그라데이션 배경을 닮게 했다. */
function skyTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 2;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  const g = ctx.createLinearGradient(0, 0, 0, 256);
  g.addColorStop(0, '#c9d6e3');
  g.addColorStop(0.55, '#eef1f4');
  g.addColorStop(1, '#f7f7f5');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 2, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function level(key) {
  if (key === 'R') return 1e6;
  const m = FLOOR.exec(key);
  return (m[1] === 'B' ? -Number(m[2]) : Number(m[2])) + (m[3] ? 0.5 : 0);
}
