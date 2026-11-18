// 360도 3D 뷰어. building.dae(층 → 종류 묶음)를 읽어 돌려 보고, 층·지붕·대지를 켜고 끈다.
// 마우스: 왼쪽 드래그 회전, 휠 확대, 오른쪽 드래그 이동.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';

const FLOOR = /^(B|F)(\d+)$/;
// 미리보기(core/tools/preview.py)에서 지붕 위로 보는 대지 요소와 같다.
const ROOF_SITE = new Set([
  'SITE_Lawn', 'SITE_RoofPlanters', 'SITE_Pergola', 'SITE_Skylights', 'SITE_SkylightFrames',
  'SITE_Equipment', 'SITE_Benches', 'SITE_Parapets', 'SITE_Penthouses',
]);
// 첫 시점: 남서쪽 위. 미리보기의 첫 그림과 같은 방향이다.
const START_AZIMUTH = THREE.MathUtils.degToRad(-55);
const START_ELEVATION = THREE.MathUtils.degToRad(30);

export function floorLabel(key) {
  const m = FLOOR.exec(key);
  if (!m) return key;
  return m[1] === 'B' ? `지하${m[2]}층` : `${m[2]}층`;
}

export class Viewer3D {
  constructor(container) {
    this.container = container;
    const style = getComputedStyle(container);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.append(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(style.getPropertyValue('--color-bg-viewer').trim() || '#eceef1');
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotateSpeed = 1.2;
    this.controls.maxPolarAngle = THREE.MathUtils.degToRad(89);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa0a8, 1.6));
    this.sun = new THREE.DirectionalLight(0xffffff, 2.2);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0005;
    this.scene.add(this.sun, this.sun.target);

    this.model = null;
    this.groups = new Map();
    this.state = { roof: true, site: true, hiddenFloors: new Set() };

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
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
    if (this.model) this.scene.remove(this.model);
    // ColladaLoader가 단위(mm → m)와 Z-위 축을 맞춰 준다.
    const collada = new ColladaLoader().parse(daeText, '');
    this.model = collada.scene;
    this.groups.clear();
    this.model.traverse((obj) => {
      if (obj.name) this.groups.set(obj.name, obj);
      if (obj.isMesh) {
        obj.castShadow = true;
        obj.receiveShadow = true;
        for (const m of [obj.material].flat()) {
          if (m && m.opacity < 1) m.depthWrite = false;
        }
      }
    });
    this.scene.add(this.model);
    this.state = { roof: true, site: true, hiddenFloors: new Set() };
    this.fit();
    return this.floors();
  }

  floors() {
    return [...this.groups.keys()]
      .filter((name) => FLOOR.test(name))
      .sort((a, b) => level(a) - level(b));
  }

  setFloorVisible(key, visible) {
    if (visible) this.state.hiddenFloors.delete(key);
    else this.state.hiddenFloors.add(key);
    this.applyVisibility();
  }

  setRoofVisible(visible) {
    this.state.roof = visible;
    this.applyVisibility();
  }

  setSiteVisible(visible) {
    this.state.site = visible;
    this.applyVisibility();
  }

  applyVisibility() {
    for (const [name, obj] of this.groups) {
      if (FLOOR.test(name)) obj.visible = !this.state.hiddenFloors.has(name);
      else if (name.endsWith('_Roofs')) obj.visible = this.state.roof;
      else if (ROOF_SITE.has(name)) obj.visible = this.state.roof;
      else if (name.startsWith('SITE_')) obj.visible = this.state.site;
    }
  }

  setAutoRotate(on) {
    this.controls.autoRotate = on;
  }

  /** 건물(층 묶음)이 화면에 꽉 차게 남서쪽 위에서 본다. 대지가 넓어도 건물 기준으로 맞춘다. */
  fit() {
    if (!this.model) return;
    this.model.updateMatrixWorld(true);
    const box = new THREE.Box3();
    for (const key of this.floors()) box.expandByObject(this.groups.get(key));
    if (box.isEmpty()) box.setFromObject(this.model);
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getBoundingSphere(new THREE.Sphere()).radius || 10;
    const distance = radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov) / 2) * 1.05;
    const dir = new THREE.Vector3(
      Math.cos(START_ELEVATION) * Math.sin(START_AZIMUTH),
      Math.sin(START_ELEVATION),
      Math.cos(START_ELEVATION) * Math.cos(START_AZIMUTH),
    );
    this.camera.position.copy(center).addScaledVector(dir, distance);
    this.camera.near = Math.max(0.05, distance / 500);
    this.camera.far = distance * 20;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.minDistance = radius * 0.2;
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

function level(key) {
  const m = FLOOR.exec(key);
  return m[1] === 'B' ? -Number(m[2]) : Number(m[2]);
}
