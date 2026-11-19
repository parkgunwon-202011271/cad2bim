# cad2bim

건축 CAD 평면(DWG 폴더)을 층이 나뉜 3D 모델(SketchUp, IFC)로 바꿉니다. 비공개 저장소입니다.

설계와 규칙은 [상세설계.md](상세설계.md), 시험 데이터는 [data/public/README.md](data/public/README.md)를 보세요.

현재 확인한 기능과 미검증 항목은 [기능 체크리스트](docs/CHECKLIST.md), 다음 작업과 완료 기준은 [TODO](docs/TODO.md)를 보세요.

## 다른 컴퓨터에서 설치

Windows 11과 Python 3.11 이상만 있으면 됩니다. 패키지 버전은 `requirements.txt`에 고정되어 있고, DWG를 읽는 LibreDWG는 `third_party/libredwg`에 함께 들어 있습니다.

```powershell
git clone git@github.com:parkgunwon-202011271/cad2bim.git
cd cad2bim
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\verify.py     # 주민센터를 변환해 기준값과 비교. 마지막 줄이 "재현 성공"이면 끝
python -m app                # 앱 실행
```

| 항목 | 저장소에 있나 | 없으면 |
|---|---|---|
| Python 패키지 | `requirements.txt`로 설치 | 실행 불가 |
| LibreDWG 0.14.8829 | 있음 (`third_party/libredwg`) | — |
| SketchUp C SDK | 없음 (재배포 금지, 선택 설치 1번) | skp만 빠지고 나머지는 출력 |
| 렌더링 패키지 | 없음 (선택 설치 2번) | 조감도만 건너뜀 |
| 실사화 환경 | 없음 (선택 설치 3번) | 실사화만 건너뜀 |

조감도와 실사화는 지금 꺼져 있습니다(`aerial_enabled`, `photo_enabled`). 앞 단계 품질을 먼저 올리기로 했기 때문입니다.

테스트: `cd core` 후 `python -m pytest -q`. SDK가 없으면 skp 테스트는 건너뜁니다. 지금 `test_slab_covers_the_floor_below_the_level` 1건이 부동소수점 비교(-199.99999999999972 ≠ -200.0)로 실패합니다. 앱과는 무관합니다.

ODA File Converter는 비회원이면 비상업 용도만 허용되므로 쓰지 않습니다.

## 앱 (프로토타입)

`python -m app`으로 창이 열립니다. DWG 폴더나 DWG·DXF 파일 여러 개를 창에 끌어다 놓으면 변환하고, 3D 모델을 360도로 돌려 볼 수 있습니다. 끌어다 놓기가 안 되면 "도면 폴더 열기"를 씁니다.

- 원칙: 도면을 그대로 옮긴다. 층 도면끼리 맞지 않는 곳(예: 1층 B안의 계단·벽이 2층 보이드와 어긋남)은 고치지 않고 그대로 둔다.
- 3D: W·A·S·D 이동, Q·E 아래·위, Shift 빠르게. 마우스는 드래그 회전, 휠 확대(커서 쪽으로), 오른쪽 드래그 옮기기. 층·지붕·대지 표시를 켜고 끕니다.
- 화각: 밖에서는 40°, 바닥에서 3m 안으로 내려오면 70°로 넓어집니다.
- 사람 시점("보기"의 버튼): 보던 자리의 바닥에 눈높이 1.6m로 섭니다. W·A·S·D 걷기, Shift 뛰기, 드래그로 둘러보기, 휠로 시야 조절, Esc로 나가기. 벽·창·난간은 막고 문은 지나가며, 0.45m 이하 턱과 계단은 오릅니다. 내부 보기(단면)는 잠시 꺼져 천장까지 보이고, 나오면 들어가기 전 시점으로 돌아갑니다.
- 가구: 도면의 가구·위생기구 레이어(FURNITURE)를 입체와 바닥 선으로 보여 줍니다("표시"에서 끄고 켬). 높이는 가까운 이름표(정수기·키오스크·소변기 등, `config/defaults.yaml`의 `furniture_labels`)로 정하고, 없으면 크기로 짐작합니다. build가 `furniture.json`으로 내보내며 모델 파일(DAE·IFC·SKP)에는 넣지 않습니다.
- 엘리베이터: 승강로 표시와 'EV'·'엘리베이터' 글자로 찾습니다. 위아래 층이 같은 자리면 바닥을 뚫어 승강로로 잇고(모델 파일에도 반영), 앱에서는 카와 승강장 문을 만듭니다. 사람 시점에서 문 앞에 서면 카가 오고, 카 안에서 숫자키(층 번호)나 PageUp/PageDown으로 층을 고릅니다. 층마다 자리가 다르면 잇지 않습니다.
- 보기 스타일: "건축"(기본, Revit의 음영+모서리처럼 재료색과 모서리 선)과 "분류 색"(기둥 파랑·문 초록 등 인식 확인용). 모델 파일 색은 바꾸지 않습니다.
- 내부 보기: 층을 고르면 그 층만 남기고 바닥 1.25m 높이에서 잘라 위에서 봅니다(난간 1.2m와 겹치지 않게). 잘린 면은 진하게 칠합니다.
- 2D: 입력 도면을 그림으로 봅니다. 숫자만 8자리 이상인 글자(학번 등)는 그리지 않습니다.
- 폴더를 놓으면 결과는 그 폴더의 `_cad2bim_out`에, 파일을 놓으면 `%LOCALAPPDATA%\cad2bim\jobs\`에 생깁니다.
- 외부 통신이 없습니다. three.js 0.170.0(MIT)도 `app/web/vendor`에 들어 있습니다.

### 화면 구조와 Figma 반영

화면은 빌드 과정이 없는 HTML·CSS·JS입니다(`app/web`). Node는 필요 없습니다.

| 파일 | 내용 | Figma에서 바꾸면 |
|---|---|---|
| `css/tokens.css` | 색·글꼴·간격·모서리. Figma Variables와 이름이 1:1 (`color/bg/surface` → `--color-bg-surface`) | 값만 고침 |
| `css/app.css` | 레이아웃(flex만 사용 = Auto Layout)과 컴포넌트 모양 | 고침 |
| `js/ui/components.js`, `js/ui/plan-viewer.js` | 컴포넌트(Button, Toolbar, Panel, ToggleList, Tabs, ProgressPanel, StatList, DropOverlay, Toast, PlanViewer). 이름이 Figma 컴포넌트와 같음 | 고침 |
| `js/main.js` | 화면 조립과 동작 | 배치가 바뀔 때만 |
| `js/viewer.js` | three.js 3D 뷰어 | 고치지 않음 |
| `js/api.js` | Python 연결 창구 | 고치지 않음 |

Python 없이 화면만 볼 때(디자인 작업용):

```powershell
cd app\web
python -m http.server 8000
# 브라우저에서 http://localhost:8000/?sample  (샘플 3D), ?sample&tab=2d (샘플 2D)
```

브라우저에서는 주민센터 샘플(`app/web/sample`, `python -m app.sample`로 다시 만듦)로 같은 흐름을 흉내 냅니다.

## 선택 설치

### 1. SketchUp C SDK

Trimble 개발자 사이트에서 약관에 동의하고 받아 `C:\SDK_Win`에 풉니다. 다른 위치라면 환경 변수 `CAD2BIM_SKETCHUP_SDK`에 지정합니다.

- SDK는 재배포가 금지되어 있어 저장소에 넣지 않습니다.
- Trimble의 서면 승인 전에는 skp 출력을 외부 고객에게 제공하지 않습니다.

### 2. 조감도 그림

```powershell
pip install -e ".[render]"
```

`config/defaults.yaml`에서 `aerial_enabled`를 `true`로 바꿉니다.

### 3. 실사화

NVIDIA 그래픽 카드(메모리 16GB에서 확인)와 저장 공간 약 12GB가 필요합니다. 모든 계산은 PC 안에서 끝나고 실행 중 인터넷을 쓰지 않습니다.

**기본 환경에 `diffusers`를 설치하지 마세요.** `huggingface_hub`가 함께 올라가 `transformers`를 쓰는 다른 프로그램이 깨집니다. 반드시 별도 가상환경에 설치합니다.

```powershell
python -m venv --system-site-packages $HOME\.cad2bim_ai
& $HOME\.cad2bim_ai\Scripts\python.exe -m pip install "diffusers==0.42.1" transformers accelerate safetensors
```

CUDA를 쓰는 PyTorch가 기본 환경에 있어야 합니다. 이어서 같은 파이썬으로 아래 모델을 내려받습니다.

| 모델 | 라이선스 |
|---|---|
| `SG161222/RealVisXL_V4.0` | OpenRAIL++ |
| `madebyollin/sdxl-vae-fp16-fix` | MIT |
| `diffusers/controlnet-depth-sdxl-1.0` | OpenRAIL++ |
| `diffusers/controlnet-canny-sdxl-1.0` | OpenRAIL++ |

다 받으면 `config/defaults.yaml`에서 `photo_enabled`를 `true`로 바꿉니다.

## 실행

입력 폴더는 `data/` 안이거나 프로젝트 밖이어야 합니다. 프로젝트의 다른 곳은 막아 두었습니다. `data/`에는 공개 도면과 AI 인식이 승인된 도면만 둡니다.

```powershell
cd core
python -m cad2bim.cli build ..\data\public\주민센터 --jobs 3
python -m cad2bim.cli compare-model ..\data\public\주민센터 "..\data\public\주민센터\주민센터 모델링.skp" --floor-height 4500
```

결과는 입력 폴더의 `_cad2bim_out`에 생깁니다. 현업에 전달할 파일은 `building_editable.skp`입니다. 설치가 맞다면 마지막 줄의 F1이 약 0.826으로 나옵니다.

## 지켜야 할 것

- 물류 도면과 출처·라이선스가 불분명한 외부 도면은 저장소에 올리지 않습니다. 로컬 `data/` 아래(`data/물류_B동/`, `data/public/외부도면/`)에 두면 `.gitignore`가 막습니다.
- 변환 결과(`_cad2bim_out`, `결과`)는 올리지 않습니다.
- `.gitignore`는 합성 도면과 `data/public/`(외부도면 제외)만 예외로 허용합니다. 다른 곳의 도면은 커밋되지 않습니다.
- 시험용 도면은 결과를 보고 규칙을 고치는 데 쓰지 않습니다.
