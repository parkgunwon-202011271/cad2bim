# cad2bim

건축 CAD 평면(DWG 폴더)을 층이 나뉜 3D 모델(SketchUp, IFC)로 바꿉니다. 비공개 저장소입니다.

설계와 규칙은 [상세설계.md](상세설계.md), 시험 데이터는 [data/public/README.md](data/public/README.md)를 보세요.

## 다른 컴퓨터에서 준비할 것

저장소에 들어 있지 않아 따로 준비해야 하는 항목입니다.

| 항목 | 필요한 때 | 없으면 |
|---|---|---|
| Python 3.11 이상과 패키지 | 항상 | 실행 불가 |
| LibreDWG | DWG 입력 | DWG를 못 읽음. DXF는 가능 |
| SketchUp C SDK | skp 출력 | skp만 빠지고 나머지는 출력 |
| 렌더링 패키지 | 조감도 그림 | 조감도만 건너뜀 |
| 실사화 환경 | 조감도 실사화 | 실사화만 건너뜀 |

조감도와 실사화는 지금 꺼져 있습니다(`aerial_enabled`, `photo_enabled`). 앞 단계 품질을 먼저 올리기로 했기 때문이며, 평소 작업에는 위 세 항목만 있으면 됩니다.

### 1. 기본 설치

```powershell
git clone git@github.com:parkgunwon-202011271/cad2bim.git
cd cad2bim\core
pip install -e ".[dev]"
python -m pytest -q
```

SDK가 없으면 skp 테스트는 건너뜁니다.

### 2. LibreDWG

Windows용 LibreDWG를 받아 `C:\libredwg`에 풉니다. 다른 위치에 두려면 `dwg2dxf`를 PATH에 넣거나 환경 변수 `LIBREDWG_DWG2DXF`에 실행 파일 경로를 지정합니다.

ODA File Converter는 비회원이면 비상업 용도만 허용되므로 쓰지 않습니다.

### 3. SketchUp C SDK

Trimble 개발자 사이트에서 약관에 동의하고 받아 `C:\SDK_Win`에 풉니다. 다른 위치라면 환경 변수 `CAD2BIM_SKETCHUP_SDK`에 지정합니다.

- SDK는 재배포가 금지되어 있어 저장소에 넣지 않습니다.
- Trimble의 서면 승인 전에는 skp 출력을 외부 고객에게 제공하지 않습니다.

### 4. 조감도 그림 (선택)

```powershell
pip install -e ".[render]"
```

`config/defaults.yaml`에서 `aerial_enabled`를 `true`로 바꿉니다.

### 5. 실사화 (선택)

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
