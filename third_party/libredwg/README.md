# LibreDWG (함께 넣은 고정 버전)

DWG를 DXF로 바꾸는 `dwg2dxf.exe`와 실행에 필요한 DLL만 넣었습니다. 다른 PC에서 따로 받지 않아도 되고, PC마다 변환 결과가 같게 하기 위해서입니다.

| 항목 | 값 |
|---|---|
| 버전 | 0.14.8829 (2026-11-04 릴리스) |
| 받은 곳 | https://github.com/LibreDWG/libredwg/releases/tag/0.14.8829 (`libredwg-0.14.8829-win64.zip`) |
| 압축 파일 SHA-256 | `419dd2581dbbb927505b71909b01c8c977f7d9afdd5a6188b7c51260ab7f28cc` (릴리스의 `dist.sha256`에는 win64 파일이 없어 받은 파일로 직접 계산) |
| 확인 | 메인 PC에 쓰던 `C:\libredwg`의 `dwg2dxf.exe`, `libredwg-0.dll`과 바이트 단위로 같음 |
| 넣은 파일 | `dwg2dxf.exe`, `libredwg-0.dll`, `libiconv-2.dll` (압축 파일 안의 것 그대로) |
| 라이선스 | GPL-3.0-or-later, 원문은 `COPYING` |
| 소스 코드 | 같은 릴리스의 `libredwg-0.14.8829.tar.xz` |

- 코어(`core/src/cad2bim/io/dwg.py`)는 이 폴더를 가장 먼저 찾습니다. 환경 변수 `LIBREDWG_DWG2DXF`가 있으면 그쪽이 우선입니다.
- 별도 프로세스로 호출하므로 cad2bim 코드와 링크되지 않습니다. 외부에 배포할 때는 이 폴더와 `COPYING`, 소스 위치 안내를 함께 둡니다.
- 버전을 바꿀 때는 `scripts/verify.py`로 결과가 달라지는지 먼저 확인합니다.
