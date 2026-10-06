# vision

책상에 고정된 Orbbec 카메라와, 카메라가 로봇 기준 어디에 있는지를 구하는 폴더입니다. 팔을 움직이지는 않습니다.

## 실행

카메라를 쓰려면 Orbbec SDK(`OrbbecViewer_*` 또는 `OrbbecSDK_*` 폴더)가 프로젝트 폴더와 같은 곳에 있어야 합니다. 예를 들어 프로젝트가 `~/AIvision`이면 SDK는 `~/OrbbecViewer_...`입니다. 코드는 그 폴더 바로 아래나 `lib/` 아래의 `libOrbbecSDK.so`를 씁니다. 노트북은 x64판, Jetson은 ARM64판이어야 합니다(`docs/jetson_setup.md`). 이 폴더는 git에 없습니다. 다른 곳에 두려면 환경변수 `ORBBEC_SDK_DIR`에 그 경로를 적습니다.

SDK가 없으면 카메라를 쓰는 명령(`yolo/pose/roi_cloud.py`, 촬영, 손눈 캘리브)이 「SDK 없음」으로 멈춥니다. 펜던트와 가상 `Arm()`은 SDK 없이도 됩니다.

프로젝트 폴더에서 실행합니다. 카메라가 필요한 명령은 Orbbec Viewer를 끈 상태에서 켭니다. 촬영은 펜던트를 먼저 Connect하고 토크를 켠 뒤 합니다. 계산만 할 때는 카메라와 펜던트가 없어도 됩니다.

```bash
python vision/handeye/vision_test.py       # 화면 확인
python vision/handeye/get_intrinsic.py     # 카메라 내부 값 저장
python vision/handeye/capture.py           # 손눈 샘플 저장
python vision/handeye/compute.py           # 카메라 위치 계산
python vision/handeye/compare_pnp_depth.py # 보드 거리와 뎁스 거리 비교
```

## 파일

| 파일 | 하는 일 |
|---|---|
| `camera.py` | 컬러 영상을 엽니다. 기본 640×480, 180° 회전. |
| `rgbd.py` | Orbbec SDK로 컬러와 뎁스를 같이 읽고, 공장 내부 파라미터를 가져옵니다. SDK 파일 로그는 꺼서 실행한 폴더에 `Log/`가 생기지 않습니다. 화면에는 에러만 나옵니다. |
| `orbbec_filters.py` | 뎁스 필터 창. 마지막 값은 `orbbec_filters.json`에 저장됩니다(git에 없음). 그 파일이 없으면 `orbbec_filters.default.json`에서 시작합니다. |
| `calib.py` | 캘리브 JSON을 읽고 씁니다. |
| `transforms.py` | 좌표 변환. 카메라 픽셀을 3D 점으로 바꿉니다. |
| `handeye/charuco.py` | 캘리브 보드를 사진에서 찾습니다. |
| `handeye/vision_test.py` | 컬러 화면만 띄웁니다. |
| `handeye/get_intrinsic.py` | 공장 내부 값을 `calib_data/intrinsics.json`에 저장합니다. |
| `handeye/capture.py` | 펜던트 팔 자세와 보드 사진을 `calib_data/handeye_tcp/`에 저장합니다. |
| `handeye/compute.py` | 저장한 샘플로 카메라의 로봇 기준 위치(`eye_to_hand.json`)를 계산합니다. |
| `handeye/compare_pnp_depth.py` | 보드로 잰 거리와 뎁스 거리가 같은지 확인합니다. |
| `calib_data/` | 내부 값, 손눈 결과, 촬영 샘플이 들어 있는 폴더입니다. 보드마다 다르므로 git에 없습니다. |

## 같이 쓰는 곳

- 촬영은 펜던트가 켜 둔 `http://127.0.0.1:8765/pose`에서 팔 끝 자세를 읽습니다. 이 스크립트는 로봇을 움직이지 않습니다.
- `yolo/`의 검출과 점군은 `camera.py`, `rgbd.py`, `calib_data/eye_to_hand.json`을 사용합니다.
