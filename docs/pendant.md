# pendant

로봇 팔을 손으로 움직이는 창입니다. 버튼을 누르면 팔이 따라가고, 브라우저에 같은 팔이 3D로 보입니다. 모터 계산, 3D 화면, 팔 설계도(URDF·메시)는 `motion/`에 있습니다.

## 실행

```bash
python pendant/main.py           # 조그 창
python pendant/main.py --grasp   # 조그 창 + 물건 위치 칸
```

## 파일

| 파일 | 하는 일 |
|---|---|
| `main.py` | 프로그램을 켭니다. 조그 창과 3D를 같이 띄웁니다. |
| `gui_app.py` | 조그 창. 방향 버튼, 관절, 좌표, 실기 연결(Connect). |
| `teach_grasp.py` | 물건 위치 칸. 어디 두고 어떻게 집을지 정하고, 픽 버튼을 누릅니다. 계산은 `motion/grasp.py`에 맡깁니다. |
| `ui_style.py` | 창 글꼴. conda에서 한글이 깨지지 않게 Noto를 씁니다. |

## 같이 쓰는 곳

- 3D 화면은 `motion/visualizer.py`, 팔 설계도와 부품 STL은 `motion/robot/`입니다.
- 실기 Connect는 이 PC에서 `lerobot-calibrate`로 만든 팔로워 캘리브를 읽습니다. 그 JSON은 이 폴더에 없습니다.
- 물건 인식은 `yolo/pose/roi_cloud.py`입니다. 물건 위치 칸은 그 결과를 받아 옵니다.
- 패키지 목록은 `docs/requirements.txt`, 설계 메모는 `docs/pendant_spec.md`입니다.
