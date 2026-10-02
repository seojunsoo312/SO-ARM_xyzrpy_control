# motion

팔이 어떻게 움직이는지 계산하고, 그 팔을 브라우저 3D로 보여 주는 폴더입니다. 버튼 창과 카메라 화면은 여기 없습니다.

## 실행

```python
from motion import Arm, load_pick_targets
```

학생 코드는 보통 `Arm`만 부릅니다. 조그 창도 이 폴더의 제어 루프를 사용합니다.

## 파일

| 파일 | 하는 일 |
|---|---|
| `arm.py` | 학생용 명령. `moveL`, `moveJ`, `grip`, `home`, `initial`. 끝나면 그 줄에서 기다립니다. |
| `controller.py` | 조그와 이동을 반복하는 루프. 펜던트 버튼이 여기에 명령을 넣습니다. |
| `robot_kinematics.py` | URDF로 팔 끝 위치를 구하고, 목표 위치로 관절 각을 계산합니다. |
| `base_frame.py` | 사람이 쓰는 좌표(+X 앞, +Y 왼쪽, +Z 위)와 URDF 좌표를 바꿉니다. |
| `hw_controller.py` | 실기 모터 연결. 이 PC의 `lerobot-calibrate` 캘리브를 읽습니다. |
| `pick_place.py` | 집어서 다른 자리에 놓는 순서. 펜던트 픽 버튼이 이 단계를 실행합니다. |
| `grasp.py` | 물체 자세에서 대기·집기 TCP를 계산합니다. 물체 축/베이스 RPY 변환, 랜덤 배치, 물체-팔 겹침도 여기 있습니다. 창은 없습니다. |
| `pick_targets.py` | 카메라에 물건 등록을 요청하고, 대기·집기·놓기 좌표를 만듭니다. |
| `visualizer.py` | 브라우저 3D(Meshcat). 지금 팔 자세를 보여 줍니다. `Arm`과 펜던트가 같이 씁니다. |
| `robot/SO101_6DOF.urdf` | 팔 설계도. 관절 이름, 길이, 어떤 부품 모양을 쓸지. |
| `robot/meshes/` | 3D에 그리는 부품 파일(STL). URDF가 가리킵니다. |
| `pose_server.py` | 지금 팔 끝 자세를 `http://127.0.0.1:8765/pose`로 내보냅니다. 손눈 촬영이 읽습니다. |
| `__init__.py` | `from motion import ...` 할 때 꺼내는 목록입니다. |

## 같이 쓰는 곳

- 조그 창과 물건 위치 칸은 `pendant/`에 있습니다. 이 폴더의 제어 루프와 3D 화면을 씁니다.
- `load_pick_targets()`는 `yolo/pose/roi_cloud.py`가 켜져 있어야 좌표를 돌려줍니다. 요청과 응답은 `yolo/pose/register_link.py`로 주고받습니다. 접근 자세 계산은 `grasp.py`입니다.
- 집기 순서의 자세한 단계는 `docs/pick_place.md`, 학생용 명령은 `docs/교육용_API_명세.md`입니다.
