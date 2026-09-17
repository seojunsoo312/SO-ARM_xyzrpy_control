# motion

팔 움직임만. GUI·YOLO·카메라 없음.

```text
yolo/  →  베이스 포즈
vision/  →  카메라, T_base_cam, apply_T
motion/  →  IK, 시리얼, go-to   ← 이 폴더
pendant/  →  조그 GUI. 여기서 motion 을 호출
```

| 파일 | 역할 |
|------|------|
| `robot_kinematics.py` | URDF FK/IK |
| `hw_controller.py` | Feetech / LeRobot 버스 |
| `controller.py` | 조그·go-to 루프 |
| `pick_place.py` | 책상 픽앤플레이스 시퀀스 (드롭 XY는 관절 보간, TCP 직선 아님) |
| `pose_server.py` | 손-눈용 TCP JSON (`GET /pose`) |

URDF·메시·캘리브 JSON 은 경로만 `pendant/` 에 남겨 두었다.

```python
from motion import Controller, Hardware, RobotKinematics, DEFAULT_URDF
```
