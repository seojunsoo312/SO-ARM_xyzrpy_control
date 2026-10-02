# cad

집는 브라켓의 모양과 집기 숫자가 있는 폴더입니다. 

## 실행

이 폴더를 직접 실행하지는 않습니다. STL 축만 보려면 프로젝트 폴더에서 아래를 켭니다. 화면의 축은 빨강 X, 초록 Y, 파랑 Z입니다.

```bash
python yolo/pose/view/view_cad.py
```

## 파일


| 파일                 | 하는 일                                                   |
| ------------------ | ------------------------------------------------------ |
| `model.yaml`       | 집기 정보. STL 이름, 축을 맞추는 회전, 화면 위치, 집는 점, 접근 거리, 내려놓는 XY. |
| `model.py`         | `model.yaml`과 STL을 읽는 코드. 다른 폴더는 yaml을 직접 읽지 않고 이 파일을 씁니다. |
| `bracket_4035.stl` | 브라켓 모양. 등록할 때 이 형상과 카메라 점군을 맞춥니다.                      |


`model.yaml`의 각 숫자는 파일 안 주석에 적혀 있습니다. 길이는 mm, 각도는 deg입니다. 축은 +X 앞, +Y 왼쪽, +Z 위입니다.

## 같이 쓰는 곳

- 형상 맞추기는 `yolo/pose/register.py`, 카메라로 위치를 구하는 창은 `yolo/pose/roi_cloud.py`입니다.
- 펜던트 물건 칸(`pendant/teach_grasp.py`), 접근 자세 계산(`motion/grasp.py`), 픽 좌표(`motion/pick_targets.py`)가 `model.py`로 집기 점, 접근 거리, 내려놓는 XY를 읽습니다.

