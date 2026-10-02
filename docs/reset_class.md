# reset_class.py

수업에서 만든 사진, 학습 결과, 카메라 캘리브를 지우고 처음부터 다시 하게 하는 스크립트입니다. 팔 코드나 창은 지우지 않습니다.

프로젝트 폴더에서 실행합니다. 지우기 전에 목록을 보여 주고, `yes`를 입력해야 지웁니다.

```bash
python reset_class.py --yolo
python reset_class.py --calib
python reset_class.py --ply
python reset_class.py --all
python reset_class.py --all -y    # 물어보지 않고 바로 삭제
```

옵션은 같이 쓸 수 있습니다. 예를 들어 `--yolo --calib`는 둘 다 지웁니다. `--all`은 세 가지를 한 번에 합니다.

## 무엇을 지우나

### `--yolo`

찍은 사진과 학습 결과입니다.


| 지우는 곳                                         | 내용                      |
| --------------------------------------------- | ----------------------- |
| `yolo/datasets/raw/images/`                   | 찍은 사진                   |
| `yolo/datasets/raw/labels/`                   | 박스 라벨                   |
| `yolo/datasets/splits/`                       | 학습/검증 목록                |
| `yolo/datasets/custom/`                       | 예전 학습용 폴더               |
| `yolo/runs/`                                  | 학습 그래프, 저장한 점군, 등록 JSON |
| `yolo/weights/best.pt`                        | 우리가 학습한 가중치             |


학습에 필요한 코드, CAD, 시작 가중치 `yolo11n.pt`는 남습니다.



### `--calib`

카메라 캘리브 결과입니다.


| 지우는 곳                                | 내용                    |
| ------------------------------------ | --------------------- |
| `vision/calib_data/handeye_tcp/`     | 손눈 캘리브 때 저장한 사진과 JSON |
| `vision/calib_data/intrinsics.json`  | 카메라 내부 파라미터           |
| `vision/calib_data/eye_to_hand.json` | 카메라와 로봇 사이 위치         |


카메라 코드 자체는 남습니다. 지운 뒤에는 내부 파라미터, 촬영, 손눈 계산을 다시 합니다.

```bash
python vision/handeye/get_intrinsic.py
python vision/handeye/capture.py
python vision/handeye/compute.py
```



### `--ply`

프로젝트 안의 `.ply` 점군 파일을 모두 지웁니다. 