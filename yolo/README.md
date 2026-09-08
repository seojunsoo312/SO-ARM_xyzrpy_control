# YOLO — 1클래스 커스텀 검출

작업대 위 **한 종류** 물건을 찍고, 박스를 그린 뒤, YOLO11n을 파인튜닝해서 Orbbec 화면에서 잡는다.

COCO에 없는 물건이 맞다. 그래서 실시간 인식은 **우리가 학습한 `weights/best.pt`** 만 쓴다. `yolo11n.pt` 는 학습 시작 가중치(전이학습)일 뿐이고, COCO 클래스로는 우리 물건을 찾지 않는다.

카메라는 `vision/camera.py` 와 같다. **640×480**, 기본 **180° 회전**. 펜던트·캘리브 스크립트와 **동시에 켜지 말 것**.

## 환경 (한 번)

YOLO·점군·펜던트는 **가상환경 하나**면 된다. 이름은 배포 때 바꿔도 되고, 코드는 환경 이름을 읽지 않는다. 지금 이 보드 실습은 `lerobot`. `setup.sh` 는 나중에 맞춘다.

시스템 `/bin/python` 은 쓰지 않는다 (비전 캘리브 전용).

```bash
# 프로젝트 루트에서
conda activate lerobot
```

학습 전에 펜던트·브라우저를 끄자. Orin Nano 메모리는 8GB다.

## 클래스 이름

기본 이름은 `target` 이다. 바꾸려면 `yolo/config.py` 의 `CLASS_NAME` 만 고친다. 라벨 txt는 클래스 인덱스 **0** 이라, `prepare.py` 전에 이름을 바꿔도 된다.

## 순서

프로젝트 루트에서, `conda activate lerobot` 한 채로:

```bash
python yolo/capture.py     # SPACE=저장  q=종료
python yolo/label.py       # 드래그=박스  n=다음  p=이전  d=마지막 삭제  r=비우기  q=종료
python yolo/prepare.py     # train/val 8:2, data.yaml
python yolo/train.py       # OOM 나면 --batch 2
python yolo/detect.py      # 실시간. q=종료
python yolo/roi_cloud.py --mask   # 인스턴스 점군 → 최상단 (지금은 로컬 면 디버그)
```

세그(마스크, 나중에 yaw)가 필요하면 박스 라벨 다음에:

```bash
python yolo/auto_seg.py      # 박스 → MobileSAM 마스크 → labels_seg
python yolo/review_seg.py    # 확인. d=마지막 삭제  r=비우기
python yolo/prepare.py --seg
python yolo/train.py --seg   # yolo11n-seg → weights/best-seg.pt
python yolo/detect.py --seg  # 마스크 + 마스크 중심
```

빈 책상은 박스 없이 두면 마스크도 안 만든다. SAM이 틀린 장만 `review_seg`에서 지운다. 전 장 폴리곤을 손으로 칠 필요는 없다.

점군 ROI는 **박스**가 기본이다 (체크리스트). 세그 마스크로 책상 점을 더 빼고 싶으면:

```bash
python yolo/roi_cloud.py --mask
python yolo/roi_cloud.py --base    # T_base_cam 있으면 베이스 mm
```

`roi_cloud.py`는 V4L2가 아니라 Orbbec SDK다. Viewer·펜던트·`detect.py`와 같이 켜지 말 것. PLY는 `yolo/runs/roi/`.

`roi_cloud.py` 처리 순서:

```
YOLO 박스 (쌓임은 --mask)
→ 전체 화면에서 RANSAC 책상 평면 추정
→ 검출마다 책상보다 위인 점군만 (합치지 않음)
→ 높이 점수로 최상단 1개
→ (현재) 윗면 슬라이스 → 로컬 면 PCA (디버그)
→ (예정) 그 ROI vs CAD → FPFH+RANSAC → Point-to-Plane ICP → xyzrpy
→ s 키로 선택 인스턴스 ply
```

6D 정본은 CAD 축 기준 \(xyzrpy\) 다. 설계는 `docs/grasp_pose_pipeline.md` 0절 수정안. CAD ply 는 `yolo/cad/` (아직 없음). 로컬 면은 등록 전까지 화면 확인용이다.

- `r`: 책상 평면 다시 추정
- `v`: 선택 인스턴스를 Open3D로 보기 (`view_cloud.py`)
- `s`: 선택 인스턴스 ply 저장
- `--slice-mm 6`: 윗면 슬라이스 두께
- `--plane-mm 6`: 책상에서 6mm 이내 제거 (얇은 물체는 낮출 것)
- `--no-plane`: 비교용으로 책상 제거 끄기
- `--base`: `vision/calib_data/eye_to_hand.json` 의 `T_base_cam` 으로 베이스 mm

학습 `data.yaml` 의 `path` 는 yaml 기준 상대경로 `datasets/custom` 이다. 머신 절대경로는 넣지 않는다.

GPU·카메라만 먼저 보고 싶으면 (우리 물건은 안 잡힘):

```bash
python yolo/detect.py --pretrained
```

## 찍을 때

- 카메라·프레임은 고정. 물건만 움직인다.
- **80장 이상**. 위치, 회전, 거리, 조명, 손·그리퍼 가림, **빈 책상**(박스 없이 저장).
- 한 화면에 같은 물건이 여러 개면 박스도 여러 개.

## 폴더

```
yolo/
├── README.md
├── setup.sh              # conda에 Jetson torch + ultralytics
├── requirements.txt
├── config.py             # 클래스 이름, 경로, batch
├── boxes.py              # xyxy ↔ YOLO 정규화 박스
├── capture.py            # 촬영 → datasets/raw/images
├── label.py              # 박스 → datasets/raw/labels
├── auto_seg.py           # 박스 → MobileSAM 마스크
├── review_seg.py         # 마스크 확인·삭제
├── prepare.py            # train/val 8:2, data.yaml  (--seg 면 폴리곤)
├── train.py              # YOLO11n / --seg 면 yolo11n-seg
├── detect.py             # 실시간. --seg 면 마스크 중심
├── depth_cloud.py        # 뎁스 → 점군, 책상 RANSAC
├── grasp_pose.py         # 인스턴스 점군 + 최상단 (+ 로컬 면 디버그)
├── register.py           # CAD vs ROI → FPFH+ICP (뼈대)
├── cad/                  # model.yaml + stl/ply (mm)
├── roi_cloud.py          # 카메라 루프. 최상단 오버레이
├── view_cloud.py         # Open3D 점군 뷰어
├── view_cad.py           # yolo/cad STL 파일 축(XYZ)
├── data.yaml             # path: datasets/custom (prepare가 씀)
├── datasets/
│   ├── raw/
│   │   ├── images/
│   │   ├── labels/
│   │   └── labels_seg/   # 폴리곤. 박스 라벨은 그대로 둠
│   └── custom/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── weights/
│   ├── best.pt           # detect 학습 결과
│   └── best-seg.pt       # seg 학습 결과
└── runs/
    ├── train/
    ├── train_seg/
    └── roi/              # roi_cloud.py 가 저장한 ply
```

지금 안 하는 것: 집기(`pick.py`), 장면 전체 FPFH, YOLOv8-Pose+PnP, 펜던트에 YOLO 넣기.  
`register.py` 는 함수 뼈대만. CAD 파일명은 `yolo/cad/model.yaml` 의 `mesh`.
