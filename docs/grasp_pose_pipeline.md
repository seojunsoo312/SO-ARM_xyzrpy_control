# 집기 포즈 파이프라인 설계

인식·점군 이후의 **인스턴스 6D 포즈 → 집기** 설계다.  
코드와 이 문서는 프로젝트 루트 (`vision/`, `yolo/`, `motion/`, `pendant/`) 한곳에 둔다.

대상: Jetson Orin Nano Super, MyCobot 280, 프레임 고정 Orbbec RGB-D (eye-to-hand).

작업 분해는 `[grasp_pose_dev_plan.md](./grasp_pose_dev_plan.md)`.

---

## 0. 수정안 (정본 변경)

이전 정본은 YOLO ROI 점군의 **로컬 윗면 + 2D PCA** 였다. CAD는 면이 실패할 때만 쓰기로 되어 있었다.

**바꾼다.** 물체 xyzrpy 의 기준(reference)은 CAD 축이다. 학습 6D(DenseFusion 등)는 쓰지 않고, Open3D **FPFH + RANSAC → Point-to-Plane ICP** 로 CAD와 ROI 점군을 맞춘다.


|                  | 이전                        | 수정안                             |
| ---------------- | ------------------------- | ------------------------------- |
| xyzrpy reference | 보이는 윗면 법선 + 장축 (CAD 축 아님) | **CAD 모델 프레임**                  |
| 초기 포즈            | (없음 / PCA)                | CAD vs ROI 점군 **FPFH + RANSAC** |
| 정밀 포즈            | 로컬 면 PCA                  | **Point-to-Plane ICP**          |
| 로컬 면 PCA         | 정본                        | 오버레이·디버그. 등록 실패 시 임시 폴백         |
| 장면 전체 FPFH       | 금지                        | **금지 유지**                       |
| YOLO ROI         | 인스턴스 점군                   | **유지.** 등록 입력은 이 구름만            |


CAD가 포즈 숫자를 넣어 주는 것이 아니다. CAD는 회전 0인 형상·축이고, FPFH+RANSAC이 [R \mid t] 를 **계산**한다.

앞단은 그대로다. YOLO로 인스턴스를 자르고, 책상 RANSAC은 바닥 제거만, `T_base_cam` 으로 베이스 mm.

YOLOv8-Pose + PnP 는 폐기한다. 모서리 키포인트 라벨·별도 학습을 하지 않는다. 초기식도 FPFH+RANSAC 만 쓴다.

---



## 1. 무엇을 푸는가

최종 목표는 작업대 위 물체의 **CAD 기준 6D** 를 구해 집어 옮기는 것이다. 장면은 아래로 넓힌다.


| 장면      | 예          | 이 설계에서             |
| ------- | ---------- | ------------------ |
| 단층 1개   | 책상 위 부품 하나 | 같은 파이프라인의 가장 쉬운 검증 |
| 단층 소량   | 떨어진 2–3개   | 인스턴스 선택이 맞는지 확인    |
| 단층 여러 개 | 안 겹침       | 작업공간·집기 반복         |
| 쌓임      | 랜덤 더미      | 최상단만 집고 반복         |


**설계 가정은 처음부터 쌓임이 가능하다.**  
**검증 장면만** 하나 → 두세 개 → 여러 개 → 쌓임 순으로 올린다.  
단층 전용 코드를 만들었다가 더미에서 갈아엎지 않는다.

---



## 2. 핵심 가정



### 버린다

- 모든 물체가 책상 면에 붙어 있다.
- roll / pitch 는 항상 0 이고, z 는 `z_desk + 물체높이` 로 고정한다.
- 픽셀 마스크 PCA 또는 박스 중심이 집기 xy 의 정본이다.
- 로컬 윗면 PCA 를 CAD 프레임 6D 의 정본으로 쓴다.
- **장면 전체**에 FPFH+RANSAC 을 돌려 물체를 찾는다.
- DenseFusion, GDR-Net, DOPE, CosyPose.
- **YOLOv8-Pose + PnP** (키포인트 라벨·학습). 폐기. 초기식 교체로도 쓰지 않는다.

책상은 물체 자세의 정답 평면이 아니다. 바닥 점을 지울 때만 쓴다.

### 남긴다

- 책상 RANSAC = 바닥 제거, 작업 공간 높이 범위.
- 카메라 프레임 고정. `p_base = T_base_cam * p_cam`. `apply_T` 는 `vision.transforms` 만.
- 프로젝트 베이스 = +X 전진, +Z 위, +Y 왼쪽 (`motion/base_frame.py`, URDF `Rz(-90°)`). 조그·place·손눈·`--base` 숫자가 이 프레임.
- 인식은 `yolo/`, 조그는 펜던트. 한 창에 넣지 않는다.
- 집기는 pre-grasp (목표보다 Z+40–50 mm) 후 하강.
- 등록 입력은 **선택한 인스턴스 점군만**.



### CAD = xyzrpy 의 reference

reference 없이 “얼마나 돌아갔는지”는 정의되지 않는다.

- CAD 점군: 모델 프레임( `model.yaml` 의 `mesh_rpy` 적용 후), 자세는 항등 (축이 yaw=0).
- ROI 점군: 카메라에 보이는 현재 형상.
- FPFH+RANSAC 의 [R \mid t] = CAD 축을 관측에 겹치려면 필요한 변환.
- xyzrpy 는 그 R, t 를 베이스(또는 카메라)에서 푼 값.

CAD 파일 안에 포즈가 들어 있는 것이 아니다.

---



## 3. 파이프라인

```text
RGB-D (Orbbec SDK, 640×480, 컬러·뎁스 동일 180° 회전)
        │
        ├─ YOLO-seg ──────────── 인스턴스 마스크 (박스는 백업)
        ├─ 전장면 책상 RANSAC ── 바닥 제거용 평면만
        ├─ CAD 메시 → mm 점군 (밀도·법선 장면과 맞춤, 한 번 로드)
        │
        └─ 인스턴스 i = 1..N
              마스크 ∩ (책상보다 위 Δh)
              → 카메라 mm 점군
              → (선택) T_base_cam → 베이스 mm
              → 높이 점수로 최상단 1개 (또는 클릭)
                    │
                    ▼
              정본: 그 ROI 점군 vs CAD
                    voxel 다운샘플 + 법선
                    FPFH
                    RANSAC global registration  →  T_init
                    Point-to-Plane ICP          →  T
                    T → xyzrpy (베이스 mm)
                    │
                    ├─ 디버그: 로컬 윗면 PCA (정본 아님)
                    └─ 디버그: 2D 마스크 PCA (정본 아님)
                    │
                    ▼
              작업공간 안이면 pre-grasp → 집기 → 상승
              더미면 같은 루프를 다음 최상단에 반복
```

막은 것: 책상 전체 점군에 CAD/FPFH 를 던지는 것. ROI가 섞이면 틀린 면에도 inlier 가 나온다.

### 3.1 센서·캘리브

- RGB-D 는 Orbbec SDK. V4L2 컬러만으로는 점군이 안 된다.
- `K`: `vision/calib_data/intrinsics.json` (640×480).
- `T_base_cam`: `vision/calib_data/eye_to_hand.json`.
- 카메라·펜던트·Viewer·`detect.py` 는 동시에 켜지 않는다. 집기 때만 **한 프로세스**에서 시리얼+카메라.



### 3.2 인스턴스와 바닥

- 쌓임·겹침이면 **세그 마스크가 기본**. 박스는 책상·옆 물체를 많이 담는다.
- `plane_foreground_mask` 는 책상보다 수 mm 위만. 물체 옆면은 이 단계에서 남는다.
- 인스턴스마다 점군을 따로. 전 검출 concatenate 한 OBB 는 쓰지 않는다.



### 3.3 최상단 선택

한 프레임 등록·집기 후보는 **하나**.

- 베이스 Z 또는 책상 위 높이 95%.
- 점 개수 하한.
- (선택) 클릭으로 인덱스 고정.

더미에서 먼저 깨지면 포즈가 아니라 **마스크·점 섞임**을 본다. 섞인 구름에 FPFH 를 돌리지 않는다.

### 3.4 정본 포즈: CAD vs ROI (Open3D)

학습 없음. conda Open3D 로 점군 연산. 뷰어 GUI 는 기존처럼 `/usr/bin/Open3D` (`view_cloud.py`). pip 뷰어는 SIGSEGV.

순서:

1. CAD → 모델 점군. 단위 **mm**, voxel·법선 방향을 장면과 같게.
2. 선택 ROI 점군도 같은 voxel.
3. 양쪽 법선, FPFH.
4. RANSAC 기반 global registration → T_\text{init}.
5. Point-to-Plane ICP → T.
6. T 를 카메라 프레임에서 구했으면 `T_base_cad = T_base_cam @ T_cam_cad`.
7. t → xyz mm, R → rpy deg.

튜닝이 핵심이다. voxel, FPFH 반경, RANSAC 거리, 법선 방향(뒤집힘). “함수 몇 줄”은 데모이고, 평면이 넓은 부품은 180° 뒤집힘을 화면에서 확인한다.

### 3.5 로컬 윗면 (디버그·폴백)

이미 `yolo.pose.debug.local_plane.local_pose` 에 있다. 등록이 실패하거나 CAD가 잠깐 없을 때 화면에만 쓴다. 로봇 명령의 정본이 아니다.

### 3.6 2D 경로 (삭제)

`2D_pose.ipynb` 는 지웠다. 마스크 PCA + 광선 교차는 쓰지 않는다.

### 3.7 집기

- 타깃 xyzrpy 는 등록된 T (베이스 mm).
- CAD 에 파지 포즈를 정해 두면 `T_base_grasp = T_base_cad @ T_cad_grasp`. 없으면 CAD 원점+축을 그대로 집기 프레임으로 쓴다 (부품에 따라 오프셋 필요).
- 작업공간 밖이면 스킵.
- pre-grasp → 하강 → 집기 → 상승 → 두기.
- 더미는 한 개 집고 루프. 다음 프레임에서 최상단을 다시 고른다.

---



## 4. 모듈과 파일

인식·점군·등록은 `yolo/` 에 둔다. 학습 스크립트는 `yolo/train/`, 점군·등록은 `yolo/pose/`.  
실행 진입은 `yolo/pose/roi_cloud.py` (확인) 와 나중의 `pick.py` (동작).

```text
Project/
├── docs/
│   ├── grasp_pose_pipeline.md    # 이 파일
│   └── grasp_pose_dev_plan.md
├── vision/
│   └── calib_data/               # K, T_base_cam
├── motion/
├── pendant/
│   └── teach_grasp.py            # Meshcat 집기 티칭 (`main.py --grasp`)
└── yolo/
    ├── train/                    # 촬영·라벨·학습·검출
    ├── pose/
    │   ├── depth_cloud.py        # 뎁스→점, 책상 RANSAC
    │   ├── instances.py          # 인스턴스 점군 + 최상단
    │   ├── register.py           # CAD 로드, FPFH+RANSAC, ICP, xyzrpy
    │   ├── roi_cloud.py          # 카메라 루프, 오버레이, ply
    │   ├── debug/local_plane.py  # 로컬 면 PCA (디버그)
    │   └── view/                 # Open3D 점군·CAD 축 뷰어
    ├── cad/                      # model.yaml + mesh (mm)
    └── runs/roi/
```

`K` / `T_base_cam` 은 `vision.calib`. 점 변환은 `vision.transforms.apply_T` 만.

### 4.1 파일별 책임


| 파일                     | 책임                          | 하지 않는 일            |
| ---------------------- | --------------------------- | ------------------ |
| `vision/calib.py`      | `K`, `T_base_cam` 로드        | 점군 수식              |
| `vision/transforms.py` | `apply_T`                   | 파일 I/O             |
| `yolo/pose/depth_cloud.py` | 뎁스→점, 책상 RANSAC, 마스크        | YOLO, `apply_T` 복제 |
| `yolo/pose/instances.py`  | 인스턴스 점군, 최상단 선택           | 카메라, 시리얼, 포즈 수식     |
| `yolo/pose/debug/local_plane.py` | (디버그) 로컬 면 PCA            | 로봇 명령                 |
| `yolo/pose/register.py`    | CAD vs ROI → T, xyzrpy      | 검출, 전 장면 FPFH      |
| `yolo/pose/roi_cloud.py`   | 루프, 오버레이, ply, `v`/`s`/`r`  | 학습, 펜던트            |
| `yolo/pose/view/`          | PLY / CAD 축 보기               | 6D 정본                 |
| `pendant/teach_grasp.py`   | Meshcat 집기 티칭               | YOLO, 카메라            |
| `pick.py`              | 등록 포즈 + `send_coords` (나중)  | 전용 GUI             |




### 4.2 지금 코드 vs 이 수정안

이미 있는 것: 인스턴스 점군, 최상단, 로컬 면 오버레이 (`instances.py`, `debug/local_plane.py`, `roi_cloud.py`).

아직 없는 것: `register.py`, `yolo/cad/` 모델, `roi_cloud` 에 T / xyzrpy 표시.

로컬 면 코드는 지우지 않는다. 정본만 등록으로 바꾼다.

### 4.3 실행 (구현 후)

프로젝트 루트, `conda activate lerobot`:

```bash
python yolo/train/detect.py --seg
python yolo/pose/roi_cloud.py --mask              # ROI + 최상단 (지금은 로컬 면)
python yolo/pose/roi_cloud.py --mask --base       # 베이스 mm
# python yolo/pose/roi_cloud.py --mask --cad    # 등록 정본. mesh 는 yolo/cad/model.yaml
# python yolo/pick.py
```

---



## 5. 단계 표

앞단계(물건, `K`, `T_base_cam`, YOLO)는 그대로다.


| 단계  | 산출                              | 실행                            | 선행                |
| --- | ------------------------------- | ----------------------------- | ----------------- |
| 5b  | 인스턴스별 점군 (책상 점 제외)              | `roi_cloud.py --mask`         | `K`, 세그, SDK      |
| 5c  | 최상단(또는 클릭) 인스턴스                 | `instances.select_topmost`    | 5b                |
| 5d  | CAD vs ROI → T, xyzrpy          | `register.py`                 | 5c, CAD(mm)       |
| 5e  | (디버그) 로컬 면 PCA                 | `debug.local_plane.local_pose` | 5c                |
| 6   | 책상 `z_desk` (바닥·작업공간)           | 펜던트                           | TCP               |
| 7   | 집기 프레임 = T 또는 `T @ T_cad_grasp` | 오버레이                          | 5d                |
| 8   | 작업공간 필터                         | 코드                            | 7                 |
| 9   | pre-grasp 집기                    | `pick.py`                     | 7, 시리얼+카메라 한 프로세스 |
| 10  | 픽앤플레이스, 더미면 9 반복                | `pick.py`                     | 9                 |


---



## 6. 포즈 출력

인스턴스 하나, 베이스 mm, 오른손 좌표:

```text
T             : 4x4           # T_base_cad
xyz_mm        : [x, y, z]     # t
rpy_deg       : [r, p, y]     # R. CAD 축 기준
fitness       : float         # ICP / RANSAC inlier
n_points      : int
source        : "fpfh_icp"
```

폴백일 때만 `source="local_plane"` (면 중심 + 법선 + PCA yaw, CAD 축 아님).

집기 예: CAD 파지 포즈가 있으면 그것을 T 로 변환. 단층에서 윗면 집기면 rpy 가 대략 `[180, 0, yaw]` 근처로 떨어질 수 있으나, 그건 결과이지 가정이 아니다.

---



## 7. 검증

코드 경로는 같다. 장면만 바꾼다.

1. **책상 위 1개, 가운데** — CAD와 ROI가 겹쳐 보이는지, xyzrpy 가 손으로 돌린 방향과 맞는지.
2. **1개, 가장자리** — 옆면이 ROI에 들어와도 등록이 뒤집히지 않는지.
3. **떨어진 2–3개** — 최상단(또는 클릭)만 등록.
4. **1개 집기** — pre-grasp 후 집힘.
5. **겹침** — 마스크가 갈라지는지. 한 구름이면 세그부터.
6. **쌓임** — 최상단 ROI만 등록·집기. 아래가 섞이면 FPFH를 의심하기 전에 마스크를 본다.

---



## 8. 일부러 넣지 않는 것

- 장면 전체 FPFH / 전역 RANSAC 으로 물체 찾기
- DenseFusion 등 학습형 RGB-D 6D
- **YOLOv8-Pose + PnP** (폐기. 이미지마다 모서리 라벨, 세그와 별도 학습)
- 로컬 면 PCA 를 CAD 6D 대신 로봇에 보내기 (정본 변경 이후)
- 펜던트에 검출·점군 UI
- 파지점에 바로 `send_coords` (pre-grasp 없이)
- `/bin/python` 으로 YOLO·점군 (캘리브 전용). 점군·Open3D 연산은 `lerobot`

---



## 9. 구현 순서

상세는 `[grasp_pose_dev_plan.md](./grasp_pose_dev_plan.md)`.

1. CAD ply (mm) 경로 고정, 모델 점군 로드.
2. `register.py`: FPFH+RANSAC → ICP → xyzrpy. 오프라인 ROI ply 로 먼저.
3. `roi_cloud.py` 에 등록 오버레이. 로컬 면은 디버그로 유지.
4. 검증 1–3 (화면만).
5. `pick.py`.
6. 겹침·쌓임.

Open3D 점군 연산은 conda. 뷰어는 `/usr/bin/Open3D`.