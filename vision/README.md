# vision

고정 Orbbec RGB(-D)와 eye-to-hand. 손-눈 캡처는 `hand_eye_calib.py`가
펜던트가 켜 둔 토크/FK(`motion.pose_server`)만 읽는다. 컬러 `K`는 ChArUco가 아니라
**SDK 공장값**이다. FK 프레임은 **tcp**.

```
Project/
  vision/
    camera.py          RGB (V4L2 또는 SDK) 640×480
    rgbd.py            Orbbec SDK D2C + 공장 K 읽기
    get_intrinsic.py   SDK → calib_data/intrinsics.json
    charuco.py         손-눈용 보드 검출
    transforms.py      invert_T, rt_to_T, unproject
    hand_eye_calib.py  캡처 + Park → T_base_cam
    calib.py           JSON 입출력
    vision_test.py     RGB 미리보기
    calib_data/
      intrinsics.json  공장 K (카메라 바꿀 때 get_intrinsic.py)
      eye_to_hand.json SO-ARM 손-눈
      handeye_tcp/     캡처 샘플
  yolo/
  motion/
  pendant/
```

## 실행

프로젝트 루트에서. Viewer·펜던트 GUI는 끄고, `ORBBEC_SDK_DIR` 은 SDK 폴더.

```bash
python vision/vision_test.py
python vision/get_intrinsic.py
python vision/hand_eye_calib.py
python vision/hand_eye_calib.py --compute-only
python yolo/detect.py
```
