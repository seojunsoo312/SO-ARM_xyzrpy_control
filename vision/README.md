# vision

고정 Orbbec RGB(-D)와 eye-to-hand. 손-눈 캡처는 `handeye/capture.py`가
펜던트가 켜 둔 토크/FK(`motion.pose_server`)만 읽는다. 계산은 `handeye/compute.py`.
컬러 `K`는 ChArUco가 아니라 **SDK 공장값**이다. FK 프레임은 **tcp**.

```
Project/
  vision/
    camera.py              RGB (V4L2 또는 SDK) 640×480
    rgbd.py                Orbbec SDK D2C + 공장 K 읽기
    calib.py               JSON 입출력
    transforms.py          invert_T, rt_to_T, unproject
    handeye/
      vision_test.py       RGB 미리보기
      get_intrinsic.py     SDK → calib_data/intrinsics.json
      charuco.py           캘리브 보드 검출
      capture.py           펜던트 TCP + ChArUco → png/json
      compute.py           저장 샘플 → T_base_cam (--align-desk)
      compare_pnp_depth.py PnP Z vs 뎁스 Z
    calib_data/
      intrinsics.json      공장 K (카메라 바꿀 때 get_intrinsic.py)
      eye_to_hand.json     SO-ARM 손-눈
      handeye_tcp/         캡처 샘플
  yolo/
  motion/
  pendant/
```

## 실행

프로젝트 루트에서. Viewer는 끄고, `ORBBEC_SDK_DIR` 은 SDK 폴더.
캡처는 펜던트 ON(Connect·토크). 계산만이면 카메라·펜던트 없이 된다.

```bash
python vision/handeye/vision_test.py
python vision/handeye/get_intrinsic.py
python vision/handeye/capture.py
python vision/handeye/compute.py
python vision/handeye/compute.py --align-desk
python vision/handeye/compare_pnp_depth.py
```
