# CAD 모델

등록(`register.py`)이 맞출 형상. 파일 축이 \(xyzrpy\) 의 0도다.

바꿀 때는 **파일만 이 폴더에 두고** `model.yaml` 의 `mesh` 를 고친다. 코드에 부품 이름을 넣지 않는다.

```yaml
mesh: 지금쓰는파일.stl
unit: mm
class: bracket
```

`class` 는 YOLO 한 클래스 이름이다. 캡처 화면에 이렇게 보인다. CAD 축과 맞추는 작업이 아니다.

단위가 mm인지 내보내기 설정을 확인한다. m 이면 스케일이 틀린다.

파일에 들어 있는 축(원점·XYZ)은 등록 없이 이렇게 본다.

```bash
python yolo/view_cad.py
python yolo/view_cad.py bracket_4035.stl
```

빨강=X 초록=Y 파랑=Z. `roi_cloud.py --cad` 축이 이것과 같아야 한다.
