# cpu-helmet-detection

GPU 없이 **일반 노트북 CPU**에서 실시간으로 동작하는 **안전모 착용 감지 AI** (진행 중)

카메라 영상에서 사람의 머리를 찾아 **안전모를 썼는지(helmet) / 안 썼는지(no_helmet)** 를 구분합니다.

![결과 예시](images/result_example1.jpg)

> 초록 박스 = helmet(착용), 빨간 박스 = no_helmet(미착용), 숫자 = 모델의 확신도(0~1)

---

## 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | YOLO로 안전모 착용/미착용 학습 + 웹캠 데모 (`1_train/`) | ✅ 완료 |
| 2 | OpenVINO / ONNX로 CPU 최적화, 속도 비교 (`2_optimize/`) | ⏳ 예정 |
| 3 | 위험 구역 경고 시스템 (`3_alert_system/`) | ⏳ 예정 |

---

## 1단계 결과: 모델 학습

### 학습 환경
| 항목 | 값 |
|---|---|
| CPU | AMD Ryzen 5 5600 6코어 (GPU 사용 안 함) |
| 소프트웨어 | Python 3.13, PyTorch 2.15 (CPU), Ultralytics 8.4.212 |
| 모델 | YOLO26n (Ultralytics, nano 크기, 파라미터 약 239만 개) |
| 학습 방법 | 사전학습 모델에서 시작하는 전이학습, 50에폭, 이미지 크기 640 |
| 학습 시간 | 약 2시간 28분 (CPU만 사용) |

### 성능 (검사용 사진 200장)
| 클래스 | Precision (정밀도) | Recall (재현율) | mAP50 | mAP50-95 |
|---|---|---|---|---|
| helmet | 0.924 | 0.827 | **0.880** | 0.693 |
| no_helmet | 0.923 | 0.603 | **0.709** | 0.439 |
| 전체 | 0.924 | 0.715 | **0.794** | 0.566 |

- **Precision (정밀도)**: 모델이 "찾았다"고 한 것 중 실제로 맞은 비율
  (높을수록 엉뚱한 곳에 박스를 덜 그림)
- **Recall (재현율)**: 실제 정답 중 모델이 찾아낸 비율
  (높을수록 놓치는 사람이 적음)
- **mAP (mean Average Precision, 평균 정밀도)**: 물체 감지 모델의 대표 점수 (0~1, 1에 가까울수록 좋음)
  - 확신도 기준을 높게도 낮게도 바꿔 가며 정밀도와 재현율을 함께 평가한 점수가 **AP**이고,
    이것을 클래스(helmet, no_helmet)별로 구해 평균 낸 것이 **mAP**입니다.
  - 그래서 "정확하게 찾는가"와 "빠뜨리지 않고 찾는가"를 한 숫자로 볼 수 있습니다.
- **IoU (겹침 정도)**: 모델이 그린 박스와 정답 박스가 얼마나 겹치는지 (0~1)
  - **mAP50**: IoU가 0.5 이상(절반 이상 겹침)이면 맞힌 것으로 계산한 mAP
  - **mAP50-95**: IoU 기준을 0.5, 0.55, … 0.95까지 점점 엄격하게 바꿔 가며 계산한 mAP의 평균
    (박스 위치까지 정확해야 높게 나와서 mAP50보다 낮게 나옴)

1에폭 학습 후 mAP50 0.606에서 50에폭 학습 후 **0.794**로 올랐습니다.
helmet(0.880)보다 no_helmet(0.709) 점수가 낮고, 특히 no_helmet의 Recall(0.603)이 낮아 미착용자를 놓치는 경우가 있습니다.

### 속도 (최적화 전 기준값, PyTorch CPU)
| 측정 | 결과 |
|---|---|
| 검사용 사진 1장 추론 시간 (200장 측정) | 약 23~32 ms (중앙값 25.6 ms) |
| 웹캠 데모 FPS | 이 PC에서 다시 측정 예정 |

→ 2단계에서 OpenVINO / ONNX로 바꾼 뒤 이 값과 비교할 예정입니다.

### 학습 그래프
![학습 그래프](images/training_curves.png)

- 검사용 데이터의 박스 손실(val/box_loss)이 끝까지 함께 줄어들어, 큰 과적합(학습 데이터만 외우는 현상)은 보이지 않습니다.
- 다만 mAP는 **30에폭 무렵부터 더 오르지 않았고**, 검사용 분류 손실(val/cls_loss)은 35에폭 이후 조금 올라갔습니다.
  → 에폭을 더 늘리기보다 데이터를 늘리는 쪽이 성능 개선에 효과적일 것으로 보입니다.
- 저장된 `best.pt`는 학습 중 점수가 가장 좋았던 에폭의 모델입니다.

### 결과 예시 (여러 명이 있는 장면)
<img src="images/result_example2.jpg" width="500">

> 오른쪽 앞의 **안전모를 안 쓴 사람(가까이 있어 크게 찍힌 얼굴)은 찾지 못했습니다.** 멀리 있는 작은 머리는 잘 찾지만, 가까운 큰 얼굴은 놓치는 한계를 보여 주는 예시입니다. (아래 한계점 2번 참고)
> 결과 사진은 확신도 0.4 이상 박스만 표시했습니다. 전체 성능은 위 표를 참고하세요.

---

## 한계점과 개선 계획

1. **no_helmet을 놓치는 경우가 있음** (Recall 0.603)
   - 미착용 데이터가 착용 데이터의 절반 정도로 적음 (학습용 박스 634개 vs 1,282개)
   - 개선: 미착용 데이터 추가, 이미지 크기 키우기, 더 큰 모델(yolo26s) 비교
2. **웹캠 앞 가까운 얼굴은 잘 못 찾음 (도메인 차이)**
   - 학습 데이터는 대부분 멀리서 찍은 공사 현장 사진이라 머리가 작게 나옴
   - 검사용 사진 예시: 앞쪽에 크게 찍힌 미착용자를 확신도 0.05까지 낮춰도 찾지 못함 (위 결과 예시)
   - 이전 모델의 웹캠 실험: 웹캠 앞에 가까이 앉으면 박스가 안 뜨고, 2~3m 떨어지면 감지됨.
     정면 각도의 맨머리를 no_helmet으로 감지한 경우도 있었으나 확신도는 약 0.5로 낮았음 (새 모델로 다시 확인 예정)
   - 개선: 정면·근거리 사진이 많은 데이터셋(예: Hard Hat Workers) 추가 학습
3. **박스 깜박임**
   - 매 화면을 따로 판단해서 확신도가 기준선 근처면 박스가 켜졌다 꺼졌다 함
   - 해결: 최근 박스를 몇 장 동안 유지하는 스무딩 적용 (`webcam.py --hold`)

---

## 실행 방법 (Windows)

### 1. 설치
```bash
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install ultralytics opencv-python
```

### 2. 데이터셋 내려받기
```bash
.\.venv\Scripts\python.exe 1_train\download_dataset.py
```
- `datasets/construction-ppe` 에 원본을 내려받고, helmet / no_helmet 두 클래스만 남긴 `datasets/helmet` 을 만듭니다.

### 3. 학습
```bash
.\.venv\Scripts\python.exe 1_train\train.py --epochs 50
```
- 결과 모델: `runs/helmet/weights/best.pt` (다시 학습하면 폴더 이름 뒤에 번호가 붙음)

### 4. 웹캠 데모
```bash
.\.venv\Scripts\python.exe 1_train\webcam.py
```
| 옵션 | 설명 |
|---|---|
| `--model 경로` | 사용할 모델 파일 (기본: `runs/helmet/weights/best.pt`) |
| `--source 경로` | 웹캠 대신 사진/영상 파일 사용 |
| `--conf 0.15` | 이 확신도보다 낮은 박스는 표시 안 함 (기본 0.4) |
| `--hold 10` | 박스가 사라져도 몇 장 동안 유지할지 (깜박임 줄이기) |

키보드: `s` 화면 저장(`images/`), `q` 또는 창의 X 버튼으로 종료

---

## 폴더 구조
```
1_train/          모델 학습 코드
  download_dataset.py   데이터셋 내려받기 + helmet/no_helmet만 남기기
  train.py              YOLO26n CPU 학습
  webcam.py             웹캠/사진 데모
2_optimize/       CPU 최적화와 속도 비교 (예정)
3_alert_system/   위험 구역 경고 시스템 (예정)
images/           README 결과 사진
```
데이터셋(`datasets/`), 학습 결과(`runs/`), 모델 파일(`*.pt`, `*.onnx`)은 용량 때문에 저장소에 올리지 않습니다.

---

## 데이터셋 출처와 라이선스

- **Construction-PPE** — Ultralytics
  - 문서: https://docs.ultralytics.com/datasets/detect/construction-ppe
  - 라이선스: **CC BY 4.0** (ultralytics 패키지의 `construction-ppe.yaml` 표기 기준)
  - 변경 사항: 12개 클래스 중 `helmet`, `no_helmet` 두 개만 사용
  - README의 결과 사진은 이 데이터셋의 검사용 사진에 모델 결과 박스를 그려 넣은 것입니다.

## 사용 기술
Python 3.13 · Ultralytics YOLO26 · PyTorch (CPU) · OpenCV
