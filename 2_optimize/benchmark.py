"""
PyTorch / ONNX / OpenVINO / OpenVINO INT8 모델의 CPU 속도를 같은 조건에서 비교하는 코드

먼저 2_optimize/export.py 를 실행해서 ONNX, OpenVINO 모델을 만들어 두어야 해요.

공정하게 재기 위해:
- 모든 모델이 같은 검사용 사진(기본 200장)을 사용해요.
- 사진은 미리 메모리에 올려 둬요. (파일 읽는 시간이 섞이지 않게)
- 처음 몇 장(워밍업)은 준비 때문에 느려서 측정에서 빼요.
- 여러 번(기본 3회) 반복하고, 매번 모델 순서를 바꿔요. (먼저 잰 모델만 유리하지 않게)
- 모델 하나를 잴 때마다 새 프로그램(프로세스)을 따로 띄워요.
  한 프로그램 안에서 여러 모델을 같이 돌리면, ONNX Runtime·OpenVINO 가 끝난 뒤에도
  남겨 두는 일꾼(스레드)들이 CPU 를 차지해서 PyTorch 가 2배 가까이 느려지는 걸 확인했어요.

실행 방법 (저장소 맨 위 폴더에서):
    .\\.venv\\Scripts\\python.exe 2_optimize\\benchmark.py
"""

import argparse
import json  # 따로 띄운 프로그램이 측정 결과를 글자로 돌려줄 때 쓰는 형식
import statistics  # 평균, 중앙값 같은 통계를 계산하는 파이썬 기본 도구
import subprocess  # 새 프로그램(프로세스)을 띄우는 파이썬 기본 도구
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # 저장소 맨 위 폴더
WEIGHTS = ROOT / "runs" / "helmet" / "weights"  # 모델 파일이 있는 폴더
VAL_IMAGES = ROOT / "datasets" / "helmet" / "images" / "val"  # 검사용 사진 폴더

# 비교할 모델들 (이름: 파일 경로)
MODELS = {
    "PyTorch": WEIGHTS / "best.pt",
    "ONNX": WEIGHTS / "best.onnx",
    "OpenVINO": WEIGHTS / "best_openvino_model",
    "INT8": WEIGHTS / "best_int8_openvino_model",  # OpenVINO INT8 양자화 모델
}


def load_images(n, imgsz):
    """검사용 사진 n장을 미리 읽어서 메모리에 올려 둬요."""
    images = []
    for path in sorted(VAL_IMAGES.glob("*.jpg"))[:n]:
        # np.fromfile + imdecode: 경로에 한글(문서)이 있어도 사진을 읽을 수 있는 방법
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        # 긴 변을 imgsz(640)로 미리 줄여 둬요. 모델에는 어차피 640 크기로 들어가서 결과는 같고,
        # 원본(2048픽셀) 200장을 그대로 두면 메모리를 너무 많이 써요.
        scale = imgsz / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        images.append(img)
    return images


def measure(model, images, imgsz, warmup):
    """사진들을 한 장씩 넣고, 장마다 '모델 추론 시간(ms)'을 모아서 돌려줘요."""
    for img in images[:warmup]:  # 워밍업: 결과는 버려요
        model(img, imgsz=imgsz, verbose=False)
    times = []
    for img in images:
        result = model(img, imgsz=imgsz, verbose=False)[0]
        # result.speed: ultralytics 가 재 준 시간(ms)
        #   preprocess(사진 준비) / inference(AI 계산) / postprocess(박스 정리)
        # 모델 형식에 따라 달라지는 건 inference 라서 이것만 비교해요
        times.append(result.speed["inference"])
    return times


def run_one(name, args):
    """[따로 띄운 프로그램 안에서 실행] 모델 하나만 재고, 결과를 JSON 글자로 출력해요."""
    images = load_images(args.images, args.imgsz)
    # task="detect": 변환한 모델은 어떤 일(물체 감지)을 하는지 파일만으로는 몰라서 알려줘요
    model = YOLO(str(MODELS[name]), task="detect")
    times = measure(model, images, args.imgsz, args.warmup)
    print("RESULT " + json.dumps(times))  # 맨 앞의 'RESULT' 로 결과 줄을 찾아요


def measure_in_new_process(name, args):
    """새 프로그램을 띄워서 모델 하나를 재고, 시간 목록을 받아 와요."""
    cmd = [sys.executable, __file__, "--only", name,  # 이 파일을 '모델 하나만 재기' 모드로 실행
           "--images", str(args.images), "--warmup", str(args.warmup), "--imgsz", str(args.imgsz)]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=200, help="측정에 쓸 검사용 사진 수")
    parser.add_argument("--rounds", type=int, default=3, help="전체 측정을 몇 번 반복할지")
    parser.add_argument("--warmup", type=int, default=10, help="측정 전에 버리는 사진 수")
    parser.add_argument("--imgsz", type=int, default=640, help="모델에 넣을 사진 크기(픽셀)")
    parser.add_argument("--only", choices=list(MODELS), help="(내부용) 이 모델 하나만 재기")
    args = parser.parse_args()

    if args.only:  # 따로 띄운 프로그램이면 모델 하나만 재고 끝
        run_one(args.only, args)
        return

    for name, path in MODELS.items():
        if not path.exists():
            raise SystemExit(f"{name} 모델이 없어요: {path}\n먼저 2_optimize/export.py 를 실행하세요.")

    print(f"사진 {args.images}장, {args.rounds}회 반복, 워밍업 {args.warmup}장 (모델마다 새 프로그램에서 측정)")
    all_times = {name: [] for name in MODELS}  # 모델별로 모든 회차의 시간을 모아 둠
    round_means = {name: [] for name in MODELS}  # 모델별로 회차마다의 평균

    names = list(MODELS)
    for r in range(args.rounds):
        # 매 회차마다 순서를 한 칸씩 돌려요 (예: 1회 P,O,V,I → 2회 O,V,I,P → 3회 V,I,P,O)
        order = names[r % len(names):] + names[:r % len(names)]
        print(f"\n[{r + 1}회차] 순서: {' → '.join(order)}")
        for name in order:
            times = measure_in_new_process(name, args)
            all_times[name].extend(times)
            round_means[name].append(statistics.mean(times))
            print(f"  {name:9s} 평균 {statistics.mean(times):5.1f} ms")

    # 결과 표 출력
    base = statistics.median(all_times["PyTorch"])  # 비교 기준: PyTorch 중앙값
    print(f"\n===== 결과 (사진 {args.images}장 × {args.rounds}회, 모델 추론 시간만) =====")
    print(f"{'모델':9s} {'중앙값':>8s} {'평균':>8s} {'FPS':>7s} {'회차별 평균':>22s} {'PyTorch 대비':>11s}")
    for name, times in all_times.items():
        med = statistics.median(times)  # 중앙값: 가운데 값 (가끔 튀는 느린 값에 덜 흔들려요)
        mean = statistics.mean(times)
        per_round = " / ".join(f"{m:.1f}" for m in round_means[name])
        print(f"{name:9s} {med:6.1f}ms {mean:6.1f}ms {1000 / med:7.1f} {per_round:>22s} {base / med:9.2f}배")
    print("※ FPS = 1000 / 중앙값(ms). 'PyTorch 대비'가 1보다 크면 그만큼 빠르다는 뜻이에요.")


if __name__ == "__main__":
    main()
