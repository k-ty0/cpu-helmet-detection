"""
학습한 모델(best.pt)을 CPU에서 빠르게 돌릴 수 있는 형식으로 바꾸는 코드

- ONNX     : 여러 프로그램에서 함께 쓸 수 있는 AI 모델 파일 형식 (best.onnx)
- OpenVINO : 인텔이 만든 CPU 가속 도구용 형식 (best_openvino_model/ 폴더)
- OpenVINO INT8 : 모델 안의 숫자를 8비트 정수로 줄인(양자화) 형식 (best_int8_openvino_model/ 폴더)

변환이 끝나면 검사용 사진 몇 장으로
"원래 모델과 변환한 모델이 같은 결과를 내는지" 간단히 확인해요.

실행 방법 (저장소 맨 위 폴더에서):
    .\\.venv\\Scripts\\python.exe 2_optimize\\export.py
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # 저장소 맨 위 폴더
DEFAULT_MODEL = ROOT / "runs" / "helmet" / "weights" / "best.pt"  # 1단계에서 학습한 모델
VAL_IMAGES = ROOT / "datasets" / "helmet" / "images" / "val"  # 검사용 사진 폴더
DATA_YAML = ROOT / "datasets" / "helmet" / "data.yaml"  # 데이터셋 설정 파일 (INT8 보정용 사진 위치)


def export_models(model_path, imgsz):
    """best.pt 를 ONNX, OpenVINO, OpenVINO INT8 형식으로 바꾸고, 만들어진 파일 경로를 돌려줘요."""
    model = YOLO(str(model_path))
    # export(내보내기): 학습한 모델을 다른 형식의 파일로 저장하는 기능
    # 결과 파일은 best.pt 와 같은 폴더(runs/helmet/weights/)에 생겨요 (runs/ 는 커밋 안 됨)
    onnx_path = model.export(format="onnx", imgsz=imgsz, device="cpu")
    openvino_path = model.export(format="openvino", imgsz=imgsz, device="cpu")

    # INT8 양자화: 숫자를 8비트 정수로 줄여서 더 작고 빠르게 만들어요 (NNCF 도구 사용)
    # 어느 범위로 줄일지 정하려고 사진들을 모델에 보여 주며 숫자 범위를 재요 = 보정(calibration)
    # split="train": 보정에는 '학습용' 사진을 써요.
    #   기본값은 검사용(val) 사진인데, 채점할 사진으로 보정하면 점수가 실제보다 좋게 나올 수 있어요.
    int8_path = model.export(format="openvino", imgsz=imgsz, device="cpu",
                             quantize=8, data=str(DATA_YAML), split="train")
    return {"PyTorch": str(model_path), "ONNX": onnx_path, "OpenVINO": openvino_path,
            "INT8": int8_path}


def compare_results(paths, imgsz, n_images=5, conf=0.4):
    """여러 형식의 모델로 같은 사진을 돌려서, 찾은 박스 수와 확신도가 비슷한지 비교해요."""
    images = sorted(VAL_IMAGES.glob("*.jpg"))[:n_images]  # 검사용 사진 앞에서 몇 장
    if not images:
        print("검사용 사진이 없어서 비교를 건너뛰어요. (1_train/download_dataset.py 먼저 실행)")
        return

    print(f"\n===== 결과 비교 (사진 {len(images)}장, 확신도 {conf} 이상) =====")
    for name, path in paths.items():
        # task="detect": 변환한 모델은 어떤 일(물체 감지)을 하는지 파일만으로는 몰라서 알려줘요
        model = YOLO(path, task="detect")
        summary = []
        for img in images:
            result = model(str(img), imgsz=imgsz, conf=conf, verbose=False)[0]
            confs = sorted((round(float(c), 2) for c in result.boxes.conf), reverse=True)
            summary.append(f"{len(confs)}개 {confs[:3]}")  # 박스 수와 가장 높은 확신도 3개
        print(f"{name:9s}: " + " | ".join(summary))
    print("→ 모든 줄의 박스 수와 확신도가 거의 같으면 변환이 잘 된 거예요.")
    print("  (INT8 은 숫자를 반올림해서 확신도가 조금 더 달라질 수 있어요)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="변환할 모델 파일(.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="모델에 넣을 사진 크기(픽셀), 학습 때와 같게")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"모델 파일이 없어요: {args.model}\n"
                         "GitHub Releases(model-v1)에서 best.pt 를 내려받아 넣어 주세요.")

    paths = export_models(args.model, args.imgsz)
    print("\n===== 만들어진 파일 =====")
    for name, path in paths.items():
        print(f"{name:9s}: {path}")

    compare_results(paths, args.imgsz)


if __name__ == "__main__":
    main()
