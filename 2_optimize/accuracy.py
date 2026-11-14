"""
PyTorch / ONNX / OpenVINO / OpenVINO INT8 모델의 정확도(mAP)를 같은 조건에서 비교하는 코드

형식만 바꾼 모델이 원래 모델과 같은 점수를 내는지 검사용 사진(200장)으로 확인해요.
먼저 2_optimize/export.py 를 실행해서 ONNX, OpenVINO 모델을 만들어 두어야 해요.

실행 방법 (저장소 맨 위 폴더에서):
    .\\.venv\\Scripts\\python.exe 2_optimize\\accuracy.py
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # 저장소 맨 위 폴더
WEIGHTS = ROOT / "runs" / "helmet" / "weights"  # 모델 파일이 있는 폴더
DATA_YAML = ROOT / "datasets" / "helmet" / "data.yaml"  # 데이터셋 설정 파일 (검사용 사진 위치)
RUNS_DIR = ROOT / "runs"  # 채점 결과 그림이 저장될 폴더 (.gitignore 로 커밋 안 됨)

# 비교할 모델들 (이름: 파일 경로)
MODELS = {
    "PyTorch": WEIGHTS / "best.pt",
    "ONNX": WEIGHTS / "best.onnx",
    "OpenVINO": WEIGHTS / "best_openvino_model",
    "INT8": WEIGHTS / "best_int8_openvino_model",  # OpenVINO INT8 양자화 모델
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640, help="모델에 넣을 사진 크기(픽셀)")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit("데이터셋이 없어요. 먼저 1_train/download_dataset.py 를 실행하세요.")

    scores = {}  # 모델별 점수를 모아 둠
    for name, path in MODELS.items():
        if not path.exists():
            raise SystemExit(f"{name} 모델이 없어요: {path}\n먼저 2_optimize/export.py 를 실행하세요.")
        print(f"\n===== {name} 채점 중... =====")
        # task="detect": 변환한 모델은 어떤 일(물체 감지)을 하는지 파일만으로는 몰라서 알려줘요
        model = YOLO(str(path), task="detect")
        # val(채점): 검사용 사진의 정답 박스와 모델이 찾은 박스를 비교해서 점수를 매겨요
        # batch=1: 변환한 모델은 한 번에 사진 1장만 받도록 만들어져서, 공정하게 셋 다 1장씩 넣어요
        metrics = model.val(data=str(DATA_YAML), imgsz=args.imgsz, batch=1, device="cpu",
                            project=str(RUNS_DIR), name=f"val_{name.lower()}", exist_ok=True,
                            plots=False, verbose=False)
        box = metrics.box  # 박스 관련 점수 모음
        scores[name] = {
            "전체": (box.mp, box.mr, box.map50, box.map),  # mp/mr: 평균 정밀도/재현율
            # 클래스별 점수: class_result(i) -> (정밀도, 재현율, mAP50, mAP50-95)
            **{model.names[i]: box.class_result(i) for i in range(len(model.names))},
        }

    # 결과 표 출력
    base = scores["PyTorch"]["전체"][2]  # 비교 기준: PyTorch 의 전체 mAP50
    print("\n===== 정확도 비교 (검사용 사진, 1장씩 채점) =====")
    print(f"{'모델':9s} {'구분':10s} {'Precision':>9s} {'Recall':>7s} {'mAP50':>7s} {'mAP50-95':>9s}")
    for name, rows in scores.items():
        for part, (p, r, m50, m) in rows.items():
            print(f"{name:9s} {part:10s} {p:9.3f} {r:7.3f} {m50:7.3f} {m:9.3f}")
        diff = rows["전체"][2] - base
        print(f"{'':9s} → 전체 mAP50 차이 (PyTorch 대비): {diff:+.3f}")
    print("※ 차이가 ±0.005 정도 이내면 형식을 바꿔도 정확도가 그대로라고 볼 수 있어요.")


if __name__ == "__main__":
    main()
