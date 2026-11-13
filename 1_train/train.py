"""
안전모 감지 모델(helmet / no_helmet)을 CPU로 학습하는 코드

먼저 download_dataset.py 를 실행해서 datasets/helmet 을 만들어 두어야 해요.

실행 방법 (저장소 맨 위 폴더에서):
    짧게 시험 학습:  .\\.venv\\Scripts\\python.exe 1_train\\train.py --epochs 3
    본 학습:        .\\.venv\\Scripts\\python.exe 1_train\\train.py --epochs 50
"""

import argparse  # 실행할 때 --epochs 같은 옵션을 받을 수 있게 해 주는 도구
from pathlib import Path

from ultralytics import YOLO  # YOLO 모델을 쉽게 쓰게 해 주는 라이브러리

ROOT = Path(__file__).resolve().parent.parent  # 저장소 맨 위 폴더
DATA_YAML = ROOT / "datasets" / "helmet" / "data.yaml"  # 데이터셋 설정 파일
RUNS_DIR = ROOT / "runs"  # 학습 결과가 저장될 폴더 (.gitignore 로 커밋 안 됨)


def main():
    # 1) 실행 옵션 읽기 (안 적으면 기본값 사용)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="사진 전체를 몇 번 반복해서 볼지")
    parser.add_argument("--imgsz", type=int, default=640, help="학습할 때 사진 크기(픽셀)")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit("데이터셋이 없어요. 먼저 1_train/download_dataset.py 를 실행하세요.")

    # 2) 미리 학습된 YOLO26 nano 모델 불러오기 (전이학습의 출발점)
    #    처음 실행하면 인터넷에서 yolo26n.pt 를 자동으로 내려받아요.
    model = YOLO("yolo26n.pt")

    # 3) 우리 데이터셋으로 추가 학습하기
    model.train(
        data=str(DATA_YAML),  # 어떤 데이터로 배울지
        epochs=args.epochs,  # 몇 번 반복할지
        imgsz=args.imgsz,  # 사진 크기
        batch=16,  # 한 번에 몇 장씩 묶어서 볼지
        device="cpu",  # GPU 없이 CPU 로 학습
        workers=4,  # 사진을 미리 읽어 오는 일꾼 수
        project=str(RUNS_DIR),  # 결과 저장 폴더
        name="helmet",  # 결과 폴더 이름 (runs/helmet, 또 하면 helmet2 ...)
        plots=True,  # 학습 그래프·결과 그림 저장 (README 에 쓸 수 있어요)
    )

    # 4) 검사용 사진(val)으로 성적 확인하기
    metrics = model.val(project=str(RUNS_DIR), name="helmet_val")  # 결과도 runs/ 안에 저장
    print("\n===== 학습 결과 =====")
    print(f"전체 mAP50: {metrics.box.map50:.3f}")  # 1에 가까울수록 잘 찾는다는 뜻
    for i, name in model.names.items():  # 클래스별(helmet, no_helmet) 성적
        print(f"  {name:10s} mAP50: {metrics.box.ap50[i]:.3f}")
    print(f"가장 좋은 모델 파일: {model.trainer.best}")


if __name__ == "__main__":  # Windows 에서는 이 줄이 꼭 있어야 학습이 안전하게 돌아가요
    main()
