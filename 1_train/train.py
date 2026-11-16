"""
안전모 감지 모델(helmet / no_helmet)을 CPU로 학습하는 코드

먼저 download_dataset.py 를 실행해서 datasets/helmet 을 만들어 두어야 해요.

실행 방법 (저장소 맨 위 폴더에서):
    짧게 시험 학습:  .\\.venv\\Scripts\\python.exe 1_train\\train.py --epochs 3
    본 학습:        .\\.venv\\Scripts\\python.exe 1_train\\train.py --epochs 50
    SHWD 를 합친 데이터로 이어서 학습 (지금 모델 best.pt 에서 출발):
        .\\.venv\\Scripts\\python.exe 1_train\\train.py --data datasets\\helmet_shwd\\data.yaml
            --weights runs\\helmet\\weights\\best.pt --name helmet_shwd --epochs 30
        (실제로는 한 줄로 이어서 써요)
    GPU 로 학습 (NVIDIA GPU + GPU 용 가상환경 .venv-gpu 필요):
        .\\.venv-gpu\\Scripts\\python.exe 1_train\\train.py --epochs 50 --device 0
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
    parser.add_argument("--data", default=str(DATA_YAML), help="데이터셋 설정 파일(data.yaml)")
    parser.add_argument("--weights", default="yolo26n.pt",
                        help="출발점 모델 (기본: 미리 학습된 YOLO26 nano / 이어서 학습하려면 best.pt)")
    parser.add_argument("--name", default="helmet", help="결과 폴더 이름 (runs/이름)")
    parser.add_argument("--device", default="cpu",
                        help="학습 장치: cpu 또는 0 (0 = 첫 번째 NVIDIA GPU, .venv-gpu 에서 실행해야 해요)")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"데이터셋이 없어요: {args.data}\n먼저 1_train/download_dataset.py 를 실행하세요.")

    # 2) 출발점 모델 불러오기 (전이학습: 이미 배운 걸 바탕으로 우리 데이터를 추가로 배워요)
    #    기본값 yolo26n.pt 는 처음 실행하면 인터넷에서 자동으로 내려받아요.
    #    best.pt 를 넣으면 안전모를 이미 배운 모델에서 이어서 배워요(파인튜닝).
    model = YOLO(args.weights)

    # 3) 우리 데이터셋으로 추가 학습하기
    model.train(
        data=args.data,  # 어떤 데이터로 배울지
        epochs=args.epochs,  # 몇 번 반복할지
        imgsz=args.imgsz,  # 사진 크기
        batch=16,  # 한 번에 몇 장씩 묶어서 볼지
        device=args.device,  # 기본은 CPU, GPU 로 하려면 --device 0
        workers=4,  # 사진을 미리 읽어 오는 일꾼 수
        project=str(RUNS_DIR),  # 결과 저장 폴더
        name=args.name,  # 결과 폴더 이름 (runs/helmet, 같은 이름으로 또 하면 helmet2 ...)
        plots=True,  # 학습 그래프·결과 그림 저장 (README 에 쓸 수 있어요)
    )

    # 4) 검사용 사진(val)으로 성적 확인하기
    metrics = model.val(project=str(RUNS_DIR), name=f"{args.name}_val")  # 결과도 runs/ 안에 저장
    print("\n===== 학습 결과 =====")
    print(f"전체 mAP50: {metrics.box.map50:.3f}")  # 1에 가까울수록 잘 찾는다는 뜻
    for i, name in model.names.items():  # 클래스별(helmet, no_helmet) 성적
        print(f"  {name:10s} mAP50: {metrics.box.ap50[i]:.3f}")
    print(f"가장 좋은 모델 파일: {model.trainer.best}")


if __name__ == "__main__":  # Windows 에서는 이 줄이 꼭 있어야 학습이 안전하게 돌아가요
    main()
