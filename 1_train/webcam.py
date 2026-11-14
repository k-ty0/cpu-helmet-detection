"""
학습한 모델로 웹캠 영상에서 안전모 착용 여부를 표시하는 코드

- 초록 박스: helmet (안전모 씀)
- 빨간 박스: no_helmet (안전모 안 씀)
- 화면 왼쪽 위: FPS (1초에 몇 장 처리하는지)

키보드:
    s : 지금 화면을 images/ 폴더에 사진으로 저장
    q 또는 ESC : 종료 (한글 입력 상태라면 ESC 가 확실해요)

실행 방법 (저장소 맨 위 폴더에서):
    웹캠:          .\\.venv\\Scripts\\python.exe 1_train\\webcam.py
    사진/영상 파일: .\\.venv\\Scripts\\python.exe 1_train\\webcam.py --source 사진.jpg
"""

import argparse
import time  # FPS 를 재기 위해 시간을 측정하는 도구
from datetime import datetime  # 저장할 사진 이름에 날짜·시간을 넣기 위한 도구
from pathlib import Path

import cv2  # OpenCV: 카메라 영상 읽기, 박스 그리기, 화면 띄우기
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # 저장소 맨 위 폴더
DEFAULT_MODEL = ROOT / "runs" / "helmet" / "weights" / "best.pt"  # 50에폭 학습 모델
IMAGES_DIR = ROOT / "images"  # 결과 사진을 저장할 폴더
WINDOW = "helmet detection"  # 화면 창 이름

# 클래스 이름별 박스 색깔 (OpenCV 는 RGB 가 아니라 BGR 순서예요)
COLORS = {"helmet": (0, 200, 0), "no_helmet": (0, 0, 255)}  # 초록, 빨강


def draw_boxes(frame, result):
    """모델이 찾은 박스들을 화면(frame)에 그려요."""
    for box in result.boxes:  # 찾은 물체 하나하나에 대해
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 박스 왼쪽 위, 오른쪽 아래 좌표
        name = result.names[int(box.cls)]  # 클래스 번호 -> 이름 (helmet / no_helmet)
        conf = float(box.conf)  # 확신도 (0~1, 클수록 확실하다는 뜻)
        color = COLORS.get(name, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 박스 그리기
        label = f"{name} {conf:.2f}"  # 예: "helmet 0.87"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),  # 박스 위에 글자 쓰기
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def save_image(frame):
    """현재 화면을 images/ 폴더에 저장해요."""
    IMAGES_DIR.mkdir(exist_ok=True)
    path = IMAGES_DIR / f"webcam_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"저장했어요: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="사용할 모델 파일(.pt)")
    parser.add_argument("--source", default="0", help="0=웹캠, 또는 사진/영상 파일 경로")
    parser.add_argument("--conf", type=float, default=0.4, help="이 확신도보다 낮은 박스는 안 그려요")
    parser.add_argument("--hold", type=int, default=10,
                        help="박스가 사라져도 몇 장 동안 유지할지 (깜박임 줄이기, 0이면 끔)")
    args = parser.parse_args()

    # 학습한 모델 불러오기 (.pt 뿐 아니라 2단계에서 만든 ONNX·OpenVINO·INT8 모델도 쓸 수 있어요)
    # task="detect": 변환한 모델은 어떤 일(물체 감지)을 하는지 파일만으로는 몰라서 알려줘요
    model = YOLO(args.model, task="detect")

    # 사진 한 장이면: 결과를 그려서 창에 띄우고 끝
    if Path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png"}:
        frame = cv2.imread(args.source)
        result = model(frame, conf=args.conf, verbose=False)[0]
        draw_boxes(frame, result)
        cv2.imshow(WINDOW, frame)
        print("아무 키나 누르면 닫혀요. (s: 저장)")
        if cv2.waitKey(0) & 0xFF == ord("s"):
            save_image(frame)
        cv2.destroyAllWindows()
        return

    # 웹캠(숫자) 또는 영상 파일 열기
    if args.source.isdigit():
        # CAP_DSHOW: Windows 에서 웹캠을 더 빨리 여는 방식
        cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit("카메라/영상을 열 수 없어요. 다른 프로그램이 카메라를 쓰고 있는지 확인하세요.")

    print("실행 중... (s: 사진 저장, q 또는 ESC: 종료)")
    last_result, missed = None, 0  # 마지막으로 박스를 찾은 결과, 그 뒤로 못 찾은 화면 수
    infer_times = []  # 장마다 모델이 걸린 시간(초)을 모아 두는 목록 -> 끝날 때 평균 FPS 계산
    while True:
        ok, frame = cap.read()  # 한 장(프레임) 읽기
        if not ok:  # 영상이 끝났거나 카메라 오류
            break

        start = time.perf_counter()  # 처리 시작 시간
        result = model(frame, conf=args.conf, verbose=False)[0]  # 안전모 찾기
        elapsed = time.perf_counter() - start  # 1장 처리에 걸린 시간(초)
        fps = 1.0 / elapsed  # 1장 처리 시간 -> 초당 장수
        infer_times.append(elapsed)

        # 깜박임 줄이기(스무딩): 박스가 잠깐 사라져도 바로 지우지 않고
        # 최근에 찾은 결과를 hold 장 동안 계속 보여줘요
        if len(result.boxes) > 0:  # 이번 화면에서 뭔가 찾았으면
            last_result, missed = result, 0  # 기억해 두고, '놓친 횟수'는 0으로
        else:
            missed += 1  # 이번 화면에선 못 찾음
        if last_result is not None and missed <= args.hold:
            draw_boxes(frame, last_result)  # 기억해 둔 박스 그리기
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),  # 왼쪽 위에 FPS 표시
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow(WINDOW, frame)  # 창에 보여주기

        key = cv2.waitKey(1)  # 키보드 입력 확인 (-1 이면 아무 키도 안 누름)
        # 27 = ESC 키: 한글 입력기(IME)를 거치지 않아서 한/영 상태와 상관없이 한 번에 종료돼요
        # (한글 상태에서는 'ㅂ' 이 글자를 조합하는 중이라 다음 키를 눌러야 들어올 때가 있어요)
        if key in (27, ord("q"), ord("Q"), ord("ㅂ")):  # 한글 입력 상태의 q 는 'ㅂ' 으로 들어와요
            break
        if key in (ord("s"), ord("S"), ord("ㄴ")):  # 한글 입력 상태의 s 는 'ㄴ'
            save_image(frame)
        # 창의 X 버튼을 눌러 창이 닫혔으면 종료 (없으면 다음 imshow 가 창을 다시 띄워요)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()  # 카메라 놓아주기 (다른 프로그램이 쓸 수 있게)
    cv2.destroyAllWindows()  # 창 닫기

    # 평균 FPS 출력 (README 에 적을 값)
    # 처음 몇 장은 준비(워밍업) 때문에 느려서 앞의 10장은 빼고 계산해요
    times = infer_times[10:]
    if times:
        avg_ms = sum(times) / len(times) * 1000  # 1장 평균 처리 시간(밀리초)
        print(f"\n===== 평균 속도 ({len(times)}장 기준, 처음 10장 제외) =====")
        print(f"1장 평균 처리 시간 (사진 준비+추론+박스 정리): {avg_ms:.1f} ms")
        print(f"평균 FPS (카메라 읽기·화면 그리기 제외): {1000 / avg_ms:.1f}")
    else:
        print("측정한 화면이 너무 적어요. 조금 더 오래 실행해 주세요.")


if __name__ == "__main__":
    main()
