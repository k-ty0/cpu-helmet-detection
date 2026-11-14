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
from collections import deque  # 앞에서 빼고 뒤에 넣기 편한 목록 (최근 결과 모아 두기)
from statistics import median  # 중간값 계산 (파이썬 기본 도구)
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


def get_boxes(result):
    """모델 결과에서 박스를 (좌표, 클래스 번호, 확신도) 목록으로 꺼내요."""
    return [(list(map(int, b.xyxy[0])), int(b.cls), float(b.conf)) for b in result.boxes]


def iou(a, b):
    """두 박스가 얼마나 겹치는지 (0=안 겹침, 1=완전히 같음). IoU = 겹친 넓이 / 합친 넓이"""
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))  # 겹친 부분의 너비
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))  # 겹친 부분의 높이
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)  # 1e-9: 0으로 나누기 방지


def smooth_box(box, cls, conf, history):
    """최근 화면들(history)에서 이 박스와 같은 자리의 박스들을 모아서
    1) 박스 위치는 지금 것 그대로, 크기는 중간값으로  2) 클래스(색)는 투표로 정해요.

    - 위치: 지금 화면의 박스 중심을 그대로 써서, 움직이면 바로 따라와요.
    - 크기(너비·높이): 최근 크기들의 중간값(median)을 써서, 한 장에서 갑자기 커지거나
      작아져도 무시돼요. 중간값 = 줄 세웠을 때 가운데 값 (튀는 값에 평균보다 덜 흔들려요)
    - 색 투표: 한 장에서 helmet 이 잠깐 나와도, 최근 대부분이 no_helmet 이면 no_helmet 으로 보여 줘요.
      표는 확신도만큼 줘요 (확실한 판단일수록 표가 커요).
    돌려주는 값: (평균 박스 좌표, 이긴 클래스, 이긴 클래스의 평균 확신도)
    """
    votes, counts = {}, {}  # 클래스 번호 -> 받은 표(확신도 합), 나온 횟수
    matched = []  # 같은 자리로 본 박스 좌표들
    for _, boxes in history:
        for xyxy, c, f in boxes:
            if iou(box, xyxy) > 0.3:  # 30% 넘게 겹치면 같은 머리로 봐요
                votes[c] = votes.get(c, 0) + f
                counts[c] = counts.get(c, 0) + 1
                matched.append(xyxy)
    if not votes:  # 최근에 같은 자리 박스가 없으면 원래 결과 그대로
        return box, cls, conf
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # 지금 박스의 중심
    w = median(b[2] - b[0] for b in matched)  # 최근 너비들의 중간값
    h = median(b[3] - b[1] for b in matched)  # 최근 높이들의 중간값
    new_box = [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]
    win = max(votes, key=votes.get)  # 표를 가장 많이 받은 클래스
    return new_box, win, votes[win] / counts[win]


def draw_boxes(frame, boxes, names):
    """박스들(좌표, 클래스 번호, 숫자)을 화면(frame)에 그려요."""
    for (x1, y1, x2, y2), cls, conf in boxes:  # 찾은 물체 하나하나에 대해
        name = names[cls]  # 클래스 번호 -> 이름 (helmet / no_helmet)
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
    parser.add_argument("--hold", type=float, default=0.5,
                        help="박스가 사라져도 몇 초 동안 유지할지 (깜박임 줄이기, 0이면 끔)")
    parser.add_argument("--smooth", type=float, default=0.5,
                        help="최근 몇 초 동안의 결과로 박스 크기(중간값)와 색(투표)을 정할지 (0이면 끔)")
    args = parser.parse_args()

    # 학습한 모델 불러오기 (.pt 뿐 아니라 2단계에서 만든 ONNX·OpenVINO·INT8 모델도 쓸 수 있어요)
    # task="detect": 변환한 모델은 어떤 일(물체 감지)을 하는지 파일만으로는 몰라서 알려줘요
    model = YOLO(args.model, task="detect")

    # 사진 한 장이면: 결과를 그려서 창에 띄우고 끝
    if Path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png"}:
        frame = cv2.imread(args.source)
        result = model(frame, conf=args.conf, verbose=False)[0]
        draw_boxes(frame, get_boxes(result), result.names)
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
    last_boxes, last_seen = [], 0.0  # 마지막으로 찾은 박스들, 그 시각
    history = deque()  # 최근 화면들의 (시각, 박스 목록) -> 색 투표에 사용
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

        # 깜박임 줄이기(스무딩) - 장 수가 아니라 '시간(초)' 기준이라 FPS 가 달라도 똑같이 동작해요
        now = time.perf_counter()
        boxes = get_boxes(result)
        history.append((now, boxes))
        while history and now - history[0][0] > args.smooth:  # smooth 초보다 오래된 결과는 버려요
            history.popleft()

        # 1) 박스 유지: 박스가 잠깐 사라져도 hold 초 동안은 마지막 박스를 계속 보여줘요
        if boxes:  # 이번 화면에서 뭔가 찾았으면
            last_boxes, last_seen = boxes, now  # 기억해 두기
        if last_boxes and now - last_seen <= args.hold:
            # 2) 위치는 지금 그대로, 최근 smooth 초 동안의 결과로 크기는 중간값, 색(helmet / no_helmet)은 투표로 정해요
            #    박스 옆 숫자는 이긴 쪽의 '최근 평균 확신도'예요
            if args.smooth > 0:
                shown = [smooth_box(xyxy, cls, conf, history) for xyxy, cls, conf in last_boxes]
            else:
                shown = last_boxes
            draw_boxes(frame, shown, result.names)  # 박스 그리기
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
