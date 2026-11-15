"""
위험 구역 경고 시스템 (3단계)

화면에 '위험 구역'을 정해 두고, 그 안에 있는 사람(머리)을 찾아서 표시해요.
    - 구역 안 + 안전모 안 씀(no_helmet) -> 위험 (빨간색, "DANGER")
      안전 우선: 최근 1초 동안 no_helmet 판단이 30% 이상 섞여 있어도 위험으로 봐요
    - 구역 안 + 안전모 씀(helmet)       -> 안전 (초록색)
    - 구역 밖                          -> 연한 색으로만 표시

1) 먼저 위험 구역 정하기 (한 번만 하면 zone.json 에 저장돼요):
    .\\.venv\\Scripts\\python.exe 3_alert_system\\alert.py --setup
        마우스 왼쪽 클릭 : 구역의 꼭짓점 추가 (3개 이상)
        r               : 다시 그리기
        Enter           : 저장하고 끝내기
        ESC             : 저장 안 하고 끝내기

2) 경고 시스템 실행:
    .\\.venv\\Scripts\\python.exe 3_alert_system\\alert.py
        ESC 또는 q : 종료

영상 파일로 할 때 (첫 장면을 멈춰 놓고 구역을 그려요, 실행하면 영상이 반복 재생돼요):
    .\\.venv\\Scripts\\python.exe 3_alert_system\\alert.py --setup --source datasets\\videos\\mixkit_1459.mp4 --zone 3_alert_system\\zone_1459.json
    .\\.venv\\Scripts\\python.exe 3_alert_system\\alert.py --source datasets\\videos\\mixkit_1459.mp4 --zone 3_alert_system\\zone_1459.json
"""

import argparse
import json  # 구역 좌표를 파일(zone.json)로 저장하고 읽는 도구
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np  # 좌표 목록을 OpenCV 가 원하는 배열 형태로 바꾸는 데 써요
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # 저장소 맨 위 폴더
sys.path.insert(0, str(ROOT / "1_train"))  # 1_train/webcam.py 의 함수를 가져다 쓰려고 경로 추가
from webcam import get_boxes, iou, smooth_box  # noqa: E402 (경로를 추가한 뒤에 불러와야 해서 여기 있어요)

ZONE_FILE = Path(__file__).resolve().parent / "zone.json"  # 위험 구역 좌표를 저장할 파일
DEFAULT_MODEL = ROOT / "runs" / "helmet_shwd" / "weights" / "best_int8_openvino_model"  # 개선 모델 INT8
WINDOW = "danger zone"

# 색깔 (OpenCV 는 BGR 순서)
RED, GREEN, YELLOW, GRAY = (0, 0, 255), (0, 200, 0), (0, 220, 255), (160, 160, 160)


def open_camera(source):
    """웹캠(숫자) 또는 영상 파일을 열어요."""
    cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("카메라/영상을 열 수 없어요. 다른 프로그램이 카메라를 쓰고 있는지 확인하세요.")
    return cap


def setup_zone(source, zone_file):
    """화면 위에 마우스로 클릭해서 위험 구역(다각형)을 정하고 zone_file 에 저장해요.

    웹캠이면 움직이는 화면 위에, 영상 파일이면 첫 장면을 멈춰 놓고 그 위에 그려요.
    """
    points = []  # 클릭한 꼭짓점들 [(x, y), ...]

    def on_mouse(event, x, y, flags, param):
        # 마우스 왼쪽 버튼을 누를 때마다 꼭짓점을 하나씩 추가해요
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cap = open_camera(source)
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)  # 이 창에서 마우스를 쓰면 on_mouse 가 불려요
    print("마우스 클릭: 꼭짓점 추가 / r: 다시 그리기 / Enter: 저장 / ESC: 취소")
    saved = False
    is_file = not source.isdigit()
    ok, still = cap.read()  # 첫 장면 (영상 파일이면 이 장면을 계속 보여 줘요)
    if not ok:
        raise SystemExit("화면을 읽지 못했어요.")
    while True:
        if is_file:
            frame = still.copy()  # 멈춘 첫 장면 위에 그리기 (copy: 원본에 선이 쌓이지 않게)
        else:
            ok, frame = cap.read()
            if not ok:
                break
        if points:
            pts = np.array(points, np.int32)
            cv2.polylines(frame, [pts], isClosed=len(points) >= 3, color=YELLOW, thickness=2)
            for p in points:
                cv2.circle(frame, p, 4, YELLOW, -1)  # 꼭짓점 점 찍기
        cv2.putText(frame, f"points: {len(points)}  (Enter: save, r: reset, ESC: cancel)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1)
        if key == 13:  # 13 = Enter 키
            if len(points) >= 3:
                h, w = frame.shape[:2]
                # 화면 크기도 같이 저장해 두면, 나중에 크기가 달라졌는지 확인할 수 있어요
                zone_file.write_text(json.dumps({"width": w, "height": h, "points": points}, indent=2),
                                     encoding="utf-8")
                print(f"저장했어요: {zone_file} (꼭짓점 {len(points)}개)")
                saved = True
                break
            print("꼭짓점이 3개 이상 있어야 구역이 돼요.")
        elif key in (ord("r"), ord("R"), ord("ㄱ")):  # 한글 입력 상태의 r 은 'ㄱ'
            points.clear()
        elif key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:  # ESC 또는 창 닫기
            break
    cap.release()
    cv2.destroyAllWindows()
    if not saved:
        print("저장하지 않고 끝냈어요.")


def load_zone(zone_file):
    """zone_file 에서 위험 구역 꼭짓점을 읽어요."""
    if not zone_file.exists():
        raise SystemExit(f"위험 구역이 없어요: {zone_file}\n먼저 --setup 으로 구역을 정해 주세요. "
                         "(--source, --zone 도 실행할 때와 똑같이 넣어 주세요)")
    # utf-8-sig: 메모장 등으로 고쳐 저장하면 파일 맨 앞에 보이지 않는 표시(BOM)가 붙을 수 있는데, 그것도 읽어 줘요
    data = json.loads(zone_file.read_text(encoding="utf-8-sig"))
    return np.array(data["points"], np.int32)


def in_zone(zone, box):
    """박스(머리)의 중심점이 위험 구역 안에 있는지 확인해요.

    cv2.pointPolygonTest: 점이 다각형 안이면 양수, 경계 위면 0, 밖이면 음수를 돌려줘요.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return cv2.pointPolygonTest(zone, (float(cx), float(cy)), False) >= 0


def no_helmet_share(box, history, names):
    """최근 화면들(history)에서 이 머리 자리에 나온 판단 중 no_helmet 표가 차지하는 비율 (0~1).

    표는 확신도만큼 줘요. 예: no_helmet 0.6 + helmet 0.4 -> 0.6
    """
    votes = {"helmet": 0.0, "no_helmet": 0.0}
    for _, boxes in history:
        for xyxy, c, f in boxes:
            if iou(box, xyxy) > 0.3:  # 30% 넘게 겹치면 같은 머리
                votes[names[c]] = votes.get(names[c], 0.0) + f
    total = votes["helmet"] + votes["no_helmet"]
    return votes["no_helmet"] / total if total else 0.0


def draw_zone(frame, zone, danger):
    """위험 구역을 반투명하게 칠해요. 위험한 사람이 있으면 빨간색, 없으면 노란색."""
    color = RED if danger else YELLOW
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], color)  # 구역을 색으로 채운 그림
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)  # 원래 화면과 25% 만 섞어서 반투명하게
    cv2.polylines(frame, [zone], True, color, 2)  # 테두리


def run(args):
    zone = load_zone(Path(args.zone))
    model = YOLO(args.model, task="detect")
    cap = open_camera(args.source)
    is_file = not args.source.isdigit()
    print("실행 중... (ESC 또는 q: 종료)")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # 영상의 초당 장 수 (모르면 30으로)
    last_boxes, last_seen = [], 0.0  # 웹캠 데모와 같은 깜박임 줄이기
    history = deque()
    keep = max(args.smooth, args.danger_window)  # 기록은 둘 중 더 긴 시간만큼 보관
    frame_no, start = 0, time.perf_counter()
    counts = {"frames": 0, "red": 0, "green": 0}  # 끝날 때 요약해서 보여 줄 숫자
    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file and not args.once:  # 영상이 끝나면 처음으로 되감아서 계속 반복 (시연용)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_boxes, history = [], deque()  # 이전 바퀴의 박스 기억은 지워요
                frame_no, start = 0, time.perf_counter()
                continue
            break
        result = model(frame, conf=args.conf, verbose=False)[0]

        # 시간 재기: 영상 파일은 '영상 속 시간'(몇 번째 장 / 초당 장 수)을 써요.
        # 컴퓨터가 영상보다 빨리 처리해도 1초는 영상 속 1초가 되게 하려고요.
        if is_file:
            now = frame_no / fps
            frame_no += 1
            wait = now - (time.perf_counter() - start)
            if wait > 0:
                time.sleep(wait)  # 너무 빨리 재생되지 않게 실제 속도에 맞춰 기다려요
        else:
            now = time.perf_counter()

        boxes = get_boxes(result)
        history.append((now, boxes))
        while history and now - history[0][0] > keep:
            history.popleft()
        recent = [h for h in history if now - h[0] <= args.smooth]  # 크기·색 정하기는 최근 smooth 초만
        if boxes:
            last_boxes, last_seen = boxes, now
        shown = []
        if last_boxes and now - last_seen <= args.hold:
            shown = [smooth_box(xyxy, cls, conf, recent) for xyxy, cls, conf in last_boxes]

        # 안전 우선 규칙: 최근 danger_window 초 동안 no_helmet 표가 danger_ratio 이상이면 '위험'
        # (안전모를 안 쓴 사람을 놓치는 것이, 쓴 사람을 한 번 더 경고하는 것보다 훨씬 위험하니까요)
        names = result.names
        people = []  # (박스, 확신도, 구역 안인지, 위험인지)
        for xyxy, cls, conf in shown:
            inside = in_zone(zone, xyxy)
            risky = names[cls] == "no_helmet" or no_helmet_share(xyxy, history, names) >= args.danger_ratio
            people.append((xyxy, cls, conf, inside, risky))
        danger = any(inside and risky for *_, inside, risky in people)
        draw_zone(frame, zone, danger)
        counts["frames"] += 1
        counts["red"] += sum(inside and risky for *_, inside, risky in people)
        counts["green"] += sum(inside and not risky for *_, inside, risky in people)

        for (x1, y1, x2, y2), cls, conf, inside, risky in people:
            name = names[cls]
            if not inside:
                color, label = GRAY, f"{name} {conf:.2f}"  # 구역 밖: 회색
            elif risky:
                color, label = RED, "DANGER no_helmet"  # 구역 안 + 안전모 안 씀(또는 의심)
            else:
                color, label = GREEN, f"{name} {conf:.2f}"  # 구역 안 + 안전모 씀
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1)
        if key in (27, ord("q"), ord("Q"), ord("ㅂ")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n===== 요약 ({counts['frames']}장) =====")
    print(f"구역 안 빨간 박스(위험): {counts['red']}개 / 구역 안 초록 박스(안전): {counts['green']}개")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="마우스로 위험 구역을 새로 정하기")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="사용할 모델 (기본: 개선 모델 INT8)")
    parser.add_argument("--source", default="0", help="0=웹캠, 또는 영상 파일 경로")
    parser.add_argument("--zone", default=str(ZONE_FILE),
                        help="위험 구역 파일 (영상마다 화면이 다르니 따로 만들면 좋아요. 예: zone_1459.json)")
    parser.add_argument("--once", action="store_true", help="영상 파일을 한 번만 재생 (기본: 끝나면 반복)")
    parser.add_argument("--conf", type=float, default=0.4, help="이 확신도보다 낮은 박스는 무시")
    parser.add_argument("--hold", type=float, default=0.5, help="박스가 사라져도 몇 초 동안 유지할지")
    parser.add_argument("--danger-window", type=float, default=1.0,
                        help="안전 우선 규칙: 최근 몇 초 동안의 판단을 볼지")
    parser.add_argument("--danger-ratio", type=float, default=0.3,
                        help="안전 우선 규칙: no_helmet 표가 이 비율 이상이면 위험으로 봐요 (0~1)")
    parser.add_argument("--smooth", type=float, default=0.5, help="최근 몇 초 결과로 크기·색을 정할지")
    args = parser.parse_args()

    if args.setup:
        setup_zone(args.source, Path(args.zone))
    else:
        run(args)


if __name__ == "__main__":
    main()
