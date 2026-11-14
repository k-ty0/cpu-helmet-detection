"""
SHWD(Safety Helmet Wearing Dataset)를 YOLO 형식으로 바꿔서
지금 데이터셋(datasets/helmet)과 합친 새 데이터셋(datasets/helmet_shwd)을 만드는 코드

왜 필요한가요?
    지금 데이터(공사장 사진)는 사람이 멀리 작게 찍혀 있어서,
    웹캠처럼 가까이 크게 찍힌 얼굴을 잘 못 찾아요.
    SHWD에는 가까이서 찍은 안전모 사진이 많아서 그 부분을 보충해 줘요.

- 출처: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset (MIT 라이선스)
- 먼저 VOC2028.zip 을 내려받아 datasets/shwd/ 에 풀어 두어야 해요.
  (풀면 datasets/shwd/VOC2028/Annotations, JPEGImages, ImageSets 가 생겨요)

만들어지는 것 (datasets/helmet_shwd/):
    train     : 원래 학습 사진 800장 + SHWD 안전모 사진 전부 + SHWD 교실 맨머리 사진 일부(기본 800장)
    val       : 원래 검사 사진 200장 그대로 (공사장 성능이 떨어지지 않았는지 비교하려고)
    val_shwd  : SHWD 검사용 사진 (가까운 얼굴 성능이 좋아졌는지 따로 확인하려고)

실행 방법 (저장소 맨 위 폴더에서):
    .\\.venv\\Scripts\\python.exe 1_train\\prepare_shwd.py
"""

import argparse
import random  # 교실 사진 중 일부를 무작위로 고르는 도구
import shutil  # 파일 복사 도구
import xml.etree.ElementTree as ET  # VOC 형식(.xml) 이름표를 읽는 파이썬 기본 도구
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 저장소 맨 위 폴더
DATASETS_DIR = ROOT / "datasets"
SHWD_DIR = DATASETS_DIR / "shwd" / "VOC2028"  # SHWD 원본 (압축 푼 곳)
HELMET_DIR = DATASETS_DIR / "helmet"  # 지금 쓰는 데이터셋 (Construction-PPE)
OUT_DIR = DATASETS_DIR / "helmet_shwd"  # 새로 만들 합친 데이터셋

# SHWD 이름 -> 우리 번호 (hat=안전모 씀, person=안전모 안 쓴 머리)
# 그 밖의 이름(예: dog 3개)은 버려요
CLASS_MAP = {"hat": 0, "person": 1}
NAMES = ["helmet", "no_helmet"]


def read_voc(xml_path):
    """VOC 이름표(.xml) 하나를 읽어서 YOLO 형식 줄 목록과 들어 있는 클래스 이름들을 돌려줘요.

    VOC 형식: 박스를 픽셀 좌표 (왼쪽 xmin, 위 ymin, 오른쪽 xmax, 아래 ymax) 로 적어요.
    YOLO 형식: "번호 중심x 중심y 너비 높이" 를 사진 크기로 나눈 0~1 사이 값으로 적어요.
    """
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)
    lines, names = [], set()
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            continue
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        # 박스가 사진 밖으로 삐져나간 경우 사진 안으로 잘라 줘요
        x1, x2 = max(0.0, x1), min(w, x2)
        y1, y2 = max(0.0, y1), min(h, y2)
        if x2 - x1 < 1 or y2 - y1 < 1:  # 크기가 거의 없는 잘못된 박스는 버려요
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h  # 중심 좌표 (0~1)
        bw, bh = (x2 - x1) / w, (y2 - y1) / h  # 너비, 높이 (0~1)
        lines.append(f"{CLASS_MAP[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        names.add(name)
    return lines, names


def copy_split(src_dir, split_src, split_dst):
    """지금 데이터셋(helmet)의 사진·이름표를 새 데이터셋으로 그대로 복사해요."""
    for kind in ("images", "labels"):
        src = src_dir / kind / split_src
        dst = OUT_DIR / kind / split_dst
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.suffix != ".cache":  # .cache 는 ultralytics 가 자동으로 만드는 임시 파일이라 빼요
                shutil.copy2(f, dst / f.name)
    return len(list((OUT_DIR / "images" / split_dst).iterdir()))


def add_shwd(ids, split_dst):
    """SHWD 사진들(ids)을 YOLO 형식으로 바꿔서 새 데이터셋의 split_dst 에 넣어요."""
    img_dst = OUT_DIR / "images" / split_dst
    lbl_dst = OUT_DIR / "labels" / split_dst
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for i in ids:
        lines, _ = read_voc(SHWD_DIR / "Annotations" / f"{i}.xml")
        new = f"shwd_{i}"  # 원래 사진과 이름이 겹치지 않게 앞에 shwd_ 를 붙여요
        shutil.copy2(SHWD_DIR / "JPEGImages" / f"{i}.jpg", img_dst / f"{new}.jpg")
        (lbl_dst / f"{new}.txt").write_text("\n".join(lines))


def pick(split_file, n_person, seed):
    """SHWD 나누기 목록(trainval/test)에서 사용할 사진을 골라요.

    - 안전모(hat)가 하나라도 있는 사진: 전부 사용
    - 맨머리(person)만 있는 사진(대부분 교실): 너무 많아서 n_person 장만 무작위로 사용
    """
    ids = (SHWD_DIR / "ImageSets" / "Main" / split_file).read_text().split()
    with_hat, person_only = [], []
    for i in ids:
        _, names = read_voc(SHWD_DIR / "Annotations" / f"{i}.xml")
        if "hat" in names:
            with_hat.append(i)
        elif "person" in names:
            person_only.append(i)
    random.Random(seed).shuffle(person_only)  # seed 를 고정하면 다시 실행해도 같은 사진이 골라져요
    return with_hat, person_only[:n_person]


def write_yaml(file_name, val_split):
    """YOLO 가 읽을 설정 파일을 만들어요. (val_split: 어떤 사진으로 채점할지)"""
    text = (
        f"path: {OUT_DIR.as_posix()}\n"
        "train: images/train\n"
        f"val: images/{val_split}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    )
    (OUT_DIR / file_name).write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", type=int, default=800, help="학습에 넣을 SHWD 교실 맨머리 사진 수")
    parser.add_argument("--val-person", type=int, default=200, help="val_shwd 에 넣을 교실 맨머리 사진 수")
    parser.add_argument("--seed", type=int, default=0, help="무작위로 고를 때 쓰는 번호 (같으면 같은 사진)")
    args = parser.parse_args()

    if not SHWD_DIR.exists():
        raise SystemExit(f"SHWD 가 없어요: {SHWD_DIR}\nVOC2028.zip 을 datasets/shwd/ 에 풀어 주세요.")
    if not (HELMET_DIR / "images" / "train").exists():
        raise SystemExit("datasets/helmet 이 없어요. 먼저 1_train/download_dataset.py 를 실행하세요.")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)  # 예전에 만든 게 있으면 지우고 새로 만들어요 (datasets/ 안의 생성물만)

    # 1) 원래 데이터 복사
    n_train = copy_split(HELMET_DIR, "train", "train")
    n_val = copy_split(HELMET_DIR, "val", "val")
    print(f"원래 데이터: 학습 {n_train}장, 검사 {n_val}장")

    # 2) SHWD 학습용(trainval) -> train 에 추가
    hat, person = pick("trainval.txt", args.person, args.seed)
    add_shwd(hat + person, "train")
    print(f"SHWD 학습용 추가: 안전모 사진 {len(hat)}장 + 교실 맨머리 사진 {len(person)}장")

    # 3) SHWD 검사용(test) -> val_shwd (학습에는 절대 안 씀)
    hat_t, person_t = pick("test.txt", args.val_person, args.seed)
    add_shwd(hat_t + person_t, "val_shwd")
    print(f"SHWD 검사용(val_shwd): 안전모 사진 {len(hat_t)}장 + 교실 맨머리 사진 {len(person_t)}장")

    # 4) 설정 파일 2개
    write_yaml("data.yaml", "val")  # 학습용: 원래 검사 사진으로 채점 (예전 모델과 비교 가능)
    write_yaml("data_shwd_val.yaml", "val_shwd")  # SHWD 검사 사진으로 따로 채점할 때
    total = len(list((OUT_DIR / "images" / "train").iterdir()))
    print(f"완료: 학습 사진 총 {total}장 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
