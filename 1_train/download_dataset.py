"""
안전모 데이터셋(Construction-PPE)을 내려받는 코드

- 출처: Ultralytics Construction-PPE 데이터셋 (CC BY 4.0)
  https://docs.ultralytics.com/datasets/detect/construction-ppe
- 내려받은 파일은 저장소 맨 위의 datasets/ 폴더에 저장돼요.
  (datasets/ 는 .gitignore 에 들어 있어서 커밋되지 않아요)

실행 방법 (저장소 맨 위 폴더에서):
    .\\.venv\\Scripts\\python.exe 1_train\\download_dataset.py
"""

import urllib.request  # 인터넷에서 파일을 내려받는 파이썬 기본 도구
import zipfile  # 압축(zip) 파일을 푸는 파이썬 기본 도구
from pathlib import Path  # 파일/폴더 경로를 쉽게 다루는 도구

# 이 파일(download_dataset.py)이 있는 폴더의 한 칸 위 = 저장소 맨 위 폴더
ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"  # 데이터셋을 둘 폴더

# Ultralytics 공식 GitHub에 올라와 있는 데이터셋 주소
# (ultralytics 패키지 안의 construction-ppe.yaml 에 적힌 주소와 같아요)
URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
ZIP_PATH = DATASETS_DIR / "construction-ppe.zip"  # 내려받은 압축 파일 이름
RAW_DIR = DATASETS_DIR / "construction-ppe"  # 압축을 푼 원본 데이터셋 폴더


def download():
    """압축 파일을 내려받아요. 이미 있으면 건너뛰어요."""
    DATASETS_DIR.mkdir(exist_ok=True)  # datasets 폴더가 없으면 만들기
    if ZIP_PATH.exists():
        print(f"이미 내려받았어요: {ZIP_PATH}")
        return
    print(f"내려받는 중... {URL}")
    urllib.request.urlretrieve(URL, ZIP_PATH)  # 주소에서 파일을 받아 저장
    print(f"내려받기 완료: {ZIP_PATH}")


def safe_extract():
    """압축을 안전하게 풀어요.

    보안: 압축 파일 안에 '../' 같은 경로가 숨어 있으면
    datasets 폴더 바깥에 파일이 만들어질 수 있어요(Zip Slip 공격).
    그래서 풀기 전에 모든 경로가 datasets 폴더 안인지 검사해요.
    """
    if RAW_DIR.exists():
        print(f"이미 압축을 풀었어요: {RAW_DIR}")
        return
    with zipfile.ZipFile(ZIP_PATH) as zf:
        base = DATASETS_DIR.resolve()
        for name in zf.namelist():
            target = (base / name).resolve()  # 풀었을 때 실제로 생길 위치
            if not target.is_relative_to(base):  # datasets 폴더 밖이면 멈추기
                raise RuntimeError(f"위험한 경로가 있어서 멈췄어요: {name}")
        zf.extractall(base)  # 검사를 통과하면 압축 풀기
    print(f"압축 풀기 완료: {RAW_DIR}")


HELMET_DIR = DATASETS_DIR / "helmet"  # helmet / no_helmet 두 가지만 남긴 새 데이터셋

# 원본 번호 -> 새 번호 (원본: 0=helmet, 6=no_helmet / 나머지 장갑·조끼 등은 버려요)
CLASS_MAP = {0: 0, 6: 1}
NEW_NAMES = ["helmet", "no_helmet"]  # 새 번호 0, 1 의 이름


def make_helmet_dataset():
    """원본에서 helmet / no_helmet 이름표만 골라 새 데이터셋을 만들어요."""
    for split in ["train", "val"]:  # train=학습용, val=검사용
        img_out = HELMET_DIR / "images" / split
        lbl_out = HELMET_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img in (RAW_DIR / "images" / split).glob("*.jpg"):
            # 사진은 그대로 복사
            (img_out / img.name).write_bytes(img.read_bytes())

            # 같은 이름의 이름표(.txt) 파일을 읽어서 필요한 줄만 남기기
            src_label = RAW_DIR / "labels" / split / (img.stem + ".txt")
            new_lines = []
            if src_label.exists():
                for line in src_label.read_text().splitlines():
                    parts = line.split()  # "번호 x y w h" 를 칸별로 나누기
                    if parts and int(parts[0]) in CLASS_MAP:
                        parts[0] = str(CLASS_MAP[int(parts[0])])  # 번호 바꾸기
                        new_lines.append(" ".join(parts))
            # 남은 줄이 없어도 빈 파일을 만들어요 (= "이 사진엔 대상 없음" 이라는 뜻)
            (lbl_out / (img.stem + ".txt")).write_text("\n".join(new_lines))

        print(f"{split}: 사진 {len(list(img_out.glob('*.jpg')))}장 준비 완료")

    # YOLO 가 읽을 설정 파일(data.yaml): 데이터 위치와 클래스 이름을 알려줘요
    yaml_text = (
        f"path: {HELMET_DIR.as_posix()}\n"  # 데이터셋 폴더 (이 컴퓨터의 실제 경로)
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NEW_NAMES))
    )
    (HELMET_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"설정 파일 저장: {HELMET_DIR / 'data.yaml'}")


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만 아래가 실행돼요
    download()
    safe_extract()
    make_helmet_dataset()
