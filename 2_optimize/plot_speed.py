"""
모델 형식별 속도 비교 그래프를 그려서 images/speed_comparison.png 로 저장하는 코드

아래 RESULTS 의 숫자는 benchmark.py 결과(사진 200장 × 3회, 중앙값)를 옮겨 적은 거예요.
다시 측정하면 RESULTS 숫자만 바꾸고 이 파일을 다시 실행하세요.

실행 방법 (저장소 맨 위 폴더에서):
    .\\.venv\\Scripts\\python.exe 2_optimize\\plot_speed.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장하는 모드
import matplotlib.pyplot as plt  # 그래프를 그리는 도구

ROOT = Path(__file__).resolve().parent.parent  # 저장소 맨 위 폴더
OUT = ROOT / "images" / "speed_comparison.png"

# (모델 이름, 1장 추론 시간 중앙값 ms, PyTorch 대비 배수) — benchmark.py 결과, AMD Ryzen 5 5600
# 배수는 benchmark.py 가 반올림 전 시간으로 계산한 값을 그대로 적어요.
# (반올림된 ms 로 다시 나누면 3.09 → 3.07 처럼 README 와 달라질 수 있어서)
RESULTS = [
    ("PyTorch (원래 모델)", 25.5, 1.00),
    ("ONNX", 21.1, 1.21),
    ("OpenVINO", 15.6, 1.64),
    ("OpenVINO INT8", 8.3, 3.09),
]

# 색깔 (차분한 한 가지 색 + 글자는 회색 계열로)
BAR = "#2a78d6"  # 막대 색 (파랑)
INK = "#0b0b0b"  # 진한 글자
MUTED = "#898781"  # 흐린 글자 (축, 설명)
GRID = "#e1e0d9"  # 눈금선
SURFACE = "#fcfcfb"  # 배경


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 기본 한글 글꼴 (없으면 한글이 네모로 깨져요)
    names = [n for n, _, _ in RESULTS]
    times = [t for _, t, _ in RESULTS]
    speedups = [s for _, _, s in RESULTS]

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(RESULTS))[::-1]  # 위에서부터 PyTorch → INT8 순서로 보이게
    ax.barh(list(y), times, height=0.55, color=BAR)

    # 막대 끝에 "시간 · 몇 배 빠른지" 적기
    for yi, t, s in zip(y, times, speedups):
        speedup = "기준" if s == 1.0 else f"{s:.2f}배 빠름"
        ax.text(t + 0.4, yi, f"{t:.1f} ms · {speedup}", va="center", fontsize=10, color=INK)

    ax.set_yticks(list(y), names, fontsize=10, color=INK)
    ax.set_xlim(0, max(times) * 1.45)  # 오른쪽에 글자 들어갈 자리 남기기
    ax.set_xlabel("사진 1장 추론 시간 (ms, 짧을수록 빠름)", fontsize=9, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)  # 세로 눈금선만 연하게
    ax.set_axisbelow(True)  # 눈금선을 막대 뒤로
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)  # 테두리 선 지우기
    ax.spines["bottom"].set_color(GRID)

    ax.set_title("모델 형식별 CPU 속도 (AMD Ryzen 5 5600, 검사용 사진 200장 × 3회 중앙값)",
                 fontsize=11, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"저장했어요: {OUT}")


if __name__ == "__main__":
    main()
