"""발표용 그림: 입력 조합별 평균 오차(가슴·허리·어깨사이길이 평균, cm). 고른 조합만 강조색."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(sys.argv[1])
table = pd.read_csv(RESULTS / "feature_search_all.csv", encoding="utf-8-sig")
mae = pd.read_csv(RESULTS / "mae.csv", encoding="utf-8-sig")
mae = mae[mae.group == "전체"].pivot(index="model", columns="target", values="mae_cm")
mlp = float(mae.loc["mlp_age_ensemble", ["가슴둘레", "허리둘레", "어깨사이길이(표면)"]].mean())

CHOSEN = "키 + 몸무게 + 나이 + BMI + 몸무게² + 나이²"
rows = [
    ("키만", "키"),
    ("몸무게만", "몸무게"),
    ("키 + 몸무게  (지금 앱)", "키 + 몸무게"),
    ("나이 없이 최고 (5개 항)", "몸무게 + BMI + 키×몸무게 + 키² + 몸무게²"),
    ("키 + 몸무게 + 나이", "키 + 몸무게 + 나이"),
    ("키 + 나이 + BMI  (3개 항 최고)", "키 + 나이 + BMI"),
    ("선택: 키·몸무게·나이·BMI·몸무게²·나이²", CHOSEN),
]
labels, values = [], []
for label, combo in rows:
    labels.append(label)
    values.append(float(table.loc[table["조합"] == combo, "점수"].iloc[0]))
labels.append("참고: MLP (성별·키·몸무게·나이)")
values.append(mlp)

SURFACE, INK, INK2, ACCENT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#c9c8c2"
plt.rcParams["font.family"] = "Malgun Gothic"
fig, ax = plt.subplots(figsize=(13, 6.4), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
y = list(range(len(labels)))[::-1]
colors = [ACCENT if l.startswith("선택") else MUTED for l in labels]
ax.barh(y, values, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
for yi, v, l in zip(y, values, labels):
    ax.text(v + 0.05, yi, f"{v:.2f}", va="center", ha="left", fontsize=13,
            color=INK, fontweight="bold" if l.startswith("선택") else "normal")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=13, color=INK)
for tick, l in zip(ax.get_yticklabels(), labels):
    if l.startswith("선택"):
        tick.set_fontweight("bold")
    if l.startswith("참고"):
        tick.set_color(INK2)
ax.set_xlim(0, max(values) * 1.12)
ax.set_xlabel("평균 오차 (cm, 가슴둘레·허리둘레·어깨사이길이 평균) — 낮을수록 정확", fontsize=12, color=INK2)
ax.tick_params(axis="x", colors=INK2, labelsize=11)
ax.tick_params(axis="y", length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.grid(axis="x", color="#ecebe8", linewidth=1)
ax.set_axisbelow(True)
ax.set_title("선형 회귀 입력 조합 255개 비교 — 사이즈코리아 8차 5,092명, 5겹 교차검증 × 3회",
             fontsize=15, color=INK, loc="left", pad=14)
fig.tight_layout()
out = RESULTS / "feature_search_chart.png"
fig.savefig(out, facecolor=SURFACE)
print(out)
