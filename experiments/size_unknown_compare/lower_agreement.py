"""하의 사이즈 일치율 — 입력 조합별로 예측 허리둘레로 고른 사이즈가 실제 허리둘레로 고른 사이즈와 같은 비율.

    python lower_agreement.py <body.csv> <앱 저장소 경로>

- 사이즈 고르는 규칙은 앱 그대로(`app/size_fit.recommend`, 하의는 허리 여유 목표 +3cm, 작으면 벌점).
- 치수표: 저장소의 실제 하의 치수표 2개(무신사 퍼티그 팬츠 S~XXL, 코듀로이 와이드 팬츠 S~L). `sample_shorts`는 예시 값이라 뺀다.
- 분할은 feature_search.py와 같다(성별 층화 5겹 × 3회). 몸 허리 = 사이즈코리아 허리둘레 ÷ 2(단면).
- 같이 보는 값: 실제보다 **작은** 사이즈를 추천한 비율(못 입는 쪽 오류).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8")
DATA, REPO = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, str(REPO / "app"))
sys.path.insert(0, str(Path(__file__).parent))
from size_fit import SizeChart, recommend  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("fs", Path(__file__).parent / "feature_search.py")
sys.argv = [sys.argv[0], DATA, str(Path(__file__).parent / "results")]
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)

CHARTS = ["musinsa_fatigue_pants_forest.json", "musinsa_pants.json"]
MODELS = {
    "키 + 몸무게 (지금 앱)": ["키", "몸무게"],
    "키 + 몸무게 2차 곡선": ["키", "몸무게", "BMI", "키×몸무게", "키²", "몸무게²"],
    "키 + 몸무게 + 나이": ["키", "몸무게", "나이"],
    "키 + 몸무게 + 나이 2차 (6개 항 선택안)": ["키", "몸무게", "나이", "BMI", "몸무게²", "나이²"],
}

df = pd.read_csv(DATA, encoding="utf-8-sig")
sex = df["sex"].to_numpy()
waist = df["waist"].to_numpy()

oof = {name: np.full((3, len(df)), np.nan) for name in MODELS}
for r in range(3):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(df, sex):
        for s in ("F", "M"):
            a = df.iloc[tr]
            a = a[a["sex"] == s]
            b = te[sex[te] == s]
            for name, combo in MODELS.items():
                model = fs.fit(fs.design(a, combo), a["waist"].to_numpy())
                oof[name][r, b] = fs.predict(model, fs.design(df.iloc[b], combo))

lines = []
for chart_file in CHARTS:
    chart = SizeChart.from_dict(json.load(open(REPO / "data" / "size_charts" / chart_file, encoding="utf-8")))
    order = list(chart.sizes)
    pick = lambda w: order.index(recommend(chart, {"waist": round(float(w) / 2, 1)}).best.size)  # noqa: E731
    actual = np.array([pick(w) for w in waist])
    sizes_txt = ", ".join(f"{k} {v['waist']:g}" for k, v in chart.sizes.items())
    lines.append(f"\n## {chart.name} (허리 단면: {sizes_txt})")
    for s, label in (("M", "남"), ("F", "여")):
        mask = sex == s
        dist = np.bincount(actual[mask], minlength=len(order)) / mask.sum()
        lines.append(f"{label} 실제 사이즈 분포: " + ", ".join(f"{order[i]} {dist[i] * 100:.1f}%" for i in range(len(order))))
    lines.append("| 방법 | 남 일치율 | 여 일치율 | 남 작게 추천 | 여 작게 추천 | 허리 오차(cm) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    row = ["최다 사이즈만 추천"]
    for s in ("M", "F"):
        mask = sex == s
        row.append(f"{np.bincount(actual[mask]).max() / mask.sum() * 100:.1f}%")
    for s in ("M", "F"):
        mask = sex == s
        most = np.bincount(actual[mask]).argmax()
        row.append(f"{(most < actual[mask]).mean() * 100:.1f}%")
    lines.append("| " + " | ".join(row) + " | - |")
    for name in MODELS:
        agree, small = {"M": [], "F": []}, {"M": [], "F": []}
        for r in range(3):
            pred = np.array([pick(w) for w in oof[name][r]])
            for s in ("M", "F"):
                mask = sex == s
                agree[s].append((pred[mask] == actual[mask]).mean())
                small[s].append((pred[mask] < actual[mask]).mean())
        mae = np.abs(oof[name] - waist[None]).mean()
        lines.append(f"| {name} | {np.mean(agree['M']) * 100:.1f}% | {np.mean(agree['F']) * 100:.1f}% | "
                     f"{np.mean(small['M']) * 100:.1f}% | {np.mean(small['F']) * 100:.1f}% | {mae:.2f} |")
print("\n".join(lines))
(Path(__file__).parent / "results" / "lower_agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
