"""저장한 계수로 치수를 예측하고 유니클로 치수표로 사이즈를 추천한다.

사용법: python predict_example.py 계수.json body.csv 키cm 몸무게kg 나이
"""
import json
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

COEFFICIENTS, BODY, HEIGHT, WEIGHT, AGE = sys.argv[1:6]
HEIGHT, WEIGHT, AGE = float(HEIGHT), float(WEIGHT), float(AGE)

CHART = {"M": (51.5, 43.0), "L": (54.5, 44.5), "XL": (58.5, 46.5)}
SIZES = list(CHART)
TARGETS = ["chest", "waist", "shoulder_surface", "shoulder"]


def feature_values(height, weight, age):
    return {
        "height": height,
        "weight": weight,
        "bmi": weight / (height / 100) ** 2,
        "height_x_weight": height * weight,
        "height_sq": height**2,
        "weight_sq": weight**2,
        "age": age,
    }


def predict(entry, height, weight, age):
    values = feature_values(height, weight, age)
    return entry["intercept"] + sum(c * values[name] for name, c in entry["coefficients"].items())


def scores(chest, shoulder):
    result = {}
    for size, (garment_chest, garment_shoulder) in CHART.items():
        chest_ease = garment_chest - chest / 2
        shoulder_ease = garment_shoulder - shoulder
        result[size] = (
            abs(chest_ease - 5.0)
            + 3.0 * max(-chest_ease, 0)
            + 0.6 * abs(shoulder_ease - 0.5)
        )
    return result


def report(label, chest, shoulder):
    table = scores(chest, shoulder)
    best = min(table, key=table.get)
    print(f"{label}: 추천 {best} | " + " ".join(f"{s} {v:.1f}" for s, v in table.items()))
    for size, (garment_chest, garment_shoulder) in CHART.items():
        print(f"   {size}: 가슴 여유 {garment_chest - chest / 2:+.1f} / 어깨 여유 {garment_shoulder - shoulder:+.1f}")
    return best


coefficients = json.load(open(COEFFICIENTS, encoding="utf-8"))["terms_age"]["sex"]
df = pd.read_csv(BODY, encoding="utf-8-sig")
df.loc[df["shoulder_surface"] > 100, "shoulder_surface"] = np.nan

report("슬라이드 착용 모델 (가슴 104, 어깨 46)", 104.0, 46.0)

for sex in ["M", "F"]:
    entry = coefficients[sex]
    predicted = {t: predict(entry[t], HEIGHT, WEIGHT, AGE) for t in TARGETS}
    print(f"\n===== 성별 {sex} / 키 {HEIGHT} / 몸무게 {WEIGHT} / 나이 {AGE}")
    print({t: round(v, 1) for t, v in predicted.items()})
    report("어깨사이길이 기준", predicted["chest"], predicted["shoulder_surface"])
    report("어깨너비 기준", predicted["chest"], predicted["shoulder"])

    # 실제 사람들의 예측 오차를 그대로 더해 사이즈가 얼마나 흔들리는지 본다.
    group = df[(df["sex"] == sex) & df["shoulder_surface"].notna()]
    fitted = {
        t: np.array([predict(entry[t], h, w, a) for h, w, a in zip(group["height"], group["weight"], group["age"])])
        for t in ["chest", "shoulder_surface"]
    }
    chest_samples = predicted["chest"] + (group["chest"].to_numpy() - fitted["chest"])
    shoulder_samples = predicted["shoulder_surface"] + (
        group["shoulder_surface"].to_numpy() - fitted["shoulder_surface"]
    )
    picks = [min(s := scores(c, r), key=s.get) for c, r in zip(chest_samples, shoulder_samples)]
    print("오차를 반영한 사이즈 비율:", {s: round(picks.count(s) / len(picks), 3) for s in SIZES})

    # 체격과 나이가 비슷한 실제 측정자와 대조한다.
    similar = group[
        group["height"].between(HEIGHT - 3, HEIGHT + 3)
        & group["weight"].between(WEIGHT - 4, WEIGHT + 4)
        & group["age"].between(AGE - 5, AGE + 5)
    ]
    print(f"비슷한 실제 측정자 {len(similar)}명 (키 ±3cm, 몸무게 ±4kg, 나이 ±5세)")
    if len(similar):
        print(similar[["chest", "waist", "shoulder_surface", "shoulder"]].describe().round(1).loc[["mean", "std", "min", "max"]].to_string())
        actual = [
            min(s := scores(c, r), key=s.get)
            for c, r in zip(similar["chest"], similar["shoulder_surface"])
        ]
        print("이 사람들의 실제 치수로 고른 사이즈:", {s: actual.count(s) for s in SIZES})
