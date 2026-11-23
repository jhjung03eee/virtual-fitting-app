"""선형 회귀 입력 조합 실험 — 후보 항 8개로 만들 수 있는 조합 255개를 같은 분할로 비교해 하나를 고른다.

    python feature_search.py <body.csv> <results 폴더>

후보 항: 키, 몸무게, 나이, BMI, 키×몸무게, 키², 몸무게², 나이²
- 성별마다, 부위마다 최소제곱 선형 회귀(공식 해). 입력은 표준화한다.
- 분할은 run_comparison.py와 같다: 성별 층화 5겹 × 3회 반복(random_state = 반복 번호).
- 평가: 부위별 평균 절대오차(cm), 자기 신고 오차(키 ±1cm, 몸무게 ±1.5kg)를 넣었을 때의 오차,
  유니클로 반팔(M·L·XL) 사이즈 일치율.

**고르는 규칙 (결과를 보기 전에 정함)**
1. 점수 = 가슴둘레·허리둘레·어깨사이길이 오차의 평균(cm). 앱이 상의(가슴·어깨)와 하의(허리)에 쓰는 세 부위다.
2. 점수가 가장 낮은 조합을 기준으로, 사람 단위 재표집(2000번) 95% 구간이 0을 포함하는 조합은 "차이 없음"으로 본다.
3. 차이 없는 조합 중 **항이 가장 적은 것**을 고른다(같으면 점수가 낮은 것). 항이 많을수록 입력 오차에 흔들리고 설명하기 어렵다.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8")

DATA, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["chest", "bust", "waist", "shoulder_surface", "shoulder"]
KO = {"chest": "가슴둘레", "bust": "젖가슴둘레", "waist": "허리둘레",
      "shoulder_surface": "어깨사이길이", "shoulder": "어깨너비"}
SCORE_TARGETS = ["chest", "waist", "shoulder_surface"]
FEATURES = {
    "키": lambda f: f["height"],
    "몸무게": lambda f: f["weight"],
    "나이": lambda f: f["age"],
    "BMI": lambda f: f["weight"] / (f["height"] / 100) ** 2,
    "키×몸무게": lambda f: f["height"] * f["weight"],
    "키²": lambda f: f["height"] ** 2,
    "몸무게²": lambda f: f["weight"] ** 2,
    "나이²": lambda f: f["age"] ** 2,
}
NAMES = list(FEATURES)
REPEATS, FOLDS = 3, 5
HEIGHT_NOISE_CM, WEIGHT_NOISE_KG = 1.0, 1.5

# run_comparison.py 와 같은 유니클로 치수표(단면 cm)와 점수식
CHART = {"M": (51.5, 43.0), "L": (54.5, 44.5), "XL": (58.5, 46.5)}


def recommend(chest_circumference, shoulder):
    scores = []
    for garment_chest, garment_shoulder in CHART.values():
        chest_ease = garment_chest - np.asarray(chest_circumference) / 2
        shoulder_ease = garment_shoulder - np.asarray(shoulder)
        scores.append(np.abs(chest_ease - 5) + 3 * np.maximum(-chest_ease, 0) + 0.6 * np.abs(shoulder_ease - 0.5))
    return np.stack(scores, axis=-1).argmin(axis=-1)


def design(frame, combo):
    return np.column_stack([FEATURES[n](frame).to_numpy(float) for n in combo])


def fit(x, y):
    """표준화한 입력으로 최소제곱. → (평균, 표준편차, 계수[절편 포함])"""
    mu, sd = x.mean(axis=0), x.std(axis=0)
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    beta, *_ = np.linalg.lstsq(z, y, rcond=None)
    return mu, sd, beta


def predict(model, x):
    mu, sd, beta = model
    return np.column_stack([np.ones(len(x)), (x - mu) / sd]) @ beta


def bootstrap_ci(diff, draws=2000, seed=0):
    diff = diff[~np.isnan(diff)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (draws, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df.loc[df["shoulder_surface"] > 100, "shoulder_surface"] = np.nan   # 결측 코드 999.9
    n = len(df)
    truth = df[TARGETS].to_numpy()
    sex = df["sex"].to_numpy()

    splits = []
    noisy_inputs = []
    for repeat in range(REPEATS):
        splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=repeat)
        for fold, (tr, te) in enumerate(splitter.split(df, sex)):
            splits.append((repeat, tr, te))
            rng = np.random.default_rng(1000 * repeat + fold)
            noisy = df.iloc[te].copy()
            noisy["height"] += rng.normal(0, HEIGHT_NOISE_CM, len(te))
            noisy["weight"] += rng.normal(0, WEIGHT_NOISE_KG, len(te))
            noisy_inputs.append(noisy)

    combos = [c for k in range(1, len(NAMES) + 1) for c in itertools.combinations(NAMES, k)]
    print("조합", len(combos), "개 · 분할", len(splits), "개")
    men = sex == "M"
    known = ~np.isnan(truth[:, TARGETS.index("shoulder_surface")])
    actual = recommend(truth[:, 0], truth[:, TARGETS.index("shoulder_surface")])
    rows, person_scores = [], {}
    for combo in combos:
        oof = np.full((REPEATS, n, len(TARGETS)), np.nan)
        oof_noisy = np.full_like(oof, np.nan)
        for (repeat, tr, te), noisy in zip(splits, noisy_inputs):
            train, test = df.iloc[tr], df.iloc[te]
            for s in ("F", "M"):
                trs, tes = train[train["sex"] == s], (test["sex"] == s).to_numpy()
                x = design(trs, combo)
                for k, target in enumerate(TARGETS):
                    ok = trs[target].notna().to_numpy()
                    model = fit(x[ok], trs[target].to_numpy()[ok])
                    oof[repeat, te[tes], k] = predict(model, design(test[tes], combo))
                    oof_noisy[repeat, te[tes], k] = predict(model, design(noisy[tes], combo))
        err = np.abs(oof - truth[None])
        err_noisy = np.abs(oof_noisy - truth[None])
        row = {"조합": " + ".join(combo), "항 수": len(combo), "나이 필요": any("나이" in c for c in combo)}
        for k, target in enumerate(TARGETS):
            row[KO[target]] = float(np.nanmean(err[:, :, k]))
            row[KO[target] + "_남"] = float(np.nanmean(err[:, men, k]))
            row[KO[target] + "_여"] = float(np.nanmean(err[:, ~men, k]))
        score_idx = [TARGETS.index(t) for t in SCORE_TARGETS]
        per_person = np.nanmean(err[:, :, score_idx].mean(axis=0), axis=1)        # 반복 평균 → 세 부위 평균
        person_scores[row["조합"]] = per_person
        row["점수"] = float(np.nanmean(per_person))
        row["점수_반복편차"] = float(np.std([np.nanmean(err[r][:, score_idx]) for r in range(REPEATS)]))
        row["점수_입력오차"] = float(np.nanmean(err_noisy[:, :, score_idx]))
        agree = []
        for r in range(REPEATS):
            pred = recommend(oof[r, :, 0], oof[r, :, TARGETS.index("shoulder_surface")])
            agree.append((pred[men & known] == actual[men & known]).mean())
        row["남 상의 일치율"] = float(np.mean(agree))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("점수").reset_index(drop=True)

    best = table.iloc[0]["조합"]
    cis = {}
    for name in table["조합"]:
        cis[name] = (0.0, 0.0, 0.0) if name == best else bootstrap_ci(person_scores[name] - person_scores[best])
    table["최고 대비 차이"] = [cis[c][0] for c in table["조합"]]
    table["차이 95% 하한"] = [cis[c][1] for c in table["조합"]]
    table["차이 95% 상한"] = [cis[c][2] for c in table["조합"]]
    table["최고와 차이 없음"] = table["차이 95% 하한"] <= 0
    same = table[table["최고와 차이 없음"]].sort_values(["항 수", "점수"])
    chosen = same.iloc[0]["조합"]
    no_age = table[~table["나이 필요"]].iloc[0]["조합"]
    table.to_csv(OUT / "feature_search_all.csv", index=False, encoding="utf-8-sig")

    # 발표용 비교 행: 대표 조합
    picks = ["키", "몸무게", "키 + 몸무게", "키 + 몸무게 + 나이", "키 + 몸무게 + BMI + 키×몸무게 + 키² + 몸무게²",
             "키 + 몸무게 + 나이 + BMI + 키×몸무게 + 키² + 몸무게²", " + ".join(NAMES), best, chosen, no_age]
    seen, lines = set(), []
    cols = ["조합", "항 수", "가슴둘레", "허리둘레", "어깨사이길이", "점수", "점수_입력오차", "남 상의 일치율", "최고 대비 차이"]
    for p in picks:
        if p in seen:
            continue
        seen.add(p)
        lines.append(table[table["조합"] == p][cols])
    summary = pd.concat(lines)
    print("\n[대표 조합] 오차 cm (5겹×3회, 사람마다 검증 때만 예측)")
    print(summary.round(3).to_string(index=False))
    print("\n[점수 상위 10개]")
    print(table.head(10)[cols + ["차이 95% 상한"]].round(3).to_string(index=False))
    print(f"\n최고 점수: {best}")
    print(f"최고와 차이 없는 조합 {int(table['최고와 차이 없음'].sum())}개 중 항이 가장 적은 것 → 선택: {chosen}")
    print(f"나이 없이 가장 좋은 조합: {no_age}")

    # 고른 조합의 계수(전체 데이터) — 앱에 넣을 값. 입력 단위: 키 cm, 몸무게 kg, 나이 세
    combo = chosen.split(" + ")
    coef = {"features": combo, "sex": {}}
    for s in ("F", "M"):
        g = df[df["sex"] == s]
        coef["sex"][s] = {}
        for target in TARGETS:
            ok = g[target].notna()
            x, y = design(g[ok], combo), g.loc[ok, target].to_numpy()
            mu, sd, beta = fit(x, y)
            raw = beta[1:] / sd
            intercept = beta[0] - float((raw * mu).sum())
            resid = y - (x @ raw + intercept)
            coef["sex"][s][target] = {"intercept": float(intercept), "coefficients": dict(zip(combo, raw.tolist())),
                                     "residual_sd": float(resid.std())}
    (OUT / "chosen_coefficients.json").write_text(json.dumps(coef, ensure_ascii=False, indent=2), encoding="utf-8")
    summary.to_csv(OUT / "feature_search_summary.csv", index=False, encoding="utf-8-sig")
    print("\n저장:", OUT / "feature_search_all.csv", OUT / "feature_search_summary.csv", OUT / "chosen_coefficients.json")


if __name__ == "__main__":
    main()
