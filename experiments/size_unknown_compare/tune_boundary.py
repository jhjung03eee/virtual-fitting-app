"""사이즈 경계 안내의 점수 차 기준값을 데이터로 정하고 검증한다.

추천 1등과 2등의 점수 차가 기준값보다 작으면 '경계'로 안내한다. 정답은 실제 치수를
같은 점수식에 넣어 고른 사이즈다. 기준값은 바깥 평가 데이터를 보지 않고 학습 데이터
안의 교차검증 예측으로 고른다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

DATA, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["chest", "shoulder_surface"]
SIZES = ["M", "L", "XL"]
CHART = np.array([[51.5, 43.0], [54.5, 44.5], [58.5, 46.5]])
ACCURACY_GOALS = [0.80, 0.85, 0.90]
PRIMARY_GOAL = 0.85
THRESHOLDS = np.round(np.arange(0, 8.01, 0.1), 1)
MIN_CONFIDENT = 50
RESIDUAL_DRAWS = 500


def size_scores(chest, shoulder, chart=CHART):
    chest_ease = chart[:, 0] - np.asarray(chest)[..., None] / 2
    shoulder_ease = chart[:, 1] - np.asarray(shoulder)[..., None]
    return (
        np.abs(chest_ease - 5.0)
        + 3.0 * np.maximum(-chest_ease, 0)
        + 0.6 * np.abs(shoulder_ease - 0.5)
    )


def top_two(scores):
    order = np.argsort(scores, axis=-1)
    first, second = order[..., 0], order[..., 1]
    gap = np.take_along_axis(scores, second[..., None], -1)[..., 0] - np.take_along_axis(
        scores, first[..., None], -1
    )[..., 0]
    return first, second, gap


def inputs(frame, use_age=True):
    h, w = frame["height"].to_numpy(), frame["weight"].to_numpy()
    columns = [h, w, w / (h / 100) ** 2, h * w, h**2, w**2]
    if use_age:
        columns.append(frame["age"].to_numpy())
    return np.column_stack(columns)


def fit(train, use_age=True):
    models = {}
    for sex, group in train.groupby("sex"):
        scaler = StandardScaler().fit(inputs(group, use_age))
        x = scaler.transform(inputs(group, use_age))
        models[sex] = (
            scaler,
            LinearRegression().fit(x, group[TARGETS].to_numpy()),
        )
    return models


def predict(models, frame, use_age=True):
    result = np.zeros((len(frame), 2))
    for sex, (scaler, model) in models.items():
        mask = (frame["sex"] == sex).to_numpy()
        if mask.any():
            result[mask] = model.predict(scaler.transform(inputs(frame[mask], use_age)))
    return result


def cross_predict(frame, seed, use_age=True):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    result = np.zeros((len(frame), 2))
    for train_index, test_index in splitter.split(frame, frame["sex"]):
        result[test_index] = predict(fit(frame.iloc[train_index], use_age), frame.iloc[test_index], use_age)
    return result


def select_threshold(gap, correct, goal):
    for threshold in THRESHOLDS:
        confident = gap >= threshold
        if confident.sum() >= MIN_CONFIDENT and correct[confident].mean() >= goal:
            return float(threshold)
    return None


def evaluate(frame, predicted, threshold, chart=CHART):
    actual = size_scores(frame["chest"], frame["shoulder_surface"], chart).argmin(axis=-1)
    first, second, gap = top_two(size_scores(predicted[:, 0], predicted[:, 1], chart))
    flagged = gap < threshold
    rows = {}
    for label, mask in {"남": frame["sex"].eq("M").to_numpy(), "여": frame["sex"].eq("F").to_numpy()}.items():
        confident = mask & ~flagged
        boundary = mask & flagged
        rows[label] = {
            "경계 비율": boundary.mean() / mask.mean(),
            "단정 일치율": (first == actual)[confident].mean() if confident.any() else np.nan,
            "경계 내 1등 일치율": (first == actual)[boundary].mean() if boundary.any() else np.nan,
            "경계 내 1·2등 포함률": ((first == actual) | (second == actual))[boundary].mean() if boundary.any() else np.nan,
        }
    return rows


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df = df[df["shoulder_surface"] < 100].reset_index(drop=True)
    print("대상", len(df), "명 (어깨사이길이 결측 코드 1건 제외)")
    men = df["sex"].eq("M").to_numpy()
    actual = size_scores(df["chest"], df["shoulder_surface"]).argmin(axis=-1)

    # 1) 기준값별 곡선: 3회 반복의 교차검증 예측을 평균한다.
    curve_rows = []
    oof = [cross_predict(df, seed) for seed in range(3)]
    for threshold in THRESHOLDS[:51]:
        for label, mask in {"남": men, "여": ~men}.items():
            shares, accuracies, pair_rates = [], [], []
            for predicted in oof:
                first, second, gap = top_two(size_scores(predicted[:, 0], predicted[:, 1]))
                confident = mask & (gap >= threshold)
                boundary = mask & (gap < threshold)
                shares.append(boundary.sum() / mask.sum())
                accuracies.append((first == actual)[confident].mean() if confident.any() else np.nan)
                pair_rates.append(((first == actual) | (second == actual))[boundary].mean() if boundary.any() else np.nan)
            curve_rows.append(
                {
                    "threshold": threshold,
                    "group": label,
                    "boundary_share": np.mean(shares),
                    "confident_accuracy": np.nanmean(accuracies),
                    "boundary_pair_coverage": np.nanmean(pair_rates) if np.isfinite(pair_rates).any() else np.nan,
                }
            )
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(OUT / "boundary_curve.csv", index=False, encoding="utf-8-sig")
    print("\n[기준값별 결과]")
    wide = curve[curve["threshold"].isin([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5])]
    print(wide.round(3).to_string(index=False))

    # 2) 중첩 검증: 학습 데이터 안에서 고른 기준값을 처음 보는 사람들에게 적용한다.
    print("\n[중첩 검증: 목표별, 남성 분포로 기준값 선택]")
    nested_rows = []
    for goal in ACCURACY_GOALS:
        chosen = []
        for seed in range(3):
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (train_index, test_index) in enumerate(splitter.split(df, df["sex"])):
                train, test = df.iloc[train_index].reset_index(drop=True), df.iloc[test_index].reset_index(drop=True)
                inner = cross_predict(train, seed=100 + 10 * seed + fold)
                train_actual = size_scores(train["chest"], train["shoulder_surface"]).argmin(axis=-1)
                first, _, gap = top_two(size_scores(inner[:, 0], inner[:, 1]))
                train_men = train["sex"].eq("M").to_numpy()
                threshold = select_threshold(gap[train_men], (first == train_actual)[train_men], goal)
                if threshold is None:
                    raise RuntimeError(f"목표 {goal}를 달성하는 기준값 없음")
                chosen.append(threshold)
                predicted = predict(fit(train), test)
                for label, values in evaluate(test, predicted, threshold).items():
                    nested_rows.append({"goal": goal, "group": label, "threshold": threshold, **values})
        print(f"목표 {goal:.0%}: 선택 기준값 평균 {np.mean(chosen):.2f}, 범위 {min(chosen)}~{max(chosen)}")
    nested = pd.DataFrame(nested_rows)
    nested.to_csv(OUT / "boundary_nested.csv", index=False, encoding="utf-8-sig")
    print(nested.groupby(["goal", "group"]).mean(numeric_only=True).round(3).to_string())

    # 3) 최종 기준값
    finals = {}
    for goal in ACCURACY_GOALS:
        values = []
        for predicted in oof:
            first, _, gap = top_two(size_scores(predicted[:, 0], predicted[:, 1]))
            values.append(select_threshold(gap[men], (first == actual)[men], goal))
        finals[goal] = round(float(np.mean(values)), 1)
    print("\n[전체 교차검증 예측 기준 최종 후보]", finals)
    final = finals[PRIMARY_GOAL]

    # 4) 입력 오차, 나이 미입력에서도 유지되는가
    rng = np.random.default_rng(0)
    print(f"\n[기준값 {final} 유지 확인]")
    noisy_rows = []
    for seed in range(3):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, test_index in splitter.split(df, df["sex"]):
            train, test = df.iloc[train_index], df.iloc[test_index].reset_index(drop=True)
            noisy = test.copy()
            noisy["height"] += rng.normal(0, 1.0, len(test))
            noisy["weight"] += rng.normal(0, 1.5, len(test))
            scenarios = {
                "나이 포함": predict(fit(train), test),
                "입력 오차": predict(fit(train), noisy),
                "나이 없음": predict(fit(train, False), test, False),
            }
            for name, predicted in scenarios.items():
                for label, values in evaluate(test, predicted, final).items():
                    noisy_rows.append({"scenario": name, "group": label, **values})
    print(pd.DataFrame(noisy_rows).groupby(["scenario", "group"]).mean(numeric_only=True).round(3).to_string())

    # 5) 점수 차 규칙과 오차 표본 확률 규칙 비교 (같은 경계 비율에서)
    print("\n[점수 차 규칙 vs 오차 반영 확률 규칙, 남성 일치율]")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    probability = np.zeros(len(df))
    gap_all = np.zeros(len(df))
    first_all = np.zeros(len(df), dtype=int)
    draw_rng = np.random.default_rng(1)
    for train_index, test_index in splitter.split(df, df["sex"]):
        train, test = df.iloc[train_index], df.iloc[test_index]
        models = fit(train)
        predicted = predict(models, test)
        residuals = train[TARGETS].to_numpy() - predict(models, train)
        first, _, gap = top_two(size_scores(predicted[:, 0], predicted[:, 1]))
        gap_all[test_index], first_all[test_index] = gap, first
        for sex in ["M", "F"]:
            sex_rows = np.where(test["sex"].to_numpy() == sex)[0]
            pool = residuals[train["sex"].to_numpy() == sex]
            draws = pool[draw_rng.integers(0, len(pool), RESIDUAL_DRAWS)]
            samples = predicted[sex_rows, None, :] + draws[None, :, :]
            picks = size_scores(samples[..., 0], samples[..., 1]).argmin(axis=-1)
            probability[test_index[sex_rows]] = (picks == first[sex_rows, None]).mean(axis=1)
    correct = first_all == actual
    rows = []
    for share in [0.2, 0.3, 0.4, 0.5, 0.6]:
        gap_cut = np.quantile(gap_all[men], share)
        probability_cut = np.quantile(probability[men], share)
        rows.append(
            {
                "경계 비율": share,
                "점수 차 규칙 일치율": correct[men & (gap_all >= gap_cut)].mean(),
                "확률 규칙 일치율": correct[men & (probability >= probability_cut)].mean(),
            }
        )
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # 6) 치수표가 달라도 같은 기준값이 통하는가 (실제 옷이 아닌 가상 변형)
    print("\n[가상 치수표 변형: 남성, 각 표에서 다시 고른 기준값과 고정 기준값 결과]")
    variants = {
        "원본": CHART,
        "가슴 -2": CHART + [-2, 0],
        "가슴 +2": CHART + [2, 0],
        "어깨 -1": CHART + [0, -1],
        "어깨 +1": CHART + [0, 1],
    }
    predicted = oof[0]
    variant_rows = []
    for name, chart in variants.items():
        chart_actual = size_scores(df["chest"], df["shoulder_surface"], chart).argmin(axis=-1)
        first, _, gap = top_two(size_scores(predicted[:, 0], predicted[:, 1], chart))
        confident = men & (gap >= final)
        variant_rows.append(
            {
                "표": name,
                "다시 고른 기준값": select_threshold(gap[men], (first == chart_actual)[men], PRIMARY_GOAL),
                "고정 기준값 경계 비율": (men & (gap < final)).sum() / men.sum(),
                "고정 기준값 단정 일치율": (first == chart_actual)[confident].mean(),
            }
        )
    print(pd.DataFrame(variant_rows).round(3).to_string(index=False))

    # 7) 구체적인 사례
    print("\n[사례]")
    first, second, gap = top_two(size_scores(np.array([104.0]), np.array([46.0])))
    print("슬라이드 모델 104/46", SIZES[first[0]], SIZES[second[0]], "점수 차", round(float(gap[0]), 2), "경계" if gap[0] < final else "단정")


if __name__ == "__main__":
    main()
