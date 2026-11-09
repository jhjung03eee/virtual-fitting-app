"""성별·키·몸무게로 신체 치수를 예측하는 선형 회귀와 MLP를 같은 분할로 비교한다.

평가는 치수 오차(cm)와, 유니클로 크루넥 반팔 치수표에서 실제 치수로 고른 사이즈와의
일치율로 한다. MLP 설정은 바깥 검증 데이터를 보지 않고 학습 데이터 안에서만 고른다.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA = sys.argv[1]
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["chest", "bust", "waist", "shoulder_surface", "shoulder"]
LABELS = {
    "chest": "가슴둘레",
    "bust": "젖가슴둘레",
    "waist": "허리둘레",
    "shoulder_surface": "어깨사이길이(표면)",
    "shoulder": "어깨너비(직선)",
}
REPEATS = 3
FOLDS = 5
MLP_SEEDS = 5
MLP_GRID = [
    {"hidden_layer_sizes": hidden, "alpha": alpha}
    for hidden in [(16,), (32, 32), (64, 64)]
    for alpha in [1e-4, 1e-2]
]
HEIGHT_NOISE_CM = 1.0  # 자기 신고 오차 가정 (Bartol et al. 2022와 같은 크기)
WEIGHT_NOISE_KG = 1.5

# 중간발표 9번 슬라이드와 같은 유니클로 치수표(단면 cm)와 점수식
CHART = {"M": (51.5, 43.0), "L": (54.5, 44.5), "XL": (58.5, 46.5)}
SIZES = list(CHART)
CHEST_TARGET, SHOULDER_TARGET, SHOULDER_WEIGHT, SMALL_PENALTY = 5.0, 0.5, 0.6, 3.0


def size_scores(chest_circumference, shoulder):
    chest_half = np.asarray(chest_circumference) / 2
    scores = []
    for garment_chest, garment_shoulder in CHART.values():
        chest_ease = garment_chest - chest_half
        shoulder_ease = garment_shoulder - np.asarray(shoulder)
        scores.append(
            np.abs(chest_ease - CHEST_TARGET)
            + SMALL_PENALTY * np.maximum(-chest_ease, 0)
            + SHOULDER_WEIGHT * np.abs(shoulder_ease - SHOULDER_TARGET)
        )
    return np.stack(scores, axis=-1)


def recommend(chest_circumference, shoulder):
    return size_scores(chest_circumference, shoulder).argmin(axis=-1)


def check_scoring():
    """슬라이드의 계산 예시를 그대로 재현하는지 확인한다."""
    got = size_scores([104.0], [46.0])[0].round(1).tolist()
    if got != [9.1, 3.7, 1.5]:
        raise AssertionError(f"슬라이드 점수와 다름: {got}")


def linear_inputs(frame, terms):
    h, w = frame["height"].to_numpy(), frame["weight"].to_numpy()
    columns = [h, w]
    if terms in ("terms", "terms_age"):
        columns += [w / (h / 100) ** 2, h * w, h**2, w**2]
    if terms in ("age", "terms_age"):
        columns += [frame["age"].to_numpy()]
    return np.column_stack(columns)


LINEAR_FEATURES = {
    "base": ["height", "weight"],
    "terms": ["height", "weight", "bmi", "height_x_weight", "height_sq", "weight_sq"],
    "terms_age": ["height", "weight", "bmi", "height_x_weight", "height_sq", "weight_sq", "age"],
}


def fit_predict_linear(train, test, terms):
    """성별별 선형 회귀. 항이 많아도 공식 해를 쓰고, 수치 안정을 위해 입력을 표준화한다."""
    result = np.full((len(test), len(TARGETS)), np.nan)
    for sex in ["F", "M"]:
        tr, te = train[train["sex"] == sex], test["sex"].to_numpy() == sex
        scaler = StandardScaler().fit(linear_inputs(tr, terms))
        for k, target in enumerate(TARGETS):
            known = tr[target].notna().to_numpy()
            model = LinearRegression().fit(
                scaler.transform(linear_inputs(tr, terms))[known], tr[target].to_numpy()[known]
            )
            result[te, k] = model.predict(scaler.transform(linear_inputs(test[te], terms)))
    return result


def predict_mean(train, test):
    result = np.full((len(test), len(TARGETS)), np.nan)
    for sex in ["F", "M"]:
        te = test["sex"].to_numpy() == sex
        result[te] = train.loc[train["sex"] == sex, TARGETS].mean().to_numpy()
    return result


def mlp_inputs(frame, use_age=False):
    columns = [(frame["sex"] == "M").astype(float), frame["height"], frame["weight"]]
    if use_age:
        columns.append(frame["age"])
    return np.column_stack(columns)


def fit_mlps(train, params, seeds, use_age=False):
    known = train[TARGETS].notna().all(axis=1)
    fit_frame = train[known]
    x_scaler = StandardScaler().fit(mlp_inputs(fit_frame, use_age))
    y_scaler = StandardScaler().fit(fit_frame[TARGETS])
    models = []
    for seed in seeds:
        model = MLPRegressor(
            **params,
            activation="relu",
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=30,
            learning_rate_init=1e-3,
            random_state=seed,
        )
        model.fit(x_scaler.transform(mlp_inputs(fit_frame, use_age)), y_scaler.transform(fit_frame[TARGETS]))
        models.append(model)
    return x_scaler, y_scaler, models, use_age


def predict_mlps(bundle, frame):
    x_scaler, y_scaler, models, use_age = bundle
    x = x_scaler.transform(mlp_inputs(frame, use_age))
    return np.stack([y_scaler.inverse_transform(m.predict(x)) for m in models])


def tune_mlp(train, seed, use_age=False):
    """학습 데이터 안에서 3겹 검증으로 설정을 고른다."""
    best, best_score = None, np.inf
    splitter = KFold(n_splits=3, shuffle=True, random_state=seed)
    known = train[train[TARGETS].notna().all(axis=1)]
    y_std = known[TARGETS].std().to_numpy()
    for params in MLP_GRID:
        scores = []
        for inner_train, inner_val in splitter.split(known):
            bundle = fit_mlps(known.iloc[inner_train], params, [seed], use_age)
            prediction = predict_mlps(bundle, known.iloc[inner_val])[0]
            scores.append(
                mean_squared_error(
                    known.iloc[inner_val][TARGETS].to_numpy() / y_std, prediction / y_std
                )
            )
        if np.mean(scores) < best_score:
            best, best_score = params, float(np.mean(scores))
    return best


def run_fold(df, repeat, fold, train_index, test_index):
    train, test = df.iloc[train_index], df.iloc[test_index]
    rng = np.random.default_rng(1000 * repeat + fold)
    noisy = test.copy()
    noisy["height"] = test["height"] + rng.normal(0, HEIGHT_NOISE_CM, len(test))
    noisy["weight"] = test["weight"] + rng.normal(0, WEIGHT_NOISE_KG, len(test))

    params = tune_mlp(train, seed=repeat * 100 + fold)
    seeds = [repeat * 100 + fold + s for s in range(MLP_SEEDS)]
    bundle = fit_mlps(train, params, seeds)
    single_runs = predict_mlps(bundle, test)
    age_params = tune_mlp(train, seed=repeat * 100 + fold, use_age=True)
    age_bundle = fit_mlps(train, age_params, seeds, use_age=True)
    age_runs = predict_mlps(age_bundle, test)
    predictions = {
        "mean": predict_mean(train, test),
        "linear": fit_predict_linear(train, test, "base"),
        "linear_terms": fit_predict_linear(train, test, "terms"),
        "linear_age": fit_predict_linear(train, test, "age"),
        "linear_terms_age": fit_predict_linear(train, test, "terms_age"),
        "mlp_single": single_runs[0],
        "mlp_ensemble": single_runs.mean(axis=0),
        "mlp_age_single": age_runs[0],
        "mlp_age_ensemble": age_runs.mean(axis=0),
        "linear|noisy": fit_predict_linear(train, noisy, "base"),
        "linear_terms|noisy": fit_predict_linear(train, noisy, "terms"),
        "linear_terms_age|noisy": fit_predict_linear(train, noisy, "terms_age"),
        "mlp_ensemble|noisy": predict_mlps(bundle, noisy).mean(axis=0),
        "mlp_age_ensemble|noisy": predict_mlps(age_bundle, noisy).mean(axis=0),
    }
    single_mae = np.nanmean(np.abs(single_runs - test[TARGETS].to_numpy()[None]), axis=1)
    return {
        "repeat": repeat,
        "fold": fold,
        "index": test_index,
        "predictions": predictions,
        "params": params,
        "single_mae_by_seed": single_mae,
    }


def bootstrap_difference(errors_a, errors_b, seed=0, draws=2000):
    """사람 단위 재표집으로 평균 절대오차 차이(a-b)의 95% 구간을 구한다."""
    valid = ~(np.isnan(errors_a) | np.isnan(errors_b))
    difference = (errors_a - errors_b)[valid]
    rng = np.random.default_rng(seed)
    means = [difference[rng.integers(0, len(difference), len(difference))].mean() for _ in range(draws)]
    return float(difference.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    check_scoring()
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    replaced = int((df["shoulder_surface"] > 100).sum())
    df.loc[df["shoulder_surface"] > 100, "shoulder_surface"] = np.nan
    print("결측 코드 제거 (어깨사이길이)", replaced, "건")

    jobs = []
    for repeat in range(REPEATS):
        splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=repeat)
        for fold, (train_index, test_index) in enumerate(splitter.split(df, df["sex"])):
            jobs.append((repeat, fold, train_index, test_index))
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_fold)(df, repeat, fold, tr, te) for repeat, fold, tr, te in jobs
    )

    model_names = list(results[0]["predictions"])
    truth = df[TARGETS].to_numpy()
    # 반복마다 전원이 한 번씩 검증 데이터가 된다: (반복, 사람, 항목)
    out_of_fold = {
        name: np.full((REPEATS, len(df), len(TARGETS)), np.nan) for name in model_names
    }
    for item in results:
        for name in model_names:
            out_of_fold[name][item["repeat"], item["index"]] = item["predictions"][name]

    masks = {"전체": np.ones(len(df), bool), "남": df["sex"].eq("M").to_numpy(), "여": df["sex"].eq("F").to_numpy()}
    mae_rows = []
    for name in model_names:
        errors = np.abs(out_of_fold[name] - truth)
        for group, mask in masks.items():
            for k, target in enumerate(TARGETS):
                by_repeat = np.nanmean(errors[:, mask, k], axis=1)
                mae_rows.append(
                    {"model": name, "group": group, "target": LABELS[target], "mae_cm": by_repeat.mean(), "repeat_sd": by_repeat.std()}
                )
    mae = pd.DataFrame(mae_rows)
    mae.to_csv(OUT / "mae.csv", index=False, encoding="utf-8-sig")
    print("\n[평균 절대오차 cm]")
    for group in masks:
        print("\n--", group)
        print(
            mae[mae["group"] == group]
            .pivot(index="model", columns="target", values="mae_cm")
            .reindex(model_names)[[LABELS[t] for t in TARGETS]]
            .round(2)
            .to_string()
        )

    print("\n[MAE 차이와 95% 구간: 앞 모델 - 뒤 모델, 음수면 앞 모델이 더 정확]")
    difference_rows = []
    for a, b in [
        ("linear_terms", "linear"),
        ("mlp_ensemble", "linear"),
        ("mlp_ensemble", "linear_terms"),
        ("mlp_single", "linear_terms"),
        ("linear_terms_age", "linear_terms"),
        ("linear_terms_age", "linear_age"),
        ("mlp_age_ensemble", "linear_terms_age"),
        ("mlp_age_single", "linear_terms_age"),
    ]:
        error_a = np.abs(out_of_fold[a] - truth).mean(axis=0)
        error_b = np.abs(out_of_fold[b] - truth).mean(axis=0)
        for group, mask in masks.items():
            for k, target in enumerate(TARGETS):
                diff, low, high = bootstrap_difference(error_a[mask, k], error_b[mask, k])
                difference_rows.append({"a": a, "b": b, "group": group, "target": LABELS[target], "diff_cm": diff, "low": low, "high": high})
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(OUT / "mae_differences.csv", index=False, encoding="utf-8-sig")
    print(differences.round(3).to_string(index=False))

    print("\n[사이즈 일치율: 실제 치수로 고른 사이즈 = 예측 치수로 고른 사이즈]")
    index_of = {t: i for i, t in enumerate(TARGETS)}
    agreement_rows = []
    for shoulder_key in ["shoulder_surface", "shoulder"]:
        known = df[["chest", shoulder_key]].notna().all(axis=1).to_numpy()
        actual = recommend(df["chest"].to_numpy(), df[shoulder_key].to_numpy())
        for group, mask in masks.items():
            use = mask & known
            counts = np.bincount(actual[use], minlength=3) / use.sum()
            print(f"\n어깨 기준 {LABELS[shoulder_key]} / {group} 실제 사이즈 분포:", dict(zip(SIZES, counts.round(3).tolist())))
            agreement_rows.append({"shoulder": LABELS[shoulder_key], "group": group, "model": "최다 사이즈만 추천", "agreement": counts.max(), "repeat_sd": 0.0})
            for name in model_names:
                per_repeat = []
                for repeat in range(REPEATS):
                    predicted = recommend(
                        out_of_fold[name][repeat, :, index_of["chest"]],
                        out_of_fold[name][repeat, :, index_of[shoulder_key]],
                    )
                    per_repeat.append((predicted[use] == actual[use]).mean())
                agreement_rows.append({"shoulder": LABELS[shoulder_key], "group": group, "model": name, "agreement": np.mean(per_repeat), "repeat_sd": np.std(per_repeat)})
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(OUT / "size_agreement.csv", index=False, encoding="utf-8-sig")
    print(agreement.pivot_table(index="model", columns=["shoulder", "group"], values="agreement").round(3).to_string())

    print("\n[BMI 구간별 MAE, 남녀 전체]")
    bmi = df["weight"] / (df["height"] / 100) ** 2
    bands = pd.cut(bmi, [0, 18.5, 25, 30, 100], labels=["<18.5", "18.5~25", "25~30", ">=30"], right=False)
    band_rows = []
    for band in bands.cat.categories:
        mask = (bands == band).to_numpy()
        for name in ["linear", "linear_terms", "mlp_ensemble", "linear_terms_age", "mlp_age_ensemble"]:
            errors = np.abs(out_of_fold[name] - truth)
            for target in ["chest", "waist", "shoulder_surface"]:
                band_rows.append({"bmi": band, "n": int(mask.sum()), "model": name, "target": LABELS[target], "mae_cm": np.nanmean(errors[:, mask, index_of[target]])})
    band_table = pd.DataFrame(band_rows)
    band_table.to_csv(OUT / "mae_by_bmi.csv", index=False, encoding="utf-8-sig")
    print(band_table.pivot_table(index=["bmi", "n", "model"], columns="target", values="mae_cm", observed=True).round(2).to_string())

    single = np.stack([item["single_mae_by_seed"] for item in results])
    print("\n[MLP 시드에 따른 항목별 MAE 변동: 같은 분할 안 시드 간 범위의 평균 cm]")
    print(dict(zip([LABELS[t] for t in TARGETS], (single.max(axis=1) - single.min(axis=1)).mean(axis=0).round(3).tolist())))
    print("\n[MLP 선택 설정 횟수]")
    print(pd.Series([str(item["params"]) for item in results]).value_counts().to_string())

    men = df[(df["sex"] == "M") & df["chest"].between(102, 106)]
    print("\n[어깨 척도 확인: 가슴둘레 102~106cm 남성]", len(men), "명")
    print(men[["shoulder", "shoulder_surface"]].describe().round(1).loc[["mean", "std", "25%", "75%"]].to_string())
    print("46cm 이상 비율 | 어깨너비", round((men["shoulder"] >= 46).mean(), 3), "| 어깨사이길이", round((men["shoulder_surface"] >= 46).mean(), 3))

    # 앱에 넣을 수 있도록 전체 데이터로 계수를 저장한다.
    # 입력 단위: 키 cm, 몸무게 kg, 나이 세. BMI = 몸무게 / (키/100)^2. 출력 단위: cm.
    coefficients = {}
    largest_gap = 0.0
    for key, features in LINEAR_FEATURES.items():
        coefficients[key] = {"features": features, "sex": {}}
        for sex in ["F", "M"]:
            group = df[df["sex"] == sex]
            coefficients[key]["sex"][sex] = {}
            for target in TARGETS:
                known = group[group[target].notna()]
                x = linear_inputs(known, key)
                scaler = StandardScaler().fit(x)
                model = LinearRegression().fit(scaler.transform(x), known[target])
                raw = model.coef_ / scaler.scale_
                intercept = model.intercept_ - float((raw * scaler.mean_).sum())
                fitted = model.predict(scaler.transform(x))
                # 저장한 계수만으로 계산한 값이 모델 예측과 같은지 확인한다.
                largest_gap = max(largest_gap, float(np.abs(x @ raw + intercept - fitted).max()))
                coefficients[key]["sex"][sex][target] = {
                    "coefficients": dict(zip(features, raw.tolist())),
                    "intercept": float(intercept),
                    "residual_sd": float((known[target] - fitted).std()),
                }
    if largest_gap > 1e-3:
        raise AssertionError(f"저장 계수 재현 오차가 큼: {largest_gap}")
    (OUT / "linear_coefficients.json").write_text(json.dumps(coefficients, indent=2), encoding="utf-8")
    print("\n[계수 저장 완료] 재현 오차 최대", largest_gap, "cm")

    print("\n[예시 예측: 저장 계수 사용, cm]")
    example = pd.DataFrame(
        [
            {"sex": "M", "height": 175.0, "weight": 75.0, "age": 25},
            {"sex": "M", "height": 175.0, "weight": 75.0, "age": 55},
            {"sex": "F", "height": 160.0, "weight": 55.0, "age": 25},
            {"sex": "F", "height": 160.0, "weight": 55.0, "age": 55},
        ]
    )
    for _, person in example.iterrows():
        one = pd.DataFrame([person])
        x = linear_inputs(one, "terms_age")[0]
        entry = coefficients["terms_age"]["sex"][person["sex"]]
        values = {
            LABELS[t]: round(float(x @ np.array(list(entry[t]["coefficients"].values())) + entry[t]["intercept"]), 1)
            for t in TARGETS
        }
        print(person["sex"], person["height"], person["weight"], person["age"], values)


if __name__ == "__main__":
    main()
