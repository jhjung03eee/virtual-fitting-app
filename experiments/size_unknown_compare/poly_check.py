"""키·몸무게만으로 1~4차 다항 곡선과 스플라인을 같은 분할(5겹×3회)로 비교한다. (조합실험 보충)

    python poly_check.py <body.csv>
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
df.loc[df["shoulder_surface"] > 100, "shoulder_surface"] = np.nan
T = ["chest", "waist", "shoulder_surface"]
KO = ["가슴둘레", "허리둘레", "어깨사이길이"]
models = {
    "1차 (직선, 지금 앱)": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "2차 곡선": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()),
    "3차 곡선": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False), LinearRegression()),
    "4차 곡선": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(4, include_bias=False), LinearRegression()),
    "스플라인 (구간별 곡선)": lambda: make_pipeline(StandardScaler(), SplineTransformer(n_knots=6, degree=3),
                                         PolynomialFeatures(2, interaction_only=True, include_bias=False),
                                         RidgeCV(alphas=np.logspace(-3, 3, 13))),
}
for inputs, label in ((["height", "weight"], "키·몸무게"), (["height", "weight", "age"], "키·몸무게·나이")):
    print(f"\n[{label}] 평균 절대오차 cm")
    for name, make in models.items():
        err = np.full((3, len(df), len(T)), np.nan)
        for r in range(3):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(df, df["sex"]):
                for s in ("F", "M"):
                    a = df.iloc[tr]; a = a[a["sex"] == s]
                    b = te[df["sex"].to_numpy()[te] == s]
                    for k, t in enumerate(T):
                        ok = a[t].notna()
                        m = make().fit(a.loc[ok, inputs], a.loc[ok, t])
                        err[r, b, k] = np.abs(m.predict(df.iloc[b][inputs]) - df.iloc[b][t].to_numpy())
        mae = np.nanmean(err, axis=(0, 1))
        print(f"  {name:22s} " + "  ".join(f"{KO[k]} {mae[k]:.3f}" for k in range(3)) + f"  | 점수 {mae.mean():.3f}")
