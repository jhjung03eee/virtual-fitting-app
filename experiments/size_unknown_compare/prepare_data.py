"""사이즈코리아 8차 1~2차년도 직접측정 시트에서 치수 예측에 쓸 열을 뽑는다."""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SOURCE = sys.argv[1]
OUTPUT = sys.argv[2]

# 시트 구조: 4행 항목명, 6행 '001. 키' 형식 열 이름, 7행부터 개인별 데이터
raw = pd.read_excel(SOURCE, sheet_name=0, header=None, engine="openpyxl")
names = raw.iloc[4].tolist()


def column_of(name):
    hits = [j for j, value in enumerate(names) if value == name]
    if len(hits) != 1:
        raise ValueError(f"{name}: 열 {len(hits)}개 발견")
    return hits[0]


body = raw.iloc[7:]
df = pd.DataFrame(
    {
        "id": body.iloc[:, 2].values,
        "sex": body.iloc[:, 3].values,
        "age": pd.to_numeric(body.iloc[:, 4], errors="coerce").values,
    }
)

# 체중은 kg, 나머지는 mm로 저장돼 있다. mm 항목만 cm로 바꾼다.
MM_COLUMNS = {
    "height": "키",
    "chest": "가슴둘레",
    "bust": "젖가슴둘레",
    "waist": "허리둘레",
    "navel_waist": "배꼽수준허리둘레",
    "shoulder": "어깨너비",
    "shoulder_surface": "어깨사이길이",
}
for key, name in MM_COLUMNS.items():
    df[key] = pd.to_numeric(body.iloc[:, column_of(name)], errors="coerce").values / 10
df["weight"] = pd.to_numeric(
    body.iloc[:, column_of("체중(몸무게)")], errors="coerce"
).values

print("행 수", len(df), "| ID 중복", df["id"].duplicated().sum())
print(df["sex"].value_counts(dropna=False).to_string())
print("\n결측값")
print(df.isna().sum().to_string())
numeric = df.drop(columns=["id", "sex"])
print("\n성별 요약")
summary = df.groupby("sex")[numeric.columns.tolist()].agg(["mean", "std", "min", "max"])
print(summary.round(1).T.to_string())

bmi = df["weight"] / (df["height"] / 100) ** 2
print("\nBMI 범위", round(bmi.min(), 1), round(bmi.max(), 1))

# 평균에서 6 표준편차 넘게 벗어난 값은 입력 오류 후보로 출력만 한다.
print("\n성별 기준 |z| > 6 값")
for sex, group in df.groupby("sex"):
    for column in numeric.columns:
        z = (group[column] - group[column].mean()) / group[column].std()
        for index in group.index[np.abs(z) > 6]:
            print(sex, column, group.loc[index, "id"], group.loc[index, column], round(z[index], 1))

print("\n상관계수 (입력 vs 대상)")
for sex, group in df.groupby("sex"):
    print(sex)
    print(
        group[["height", "weight", "age", "chest", "bust", "waist", "shoulder", "shoulder_surface"]]
        .corr()
        .loc[["height", "weight", "age"], ["chest", "bust", "waist", "shoulder", "shoulder_surface"]]
        .round(2)
        .to_string()
    )

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print("\n저장", OUTPUT, df.shape)
