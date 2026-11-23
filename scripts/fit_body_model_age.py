"""치수를 모를 때의 회귀식에 **나이**를 더한 식을 학습해 기존 모델 파일에 덧붙인다 (docs/PLAN.md F-40).

    python scripts/fit_body_model_age.py <body.csv> [--model data/body_models/sizekorea8_2020.json]

키·몸무게 식(`scripts/fit_body_model.py`, MDIS 원자료)은 그대로 두고, 성별마다 `parts_age`·`resid_corr_age`·`age_range` 를 추가한다.
앱(`app/size_unknown.predict`)은 나이를 넣으면 `parts_age`, 비우면 기존 `parts` 를 쓴다.

데이터: 사이즈코리아 8차(2020~24) **공개용** 치수 데이터 엑셀(sizekorea.kr)에서 뽑은 CSV.
`experiments/size_unknown_compare/prepare_data.py`(PR #6)가 만드는 형식: sex(M/F) · height · weight(kg) · age · shoulder(어깨너비) ·
bust(젖가슴둘레) · waist(허리둘레), 길이는 cm. **원자료는 저장소에 올리지 않는다** — 계수만 커밋한다.
공개용 데이터로 맞춘 키·몸무게 식은 MDIS 식과 예측값이 0.1cm 안으로 같다(같은 조사, 같은 부위) — 확인은 F-40.

왜 나이인가(F-40): 입력 조합 255개·곡선(2~4차)·신경망을 비교하니 곡선·추가 항은 오차를 0.4mm 이하로만 줄였고,
나이만 허리 오차를 17% 줄이고 하의 사이즈 일치율을 3~6%p 올렸다. 그래서 직선 a + b·키 + c·몸무게 + d·나이.
"""
import argparse
import csv
import json
import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = {'shoulder': 'shoulder', 'chest': 'bust', 'waist': 'waist'}   # 모델 부위 → CSV 열 (fit_body_model 과 같은 정의)


def load(path):
    people = {'m': [], 'f': []}
    with open(path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            sx = {'M': 'm', 'F': 'f'}.get(r['sex'].strip())
            try:
                v = {k: float(r[k]) for k in ('height', 'weight', 'age', *PARTS.values())}
            except (KeyError, ValueError):
                continue
            if sx and 120 <= v['height'] <= 220 and 30 <= v['weight'] <= 200:
                people[sx].append(v)
    return people


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--model', default=os.path.join(ROOT, 'data', 'body_models', 'sizekorea8_2020.json'))
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    with open(a.model, encoding='utf-8') as f:
        model = json.load(f)
    people = load(a.data)
    rng = random.Random(a.seed)
    model['age_method'] = ('나이 추가 식: 성별·부위별 선형 회귀 a + b·키(cm) + c·몸무게(kg) + d·나이(세). '
                           '사이즈코리아 8차(2020~24) 공개용 치수 데이터, 무작위 80/20 seed '
                           f'{a.seed}. scripts/fit_body_model_age.py')
    for sx, ps in people.items():
        rng.shuffle(ps)
        cut = int(len(ps) * 0.8)
        train, test = ps[:cut], ps[cut:]
        design = lambda rows: np.array([[1.0, p['height'], p['weight'], p['age']] for p in rows])   # noqa: E731
        xtr, xte = design(train), design(test)
        parts, resid = {}, {}
        print(f'\n[{sx}] 학습 {len(train)}명 · 평가 {len(test)}명')
        for part, col in PARTS.items():
            ytr = np.array([p[col] for p in train])
            yte = np.array([p[col] for p in test])
            c, *_ = np.linalg.lstsq(xtr, ytr, rcond=None)
            res, err = ytr - xtr @ c, yte - xte @ c
            parts[part] = {'a': round(float(c[0]), 4), 'b_height': round(float(c[1]), 5),
                           'c_weight': round(float(c[2]), 5), 'd_age': round(float(c[3]), 5),
                           'resid_sd_cm': round(float(np.sqrt((res ** 2).sum() / (len(res) - 4))), 2),
                           'test_mae_cm': round(float(np.abs(err).mean()), 2),
                           'test_rmse_cm': round(float(np.sqrt((err ** 2).mean())), 2)}
            resid[part] = res
            print(f'  {part:8s} 평가 MAE {parts[part]["test_mae_cm"]:.2f}cm (나이 1살당 {c[3]:+.3f}cm)')
        names = list(PARTS)
        corr = {f'{p}|{q}': round(float(np.corrcoef(resid[p], resid[q])[0, 1]), 3)
                for i, p in enumerate(names) for q in names[i + 1:]}
        ages = [p['age'] for p in ps]
        model['sex'][sx].update({'parts_age': parts, 'resid_corr_age': corr,
                                 'age_range': [int(min(ages)), int(max(ages))],
                                 'n_train_age': len(train), 'n_test_age': len(test)})
    with open(a.model, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(model, f, ensure_ascii=False, indent=1)
    print('\n저장:', a.model)


if __name__ == '__main__':
    main()
