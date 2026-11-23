"""치수를 모를 때의 회귀식 학습: 키·몸무게 → 어깨너비·젖가슴둘레·허리둘레 (Bartol 외, Sensors 2022 방식).

결과 = data/body_models/sizekorea8_2020.json (회귀 계수·잔차 표준편차·부위 간 잔차 상관·평가 오차).
앱(app/size_unknown.py)이 이 파일만 읽는다. **원자료(CSV)는 저장소에 올리지 않는다** — 계수만 커밋한다.

데이터: 사이즈코리아 8차(2020) 직접측정 공공용 원자료. 국가데이터처 MDIS(mdis.mods.go.kr) →
한국인인체치수조사 → ONE-ID 로그인 후 내려받기. **필요한 항목을 한 번에 골라 한 파일로** 받는다
(항목을 나눠 받으면 행 순서가 달라져 합칠 수 없다 — 2026-11-09 실제로 겪음).
필요한 열: 성별(1=남, 2=여) · 002 키(mm) · 125 몸무게(kg) · 088 어깨사이너비 · 042 젖가슴둘레 · 045 허리둘레 (mm).

방법: 성별·부위마다 최소제곱 a + b·키 + c·몸무게. 무작위 80/20 분할(seed 42), 평가는 학습에 안 쓴 20%.

    python scripts/fit_body_model.py <사이즈코리아.csv> [--out data/body_models/sizekorea8_2020.json]
"""
import argparse
import csv
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLUMNS = {'sex': '성별', 'height': '002. 키', 'weight': '125. 몸무게',
           'shoulder': '088. 어깨사이너비', 'chest': '042. 젖가슴둘레', 'waist': '045. 허리둘레'}
TARGETS = ('shoulder', 'chest', 'waist')
RANGE_CM = {'height': (120, 220), 'weight': (30, 200), 'shoulder': (25, 60), 'chest': (60, 150), 'waist': (50, 150)}
MISSING = '9999'   # 사이즈코리아 결측 표기 (여성 전용 항목 등)


def read_rows(path):
    for enc in ('utf-8-sig', 'cp949'):
        try:
            with open(path, encoding=enc, newline='') as f:
                rows = list(csv.reader(f))
            return [h.strip() for h in rows[0]], rows[1:]
        except UnicodeDecodeError:
            continue
    raise SystemExit('파일 인코딩을 읽지 못했습니다 (UTF-8/CP949 아님)')


def ols(xs, ys):
    """절편·키·몸무게 3×3 정규방정식 (numpy 없이)."""
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for (h, w), y in zip(xs, ys):
        row = (1.0, h, w)
        for i in range(3):
            b[i] += row[i] * y
            for j in range(3):
                A[i][j] += row[i] * row[j]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(A[r][i]))
        A[i], A[p] = A[p], A[i]
        b[i], b[p] = b[p], b[i]
        for r in range(i + 1, 3):
            f = A[r][i] / A[i][i]
            for c in range(i, 3):
                A[r][c] -= f * A[i][c]
            b[r] -= f * b[i]
    coef = [0.0] * 3
    for i in reversed(range(3)):
        coef[i] = (b[i] - sum(A[i][c] * coef[c] for c in range(i + 1, 3))) / A[i][i]
    return coef


def load_people(path):
    header, rows = read_rows(path)
    missing = [c for c in COLUMNS.values() if c not in header]
    if missing:
        raise SystemExit(f'열이 없습니다: {missing}\n파일의 열: {header}')
    idx = {k: header.index(c) for k, c in COLUMNS.items()}
    people = {'m': [], 'f': []}
    for r in rows:
        sx = {'1': 'm', '2': 'f'}.get(r[idx['sex']].strip())
        if sx is None or any(r[idx[k]].strip() in ('', MISSING) for k in idx):
            continue
        v = {k: float(r[idx[k]]) for k in idx if k != 'sex'}
        for k in ('height',) + TARGETS:
            v[k] /= 10.0              # mm → cm
        if all(RANGE_CM[k][0] <= v[k] <= RANGE_CM[k][1] for k in v):
            people[sx].append(v)
    return people


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'body_models', 'sizekorea8_2020.json'))
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    people = load_people(a.data)
    model = {'source': '사이즈코리아 8차(2020) 직접측정 공공용 원자료 (국가데이터처 MDIS). 원자료는 저장소에 올리지 않는다 — 여기는 회귀 계수만',
             'method': 'Bartol 외(Sensors 2022) 방식: 성별·부위별 선형 회귀 a + b·키(cm) + c·몸무게(kg)',
             'split': f'무작위 80/20, seed {a.seed}', 'columns': COLUMNS, 'sex': {}}
    rng = random.Random(a.seed)
    for sx, ps in people.items():
        rng.shuffle(ps)
        cut = int(len(ps) * 0.8)
        train, test = ps[:cut], ps[cut:]
        out = {'n_train': len(train), 'n_test': len(test), 'parts': {}}
        resid = {}
        print(f'\n[{sx}] 학습 {len(train)}명 · 평가 {len(test)}명')
        for t in TARGETS:
            c = ols([(p['height'], p['weight']) for p in train], [p[t] for p in train])
            pred = lambda p: c[0] + c[1] * p['height'] + c[2] * p['weight']   # noqa: E731
            res = [p[t] - pred(p) for p in train]
            err = [p[t] - pred(p) for p in test]
            sd = (sum(e * e for e in res) / (len(res) - 3)) ** 0.5
            mae = sum(abs(e) for e in err) / len(err)
            rmse = (sum(e * e for e in err) / len(err)) ** 0.5
            out['parts'][t] = {'a': round(c[0], 4), 'b_height': round(c[1], 5), 'c_weight': round(c[2], 5),
                               'resid_sd_cm': round(sd, 2), 'test_mae_cm': round(mae, 2), 'test_rmse_cm': round(rmse, 2)}
            resid[t] = res
            print(f'  {t:8s} 평가 MAE {mae:.2f}cm · RMSE {rmse:.2f}cm')
        corr = {}
        for i, t1 in enumerate(TARGETS):
            for t2 in TARGETS[i + 1:]:
                x, y = resid[t1], resid[t2]
                corr[f'{t1}|{t2}'] = round(sum(p * q for p, q in zip(x, y))
                                           / (sum(p * p for p in x) * sum(q * q for q in y)) ** 0.5, 3)
        out['resid_corr'] = corr
        model['sex'][sx] = out
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(model, f, ensure_ascii=False, indent=1)
    print('\n저장:', a.out)


if __name__ == '__main__':
    main()
