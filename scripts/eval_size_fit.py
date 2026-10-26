"""사이즈 추천 정확도 측정 — 신체 치수 방식(app/size_fit.py).

팀원이 실제로 입어 본 옷의 치수와 착용감(작음/맞음/큼)을 정답으로 두고, 신체 치수로 그 착용감을
몇 % 맞히는지 잰다. 입력 양식은 data/fit_eval/README.md. 추천은 **신체 정보만** 쓴다(2026-10-26 결정,
기준 옷 비교 방식은 뺐다 — docs/PLAN.md F-30).

    python scripts/eval_size_fit.py                    # data/fit_eval/people.csv, garments.csv
    python scripts/eval_size_fit.py --sweep            # 주 항목 여유분의 작음·큼 경계값별 정확도도 출력

**예측 방법**: 옷 한 벌(입은 사이즈)의 여유분 = 옷 단면 − 몸 단면. 주 항목(상의 가슴→어깨, 하의 허리)의
핏 라벨을 작음/맞음/큼으로 묶는다(BODY_LABEL_TO_VERDICT). 레귤러핏 옷 기준이므로 오버핏으로 나온 옷은
기록에서 빼거나 material 칸에 적어 따로 본다.

--sweep 은 주 항목 여유분을 직접 경계값으로 잘라(작음 < a, 큼 > b) 정확도가 가장 높은 구간을 찾는다.
`size_fit.FIT_BANDS` 의 레귤러 구간(가슴 +3~7cm 등)을 보정하는 근거가 된다. **같은 데이터로 고르는 것이라
낙관적**이다 — 기록이 쌓이면 사람 단위로 나눠(한 명 빼고 고른 값을 그 사람에게 적용) 다시 재야 한다.
"""
import argparse
import csv
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

from body_profile import BodyProfile  # noqa: E402
from size_fit import SizeChart, evaluate_size  # noqa: E402

EVAL_DIR = os.path.join(ROOT, 'data', 'fit_eval')
VERDICTS = ('작음', '맞음', '큼')
# 사이즈 판정 항목(size_fit.SIZE_DIMENSIONS, 2026-10-26): 상의 총장·어깨·가슴, 하의 총장·허리
GARMENT_DIMS = ('shoulder', 'chest', 'length', 'waist')
MAIN_DIMENSION = {'upper': ('chest', 'shoulder'), 'lower': ('waist',)}
BODY_LABEL_TO_VERDICT = {
    '작음': '작음', '타이트': '맞음', '레귤러': '맞음', '루즈': '큼', '오버핏': '큼',
}


def _num(value) -> Optional[float]:
    value = (value or '').strip()
    return float(value) if value else None


def load_people(path) -> Dict[str, Dict[str, float]]:
    """사람별 몸 치수를 치수표와 같은 단면 기준으로 돌려준다."""
    people = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            profile = BodyProfile(
                height_cm=_num(row.get('height_cm')) or 0.0,
                shoulder_cm=_num(row.get('shoulder_cm')),
                chest_circumference_cm=_num(row.get('chest_circumference_cm')),
                waist_circumference_cm=_num(row.get('waist_circumference_cm')),
            )
            people[row['person'].strip()] = profile.to_chart_dimensions()
    return people


def load_garments(path) -> List[dict]:
    garments = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            felt = (row.get('felt') or '').strip()
            if felt not in VERDICTS:
                raise ValueError(f'{path}:{i} felt 는 {"/".join(VERDICTS)} 중 하나여야 한다: {felt!r}')
            category = (row.get('category') or '').strip()
            if category not in ('upper', 'lower'):
                raise ValueError(f'{path}:{i} category 는 upper/lower: {category!r}')
            dims = {d: _num(row.get(d)) for d in GARMENT_DIMS}
            garments.append({
                'id': i,
                'person': row['person'].strip(),
                'garment': (row.get('garment') or '').strip(),
                'category': category,
                'material': (row.get('material') or '').strip(),
                'dims': {d: v for d, v in dims.items() if v is not None},
                'felt': felt,
            })
    return garments


def main_ease(garment: dict, body: Dict[str, float]) -> Optional[Tuple[str, float]]:
    """주 항목 (이름, 여유분). 비교할 수 없으면 None."""
    for dim in MAIN_DIMENSION[garment['category']]:
        if dim in garment['dims'] and dim in body:
            return dim, garment['dims'][dim] - body[dim]
    return None


def predict_by_body(garment: dict, body: Dict[str, float]) -> Optional[str]:
    """신체 치수 방식(size_fit 의 핏 라벨). 주 항목을 비교할 수 없으면 None(측정 불가)."""
    chart = SizeChart(name=garment['garment'], category=garment['category'],
                      sizes={'worn': garment['dims']})
    labels = {d.dimension: d.label for d in evaluate_size(chart, 'worn', body).dimensions}
    for dim in MAIN_DIMENSION[garment['category']]:
        if dim in labels and labels[dim] in BODY_LABEL_TO_VERDICT:
            return BODY_LABEL_TO_VERDICT[labels[dim]]
    return None


def score(results: List[Tuple[str, str]]) -> dict:
    """(정답, 예측) 목록 → 정확도와 혼동 행렬."""
    confusion = Counter(results)
    correct = sum(n for (truth, pred), n in confusion.items() if truth == pred)
    return {'n': len(results), 'correct': correct,
            'accuracy': correct / len(results) if results else None, 'confusion': confusion}


def evaluate(people, garments) -> dict:
    body_results, skipped = [], 0
    for g in garments:
        pred = predict_by_body(g, people.get(g['person'], {}))
        if pred is None:
            skipped += 1
        else:
            body_results.append((g['felt'], pred))
    return {
        'body': dict(score(body_results), skipped=skipped),
        'baseline': score([(g['felt'], '맞음') for g in garments]),
    }


def sweep(people, garments, category='upper') -> List[Tuple[float, float, float, int]]:
    """주 항목 여유분을 (작음 < a, 큼 > b)로 잘랐을 때의 정확도. 낙관적 추정이다(모듈 설명 참고)."""
    points = []
    for g in garments:
        if g['category'] != category:
            continue
        hit = main_ease(g, people.get(g['person'], {}))
        if hit:
            points.append((g['felt'], hit[1]))
    rows = []
    for a in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
        for b in (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0):
            results = [(felt, '작음' if e < a else '큼' if e > b else '맞음') for felt, e in points]
            s = score(results)
            if s['n']:
                rows.append((a, b, s['accuracy'], s['n']))
    return sorted(rows, key=lambda r: -r[2])


def format_score(title, s) -> str:
    if not s['n']:
        return f'{title}: 측정할 데이터 없음'
    lines = [f"{title}: {s['accuracy']:.0%} ({s['correct']}/{s['n']})"]
    lines.append('    정답\\예측  ' + '  '.join(f'{v:>4}' for v in VERDICTS))
    for truth in VERDICTS:
        lines.append(f'    {truth:>6}    ' + '  '.join(f"{s['confusion'][(truth, p)]:>4}" for p in VERDICTS))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--people', default=os.path.join(EVAL_DIR, 'people.csv'))
    parser.add_argument('--garments', default=os.path.join(EVAL_DIR, 'garments.csv'))
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args(argv)

    for path in (args.people, args.garments):
        if not os.path.exists(path):
            print(f'{path} 가 없다. data/fit_eval/README.md 의 양식대로 만들어 주세요.')
            return 1
    people = load_people(args.people)
    garments = load_garments(args.garments)
    result = evaluate(people, garments)

    print(f'사람 {len(people)}명 · 옷 {len(garments)}벌 · 착용감 {dict(Counter(g["felt"] for g in garments))}\n')
    print(format_score('기준선 (전부 "맞음"으로 찍기)', result['baseline']))
    print(format_score('신체 치수 방식', result['body']))
    if result['body']['skipped']:
        print(f"    (몸 치수·주 항목이 없어 뺀 옷 {result['body']['skipped']}벌)")

    if args.sweep:
        for category, name in (('upper', '상의 가슴(없으면 어깨)'), ('lower', '하의 허리')):
            rows = sweep(people, garments, category)
            if not rows:
                continue
            print(f'\n{name} 여유분 경계값별 정확도 (상위 5개, 같은 데이터로 골라 낙관적):')
            for a, b, acc, n in rows[:5]:
                print(f'  작음 < {a:+.0f}cm  큼 > {b:+.0f}cm  →  {acc:.0%} (n={n})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
