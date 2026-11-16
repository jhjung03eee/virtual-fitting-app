"""경계 케이스가 얼마나 자주 생기나 — 치수표별로 '점수 차 2.0 미만' 구간을 재 본다. GPU 불필요.

    python scripts/eval_size_boundary.py

왜 (2026-11-16, 교수님 피드백 "점수가 5:5로 갈리면 어떻게 하나"):
PR #6 이 중첩 검증으로 **점수 차 2.0 미만**을 경계 안내 기준으로 제안했다(유니클로 반팔, 남성 41.9%).
그 기준을 우리 치수표 9개에 그대로 대 보면 어떻게 되는지 확인한다.

**재는 방법.** 사용자가 주 치수 하나만 넣은 경우(상의=가슴둘레, 하의=허리둘레)를 가정하고,
그 값을 0.5cm 씩 훑으며 `size_fit.recommend` 를 그대로 부른다. 1·2위 점수 차가 2.0 미만이면 경계로 센다.
점수식·가중치·여유 목표는 건드리지 않는다 — 기준값이 치수표에 따라 어떻게 달라지는지만 본다.

**한계.** 한국인 체형 분포로 가중치를 주지 않았다(사이즈코리아 원자료는 저장소에 없다).
그래서 '사람의 몇 %'가 아니라 '측정 범위의 몇 %'다. PR #6 의 41.9% 와 직접 비교할 수 없다.
어깨·총장을 같이 넣으면 분리가 더 되므로, 이 값은 '주 치수만 아는 최악의 경우'로 읽어야 한다.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

from size_fit import IDEAL_EASE, SizeChart, recommend          # noqa: E402

BOUNDARY = 2.0          # PR #6 이 제안한 경계 기준 (점수 차)
STEP = 0.5              # 훑는 간격 (cm, 둘레)
KEY = {'upper': 'chest', 'lower': 'waist'}


def body_range(chart, key):
    """이 치수표가 실제로 감당하는 체형 범위(둘레 cm).

    가장 작은 사이즈가 딱 맞는 몸 ~ 가장 큰 사이즈가 딱 맞는 몸. 그 밖은 어느 사이즈를 골라도
    '전부 작다/전부 크다'라서 경계 판정을 따질 일이 아니다 — 범위를 넓게 잡으면 비율이 희석된다.
    """
    ideal = IDEAL_EASE[chart.category][key]
    vals = [chart.sizes[s][key] for s in chart.sizes if key in chart.sizes[s]]
    return (min(vals) - ideal) * 2, (max(vals) - ideal) * 2


def scan(chart):
    """주 치수만 아는 사용자를 가정하고 훑는다. → (경계 비율, 동점 지점들, 구간들, 사이즈 간격)"""
    key = KEY[chart.category]
    if key not in chart.dimensions():
        return None
    lo, hi = body_range(chart, key)
    rows, ties = [], []
    c = lo
    while c <= hi + 1e-9:
        rec = recommend(chart, {key: c / 2})        # 둘레 → 단면
        ranked = sorted((f.score for f in rec.ranked))
        diff = (ranked[1] - ranked[0]) if len(ranked) > 1 else float('inf')
        best = rec.best.size if rec.best else None
        rows.append((c, diff, best))
        if diff < 1e-9:
            ties.append((c, best))
        c += STEP

    inside = [r for r in rows if r[1] < BOUNDARY]
    # 연속 구간으로 묶기
    bands, start, prev = [], None, None
    for c, d, _b in rows:
        if d < BOUNDARY:
            if start is None:
                start = c
            prev = c
        elif start is not None:
            bands.append((start, prev)); start = None
    if start is not None:
        bands.append((start, prev))

    # 사이즈 간격 (둘레 기준): 인접 사이즈의 주 치수 차 x 2
    vals = [chart.sizes[s][key] for s in chart.sizes if key in chart.sizes[s]]
    steps = [abs(b - a) * 2 for a, b in zip(vals, vals[1:])]
    gap = min(steps) if steps else None
    return len(inside) / len(rows), ties, bands, gap


def main():
    paths = sorted(glob.glob(os.path.join(ROOT, 'data', 'size_charts', '*.json')))
    print(f'경계 기준: 1·2위 점수 차 {BOUNDARY} 미만 (PR #6 제안)')
    print(f'가정: 사용자가 주 치수 하나만 입력 (상의=가슴둘레, 하의=허리둘레)\n')
    print(f'{"치수표":<34} {"부위":<6} {"커버 체형":>14} {"간격":>6} {"경계비율":>8}  동점')
    print('-' * 92)
    worst = []
    for p in paths:
        try:
            chart = SizeChart.from_dict(json.load(open(p, encoding='utf-8')))
        except Exception as e:
            print(f'{os.path.basename(p):<34} 읽기 실패: {e}')
            continue
        out = scan(chart)
        if out is None:
            print(f'{chart.name[:33]:<34} {chart.category:<6} 주 치수 없음 — 건너뜀')
            continue
        ratio, ties, bands, gap = out
        lo, hi = body_range(chart, KEY[chart.category])
        tie_txt = ', '.join(f'{c:g}→{b}' for c, b in ties[:4]) or '없음'
        gap_txt = f'{gap:g}cm' if gap else '-'
        print(f'{chart.name[:33]:<34} {chart.category:<6} {lo:5.0f}~{hi:<5.0f}cm {gap_txt:>6} '
              f'{ratio*100:7.0f}%  {tie_txt}')
        worst.append((ratio, chart.name, bands, gap))

    print()
    if worst:
        avg = sum(r for r, *_ in worst) / len(worst)
        print(f'평균 경계 비율: {avg*100:.0f}%')
        hi = max(worst)
        print(f'가장 높은 치수표: {hi[1]} ({hi[0]*100:.0f}%)')
        print()
        # 주 치수 하나만 쓰면 점수는 |옷단면 - 몸단면 - 목표여유| 이고 가중치가 1.0 이라,
        # 인접 두 사이즈의 점수 차가 BOUNDARY 미만인 구간의 폭은 단면 BOUNDARY = 둘레 2*BOUNDARY 로
        # **사이즈 간격과 무관하게 고정**이다. 그래서 간격이 좁은 치수표일수록 경계 비율이 높다.
        print(f'경계 구간의 폭은 둘레 {2 * BOUNDARY:g}cm 로 고정이다 (주 치수 하나만 쓸 때).')
        print('그래서 사이즈 간격이 4cm 인 구간은 통째로 경계가 된다 — 간격이 좁을수록 비율이 올라간다.')
        print('주 치수 하나만 아는 경우의 값이므로, 어깨·총장을 같이 받으면 낮아진다.')


if __name__ == '__main__':
    main()