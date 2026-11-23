"""치수를 모를 때(키·몸무게) 추천이 '치수를 알 때' 추천과 얼마나 같은지 — 사이즈코리아 평가용 20%로 (F-37).

정답 = 그 사람의 실제 치수(088 어깨사이너비·042 젖가슴둘레·045 허리둘레)를 넣었을 때 size_fit.recommend 가 내는 사이즈.
비교:
  A 평균만   : 성별 평균 치수를 넣었을 때 (아무 정보 없이 찍기)
  B 한 점    : 회귀 예측값 그대로
  C 확률 최대 : app/size_unknown.recommend_unknown (앱이 쓰는 것)
  D C + 줄자 1개 : C 의 확률이 60% 미만이면 주 부위(상의 가슴, 하의 허리)만 실제 값으로 바꿔 다시 추천
분할은 scripts/fit_body_model.py 와 같다(같은 필터·seed) — 학습에 안 쓴 사람만 쓴다. 원자료는 저장소에 없다.

    python scripts/eval_size_unknown.py <사이즈코리아.csv> [--samples 500] [--out 결과.json]
"""
import argparse
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from size_fit import SizeChart, recommend  # noqa: E402
import size_unknown as su  # noqa: E402
from fit_body_model import load_people, TARGETS  # noqa: E402

FIELD = {'shoulder': 'shoulder_cm', 'chest': 'chest_circumference_cm', 'waist': 'waist_circumference_cm'}


def best(chart, profile):
    r = recommend(chart, su.to_chart_body(profile))
    return r.best.size if r.best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--samples', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    people = load_people(a.data)
    rng = random.Random(a.seed)
    split = {}
    for sx, ps in people.items():
        rng.shuffle(ps)
        cut = int(len(ps) * 0.8)
        split[sx] = {'test': ps[cut:], 'mean': {t: sum(p[t] for p in ps[:cut]) / cut for t in TARGETS}}
    charts = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'data', 'size_charts', '*.json'))):
        with open(path, encoding='utf-8') as f:
            c = SizeChart.from_dict(json.load(f))
        if c.category in su.PREDICT:
            charts.append((os.path.basename(path), c))
    result = []
    for fname, chart in charts:
        order = list(chart.sizes)
        parts = list(su.PREDICT[chart.category].values())
        main_part = 'chest' if chart.category == 'upper' else 'waist'
        for sx in ('m', 'f'):
            st = {'chart': fname, 'name': chart.name, 'category': chart.category, 'sex': sx, 'sizes': order,
                  'n': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'C_adj': 0, 'p_true': 0.0,
                  'hi_n': 0, 'hi_hit': 0, 'lo_n': 0, 'lo_hit': 0, 'lo_tape_hit': 0}
            for p in split[sx]['test']:
                base = {'height_cm': p['height']}
                known = dict(base, **{FIELD[t]: p[t] for t in parts})
                avg = dict(base, **{FIELD[t]: split[sx]['mean'][t] for t in parts})
                K = best(chart, known)
                u = su.recommend_unknown(chart, sx, p['height'], p['weight'], samples=a.samples)
                B, C = u.point_best, u.recommendation.best.size
                conf = u.probabilities[C] >= su.CONFIDENT
                if conf:
                    D = C
                else:
                    D = su.recommend_unknown(chart, sx, p['height'], p['weight'],
                                             measured={FIELD[main_part]: p[main_part]}, samples=a.samples).recommendation.best.size
                st['n'] += 1
                st['A'] += best(chart, avg) == K
                st['B'] += B == K
                st['C'] += C == K
                st['D'] += D == K
                st['C_adj'] += abs(order.index(C) - order.index(K)) <= 1
                st['p_true'] += u.probabilities[K]
                st['hi_n' if conf else 'lo_n'] += 1
                st['hi_hit' if conf else 'lo_hit'] += C == K
                if not conf:
                    st['lo_tape_hit'] += D == K
            result.append(st)
            f = lambda x, n: f'{100 * x / n:5.1f}%' if n else '   - '   # noqa: E731
            print(f"{chart.name[:22]:22s} {sx} n={st['n']} 일치 A {f(st['A'], st['n'])} B {f(st['B'], st['n'])} "
                  f"C {f(st['C'], st['n'])} D {f(st['D'], st['n'])} | 확신≥60% {st['hi_n']}명 {f(st['hi_hit'], st['hi_n'])}"
                  f" · <60% {st['lo_n']}명 {f(st['lo_hit'], st['lo_n'])} → 줄자 {f(st['lo_tape_hit'], st['lo_n'])}", flush=True)
    tot = {k: sum(s[k] for s in result) for k in ('n', 'A', 'B', 'C', 'D', 'C_adj', 'hi_n', 'hi_hit', 'lo_n', 'lo_hit', 'lo_tape_hit')}
    print('\n전체 (치수표×성별 합산, 사람-치수표 쌍 %d개)' % tot['n'])
    for k, name in (('A', '평균만'), ('B', '한 점'), ('C', '확률 최대'), ('D', '확률 최대 + 낮으면 줄자 1개')):
        print(f'  {name:22s} {100 * tot[k] / tot["n"]:.1f}%')
    print(f'  확률 최대 ±1사이즈 {100 * tot["C_adj"] / tot["n"]:.1f}%')
    print(f'  확신≥60%: {100 * tot["hi_n"] / tot["n"]:.1f}% 의 경우, 적중 {100 * tot["hi_hit"] / tot["hi_n"]:.1f}%')
    print(f'  확신<60%: {100 * tot["lo_n"] / tot["n"]:.1f}% 의 경우, 적중 {100 * tot["lo_hit"] / tot["lo_n"]:.1f}% → 줄자 1개 {100 * tot["lo_tape_hit"] / tot["lo_n"]:.1f}%')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fo:
            json.dump({'samples': a.samples, 'total': tot, 'rows': result}, fo, ensure_ascii=False, indent=1)
        print('저장:', a.out)


if __name__ == '__main__':
    main()
