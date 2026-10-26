"""scripts/eval_size_fit.py 단위 테스트 — 가짜 착용 기록으로 계산 경로만 확인한다.

여기 숫자는 테스트용으로 만든 값이다. 실제 정확도는 팀원 기록으로 잰다.
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'app'))

import eval_size_fit as ev  # noqa: E402

PEOPLE = """person,height_cm,shoulder_cm,chest_circumference_cm,waist_circumference_cm
p1,175,44,96,80
"""

# p1 은 가슴단면 48(둘레 96). 여유 +6 맞음 · +2 작음 · +12 큼 · +7 맞음 · 하의 허리 +3 맞음
GARMENTS = """person,garment,category,material,shoulder,chest,length,waist,felt,note
p1,기준 티,upper,면티,46,54,70,,맞음,
p1,작은 티,upper,면티,43,50,66,,작음,
p1,큰 티,upper,면티,52,60,74,,큼,
p1,비슷한 셔츠,upper,셔츠,46,55,72,,맞음,
p1,면바지,lower,슬랙스,,,100,43,맞음,
"""


def write(tmp, name, text):
    path = os.path.join(tmp, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


class TestEvalSizeFit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.people = ev.load_people(write(self.tmp, 'people.csv', PEOPLE))
        self.garments = ev.load_garments(write(self.tmp, 'garments.csv', GARMENTS))

    def test_people_are_converted_to_flat(self):
        self.assertEqual(self.people['p1']['chest'], 48.0)
        self.assertEqual(self.people['p1']['waist'], 40.0)

    def test_blank_dimensions_are_dropped(self):
        self.assertEqual(set(self.garments[0]['dims']), {'shoulder', 'chest', 'length'})
        self.assertEqual(set(self.garments[4]['dims']), {'length', 'waist'})

    def test_body_prediction(self):
        preds = [ev.predict_by_body(g, self.people['p1']) for g in self.garments]
        # 작은 티(+2 타이트)·셔츠(+7 루즈)는 지금 구간으로는 틀린다
        self.assertEqual(preds, ['맞음', '맞음', '큼', '큼', '맞음'])

    def test_evaluate_against_baseline(self):
        result = ev.evaluate(self.people, self.garments)
        self.assertEqual(result['body']['accuracy'], 0.6)
        self.assertEqual(result['baseline']['accuracy'], 0.6)   # 맞음 3/5

    def test_sweep_finds_better_bands(self):
        """경계를 데이터에 맞추면(작음 < +3, 큼 > +7~11) 상의 4벌을 모두 맞힌다."""
        best = ev.sweep(self.people, self.garments, 'upper')[0]
        self.assertEqual(best[2], 1.0)
        self.assertEqual(best[3], 4)

    def test_bad_felt_is_rejected(self):
        bad = GARMENTS.replace('작음,', '좀작음,', 1)
        with self.assertRaises(ValueError):
            ev.load_garments(write(self.tmp, 'bad.csv', bad))

    def test_main_runs(self):
        code = ev.main(['--people', os.path.join(self.tmp, 'people.csv'),
                        '--garments', os.path.join(self.tmp, 'garments.csv'), '--sweep'])
        self.assertEqual(code, 0)


if __name__ == '__main__':
    unittest.main()
