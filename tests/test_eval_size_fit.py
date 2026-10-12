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

PEOPLE = """person,height_cm,shoulder_cm,chest_circumference_cm,waist_circumference_cm,hip_circumference_cm
p1,175,44,96,,
"""

# p1 은 가슴단면 54 를 '맞음'으로 입는다(몸 단면 48 대비 +6, 레귤러).
GARMENTS = """person,garment,category,material,shoulder,chest,length,sleeve,waist,hip,thigh,felt,note
p1,기준 티,upper,면티,46,54,70,,,,,맞음,
p1,작은 티,upper,면티,43,50,66,,,,,작음,
p1,큰 티,upper,면티,52,60,74,,,,,큼,
p1,비슷한 셔츠,upper,셔츠,46,55,72,,,,,맞음,
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
        self.assertNotIn('waist', self.people['p1'])

    def test_blank_dimensions_are_dropped(self):
        self.assertEqual(set(self.garments[0]['dims']), {'shoulder', 'chest', 'length'})

    def test_body_prediction(self):
        preds = [ev.predict_by_body(g, self.people['p1']) for g in self.garments]
        self.assertEqual(preds, ['맞음', '맞음', '큼', '큼'])  # 작은 티(+2 타이트)·셔츠(+7 루즈)는 틀림

    def test_reference_pairs_exclude_self(self):
        pairs = ev.reference_pairs(self.garments)
        self.assertTrue(all(t is not r for t, r in pairs))
        # 맞음 2벌이 기준: 기준 티(1쌍) + 셔츠(1쌍) + 나머지 2벌 × 2
        self.assertEqual(len(pairs), 6)

    def test_reference_beats_body_on_this_data(self):
        result = ev.evaluate(self.people, self.garments)
        self.assertEqual(result['reference']['accuracy'], 1.0)
        self.assertEqual(result['body']['accuracy'], 0.5)
        self.assertEqual(result['reference_same_material']['n'], 2)  # 기준 티 ↔ 작은 티·큰 티

    def test_bad_felt_is_rejected(self):
        bad = GARMENTS.replace('작음,', '좀작음,', 1)
        with self.assertRaises(ValueError):
            ev.load_garments(write(self.tmp, 'bad.csv', bad))

    def test_sweep_restores_constants(self):
        before = (ev.reference_fit.VERDICT_SMALL_BELOW, ev.reference_fit.VERDICT_LARGE_ABOVE)
        rows = ev.sweep(self.garments)
        self.assertTrue(rows)
        self.assertEqual(before, (ev.reference_fit.VERDICT_SMALL_BELOW, ev.reference_fit.VERDICT_LARGE_ABOVE))

    def test_main_runs(self):
        code = ev.main(['--people', os.path.join(self.tmp, 'people.csv'),
                        '--garments', os.path.join(self.tmp, 'garments.csv'), '--sweep'])
        self.assertEqual(code, 0)


if __name__ == '__main__':
    unittest.main()
