"""app/size_unknown.py — 키·몸무게로 치수를 예측해 확률이 가장 큰 사이즈를 추천하는지."""
import glob
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

from size_fit import SizeChart, recommend  # noqa: E402
import size_unknown as su  # noqa: E402


def charts():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'data', 'size_charts', '*.json'))):
        with open(path, encoding='utf-8') as f:
            out.append(SizeChart.from_dict(json.load(f)))
    return out


PEOPLE = [('남', 165, 60), ('남', 175, 70), ('남', 183, 85), ('여', 155, 48), ('여', 162, 55), ('여', 170, 68)]


class TestSizeUnknown(unittest.TestCase):
    def test_model_file_has_both_sexes_and_parts(self):
        m = su.load_model()
        for sx in ('m', 'f'):
            self.assertEqual(set(m['sex'][sx]['parts']), {'shoulder', 'chest', 'waist'})
            self.assertIn('shoulder|chest', m['sex'][sx]['resid_corr'])

    def test_prediction_is_plausible(self):
        p = su.predict('upper', '남', 175, 70)
        self.assertTrue(35 <= p.profile['shoulder_cm'] <= 45)
        self.assertTrue(85 <= p.profile['chest_circumference_cm'] <= 105)
        heavier = su.predict('upper', '남', 175, 85)
        self.assertGreater(heavier.profile['chest_circumference_cm'], p.profile['chest_circumference_cm'])

    def test_measured_field_is_not_predicted(self):
        p = su.predict('upper', '여', 160, 55, measured={'chest_circumference_cm': 90})
        self.assertEqual(p.profile['chest_circumference_cm'], 90)
        self.assertNotIn('chest_circumference_cm', p.sd)

    def test_probabilities_sum_to_one_and_are_reproducible(self):
        for chart in charts():
            if chart.category not in su.PREDICT:
                continue
            for sex, h, w in PEOPLE:
                a = su.recommend_unknown(chart, sex, h, w, samples=300)
                b = su.recommend_unknown(chart, sex, h, w, samples=300)
                self.assertAlmostEqual(sum(a.probabilities.values()), 1.0, places=9)
                self.assertEqual(a.probabilities, b.probabilities)
                best = a.recommendation.best.size
                self.assertEqual(best, max(chart.sizes, key=lambda s: (a.probabilities[s], -list(chart.sizes).index(s))))

    def test_all_measured_equals_plain_recommend(self):
        """줄자로 다 재면 흔들 항목이 없어 기존 size_fit.recommend 와 같은 답이어야 한다."""
        for chart in charts():
            if chart.category != 'upper':
                continue
            measured = {'shoulder_cm': 44, 'chest_circumference_cm': 96}
            r = su.recommend_unknown(chart, '남', 175, 70, measured=measured, samples=50)
            plain = recommend(chart, {'shoulder': 44.0, 'chest': 48.0})
            self.assertEqual(r.recommendation.best.size, plain.best.size)
            self.assertEqual(r.probabilities[plain.best.size], 1.0)

    def test_age_model_in_file(self):
        """F-40: 나이 식은 기존 키·몸무게 식 옆에 덧붙인다(기존 식은 그대로)."""
        m = su.load_model()
        for sx in ('m', 'f'):
            self.assertEqual(set(m['sex'][sx]['parts_age']), {'shoulder', 'chest', 'waist'})
            self.assertIn('d_age', m['sex'][sx]['parts_age']['waist'])
            lo, hi = m['sex'][sx]['age_range']
            self.assertTrue(lo <= 25 <= hi)

    def test_age_is_optional_and_older_means_larger_waist(self):
        base = su.predict('lower', '남', 175, 70)
        self.assertEqual(base.profile, su.predict('lower', '남', 175, 70, age=None).profile)   # 비우면 기존 식
        young = su.predict('lower', '남', 175, 70, age=25)
        old = su.predict('lower', '남', 175, 70, age=55)
        self.assertGreater(old.profile['waist_circumference_cm'], young.profile['waist_circumference_cm'] + 3)
        self.assertTrue(any('나이' in s for s in young.sources))
        self.assertFalse(any('나이' in s for s in base.sources))

    def test_age_changes_recommendation_reproducibly(self):
        chart = next(c for c in charts() if c.category == 'lower')
        a = su.recommend_unknown(chart, '여', 162, 58, samples=300, age=60)
        b = su.recommend_unknown(chart, '여', 162, 58, samples=300, age=60)
        self.assertEqual(a.probabilities, b.probabilities)
        self.assertAlmostEqual(sum(a.probabilities.values()), 1.0, places=9)

    def test_age_out_of_range(self):
        with self.assertRaises(ValueError):
            su.predict('upper', '남', 175, 70, age=12)
        with self.assertRaises(ValueError):
            su.predict('upper', '남', 175, 70, age=90)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            su.predict('upper', '', 175, 70)
        with self.assertRaises(ValueError):
            su.predict('upper', '남', 17.5, 70)
        with self.assertRaises(ValueError):
            su.predict('upper', '남', 175, 700)


if __name__ == '__main__':
    unittest.main()
