"""size_fit 단위 테스트 — mediapipe/torch 없이 순수 파이썬으로 돈다.

    python -m unittest discover -s tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from size_fit import (  # noqa: E402
    SizeChart, classify_ease, evaluate_size, recommend,
)

# 국내 쇼핑몰 상의 치수표 형태의 예시 (단면 cm)
UPPER = SizeChart(
    name='샘플 반팔 티셔츠',
    category='upper',
    sizes={
        'S': {'chest': 49.0, 'shoulder': 43.0, 'length': 66.0},
        'M': {'chest': 52.0, 'shoulder': 45.0, 'length': 68.0},
        'L': {'chest': 55.0, 'shoulder': 47.0, 'length': 70.0},
        'XL': {'chest': 58.0, 'shoulder': 49.0, 'length': 72.0},
    },
    source='예시 데이터',
)

LOWER = SizeChart(
    name='샘플 반바지',
    category='lower',
    sizes={
        'S': {'waist': 36.0, 'hip': 50.0},
        'M': {'waist': 38.0, 'hip': 52.0},
        'L': {'waist': 40.0, 'hip': 54.0},
    },
    source='예시 데이터',
)


class TestClassifyEase(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(classify_ease('upper', 'chest', -1.0), '작음')
        self.assertEqual(classify_ease('upper', 'chest', 1.0), '타이트')
        self.assertEqual(classify_ease('upper', 'chest', 5.0), '레귤러')
        self.assertEqual(classify_ease('upper', 'chest', 9.0), '루즈')
        self.assertEqual(classify_ease('upper', 'chest', 20.0), '오버핏')

    def test_band_boundaries_are_half_open(self):
        """경계값이 어느 한쪽에만 속해야 한다 (구간이 겹치거나 비면 안 됨)."""
        self.assertEqual(classify_ease('upper', 'chest', 3.0), '레귤러')
        self.assertEqual(classify_ease('upper', 'chest', 7.0), '루즈')

    def test_unknown_dimension(self):
        self.assertEqual(classify_ease('upper', 'sleeve', 3.0), '기준없음')


class TestEvaluateSize(unittest.TestCase):
    def test_ease_is_garment_minus_body(self):
        body = {'chest': 48.0, 'shoulder': 44.0}
        fit = evaluate_size(UPPER, 'M', body)
        eases = {d.dimension: d.ease_cm for d in fit.dimensions}
        self.assertEqual(eases['chest'], 4.0)
        self.assertEqual(eases['shoulder'], 1.0)

    def test_undersize_is_not_wearable(self):
        body = {'chest': 56.0}  # M(52)보다 몸이 큼
        self.assertFalse(evaluate_size(UPPER, 'M', body).wearable)
        self.assertTrue(evaluate_size(UPPER, 'XL', body).wearable)

    def test_dimension_without_body_measure_is_skipped(self):
        """몸 치수가 없는 항목(length)은 판정에 들어가면 안 된다."""
        fit = evaluate_size(UPPER, 'M', {'chest': 48.0})
        self.assertEqual([d.dimension for d in fit.dimensions], ['chest'])


class TestRecommend(unittest.TestCase):
    def test_picks_regular_fit(self):
        body = {'chest': 47.0, 'shoulder': 44.5}   # M이면 가슴 +5.0 (레귤러)
        rec = recommend(UPPER, body)
        self.assertEqual(rec.best.size, 'M')
        self.assertTrue(rec.best.wearable)

    def test_larger_body_gets_larger_size(self):
        small = recommend(UPPER, {'chest': 45.0, 'shoulder': 42.0}).best.size
        large = recommend(UPPER, {'chest': 53.0, 'shoulder': 48.0}).best.size
        order = list(UPPER.sizes)
        self.assertLess(order.index(small), order.index(large))

    def test_all_too_small_is_flagged(self):
        rec = recommend(UPPER, {'chest': 70.0})
        self.assertFalse(rec.best.wearable)
        self.assertTrue(any('작습니다' in n for n in rec.notes))
        self.assertIn('더 큰 사이즈', rec.message)

    def test_ranked_is_ordered_and_complete(self):
        rec = recommend(UPPER, {'chest': 47.0})
        self.assertEqual(len(rec.ranked), len(UPPER.sizes))
        scores = [f.score for f in rec.ranked if f.wearable]
        self.assertEqual(scores, sorted(scores))

    def test_no_common_dimension(self):
        rec = recommend(UPPER, {'inseam': 70.0})
        self.assertIsNone(rec.best)
        self.assertTrue(rec.notes)

    def test_missing_dimension_is_noted(self):
        rec = recommend(UPPER, {'chest': 47.0})  # shoulder/length 없음
        self.assertTrue(any('제외한 항목' in n for n in rec.notes))

    def test_notes_use_korean_dimension_names(self):
        """안내 문구에 'length'라고 적히면 무슨 항목인지 알 수 없다."""
        rec = recommend(UPPER, {'chest': 47.0})
        note = next(n for n in rec.notes if '제외한 항목' in n)
        self.assertIn('총장', note)
        self.assertNotIn('length', note)

    def test_unused_body_input_is_noted(self):
        """치수표에 없는 항목을 입력하면 조용히 버리지 말고 알려야 한다.

        래글런 소매 상의는 어깨 솔기가 없어 치수표에 어깨너비가 아예 없다.
        사용자가 입력한 값이 왜 반영되지 않았는지 알 수 있어야 한다.
        """
        chart = SizeChart(name='래글런', category='upper', sizes={
            'S': {'chest': 56.0}, 'M': {'chest': 58.0},
        })
        rec = recommend(chart, {'chest': 48.0, 'shoulder': 45.0})
        self.assertIsNotNone(rec.best)
        note = next(n for n in rec.notes if '사용하지 않았습니다' in n)
        self.assertIn('어깨너비', note)

    def test_lower_category(self):
        rec = recommend(LOWER, {'waist': 35.0, 'hip': 48.0})
        self.assertTrue(rec.best.wearable)
        self.assertIn(rec.best.size, LOWER.sizes)


class TestSizeChart(unittest.TestCase):
    def test_from_dict_roundtrip(self):
        chart = SizeChart.from_dict({
            'name': 'x', 'category': 'upper',
            'sizes': {'M': {'chest': 50.0}},
        })
        self.assertEqual(chart.dimensions(), ['chest'])
        self.assertEqual(chart.source, '')


if __name__ == '__main__':
    unittest.main()
