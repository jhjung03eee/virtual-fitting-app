"""reference_fit 단위 테스트 — 기준 옷 비교 추천. 순수 파이썬으로 돈다.

    python -m unittest discover -s tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from reference_fit import (  # noqa: E402
    ReferenceGarment, compare_size, label_diff, recommend_by_reference, verdict,
)
from size_fit import SizeChart  # noqa: E402

UPPER = SizeChart(
    name='샘플 반팔 티셔츠',
    category='upper',
    sizes={
        'S': {'chest': 49.0, 'shoulder': 43.0, 'length': 66.0},
        'M': {'chest': 52.0, 'shoulder': 45.0, 'length': 68.0},
        'L': {'chest': 55.0, 'shoulder': 47.0, 'length': 70.0},
        'XL': {'chest': 58.0, 'shoulder': 49.0, 'length': 72.0},
    },
)

LOWER = SizeChart(
    name='샘플 바지',
    category='lower',
    sizes={
        'S': {'waist': 36.0, 'hip': 50.0, 'length': 100.0},
        'M': {'waist': 38.0, 'hip': 52.0, 'length': 102.0},
        'L': {'waist': 40.0, 'hip': 54.0, 'length': 104.0},
    },
)


class TestLabelDiff(unittest.TestCase):
    def test_within_tolerance_is_same(self):
        self.assertEqual(label_diff('chest', 1.0), '거의 같음')
        self.assertEqual(label_diff('chest', -0.5), '거의 같음')

    def test_slight_and_clear(self):
        self.assertEqual(label_diff('chest', 2.0), '조금 큼')
        self.assertEqual(label_diff('chest', -2.0), '조금 작음')
        self.assertEqual(label_diff('chest', 4.0), '큼')

    def test_length_is_more_lenient(self):
        """기장은 재는 위치에 따라 2cm 정도는 쉽게 달라진다."""
        self.assertEqual(label_diff('length', 2.0), '거의 같음')
        self.assertEqual(label_diff('chest', 2.0), '조금 큼')


class TestVerdict(unittest.TestCase):
    def test_same_is_fit(self):
        self.assertEqual(verdict('upper', {'chest': 0.0, 'shoulder': 0.0}), '맞음')

    def test_small_and_large(self):
        self.assertEqual(verdict('upper', {'chest': -3.0, 'shoulder': -2.0}), '작음')
        self.assertEqual(verdict('upper', {'chest': 4.0, 'shoulder': 3.0}), '큼')

    def test_large_side_is_more_lenient(self):
        """+2cm 는 맞음이지만 -2cm 는 작음 — 작은 옷은 못 입는다."""
        self.assertEqual(verdict('upper', {'chest': 2.0}), '맞음')
        self.assertEqual(verdict('upper', {'chest': -2.0}), '작음')

    def test_falls_back_to_common_dimensions(self):
        """판정 항목(가슴 등)이 없으면 있는 항목으로라도 판정한다."""
        self.assertEqual(verdict('upper', {'sleeve': -5.0}), '작음')


class TestCompareSize(unittest.TestCase):
    def test_diff_is_garment_minus_reference(self):
        ref = ReferenceGarment('upper', {'chest': 51.0})
        c = compare_size(UPPER, 'M', ref)
        self.assertEqual(c.diffs[0].diff_cm, 1.0)

    def test_identical_has_zero_score(self):
        ref = ReferenceGarment.from_chart(UPPER, 'M')
        self.assertEqual(compare_size(UPPER, 'M', ref).score, 0.0)

    def test_tighter_is_worse_than_looser(self):
        """같은 3cm 차이라도 기준보다 작은 쪽이 점수가 나빠야 한다."""
        ref = ReferenceGarment('upper', {'chest': 52.0})
        small = compare_size(UPPER, 'S', ref)  # -3
        large = compare_size(UPPER, 'L', ref)  # +3
        self.assertGreater(small.score, large.score)

    def test_unmeasured_dimension_is_skipped(self):
        ref = ReferenceGarment('upper', {'chest': 52.0})
        dims = [d.dimension for d in compare_size(UPPER, 'M', ref).diffs]
        self.assertEqual(dims, ['chest'])


class TestRecommendByReference(unittest.TestCase):
    def test_picks_closest_size(self):
        ref = ReferenceGarment('upper', {'chest': 54.5, 'shoulder': 46.5, 'length': 69.0})
        rec = recommend_by_reference(UPPER, ref)
        self.assertEqual(rec.best.size, 'L')
        self.assertEqual(len(rec.ranked), 4)

    def test_same_chart_recommends_same_size(self):
        for size in UPPER.sizes:
            ref = ReferenceGarment.from_chart(UPPER, size)
            self.assertEqual(recommend_by_reference(UPPER, ref).best.size, size)

    def test_between_sizes_prefers_larger(self):
        """기준 옷이 두 사이즈 정확히 가운데면 큰 쪽(입을 수 있는 쪽)을 고른다."""
        ref = ReferenceGarment('upper', {'chest': 53.5})
        self.assertEqual(recommend_by_reference(UPPER, ref).best.size, 'L')

    def test_lower(self):
        ref = ReferenceGarment('lower', {'waist': 38.5, 'length': 102.0})
        self.assertEqual(recommend_by_reference(LOWER, ref).best.size, 'M')

    def test_hip_is_not_used(self):
        """하의 판정은 총장·허리만(2026-10-26). 엉덩이는 치수표·기준 옷에 있어도 안 쓴다."""
        ref = ReferenceGarment('lower', {'waist': 38.0, 'hip': 60.0})   # 엉덩이만 크게 다름
        rec = recommend_by_reference(LOWER, ref)
        self.assertEqual(rec.best.size, 'M')
        self.assertEqual([d.dimension for d in rec.best.diffs], ['waist'])

    def test_upper_uses_length_shoulder_chest(self):
        ref = ReferenceGarment('upper', {'chest': 52.0, 'shoulder': 45.0, 'length': 68.0})
        dims = [d.dimension for d in recommend_by_reference(UPPER, ref).best.diffs]
        self.assertEqual(dims, ['chest', 'length', 'shoulder'])

    def test_category_mismatch(self):
        ref = ReferenceGarment('lower', {'waist': 38.0})
        rec = recommend_by_reference(UPPER, ref)
        self.assertIsNone(rec.best)
        self.assertIn('종류', rec.notes[0])

    def test_no_common_dimension(self):
        ref = ReferenceGarment('upper', {'sleeve': 20.0})
        rec = recommend_by_reference(UPPER, ref)
        self.assertIsNone(rec.best)
        self.assertIn('가슴단면', rec.notes[0])

    def test_notes_missing_and_unused(self):
        ref = ReferenceGarment('upper', {'chest': 52.0, 'sleeve': 20.0})
        notes = ' '.join(recommend_by_reference(UPPER, ref).notes)
        self.assertIn('어깨너비', notes)
        self.assertIn('소매길이', notes)

    def test_all_too_small_is_noted(self):
        ref = ReferenceGarment('upper', {'chest': 65.0, 'shoulder': 55.0})
        rec = recommend_by_reference(UPPER, ref)
        self.assertEqual(rec.best.size, 'XL')
        self.assertIn('모든 사이즈가 기준 옷보다 작습니다.', rec.notes)

    def test_from_inputs_drops_blanks(self):
        ref = ReferenceGarment.from_inputs('upper', chest=52, shoulder=None, length=0, sleeve='')
        self.assertEqual(ref.dims, {'chest': 52.0})


if __name__ == '__main__':
    unittest.main()
