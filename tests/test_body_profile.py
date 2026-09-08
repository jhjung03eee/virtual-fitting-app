"""body_profile 테스트 — 둘레/단면 변환과 입력 검증이 핵심."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from body_profile import BodyProfile  # noqa: E402
from size_fit import SizeChart, recommend  # noqa: E402


class TestConversion(unittest.TestCase):
    def test_circumference_becomes_half(self):
        """가슴둘레 96 -> 가슴단면 48. 이걸 틀리면 사이즈 추천이 통째로 어긋난다."""
        p = BodyProfile(height_cm=175, chest_circumference_cm=96.0)
        self.assertEqual(p.to_chart_dimensions()['chest'], 48.0)

    def test_shoulder_is_not_halved(self):
        """어깨너비는 둘레가 아니라 폭이므로 그대로 써야 한다."""
        p = BodyProfile(height_cm=175, shoulder_cm=45.0)
        self.assertEqual(p.to_chart_dimensions()['shoulder'], 45.0)

    def test_missing_values_are_omitted(self):
        p = BodyProfile(height_cm=175, chest_circumference_cm=96.0)
        dims = p.to_chart_dimensions()
        self.assertIn('chest', dims)
        self.assertNotIn('waist', dims)
        self.assertNotIn('hip', dims)

    def test_all_fields(self):
        p = BodyProfile(height_cm=175, shoulder_cm=44.0, chest_circumference_cm=96.0,
                        waist_circumference_cm=80.0, hip_circumference_cm=94.0)
        self.assertEqual(p.to_chart_dimensions(),
                         {'shoulder': 44.0, 'chest': 48.0, 'waist': 40.0, 'hip': 47.0})


class TestValidation(unittest.TestCase):
    def test_reasonable_profile_has_no_problems(self):
        p = BodyProfile(height_cm=175, shoulder_cm=44.0, chest_circumference_cm=96.0)
        self.assertEqual(p.validate(), [])

    def test_unit_typo_is_caught(self):
        """96을 9.6으로 잘못 넣는 실수를 잡아야 한다."""
        p = BodyProfile(height_cm=175, chest_circumference_cm=9.6)
        problems = p.validate()
        self.assertEqual(len(problems), 1)
        self.assertIn('가슴둘레', problems[0])
        self.assertIn('cm', problems[0])

    def test_absurd_height(self):
        self.assertTrue(BodyProfile(height_cm=17.5).validate())

    def test_non_numeric(self):
        p = BodyProfile(height_cm=175, shoulder_cm='사십사')  # type: ignore[arg-type]
        self.assertTrue(any('숫자가 아닙니다' in x for x in p.validate()))

    def test_missing_fields_listed(self):
        p = BodyProfile(height_cm=175, chest_circumference_cm=96.0)
        missing = p.missing_fields()
        self.assertIn('어깨너비', missing)
        self.assertNotIn('가슴둘레', missing)


class TestFromInputs(unittest.TestCase):
    def test_blank_inputs_are_dropped(self):
        p = BodyProfile.from_inputs(height_cm=175, shoulder_cm='', chest_circumference_cm=96,
                                    waist_circumference_cm=0, name='정지훈')
        self.assertIsNone(p.shoulder_cm)
        self.assertIsNone(p.waist_circumference_cm)
        self.assertEqual(p.chest_circumference_cm, 96.0)
        self.assertEqual(p.name, '정지훈')

    def test_height_required(self):
        with self.assertRaises(ValueError):
            BodyProfile.from_inputs(chest_circumference_cm=96)


class TestPersistence(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        p = BodyProfile(height_cm=175, shoulder_cm=44.0, chest_circumference_cm=96.0,
                        name='테스트')
        with tempfile.TemporaryDirectory() as d:
            path = p.save(os.path.join(d, 'profile.json'))
            self.assertTrue(os.path.exists(path))
            loaded = BodyProfile.load(path)
        self.assertEqual(loaded, p)

    def test_load_ignores_unknown_keys(self):
        """예전 버전 파일에 없던 키가 있어도 깨지지 않아야 한다."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'p.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'height_cm': 170, 'legacy_field': 1}, f)
            loaded = BodyProfile.load(path)
        self.assertEqual(loaded.height_cm, 170)


class TestEndToEnd(unittest.TestCase):
    """입력한 둘레가 사이즈 추천까지 제대로 흘러가는지."""

    CHART = SizeChart(
        name='샘플 티셔츠', category='upper',
        sizes={
            'S': {'chest': 49.0, 'shoulder': 43.0},
            'M': {'chest': 52.0, 'shoulder': 45.0},
            'L': {'chest': 55.0, 'shoulder': 47.0},
        },
    )

    def test_recommendation_from_profile(self):
        # 가슴둘레 94 -> 단면 47. M(52)이면 여유 +5 로 레귤러
        p = BodyProfile(height_cm=175, chest_circumference_cm=94.0, shoulder_cm=44.5)
        rec = recommend(self.CHART, p.to_chart_dimensions())
        self.assertEqual(rec.best.size, 'M')

    def test_bigger_person_gets_bigger_size(self):
        small = BodyProfile(height_cm=170, chest_circumference_cm=88.0)
        large = BodyProfile(height_cm=185, chest_circumference_cm=104.0)
        s = recommend(self.CHART, small.to_chart_dimensions()).best.size
        l = recommend(self.CHART, large.to_chart_dimensions()).best.size
        order = list(self.CHART.sizes)
        self.assertLess(order.index(s), order.index(l))

    def test_partial_profile_still_recommends(self):
        """가슴둘레만 알아도 추천이 나와야 한다."""
        p = BodyProfile(height_cm=175, chest_circumference_cm=94.0)
        rec = recommend(self.CHART, p.to_chart_dimensions())
        self.assertIsNotNone(rec.best)
        self.assertTrue(any('제외한 항목' in n for n in rec.notes))


if __name__ == '__main__':
    unittest.main()
