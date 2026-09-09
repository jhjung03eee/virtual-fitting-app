"""치수표 추출 결과 검증층 테스트.

Vision 모델이 실제로 뱉을 법한 결과(오독, 둘레 표기, 빈 칸, 사이즈 뒤집힘)를
직접 만들어 넣는다. API도 GPU도 쓰지 않는다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from chart_extract import (  # noqa: E402
    build_chart, check_monotonic, check_ranges, normalize_label,
    normalize_size_name, normalize_sizes, size_sort_key,
)
from size_fit import recommend  # noqa: E402


GOOD = {
    'name': '오버핏 반팔 티셔츠',
    'category': 'upper',
    'sizes': {
        'S': {'가슴단면': 52, '어깨너비': 46, '총장': 68},
        'M': {'가슴단면': 55, '어깨너비': 48, '총장': 70},
        'L': {'가슴단면': 58, '어깨너비': 50, '총장': 72},
    },
}


class TestNormalizeLabel(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(normalize_label('가슴단면'), 'chest')
        self.assertEqual(normalize_label('어깨너비'), 'shoulder')
        self.assertEqual(normalize_label('총장'), 'length')
        self.assertEqual(normalize_label('밑위'), 'rise')

    def test_spacing_and_units(self):
        self.assertEqual(normalize_label('가슴 단면'), 'chest')
        self.assertEqual(normalize_label('어깨너비(cm)'), 'shoulder')
        self.assertEqual(normalize_label(' 총장 '), 'length')

    def test_misread_recovered(self):
        """OCR/VLM이 한 글자 틀려도 고정 어휘라 복구된다."""
        self.assertEqual(normalize_label('가슴단먼'), 'chest')
        self.assertEqual(normalize_label('어깨너바'), 'shoulder')

    def test_unknown_is_none(self):
        self.assertIsNone(normalize_label(''))
        self.assertIsNone(normalize_label('상품코드'))


class TestSizeNames(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_size_name('small'), 'S')
        self.assertEqual(normalize_size_name(' m '), 'M')
        self.assertEqual(normalize_size_name('2XL'), 'XXL')

    def test_alpha_order(self):
        got = sorted(['L', 'S', 'XL', 'M'], key=size_sort_key)
        self.assertEqual(got, ['S', 'M', 'L', 'XL'])

    def test_numeric_order(self):
        """숫자 사이즈는 문자열이 아니라 숫자로 정렬해야 한다 (100 > 95)."""
        got = sorted(['100', '95', '90'], key=size_sort_key)
        self.assertEqual(got, ['90', '95', '100'])

    def test_free_size_goes_last(self):
        self.assertEqual(sorted(['FREE', 'M'], key=size_sort_key), ['M', 'FREE'])


class TestNormalizeSizes(unittest.TestCase):
    def test_values_parsed(self):
        sizes, problems = normalize_sizes(GOOD['sizes'])
        self.assertEqual(sizes['M'], {'chest': 55.0, 'shoulder': 48.0, 'length': 70.0})
        self.assertEqual(problems, [])

    def test_string_values_with_units(self):
        sizes, _ = normalize_sizes({'M': {'가슴단면': '55cm', '총장': '70.5'}})
        self.assertEqual(sizes['M'], {'chest': 55.0, 'length': 70.5})

    def test_unknown_label_dropped_with_note(self):
        sizes, problems = normalize_sizes({'M': {'가슴단면': 55, '상품코드': 'A12'}})
        self.assertEqual(sizes['M'], {'chest': 55.0})
        self.assertTrue(any('상품코드' in p for p in problems))

    def test_empty_cell_dropped(self):
        """빈 칸을 0으로 채우면 '가슴단면 0cm'가 되어 추천이 망가진다."""
        sizes, problems = normalize_sizes({'M': {'가슴단면': 55, '어깨너비': '-'}})
        self.assertEqual(sizes['M'], {'chest': 55.0})
        self.assertTrue(problems)


class TestChecks(unittest.TestCase):
    def test_monotonic_ok(self):
        sizes, _ = normalize_sizes(GOOD['sizes'])
        self.assertEqual(check_monotonic(sizes), [])

    def test_monotonic_catches_misread(self):
        """L의 가슴단면을 58 대신 38로 잘못 읽은 경우."""
        broken = dict(GOOD['sizes'])
        broken['L'] = {'가슴단면': 38, '어깨너비': 50, '총장': 72}
        sizes, _ = normalize_sizes(broken)
        problems = check_monotonic(sizes)
        self.assertEqual(len(problems), 1)
        self.assertIn('L', problems[0])
        self.assertIn('chest', problems[0])

    def test_range_ok(self):
        sizes, _ = normalize_sizes(GOOD['sizes'])
        self.assertEqual(check_ranges(sizes), [])

    def test_range_detects_circumference_chart(self):
        """치수표가 단면이 아니라 둘레로 적힌 경우 반으로 나누라고 안내한다."""
        sizes, _ = normalize_sizes({'M': {'가슴단면': 110, '총장': 70}})
        problems = check_ranges(sizes)
        self.assertEqual(len(problems), 1)
        self.assertIn('둘레', problems[0])
        self.assertIn('55', problems[0])

    def test_range_detects_nonsense(self):
        sizes, _ = normalize_sizes({'M': {'가슴단면': 4, '총장': 70}})
        problems = check_ranges(sizes)
        self.assertTrue(any('벗어납니다' in p for p in problems))


class TestBuildChart(unittest.TestCase):
    def test_good_chart(self):
        chart, problems = build_chart(GOOD)
        self.assertIsNotNone(chart)
        self.assertEqual(chart.category, 'upper')
        self.assertEqual(chart.name, '오버핏 반팔 티셔츠')
        self.assertEqual(sorted(chart.sizes), ['L', 'M', 'S'])
        self.assertEqual(problems, [])

    def test_result_feeds_recommend(self):
        """추출한 치수표가 기존 추천 로직에 그대로 들어가야 의미가 있다."""
        chart, _ = build_chart(GOOD)
        rec = recommend(chart, {'chest': 48.0, 'shoulder': 45.0})
        self.assertIsNotNone(rec.best)
        self.assertIn(rec.best.size, chart.sizes)

    def test_extra_dimensions_kept_but_ignored(self):
        """소매길이는 몸 치수에 없으니 판정에서 빠질 뿐 오류가 아니다."""
        payload = dict(GOOD)
        payload['sizes'] = {s: dict(row, 소매길이=22) for s, row in GOOD['sizes'].items()}
        chart, problems = build_chart(payload)
        self.assertIn('sleeve', chart.sizes['M'])
        self.assertEqual(problems, [])
        rec = recommend(chart, {'chest': 48.0})
        self.assertIsNotNone(rec.best)

    def test_error_payload(self):
        chart, problems = build_chart({'error': '표가 흐려서 읽을 수 없음'})
        self.assertIsNone(chart)
        self.assertTrue(any('흐려서' in p for p in problems))

    def test_no_sizes(self):
        chart, problems = build_chart({'name': 'x', 'category': 'upper', 'sizes': {}})
        self.assertIsNone(chart)
        self.assertTrue(problems)

    def test_not_a_dict(self):
        chart, problems = build_chart('아무 말')
        self.assertIsNone(chart)
        self.assertTrue(problems)

    def test_category_guessed_for_lower(self):
        chart, problems = build_chart({
            'name': '데님 팬츠',
            'sizes': {
                'M': {'허리단면': 38, '허벅지단면': 30, '밑위': 28, '총장': 100},
                'L': {'허리단면': 40, '허벅지단면': 32, '밑위': 29, '총장': 102},
            },
        })
        self.assertEqual(chart.category, 'lower')
        self.assertTrue(any('추정' in p for p in problems))

    def test_chart_built_even_with_warnings(self):
        """경고가 있어도 표는 만들어야 사용자가 화면에서 고칠 수 있다."""
        broken = dict(GOOD, sizes=dict(GOOD['sizes']))
        broken['sizes']['L'] = {'가슴단면': 38, '어깨너비': 50, '총장': 72}
        chart, problems = build_chart(broken)
        self.assertIsNotNone(chart)
        self.assertTrue(problems)

    def test_unusable_chart_warns(self):
        chart, problems = build_chart({
            'category': 'upper',
            'sizes': {'M': {'소매길이': 22}, 'L': {'소매길이': 23}},
        })
        self.assertTrue(any('사이즈 추천이 안 됩니다' in p for p in problems))


if __name__ == '__main__':
    unittest.main()
