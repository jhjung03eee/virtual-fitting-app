"""치수표 추출 결과 검증층 테스트.

Vision 모델이 실제로 뱉을 법한 결과(오독, 둘레 표기, 빈 칸, 사이즈 뒤집힘)를
직접 만들어 넣는다. API도 GPU도 쓰지 않는다.

`TestRealMusinsaCharts`는 `data/samples/`에 받아둔 **실제 무신사 치수표 3건**을
사람이 눈으로 읽어 옮긴 것이다. 이미지 자체는 저작권 때문에 커밋하지 않지만
표에 적힌 숫자와 항목명은 여기 남겨 회귀 테스트로 쓴다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from chart_extract import (  # noqa: E402
    build_chart, check_monotonic, check_ranges, is_ignored_row, normalize_label,
    normalize_size_name, normalize_sizes, size_sort_key, split_variant_size,
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

    def test_sleeve_family_kept_apart(self):
        """소매 관련 세 항목은 서로 다른 치수다. 뭉뚱그리면 안 된다."""
        self.assertEqual(normalize_label('소매길이'), 'sleeve')
        self.assertEqual(normalize_label('소매부리단면'), 'cuff')
        self.assertEqual(normalize_label('암홀'), 'armhole')

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

    def test_color_prefix_split(self):
        """무신사는 색상별로 행을 나눈다: '화이트S'."""
        self.assertEqual(split_variant_size('화이트S'), ('화이트', 'S'))
        self.assertEqual(split_variant_size('블랙 M'), ('블랙', 'M'))
        self.assertEqual(split_variant_size('L'), ('', 'L'))

    def test_digit_prefixed_size_not_split(self):
        """'2XL'을 변형 '2' + 'XL'로 쪼개면 안 된다."""
        self.assertEqual(split_variant_size('2XL'), ('', 'XXL'))

    def test_alpha_order(self):
        got = sorted(['L', 'S', 'XL', 'M'], key=size_sort_key)
        self.assertEqual(got, ['S', 'M', 'L', 'XL'])

    def test_numeric_order(self):
        """숫자 사이즈는 문자열이 아니라 숫자로 정렬해야 한다 (100 > 95)."""
        got = sorted(['100', '95', '90'], key=size_sort_key)
        self.assertEqual(got, ['90', '95', '100'])

    def test_free_size_goes_last(self):
        self.assertEqual(sorted(['FREE', 'M'], key=size_sort_key), ['M', 'FREE'])

    def test_ignored_rows(self):
        self.assertTrue(is_ignored_row('내 사이즈'))
        self.assertTrue(is_ignored_row('cm'))
        self.assertFalse(is_ignored_row('M'))


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

    def test_empty_cell_dropped_quietly(self):
        """빈 칸('-')은 정상이다. 0으로 채우면 '가슴단면 0cm'가 되어 추천이 망가진다."""
        sizes, problems = normalize_sizes({'M': {'가슴단면': 55, '소매부리단면': '-'}})
        self.assertEqual(sizes['M'], {'chest': 55.0})
        self.assertEqual(problems, [])

    def test_ui_row_skipped(self):
        sizes, problems = normalize_sizes({
            '내 사이즈': {'가슴단면': '사이즈를 직접 입력해주세요'},
            'M': {'가슴단면': 55},
        })
        self.assertEqual(sorted(sizes), ['M'])
        self.assertEqual(problems, [])

    def test_identical_colors_merged(self):
        sizes, problems = normalize_sizes({
            '화이트S': {'가슴단면': 55}, '화이트M': {'가슴단면': 58},
            '블랙S': {'가슴단면': 55}, '블랙M': {'가슴단면': 58},
        })
        self.assertEqual(sorted(sizes), ['M', 'S'])
        self.assertTrue(any('합쳤습니다' in p for p in problems))

    def test_differing_colors_kept_separate(self):
        sizes, problems = normalize_sizes({
            '화이트M': {'가슴단면': 58}, '블랙M': {'가슴단면': 60},
        })
        self.assertEqual(sorted(sizes), ['블랙M', '화이트M'])
        self.assertTrue(any('따로' in p for p in problems))


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

    def test_monotonic_compares_within_same_color(self):
        """색상별 행이 남아 있으면 같은 색끼리만 비교해야 한다."""
        sizes, _ = normalize_sizes({
            '화이트S': {'가슴단면': 55}, '화이트M': {'가슴단면': 58},
            '블랙S': {'가슴단면': 54}, '블랙M': {'가슴단면': 57},
        })
        self.assertEqual(check_monotonic(sizes), [])

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


class TestRealMusinsaCharts(unittest.TestCase):
    """data/samples/ 에 받아둔 실제 무신사 치수표를 옮긴 것."""

    # tshirts_size.jpg — 색상별로 행이 나뉘고, 빈 칸이 '-'로 채워져 있다
    TSHIRT = {
        'name': '반소매 티셔츠',
        'category': 'upper',
        'unit': 'cm',
        'sizes': {
            '내 사이즈': {'총장': '사이즈를 직접 입력해주세요'},
            '화이트S': {'총장': 67, '어깨너비': 52, '가슴단면': 55, '소매길이': 23,
                      '밑단단면': '-', '소매부리단면': '-', '암홀': '-'},
            '화이트M': {'총장': 71, '어깨너비': 54, '가슴단면': 58, '소매길이': 24,
                      '밑단단면': '-', '소매부리단면': '-', '암홀': '-'},
            '화이트L': {'총장': 74, '어깨너비': 56, '가슴단면': 61, '소매길이': 25,
                      '밑단단면': '-', '소매부리단면': '-', '암홀': '-'},
            '블랙S': {'총장': 67, '어깨너비': 52, '가슴단면': 55, '소매길이': 23},
            '블랙M': {'총장': 71, '어깨너비': 54, '가슴단면': 58, '소매길이': 24},
            '블랙L': {'총장': 74, '어깨너비': 56, '가슴단면': 61, '소매길이': 25},
        },
    }

    # pants_size.jpg
    PANTS = {
        'name': '코듀로이 와이드 팬츠',
        'category': 'lower',
        'unit': 'cm',
        'sizes': {
            '내 사이즈': {'총장': '사이즈를 직접 입력해주세요'},
            'S': {'총장': 106, '허리단면': 40, '엉덩이단면': 52, '허벅지단면': 30,
                  '밑위': 30, '밑단단면': 22},
            'M': {'총장': 108, '허리단면': 42, '엉덩이단면': 54, '허벅지단면': 31,
                  '밑위': 31, '밑단단면': 22.5},
            'L': {'총장': 110, '허리단면': 44, '엉덩이단면': 56, '허벅지단면': 32,
                  '밑위': 32, '밑단단면': 23},
        },
    }

    # shirts_size.jpg — 래글런 소매라 치수표에 '어깨너비'가 아예 없다.
    # 측정 항목이 가슴단면·총장 둘뿐인 표도 추천이 되어야 한다.
    SHIRT = {
        'name': '긴소매 래글런 티셔츠',
        'category': 'upper',
        'unit': 'cm',
        'sizes': {
            '내 사이즈': {'총장': '사이즈를 직접 입력해주세요'},
            'S': {'총장': 69, '가슴단면': 56, '소매부리단면': '-', '전체소매길이': '-',
                  '밑단단면': '-', '암홀': '-'},
            'M': {'총장': 72, '가슴단면': 58, '소매부리단면': '-', '전체소매길이': '-',
                  '밑단단면': '-', '암홀': '-'},
            'L': {'총장': 73, '가슴단면': 60, '소매부리단면': '-', '전체소매길이': '-',
                  '밑단단면': '-', '암홀': '-'},
            'XL': {'총장': 75, '가슴단면': 62, '소매부리단면': '-', '전체소매길이': '-',
                   '밑단단면': '-', '암홀': '-'},
        },
    }

    # shirts_size.jpg 를 처음 캡처했을 때 — '기준표 사이즈' 탭을 잘못 잡은 경우.
    # 실측 치수가 없어 추천에 쓸 수 없다.
    COUNTRY_TABLE = {
        'sizes': {
            'S': {'한국': 90, '미국': '90-95', '영국': 15, '일본': 38, '프랑스': '42, 44', '유럽': 46},
            'M': {'한국': 95, '미국': '95-100', '영국': '15.5-16', '일본': 40, '프랑스': '46, 48', '유럽': 48},
            'L': {'한국': 100, '미국': '100-105', '영국': 16.5, '일본': 42, '프랑스': '50, 52', '유럽': 50},
        },
    }

    def test_tshirt_clean(self):
        chart, problems = build_chart(self.TSHIRT)
        self.assertEqual(chart.category, 'upper')
        # 색상별 행이 합쳐져 S/M/L 세 개만 남아야 한다
        self.assertEqual(sorted(chart.sizes, key=lambda s: ['S', 'M', 'L'].index(s)),
                         ['S', 'M', 'L'])
        self.assertEqual(chart.sizes['M'],
                         {'length': 71.0, 'shoulder': 54.0, 'chest': 58.0, 'sleeve': 24.0})
        # 남는 경고는 '색상을 합쳤다'는 안내 하나뿐이어야 한다
        self.assertEqual(len(problems), 1)
        self.assertIn('합쳤습니다', problems[0])

    def test_tshirt_recommends(self):
        """가슴둘레 96(=단면 48)인 사람에게 실제로 사이즈가 나와야 한다."""
        chart, _ = build_chart(self.TSHIRT)
        rec = recommend(chart, {'chest': 48.0, 'shoulder': 45.0})
        self.assertIsNotNone(rec.best)
        self.assertTrue(rec.best.wearable)

    def test_pants_clean(self):
        chart, problems = build_chart(self.PANTS)
        self.assertEqual(chart.category, 'lower')
        self.assertEqual(sorted(chart.sizes), ['L', 'M', 'S'])
        self.assertEqual(chart.sizes['M']['waist'], 42.0)
        self.assertEqual(chart.sizes['M']['rise'], 31.0)
        self.assertEqual(problems, [])

    def test_pants_recommends(self):
        chart, _ = build_chart(self.PANTS)
        rec = recommend(chart, {'waist': 40.0, 'hip': 49.0})
        self.assertIsNotNone(rec.best)
        self.assertTrue(rec.best.wearable)

    def test_shirt_raglan_without_shoulder(self):
        chart, problems = build_chart(self.SHIRT)
        self.assertEqual(sorted(chart.sizes, key=lambda x: ['S', 'M', 'L', 'XL'].index(x)),
                         ['S', 'M', 'L', 'XL'])
        self.assertEqual(chart.sizes['M'], {'length': 72.0, 'chest': 58.0})
        self.assertNotIn('shoulder', chart.sizes['M'])
        self.assertEqual(problems, [])

    def test_shirt_recommends_and_reports_unused_shoulder(self):
        """어깨너비를 입력해도 표에 없으면 쓰지 않았다고 알려야 한다."""
        chart, _ = build_chart(self.SHIRT)
        rec = recommend(chart, {'chest': 48.0, 'shoulder': 45.0})
        self.assertIsNotNone(rec.best)
        self.assertTrue(any('어깨너비' in n and '사용하지 않았습니다' in n
                            for n in rec.notes))

    def test_full_sleeve_label(self):
        """'전체소매길이'가 '소매부리단면'과 섞이면 안 된다."""
        self.assertEqual(normalize_label('전체소매길이'), 'sleeve')

    def test_country_table_rejected_with_guidance(self):
        """환산표를 올리면 '실측 사이즈 탭을 캡처하라'고 안내해야 한다."""
        chart, problems = build_chart(self.COUNTRY_TABLE)
        self.assertIsNone(chart)
        self.assertEqual(len(problems), 1)
        self.assertIn('실측', problems[0])

    def test_country_table_reported_by_model(self):
        chart, problems = build_chart({'error': '국가별 환산표'})
        self.assertIsNone(chart)
        self.assertIn('실측', problems[0])


if __name__ == '__main__':
    unittest.main()
