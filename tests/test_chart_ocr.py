"""OCR 상자에서 표를 복원하는 테스트. EasyOCR을 부르지 않고 상자를 직접 만든다."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from chart_extract import build_chart  # noqa: E402
from chart_ocr import read_chart_image, table_from_boxes  # noqa: E402
from chart_upload import ChartReadError  # noqa: E402


def box(x, y, text, confidence=0.99, width=40, height=20):
    left, top = x - width / 2, y - height / 2
    return ([[left, top], [left + width, top], [left + width, top + height], [left, top + height]], text, confidence)


def table(rows, confidence=0.99, step_x=100, step_y=40):
    return [box(50 + c * step_x, 30 + r * step_y, text, confidence)
            for r, row in enumerate(rows) for c, text in enumerate(row) if text]


SIZE_ROWS = [
    ['사이즈', '가슴단면', '어깨너비', '총장'],
    ['M', '51.5', '43', '68'],
    ['L', '54.5', '44.5', '71'],
    ['XL', '58.5', '46.5', '74'],
]
SIZE_COLUMNS = [
    ['cm', 'M', 'L', 'XL'],
    ['가슴단면', '51.5', '54.5', '58.5'],
    ['어깨너비', '43', '44.5', '46.5'],
    ['총장', '68', '71', '74'],
]


class TestTableFromBoxes(unittest.TestCase):
    def check_uniqlo(self, boxes):
        extracted, notes = table_from_boxes(boxes)
        chart, problems = build_chart(extracted)
        self.assertEqual(chart.category, 'upper')
        self.assertEqual(chart.sizes['M'], {'chest': 51.5, 'shoulder': 43.0, 'length': 68.0})
        self.assertEqual(chart.sizes['XL'], {'chest': 58.5, 'shoulder': 46.5, 'length': 74.0})
        self.assertEqual(notes + problems, [])

    def test_sizes_in_rows(self):
        self.check_uniqlo(table(SIZE_ROWS))

    def test_sizes_in_columns(self):
        self.check_uniqlo(table(SIZE_COLUMNS))

    def test_order_of_boxes_does_not_matter(self):
        self.check_uniqlo(list(reversed(table(SIZE_ROWS))))

    def test_slightly_tilted_row(self):
        boxes = table(SIZE_ROWS)
        tilted = [([[p[0], p[1] + index * 0.3] for p in corners], text, conf)
                  for index, (corners, text, conf) in enumerate(boxes)]
        self.check_uniqlo(tilted)

    def test_merged_box_is_split(self):
        boxes = table(SIZE_ROWS[:1])
        boxes += [box(200, 70, 'M 51.5 43 68', width=340)]
        boxes += table([[], [], ['L', '54.5', '44.5', '71'], ['XL', '58.5', '46.5', '74']])
        extracted, _ = table_from_boxes(boxes)
        self.assertEqual(set(extracted['sizes']), {'M', 'L', 'XL'})
        self.assertEqual(len(extracted['sizes']['M']), 3)

    def test_comma_and_unit(self):
        rows = [['사이즈', '가슴단면', '총장'], ['M', '51,5', '68cm'], ['L', '54.5', '71']]
        extracted, _ = table_from_boxes(table(rows))
        self.assertEqual(extracted['sizes']['M'], {'가슴단면': 51.5, '총장': 68.0})

    def test_blank_cell_stays_blank(self):
        rows = [['사이즈', '가슴단면', '총장'], ['M', '-', '68'], ['L', '54.5', '71']]
        extracted, notes = table_from_boxes(table(rows))
        self.assertEqual(extracted['sizes']['M'], {'총장': 68.0})
        self.assertEqual(notes, [])

    def test_numeric_sizes(self):
        rows = [['사이즈', '허리단면', '밑위'], ['28', '37', '25'], ['30', '39', '26']]
        extracted, _ = table_from_boxes(table(rows))
        self.assertEqual(extracted['category'], 'lower')
        self.assertEqual(extracted['sizes']['30']['허리단면'], 39.0)

    def test_note_below_table_is_ignored(self):
        boxes = table(SIZE_ROWS) + table([[], [], [], [], ['측정 방법에 따라 1~3 오차', '', '']])
        extracted, _ = table_from_boxes(boxes)
        self.assertEqual(set(extracted['sizes']), {'M', 'L', 'XL'})

    def test_low_confidence_number_asks_to_check(self):
        boxes = table(SIZE_ROWS)
        boxes = [(corners, text, 0.2 if text == '54.5' else conf) for corners, text, conf in boxes]
        _, notes = table_from_boxes(boxes)
        self.assertTrue(any('54.5' in note and '확인' in note for note in notes))

    def test_garbled_number_is_reported_not_guessed(self):
        rows = [['사이즈', '가슴단면', '총장'], ['M', '5l.5', '68'], ['L', '54.5', '71']]
        extracted, notes = table_from_boxes(table(rows))
        self.assertNotIn('가슴단면', extracted['sizes']['M'])
        self.assertTrue(any('5l.5' in note for note in notes))

    def test_misread_order_is_caught_by_validation(self):
        rows = [['사이즈', '가슴단면', '총장'], ['M', '54.5', '68'], ['L', '51.5', '71']]
        extracted, _ = table_from_boxes(table(rows))
        _, problems = build_chart(extracted)
        self.assertTrue(any('큰 사이즈가 더 작습니다' in problem for problem in problems))

    def test_country_chart(self):
        rows = [['한국', '미국', '유럽'], ['90', 'S', '44'], ['95', 'M', '46']]
        extracted, _ = table_from_boxes(table(rows))
        chart, problems = build_chart(extracted)
        self.assertIsNone(chart)
        self.assertIn('실측 사이즈', problems[0])

    def test_no_table(self):
        chart, problems = build_chart(table_from_boxes(table([['봄 신상품'], ['무료 배송']]))[0])
        self.assertIsNone(chart)
        self.assertIn('표를 찾지 못했', problems[0])

    def test_empty(self):
        self.assertIn('error', table_from_boxes([])[0])


class TestRecovery(unittest.TestCase):
    def test_missed_row_name_is_recovered(self):
        rows = [row[:] for row in SIZE_ROWS]
        rows[2][0] = ''
        extracted, notes = table_from_boxes(table(rows), recover=lambda *area: 'L')
        self.assertEqual(extracted['sizes']['L']['가슴단면'], 54.5)
        self.assertEqual(notes, [])

    def test_unrecovered_row_warns(self):
        rows = [row[:] for row in SIZE_ROWS]
        rows[2][0] = ''
        extracted, notes = table_from_boxes(table(rows), recover=lambda *area: None)
        self.assertNotIn('L', extracted['sizes'])
        self.assertTrue(any('사이즈 이름' in note for note in notes))

    def test_missed_column_name_is_recovered(self):
        rows = [row[:] for row in SIZE_COLUMNS]
        rows[0][2] = ''
        extracted, _ = table_from_boxes(table(rows), recover=lambda *area: 'L')
        self.assertEqual(extracted['sizes']['L']['어깨너비'], 44.5)

    def test_only_one_column_name_found(self):
        rows = [row[:] for row in SIZE_COLUMNS]
        rows[0][2] = rows[0][3] = ''
        answers = iter(['L', 'XL'])
        extracted, _ = table_from_boxes(table(rows), recover=lambda *area: next(answers))
        self.assertEqual(set(extracted['sizes']), {'M', 'L', 'XL'})

    def test_recovered_garbage_is_rejected(self):
        rows = [row[:] for row in SIZE_ROWS]
        rows[2][0] = ''
        extracted, _ = table_from_boxes(table(rows), recover=lambda *area: 'EOE')
        self.assertNotIn('EOE', extracted['sizes'])

    def test_misread_header_is_matched_loosely(self):
        rows = [['사이즈', '허리단면', '얼덩이', '총장'], ['S', '40', '50', '106'], ['M', '42', '52', '108']]
        extracted, notes = table_from_boxes(table(rows))
        chart, _ = build_chart(extracted)
        self.assertEqual(chart.sizes['S']['hip'], 50.0)
        self.assertEqual(notes, [])

    def test_ambiguous_header_is_not_guessed(self):
        # 문위는 밑위와 흉위(가슴)에 똑같이 가깝다. 가슴으로 넣으면 추천이 틀어진다
        rows = [['사이즈', '허리단면', '문위', '총장'], ['S', '40', '30', '106'], ['M', '42', '31', '108']]
        extracted, notes = table_from_boxes(table(rows))
        chart, _ = build_chart(extracted)
        self.assertNotIn('chest', chart.sizes['S'])
        self.assertNotIn('rise', chart.sizes['S'])
        self.assertTrue(any('문위' in note for note in notes))

    def test_unknown_header_warns_instead_of_silently_dropping(self):
        rows = [['사이즈', '허리단면', '%%', '총장'], ['S', '40', '30', '106'], ['M', '42', '31', '108']]
        _, notes = table_from_boxes(table(rows))
        self.assertTrue(any('%%' in note for note in notes))


class FakeDigitReader:
    def __init__(self, answers):
        self.answers = iter(answers)

    def recognize(self, crop, allowlist, detail, paragraph):
        return [(None, next(self.answers), 0.99)]


class FakeReader:
    def __init__(self, boxes=None, error=None):
        self.boxes, self.error = boxes, error
        self.shape = None

    def readtext(self, array, detail, paragraph):
        self.shape = array.shape
        if self.error:
            raise self.error
        return self.boxes


class TestReadChartImage(unittest.TestCase):
    def test_numbers_are_reread(self):
        from PIL import Image
        boxes = table([['사이즈', '가슴단면', '총장'], ['M', '57.5', '706'], ['L', '54.5', '108']])
        chart, _ = read_chart_image(Image.new('RGB', (2000, 1000), 'white'), FakeReader(boxes),
                                    FakeDigitReader(['51.5', '106', '54.5', '108']))
        self.assertEqual(chart.sizes['M'], {'chest': 51.5, 'length': 106.0})

    def test_small_image_is_enlarged(self):
        from PIL import Image
        reader = FakeReader(table(SIZE_ROWS))
        chart, _ = read_chart_image(Image.new('RGB', (400, 200), 'white'), reader)
        self.assertEqual(reader.shape[:2], (400, 800))
        self.assertEqual(chart.sizes['L']['chest'], 54.5)

    def test_reader_failure_becomes_message(self):
        from PIL import Image
        with self.assertRaises(ChartReadError):
            read_chart_image(Image.new('RGB', (40, 20)), FakeReader(error=RuntimeError('fail')))


if __name__ == '__main__':
    unittest.main()
