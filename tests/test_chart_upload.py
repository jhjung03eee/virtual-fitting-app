"""치수표 사진 읽기와 입력 방식 선택 테스트. 실제 API를 부르지 않고 가짜 클라이언트를 쓴다."""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from PIL import Image  # noqa: E402

from chart_upload import (  # noqa: E402
    ChartReadError, chart_from_text, chart_to_json, encode_image, extract_from_image, parse_model_json,
    read_chart_image, save_chart,
)
from size_unknown import size_input_mode  # noqa: E402


GOOD_ANSWER = json.dumps({
    'name': '테스트 반팔',
    'category': 'upper',
    'unit': 'cm',
    'sizes': {
        'M': {'가슴단면': 51.5, '어깨너비': 43, '총장': 68},
        'L': {'가슴단면': 54.5, '어깨너비': 44.5, '총장': 71},
    },
}, ensure_ascii=False)


class FakeClient:
    def __init__(self, text=GOOD_ANSWER, stop_reason='end_turn'):
        self.requests = []
        self.response = SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type='thinking', thinking=''), SimpleNamespace(type='text', text=text)],
        )
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def sample_image():
    return Image.new('RGB', (40, 30), 'white')


class TestParseModelJson(unittest.TestCase):
    def test_code_fence_and_sentence(self):
        parsed = parse_model_json('결과입니다.\n```json\n{"sizes": {"M": {}}}\n```')
        self.assertEqual(parsed, {'sizes': {'M': {}}})

    def test_no_json(self):
        with self.assertRaises(ChartReadError):
            parse_model_json('표를 찾을 수 없습니다')

    def test_broken_json(self):
        with self.assertRaises(ChartReadError):
            parse_model_json('{"sizes": ')


class TestExtract(unittest.TestCase):
    def test_request_has_image_and_prompt(self):
        client = FakeClient()
        extract_from_image(sample_image(), client)
        content = client.requests[0]['messages'][0]['content']
        self.assertEqual(content[0]['type'], 'image')
        self.assertEqual(content[0]['source']['media_type'], 'image/png')
        self.assertIn('실측 치수표', content[1]['text'])

    def test_good_answer_becomes_chart(self):
        chart, problems = read_chart_image(sample_image(), FakeClient())
        self.assertEqual(chart.sizes['L']['chest'], 54.5)
        self.assertEqual(problems, [])

    def test_refusal(self):
        with self.assertRaises(ChartReadError):
            extract_from_image(sample_image(), FakeClient(stop_reason='refusal'))

    def test_cut_off(self):
        with self.assertRaises(ChartReadError):
            extract_from_image(sample_image(), FakeClient(stop_reason='max_tokens'))

    def test_country_chart_is_rejected(self):
        answer = json.dumps({'error': '국가별 환산표'}, ensure_ascii=False)
        chart, problems = read_chart_image(sample_image(), FakeClient(answer))
        self.assertIsNone(chart)
        self.assertIn('실측 사이즈', problems[0])

    def test_large_image_is_shrunk(self):
        media_type, data = encode_image(Image.new('RGB', (5000, 100), 'white'))
        self.assertEqual(media_type, 'image/png')
        self.assertTrue(data)


class TestSaveAndEdit(unittest.TestCase):
    def test_save_does_not_overwrite(self):
        chart, _ = read_chart_image(sample_image(), FakeClient())
        with tempfile.TemporaryDirectory() as directory:
            first, second = save_chart(chart, directory), save_chart(chart, directory)
            self.assertNotEqual(first, second)
            self.assertTrue(first.startswith('uploaded_'))
            with open(os.path.join(directory, first), encoding='utf-8') as file:
                self.assertEqual(json.load(file)['sizes']['M']['shoulder'], 43)

    def test_saved_json_round_trip_keeps_values(self):
        chart, _ = read_chart_image(sample_image(), FakeClient())
        again, problems = chart_from_text(chart_to_json(chart))
        self.assertEqual(again.sizes, chart.sizes)
        self.assertEqual(problems, [])

    def test_edited_value_is_used_and_checked(self):
        chart, _ = read_chart_image(sample_image(), FakeClient())
        data = json.loads(chart_to_json(chart))
        data['sizes']['L']['chest'] = 50
        edited, problems = chart_from_text(json.dumps(data))
        self.assertEqual(edited.sizes['L']['chest'], 50)
        self.assertTrue(any('큰 사이즈가 더 작습니다' in p for p in problems))

    def test_bad_text(self):
        chart, problems = chart_from_text('{')
        self.assertIsNone(chart)
        self.assertIn('JSON', problems[0])


class TestSizeInputMode(unittest.TestCase):
    EMPTY = {'shoulder_cm': None, 'chest_circumference_cm': None, 'waist_circumference_cm': None}

    def test_no_measurements_uses_estimate(self):
        self.assertEqual(size_input_mode('upper', '남', 172, 68, self.EMPTY), 'estimate')

    def test_all_measured_skips_estimate(self):
        measured = dict(self.EMPTY, shoulder_cm=45, chest_circumference_cm=96)
        self.assertEqual(size_input_mode('upper', '남', 172, 68, measured), 'measured')

    def test_partial_measurement_fills_rest(self):
        measured = dict(self.EMPTY, chest_circumference_cm=96)
        self.assertEqual(size_input_mode('upper', '남', 172, 68, measured), 'estimate')

    def test_partial_without_weight_keeps_old_behavior(self):
        measured = dict(self.EMPTY, chest_circumference_cm=96)
        self.assertEqual(size_input_mode('upper', None, 172, None, measured), 'measured')

    def test_nothing_entered(self):
        self.assertEqual(size_input_mode('upper', None, 175, None, self.EMPTY), 'missing')

    def test_lower_needs_only_waist(self):
        measured = dict(self.EMPTY, waist_circumference_cm=80)
        self.assertEqual(size_input_mode('lower', '남', 172, 68, measured), 'measured')


if __name__ == '__main__':
    unittest.main()
