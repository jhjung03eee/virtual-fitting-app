"""GPU·Gradio 없이 사이즈 추천 카드의 표시를 검사한다."""
import ast
import html
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

from size_fit import SizeChart, ko, recommend  # noqa: E402


class TestSizeAdvice(unittest.TestCase):
    def test_unscorable_size_is_shown_as_unknown(self):
        chart = SizeChart(name='치수가 빠진 상의', category='upper', sizes={
            'S': {'length': 66.0}, 'M': {'chest': 52.0, 'length': 68.0},
        })
        rec = recommend(chart, {'chest': 48.0})

        # gradio_app 전체를 임포트하면 GPU 의존성이 필요하므로 함수만 실행한다.
        path = os.path.join(ROOT, 'app', 'gradio_app.py')
        with open(path, encoding='utf-8') as f:
            tree = ast.parse(f.read())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'size_advice')
        scope = {
            'html': html, 'ko': ko,
            'FIT_TONE': {'레귤러': 'ok'},
            '_recommend': lambda *args: (chart, rec, None),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), path, 'exec'), scope)
        card = scope['size_advice'](None, None, None, None, None)

        self.assertIn('class="fc-size__big">M</span>', card)
        self.assertIn('class="fc-chip is-unknown">S</span>', card)
        self.assertIn('S <small>(판단 불가)</small></td><td>-</td>', card)
        self.assertIn('판단할 수 없는 사이즈: S', card)


if __name__ == '__main__':
    unittest.main()
