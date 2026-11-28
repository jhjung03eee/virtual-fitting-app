"""GPU 없이 화면만 확인하는 로컬 실행. 합성 엔진(fashn_core)만 가짜로 바꾸고 나머지는 실제 코드를 쓴다.

    pip install "gradio>=6,<7" mediapipe opencv-python-headless numpy pillow
    python scripts/ui_preview.py [--sample 합성결과.png] [--port 7873]
    → http://127.0.0.1:7873

- 「피팅 해보기」는 --sample 로 준 사진을 합성 결과처럼 돌려준다. 없으면 올린 인물 사진을 그대로 돌려준다.
- 사진 검사, 사이즈 추천, 치수 추정, 사이즈별 예상 모습은 실제로 동작한다(mediapipe 필요).
- 치수표 사진 읽기(easyocr)는 설치하지 않으면 안내 문구만 나온다.
"""
import argparse
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', help='합성 결과 대신 보여 줄 사진 경로')
    ap.add_argument('--port', type=int, default=7873)
    a = ap.parse_args()

    from PIL import Image

    sample = Image.open(a.sample).convert('RGB') if a.sample else None
    stub = types.ModuleType('fashn_core')
    stub.CATEGORY_BY_CLOTH_TYPE = {'upper': 'tops', 'lower': 'bottoms', 'overall': 'one-pieces'}
    stub.DEFAULT_STEPS, stub.DEFAULT_GUIDANCE, stub.DEFAULT_SEED = 25, 2.5, 42
    stub.load_models = lambda: None
    stub.warmup = lambda: 0
    stub.try_on = lambda **kw: ((sample or kw['person'].convert('RGB')), {'total_s': 0.0})
    sys.modules['fashn_core'] = stub

    import gradio_app

    options = dict(gradio_app.STYLE) if gradio_app.GRADIO6 else {}
    print(f'화면 확인용 실행: http://127.0.0.1:{a.port}  (합성은 가짜)', flush=True)
    gradio_app.demo.queue().launch(server_name='127.0.0.1', server_port=a.port, share=False, show_error=True, **options)


if __name__ == '__main__':
    main()
