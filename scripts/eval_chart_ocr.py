"""치수표 OCR 정확도 평가.

저장소 치수표(data/size_charts/*.json)를 표 이미지로 그려 읽거나, 실제 캡처를 읽어 정답과 칸 단위로 비교한다.

    python scripts/eval_chart_ocr.py --font C:/Windows/Fonts/malgun.ttf
    python scripts/eval_chart_ocr.py --photos 캡처폴더     # 폴더의 musinsa_tshirt.png ↔ data/size_charts/musinsa_tshirt.json
"""
import argparse
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from chart_extract import LABEL_ALIASES  # noqa: E402
from chart_ocr import read_chart_image  # noqa: E402

CHART_DIR = os.path.join(ROOT, 'data', 'size_charts')
IMAGE_TYPES = ('.png', '.jpg', '.jpeg', '.webp')


def load_truth(path):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


def render(truth, font_path, layout, font_size, jpeg_quality=None):
    sizes = truth['sizes']
    dimensions = [d for d in sorted({d for row in sizes.values() for d in row}) if d in LABEL_ALIASES]
    names = [LABEL_ALIASES[d][0] for d in dimensions]
    if layout == 'rows':
        grid = [['사이즈'] + names] + [[size] + [f"{sizes[size][d]:g}" if d in sizes[size] else '-' for d in dimensions]
                                      for size in sizes]
    else:
        grid = [['cm'] + list(sizes)] + [[name] + [f"{sizes[size][d]:g}" if d in sizes[size] else '-' for size in sizes]
                                         for d, name in zip(dimensions, names)]
    font = ImageFont.truetype(font_path, font_size)
    cell_w, cell_h = font_size * 6, font_size * 2
    image = Image.new('RGB', (cell_w * len(grid[0]) + 40, cell_h * len(grid) + 40), 'white')
    draw = ImageDraw.Draw(image)
    for r, row in enumerate(grid):
        for c, text in enumerate(row):
            draw.text((20 + c * cell_w + cell_w / 2, 20 + r * cell_h + cell_h / 2), text, fill='black', font=font, anchor='mm')
            draw.rectangle((20 + c * cell_w, 20 + r * cell_h, 20 + (c + 1) * cell_w, 20 + (r + 1) * cell_h), outline='#cccccc')
    if jpeg_quality:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=jpeg_quality)
        buffer.seek(0)
        image = Image.open(buffer).convert('RGB')
    return image


def compare(truth, chart):
    """정답 칸 중 맞은 수, 틀린 수, 빠진 수."""
    right = wrong = missing = 0
    for size, row in truth['sizes'].items():
        for dimension, value in row.items():
            if dimension not in LABEL_ALIASES:
                continue
            got = (chart.sizes.get(size, {}) if chart else {}).get(dimension)
            if got is None:
                missing += 1
            elif abs(got - value) < 0.01:
                right += 1
            else:
                wrong += 1
    return right, wrong, missing


def evaluate(name, truth, image, totals):
    start = time.perf_counter()
    chart, problems = read_chart_image(image)
    took = time.perf_counter() - start
    right, wrong, missing = compare(truth, chart)
    perfect = wrong == 0 and missing == 0
    warned = bool(problems)
    totals['cells'] += right + wrong + missing
    totals['right'] += right
    totals['wrong'] += wrong
    totals['missing'] += missing
    totals['charts'] += 1
    totals['perfect'] += perfect
    totals['silent_errors'] += (not perfect) and not warned
    totals['seconds'] += took
    print(f"{name}: 맞음 {right} / 틀림 {wrong} / 빠짐 {missing} / 경고 {len(problems)} / {took:.1f}초")
    if not perfect:
        for problem in problems[:3]:
            print('   경고:', problem)


def new_totals():
    return dict(cells=0, right=0, wrong=0, missing=0, charts=0, perfect=0, silent_errors=0, seconds=0.0)


def report(label, totals):
    if not totals['charts']:
        return
    print(f"[{label}] 표 {totals['charts']}개 중 전부 일치 {totals['perfect']}개 · "
          f"칸 {totals['right']}/{totals['cells']} ({totals['right'] / totals['cells']:.1%}) · "
          f"틀림 {totals['wrong']} · 빠짐 {totals['missing']} · 경고 없는 오류 {totals['silent_errors']}개 · "
          f"표당 {totals['seconds'] / totals['charts']:.1f}초")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font', help='합성 이미지를 그릴 한글 글꼴')
    parser.add_argument('--photos', help='치수표 JSON과 같은 이름의 실제 캡처가 있는 폴더')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    if args.photos:
        totals = new_totals()
        for filename in sorted(os.listdir(args.photos)):
            stem, extension = os.path.splitext(filename)
            truth_path = os.path.join(CHART_DIR, stem + '.json')
            if extension.lower() in IMAGE_TYPES and os.path.exists(truth_path):
                image = Image.open(os.path.join(args.photos, filename)).convert('RGB')
                evaluate(filename, load_truth(truth_path), image, totals)
        report('실제 캡처', totals)
        return
    if not args.font:
        parser.error('--font 또는 --photos 가 필요합니다')
    conditions = [
        ('행 표 · 또렷함', 'rows', 28, None),
        ('열 표 · 또렷함', 'columns', 28, None),
        ('행 표 · 작은 글자', 'rows', 12, None),
        ('열 표 · 작은 글자', 'columns', 12, None),
        ('행 표 · 작은 글자 · JPEG 압축', 'rows', 12, 40),
    ]
    charts = sorted(f for f in os.listdir(CHART_DIR) if f.endswith('.json') and not f.startswith('uploaded_'))
    for label, layout, font_size, quality in conditions:
        totals = new_totals()
        for filename in charts:
            truth = load_truth(os.path.join(CHART_DIR, filename))
            image = render(truth, args.font, layout, font_size, quality)
            evaluate(f'{label} · {filename}', truth, image, totals)
        report(label, totals)


if __name__ == '__main__':
    main()
