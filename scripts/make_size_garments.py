"""사이즈별 옷 사진 만들기 — 치수표 비율만큼 옷 사진 속 옷 크기를 바꾼다 (docs/PLAN.md F-28 실험 입력).

질문: 사이즈를 먼저 추천하고, **그 사이즈의 옷 정보로** 합성할 수 있는가?
합성 모델에는 사이즈를 넣을 입구가 없고 옷 사진만 받는다. 그래서 옷 사진 자체를 사이즈에 맞게 바꿔 넣어 보고,
결과 속 옷 크기가 치수표 차이만큼 실제로 변하는지 잰다(`scripts/measure_size_effect.py`).

만드는 법:
  - 원본 상품 사진을 **기준 사이즈(기본 M)** 로 가정한다. 상품 사진이 실제로 몇 사이즈인지는 모른다 —
    사이즈 사이의 **비율**만 비교하므로 가정이 틀려도 결론은 같다.
  - 가로는 가슴단면 비율, 세로는 총장 비율로 늘리거나 줄인다(예: XXL = 가로 64/54, 세로 78/66).
  - **캔버스 크기는 모든 사이즈가 같다**(원본의 1.25배, 배경색으로 채움). 그래야 옷이 화면에서 차지하는
    크기만 달라진다. 옷의 윗단(목 부분)은 같은 높이에 맞춘다.

    python scripts/make_size_garments.py
    → data/samples/size_garments/  (상품 사진이라 커밋 제외. Kaggle 데이터셋 vfa-size-garments 로 올린다)
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

from color_match import foreground_mask  # noqa: E402

CANVAS_SCALE = 1.25
EDGE_GROW_PX = 3      # 옷 테두리의 밝은 픽셀이 배경으로 분류돼 잘리지 않게
EDGE_FEATHER_PX = 1.5


def size_scales(sizes, size, base='M'):
    """(가로 배율, 세로 배율) = (가슴 비율, 총장 비율)."""
    return sizes[size]['chest'] / sizes[base]['chest'], sizes[size]['length'] / sizes[base]['length']


def garment_box(mask):
    """옷 영역의 테두리 상자. 배경의 작은 얼룩에 끌려가지 않게 양 끝 0.2%는 버린다."""
    ys, xs = np.nonzero(mask)
    x0, x1 = np.percentile(xs, [0.2, 99.8])
    y0, y1 = np.percentile(ys, [0.2, 99.8])
    return int(x0), int(y0), int(x1) + 1, int(y1) + 1


def scale_garment(image, sx, sy, canvas_scale=CANVAS_SCALE):
    """옷만 (sx, sy) 배로 키운 사진. 캔버스는 원본의 canvas_scale 배이고 배경색으로 채운다."""
    image = image.convert('RGB')
    mask, background = foreground_mask(image)
    box = garment_box(mask)
    alpha = Image.fromarray(mask.astype(np.uint8) * 255, 'L')
    alpha = alpha.filter(ImageFilter.MaxFilter(2 * EDGE_GROW_PX + 1)).filter(ImageFilter.GaussianBlur(EDGE_FEATHER_PX))

    crop, crop_alpha = image.crop(box), alpha.crop(box)
    size = (max(1, round(crop.width * sx)), max(1, round(crop.height * sy)))
    crop, crop_alpha = crop.resize(size, Image.LANCZOS), crop_alpha.resize(size, Image.LANCZOS)

    width, height = round(image.width * canvas_scale), round(image.height * canvas_scale)
    canvas = Image.new('RGB', (width, height), tuple(int(v) for v in background))
    # 윗단 높이는 모든 사이즈에서 같게: 원본 윗단 위치를 캔버스 가운데 정렬 기준으로 옮긴 곳
    top = box[1] + (height - image.height) // 2
    left = (width - size[0]) // 2
    if top + size[1] > height:
        raise ValueError(f'옷이 캔버스 아래로 넘친다 ({sx:.2f}x{sy:.2f}). canvas_scale 을 키울 것')
    canvas.paste(crop, (left, top), crop_alpha)
    return canvas


def main():
    chart = json.load(open(os.path.join(ROOT, 'data', 'size_charts', 'musinsa_basic_tee_khaki.json'), encoding='utf-8'))
    source = ImageOps.exif_transpose(Image.open(os.path.join(ROOT, 'data', 'samples', 'set2', 'tee_khaki.jpg')))
    out = os.path.join(ROOT, 'data', 'samples', 'size_garments')
    os.makedirs(out, exist_ok=True)
    thumbs = []
    for size in chart['sizes']:
        sx, sy = size_scales(chart['sizes'], size)
        image = scale_garment(source, sx, sy)
        image.save(os.path.join(out, f'garment_tee_khaki_size_{size}.jpg'), quality=95)
        thumbs.append(image.resize((image.width // 5, image.height // 5)))
        print(f'{size:4s} 가로 x{sx:.3f} 세로 x{sy:.3f}  ({image.size[0]}x{image.size[1]})')
    strip = Image.new('RGB', (sum(t.width for t in thumbs), thumbs[0].height), 'white')
    x = 0
    for t in thumbs:
        strip.paste(t, (x, 0))
        x += t.width
    strip.save(os.path.join(out, '_sizes.jpg'), quality=90)
    print('저장:', out)


if __name__ == '__main__':
    main()
