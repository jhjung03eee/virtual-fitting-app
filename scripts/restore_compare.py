"""FASHN 결과 vs 원본 되돌리기(app/restore_original.py) 전후 비교 이미지를 만든다. GPU 불필요.

이미 받아 둔 FASHN 결과(팀 공통 조건: 팀원 02 × 옷 3벌 × 시드 2개, docs/MODEL_STUDY.md)를 쓴다.

    python scripts/restore_compare.py
    → data/samples/results/restore_original/  (data/samples 는 커밋 제외 — 사람 얼굴)
       _full.jpg   옷마다 원본 | FASHN | 되돌리기 | 결과를 쓴 자리(새 옷·원래 옷 자리·바뀐 피부·경계 여유로 색 구분)
       _zoom.jpg   얼굴·경계(상의 밑단/바지 허리) 확대 (FASHN | 되돌리기)
       <옷>_s<시드>.png  되돌리기 결과 한 장씩

수치(얼굴 선명도)는 라플라시안 분산이다. 클수록 잔결이 많다 — '더 자연스럽다'의 근거가 아니라
FASHN이 얼굴을 매끈하게 만드는 정도를 숫자로 보이는 용도.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

from restore_original import restore_original, segment  # noqa: E402

PERSON = os.path.join(ROOT, 'data', 'person', 'full_upload', 'person_team02.jpg')
RESULTS = os.path.join(ROOT, 'data', 'samples', 'results', 'fashn_baseline', 'fashn_t4x2')
OUT = os.path.join(ROOT, 'data', 'samples', 'results', 'restore_original')
GARMENTS = {'shirt_blue': 'tops', 'tee_khaki': 'tops', 'pants_forest': 'bottoms'}
SEEDS = (42, 123)
SCALE = 2.0
# 패널을 절반으로 줄여 붙이므로 글씨를 크게 쓴다
FONT = ImageFont.load_default(size=36)


def sharpness(image, box):
    gray = np.asarray(image.crop(box).convert('L'), dtype=np.float32)
    lap = gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return float(lap.var())


def class_box(classes, cls, pad, scale):
    ys, xs = np.nonzero(classes == cls)
    return tuple(int(v * scale) for v in (xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad))


def label(image, text):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    right, bottom = draw.textbbox((10, 6), text, font=FONT)[2:]
    draw.rectangle((0, 0, right + 10, bottom + 8), fill='white')
    draw.text((10, 6), text, fill='black', font=FONT)
    return image


# FASHN 결과를 쓴 자리를 왜 덮었는지에 따라 나눠 칠한다. 한 색으로 칠하면 '옷보다 크게 땄다'로만 보인다.
AREA_COLORS = {
    'new': ((230, 30, 30), 'new garment'),            # 결과에서 옷인 곳
    'old': ((30, 90, 255), 'old clothes only'),       # 원래 옷만 있던 곳 — 되돌리면 원래 소매가 다시 보인다
    'skin': ((255, 150, 0), 'changed skin'),          # 반팔↔긴팔 등으로 바뀐 팔·목·손
    'buffer': ((240, 220, 0), 'seam buffer'),         # 이음매를 옷에서 띄우는 여유 띠(배경끼리라 티가 안 남)
}


def area_overlay(restored, mask, person_classes, result_classes):
    """마스크를 새 옷 / 원래 옷 자리 / 바뀐 피부 / 경계 여유로 나눠 칠한다. (이미지, 마스크 대비 비율)."""
    m = np.asarray(mask.resize(person_classes.shape[::-1], Image.BILINEAR)) > 127
    new = m & (result_classes == 4)
    old = m & (person_classes == 4) & ~new
    skin = m & ((person_classes == 2) | (result_classes == 2)) & ~new & ~old
    areas = {'new': new, 'old': old, 'skin': skin, 'buffer': m & ~new & ~old & ~skin}
    paint = np.zeros(person_classes.shape + (3,), np.uint8)
    alpha = np.zeros(person_classes.shape, np.uint8)
    for name, area in areas.items():
        paint[area] = AREA_COLORS[name][0]
        alpha[area] = 140
    paint = Image.fromarray(paint).resize(restored.size, Image.NEAREST)
    alpha = Image.fromarray(alpha).resize(restored.size, Image.NEAREST)
    total = max(int(m.sum()), 1)
    return Image.composite(paint, restored, alpha), {k: float(v.sum()) / total for k, v in areas.items()}


def legend(image):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    step = 48
    y = image.height - step * len(AREA_COLORS) - 12
    draw.rectangle((0, y - 12, 380, image.height), fill='white')
    for color, text in AREA_COLORS.values():
        draw.rectangle((14, y, 48, y + 34), fill=color)
        draw.text((62, y), text, fill='black', font=FONT)
        y += step
    return image


def row(images, gap=8):
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    h = max(i.height for i in images)
    canvas = Image.new('RGB', (w, h), 'white')
    x = 0
    for i in images:
        canvas.paste(i, (x, 0))
        x += i.width + gap
    return canvas


def column(images, gap=8):
    w = max(i.width for i in images)
    canvas = Image.new('RGB', (w, sum(i.height for i in images) + gap * (len(images) - 1)), 'white')
    y = 0
    for i in images:
        canvas.paste(i, (0, y))
        y += i.height + gap
    return canvas


def main():
    os.makedirs(OUT, exist_ok=True)
    person = ImageOps.exif_transpose(Image.open(PERSON)).convert('RGB')
    full_rows, zoom_rows, lines = [], [], []
    for key, category in GARMENTS.items():
        for seed in SEEDS:
            path = glob.glob(os.path.join(RESULTS, f'*__{key}__*_s{seed}_*.png'))[0]
            result = Image.open(path).convert('RGB')
            restored, mask = restore_original(person, result, category, SCALE)
            if mask is None:
                sys.exit('mediapipe 가 없어 옷 영역을 구분할 수 없다')
            restored.save(os.path.join(OUT, f'{key}_s{seed}.png'))

            big = result.resize(restored.size, Image.LANCZOS)
            original = person.resize(restored.size, Image.LANCZOS)
            classes = segment(result)
            person_classes = segment(person.resize(result.size, Image.LANCZOS))
            overlay, shares = area_overlay(restored, mask, person_classes, classes)
            panels = [label(original, 'original'), label(big, 'FASHN'),
                      label(restored, 'restored'), legend(label(overlay, 'FASHN area'))]
            full_rows.append(row([p.resize((p.width // 2, p.height // 2)) for p in panels]))

            face = class_box(classes, 3, 12, SCALE)
            # FASHN 결과와 원본이 만나는 경계(상의는 밑단, 하의는 허리)가 가장 티 나기 쉬운 곳이다
            m = np.asarray(mask) > 127
            ys, xs = np.nonzero(m)
            seam_y = ys.max() if category == 'tops' else ys.min()
            cx = int(np.median(xs))
            seam = (cx - 200, seam_y - 90, cx + 200, seam_y + 90)
            zoom_rows.append(row([label(big.crop(face), f'{key} s{seed} FASHN'), label(restored.crop(face), 'restored'),
                                  label(big.crop(seam), 'FASHN'), label(restored.crop(seam), 'restored (seam)')]))
            covered = float((np.asarray(mask) > 127).mean())
            lines.append(f'{key:13s} s{seed:<4d} 얼굴 선명도 FASHN {sharpness(big, face):6.1f} → 되돌리기 '
                         f'{sharpness(restored, face):6.1f} | FASHN 결과를 쓴 면적 {covered:.0%} '
                         f"(새 옷 {shares['new']:.0%} · 원래 옷 자리 {shares['old']:.0%} · "
                         f"바뀐 피부 {shares['skin']:.0%} · 경계 여유 {shares['buffer']:.0%})")
            print(lines[-1], flush=True)

    column(full_rows).save(os.path.join(OUT, '_full.jpg'), quality=90)
    column(zoom_rows).save(os.path.join(OUT, '_zoom.jpg'), quality=92)
    with open(os.path.join(OUT, '_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('저장:', OUT)


if __name__ == '__main__':
    main()
