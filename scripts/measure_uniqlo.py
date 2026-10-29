"""유니클로 실측(docs/PLAN.md F-31) — 합성 결과와 실제로 입은 M·L·XL을 같은 축에서 잰다. GPU 불필요.

    python scripts/measure_uniqlo.py [<합성 결과 폴더> ...]
    → data/samples/results/uniqlo_compare/  (사람 얼굴이라 커밋 제외)

입력은 `scripts/make_uniqlo_inputs.py` 가 같은 규칙으로 자른 사진(`data/samples/uniqlo_inputs/`).
- 실제 착용(worn_uniqlo_M/L/XL): 사진마다 자세가 조금 달라서 **각 사진의 자세**로 어깨선·몸통 길이를 잡는다.
- 합성 결과: 넣었던 원래 사진(person_uniqlo_origN)의 자세를 쓴다(합성은 사람 위치를 바꾸지 않는다, F-25).
- 옷 영역 = '옷' 분류 ∩ 파란 기(B − R > 8)인 픽셀. 남색 반팔은 파랗고 갈색 바지·피부는 붉다.
  F-28의 '어두운 픽셀' 규칙은 갈색 바지도 어두워 여기서는 쓸 수 없다.
재는 값은 F-28과 같다(몸판 폭 · 소매 폭 · 기장, 모두 몸통 길이로 나눈 값) — `measure_size_effect.measure`.
"""
import glob
import os
import re
import sys

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from measure_size_effect import body_frame, draw, measure  # noqa: E402
from photo_check import run_pose  # noqa: E402
from restore_original import segment  # noqa: E402

INPUTS = os.path.join(ROOT, 'data', 'samples', 'uniqlo_inputs')
OUT = os.path.join(ROOT, 'data', 'samples', 'results', 'uniqlo_compare')
BLUE_MARGIN = 8.0
SIZES = ('M', 'L', 'XL')
NAME = re.compile(r'^(?P<group>[A-Z])__uniqlo_(?P<orig>orig\d)__uniqlo_navy__.*_s(?P<seed>\d+)_')
MODEL_OF_GROUP = {'P': 'FASHN', 'H': 'FitDiT'}
METRICS = (('chest', '몸판 폭'), ('sleeve', '소매 폭'), ('length', '기장'))


def navy_mask(image, frame):
    classes = segment(image)
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    blue = (rgb[..., 2] - rgb[..., 0]) > BLUE_MARGIN
    mask = Image.fromarray(((classes == 4) & blue).astype(np.uint8) * 255, 'L')
    k = max(3, image.width // 120) | 1
    # 크기 표시 스티커(흰색) 같은 구멍을 메우고 작은 점을 지운다
    mask = mask.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
    mask = mask.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))
    return np.asarray(mask) > 0


def frame_of(image):
    landmarks, _ = run_pose(image)
    return body_frame(landmarks)


def numbers(info):
    return {k: info[k] for k, _ in METRICS}


def main(folders):
    os.makedirs(OUT, exist_ok=True)
    rows = {}     # 이름 → (이미지, 측정)
    for size in SIZES:
        image = Image.open(os.path.join(INPUTS, f'worn_uniqlo_{size}.jpg'))
        rows[f'실제 {size}'] = (image, measure(image, frame_of(image), navy_mask))

    frames = {}
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(folder, '*.png'))):
            m = NAME.match(os.path.basename(path))
            if not m or m['group'] not in MODEL_OF_GROUP:
                continue
            orig = m['orig']
            if orig not in frames:
                frames[orig] = frame_of(Image.open(os.path.join(INPUTS, f'person_uniqlo_{orig}.jpg')))
            image = Image.open(path)
            rows[f"{MODEL_OF_GROUP[m['group']]} {orig} s{m['seed']}"] = (image, measure(image, frames[orig], navy_mask))

    lines = ['| 사진 | ' + ' | '.join(n for _, n in METRICS) + ' |', '|---|' + '---|' * len(METRICS)]
    for name, (_img, info) in rows.items():
        lines.append(f'| {name} | ' + ' | '.join(f'{info[k]:.3f}' for k, _ in METRICS) + ' |')
    # 모델별 평균이 실제 어느 사이즈에 가까운가 (세 값의 상대 오차 합)
    real = {s: numbers(rows[f'실제 {s}'][1]) for s in SIZES}
    for model in MODEL_OF_GROUP.values():
        group = [numbers(info) for name, (_i, info) in rows.items() if name.startswith(model)]
        if not group:
            continue
        mean = {k: float(np.mean([g[k] for g in group])) for k, _ in METRICS}
        spread = {k: float(np.std([g[k] for g in group]) / mean[k]) for k, _ in METRICS}
        dist = {s: sum(abs(mean[k] - real[s][k]) / real[s][k] for k, _ in METRICS) for s in SIZES}
        nearest = min(dist, key=dist.get)
        lines.append(f'\n{model} 평균 ' + ' · '.join(f'{n} {mean[k]:.3f}(흔들림 {spread[k]:.0%})' for k, n in METRICS)
                     + f' → 실제 {nearest}에 가장 가깝다 (' + ', '.join(f'{s} {d:.2f}' for s, d in dist.items()) + ')')
    m, xl = real['M'], real['XL']
    lines.append('\n실제 M → XL 변화: ' + ' · '.join(f'{n} {xl[k] / m[k] - 1:+.0%}' for k, n in METRICS))

    for name, (image, info) in rows.items():
        draw(image, info).save(os.path.join(OUT, f'{name.replace(" ", "_")}.jpg'), quality=90)
    with open(os.path.join(OUT, '_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('저장:', OUT)


if __name__ == '__main__':
    main(sys.argv[1:])
