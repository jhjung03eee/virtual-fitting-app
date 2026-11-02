"""사이즈 비율 변환 실험(docs/PLAN.md F-35) — 합성 결과를 M·L·XL로 늘려 실제로 입은 사진과 비교한다. GPU 불필요.

    python scripts/exp_size_warp.py <FASHN 결과 폴더>
    → data/samples/results/size_warp/  (사람 얼굴이라 커밋 제외)

입력은 F-31과 같다: 유니클로 남색 반팔 FASHN 결과 8장(원래 사진 4장 × 시드 42·123)과
같은 사람이 실제로 입은 M·L·XL(`data/samples/uniqlo_inputs/`), 치수표 `uniqlo_crew_tee_navy.json`.

**목표 크기 정하기.** 재는 단위는 F-31과 같다(몸통 길이로 나눈 값). 치수표의 cm를 이 단위로 바꾸는 비율 k
(= 실제 착용 측정값 ÷ 치수표 cm)는 사람마다 하나인 상수다. k를 **목표 사이즈를 뺀 나머지 두 사이즈의 실제 사진**으로
정한다(leave-one-out) — 맞히려는 사진을 보고 목표를 정하면 답을 보고 푸는 셈이라서.
앱에서는 실제 착용 사진이 없으므로 k를 키(사진 속 키 픽셀 ÷ 입력한 키 cm)로 잡아야 한다 — 이 실험 밖.

**판정.** 변환한 그림을 같은 규칙으로 다시 재서 실제 사진과의 오차(%)를 변환 전과 비교한다.
실제 사진끼리도 자세가 달라 측정이 흔들린다(F-31: 실제 L 몸판 폭이 M보다 작게 잼). 그 흔들림보다 작으면 충분히 맞은 것으로 본다.
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from measure_size_effect import measure  # noqa: E402
from measure_uniqlo import INPUTS, SIZES, frame_of, navy_mask  # noqa: E402
from size_warp import scales_for_size, warp_garment  # noqa: E402

CHART = os.path.join(ROOT, 'data', 'size_charts', 'uniqlo_crew_tee_navy.json')
OUT = os.path.join(ROOT, 'data', 'samples', 'results', 'size_warp')
NAME = re.compile(r'^P__uniqlo_(?P<orig>orig\d)__uniqlo_navy__.*_s(?P<seed>\d+)_')
KEYS = ('chest', 'length')
KO = {'chest': '몸판 폭', 'length': '기장'}


def calibrate(real, chart, sizes):
    """치수표 cm → 측정 단위 비율 k (항목별, 주어진 사이즈들의 평균)."""
    return {k: float(np.mean([real[s][k] / chart[s][k] for s in sizes])) for k in KEYS}


def err(a, b):
    return {k: 100.0 * (a[k] - b[k]) / b[k] for k in KEYS}


def label(image, text, font):
    image = image.convert('RGB').copy()
    d = ImageDraw.Draw(image)
    d.rectangle([0, 0, image.width, 46], fill=(255, 255, 255))
    d.text((12, 8), text, fill=(0, 0, 0), font=font)
    return image


def main(folder):
    os.makedirs(OUT, exist_ok=True)
    chart = json.load(open(CHART, encoding='utf-8'))['sizes']
    real_img, real = {}, {}
    for s in SIZES:
        real_img[s] = Image.open(os.path.join(INPUTS, f'worn_uniqlo_{s}.jpg')).convert('RGB')
        real[s] = measure(real_img[s], frame_of(real_img[s]), navy_mask)

    frames, rows, seconds, shown = {}, [], [], {}
    for path in sorted(glob.glob(os.path.join(folder, 'P__*.png'))):
        m = NAME.match(os.path.basename(path))
        if not m:
            continue
        orig = m['orig']
        if orig not in frames:
            frames[orig] = frame_of(Image.open(os.path.join(INPUTS, f'person_uniqlo_{orig}.jpg')))
        frame = frames[orig]
        result = Image.open(path).convert('RGB')
        now = measure(result, frame, navy_mask)
        shoulder_y = now['_rows'][3]
        center_x = frame[2] * result.width
        for s in SIZES:
            k = calibrate(real, chart, [o for o in SIZES if o != s])
            target = {key: k[key] * chart[s][key] for key in KEYS}
            scale_x, scale_y = scales_for_size(now, target)
            start = time.perf_counter()
            warped, _ = warp_garment(result, now['_mask'], shoulder_y, center_x, scale_x, scale_y)
            seconds.append(time.perf_counter() - start)
            after = measure(warped, frame, navy_mask)
            rows.append(dict(name=f'{orig} s{m["seed"]}', size=s, scale_x=scale_x, scale_y=scale_y,
                             before=err(now, real[s]), after=err(after, real[s]), target=err(target, real[s])))
            warped.save(os.path.join(OUT, f'{orig}_s{m["seed"]}_to_{s}.png'))
            if orig == 'orig1' and m['seed'] == '42':
                shown[s] = warped
                shown['합성'] = result

    # 표: 사이즈별 평균 |오차| — 변환 전 / 목표값 자체 / 변환 후
    lines = ['| 목표 | 배율(가로×세로) | 몸판 폭 오차 전→후 | 기장 오차 전→후 | 목표값 자체 오차(몸판·기장) |',
             '|---|---|---|---|---|']
    for s in SIZES:
        rs = [r for r in rows if r['size'] == s]
        mean = lambda part, key: float(np.mean([abs(r[part][key]) for r in rs]))
        lines.append(f"| {s} | {np.mean([r['scale_x'] for r in rs]):.2f} × {np.mean([r['scale_y'] for r in rs]):.2f} "
                     f"| {mean('before', 'chest'):.1f}% → {mean('after', 'chest'):.1f}% "
                     f"| {mean('before', 'length'):.1f}% → {mean('after', 'length'):.1f}% "
                     f"| {mean('target', 'chest'):.1f}% · {mean('target', 'length'):.1f}% |")
    order = sum(1 for name in {r['name'] for r in rows}
                if all(next(r for r in rows if r['name'] == name and r['size'] == a)['scale_y']
                       < next(r for r in rows if r['name'] == name and r['size'] == b)['scale_y']
                       for a, b in zip(SIZES, SIZES[1:])))
    real_spread = {key: 100.0 * (real['L'][key] / real['M'][key] - chart['L'][key] / chart['M'][key])
                   for key in KEYS}
    lines += ['', f'결과 {len({r["name"] for r in rows})}장 × 사이즈 {len(SIZES)}개 · 변환 한 번 평균 '
              f'{np.mean(seconds):.2f}초(노트북 CPU) · 기장이 M<L<XL 순서로 늘어난 장 {order}장',
              f"실제 사진 측정 흔들림 참고: 실제 L/M 비율이 치수표 비율과 다른 정도 — 몸판 폭 {real_spread['chest']:+.1f}%p · "
              f"기장 {real_spread['length']:+.1f}%p"]
    open(os.path.join(OUT, '_summary.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # 그림: 합성 | M 변환 · 실제 M | L 변환 · 실제 L | XL 변환 · 실제 XL (orig1 시드 42)
    font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 26) if os.name == 'nt' else ImageFont.load_default()
    h = 900
    fit = lambda im: im.resize((int(im.width * h / im.height), h), Image.LANCZOS)  # 크기를 맞춘 뒤 이름표(실제 사진이 커서)
    cells = [label(fit(shown['합성']), '합성 그대로', font)]
    for s in SIZES:
        cells += [label(fit(shown[s]), f'→ {s} 변환', font), label(fit(real_img[s]), f'실제 {s}', font)]
    sheet = Image.new('RGB', (sum(c.width for c in cells) + 12 * (len(cells) + 1), h + 24), 'white')
    x = 12
    for c in cells:
        sheet.paste(c, (x, 12))
        x += c.width + 12
    sheet.save(os.path.join(OUT, '_compare_orig1_s42.jpg'), quality=92)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(ROOT, 'data', 'samples', 'results', 'uniqlo_runs', 'fashn', 'fashn_t4x2'))
