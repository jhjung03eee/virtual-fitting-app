"""사이즈별 옷 사진(F-28)으로 합성한 결과에서 옷 크기를 재, 치수표 차이만큼 변했는지 본다. GPU 불필요.

    python scripts/measure_size_effect.py <결과 폴더> [<결과 폴더> ...]
    → data/samples/results/size_effect/  (사람 얼굴이라 커밋 제외)

**재는 법**
  - 옷 영역 = 결과 사진에서 '옷'으로 분류되고(MediaPipe selfie_multiclass, app/restore_original.segment)
    어두운 픽셀(카키). 회색 바지·피부·배경은 빠진다. 폭은 그 줄의 옷 픽셀 개수.
  - 기준점 = **원본 인물 사진**의 어깨·엉덩이(같은 사람이라 모든 결과에 같은 값을 쓴다). 몸통 길이 = 어깨 가운데~엉덩이 가운데.
  - 몸판 폭: 어깨에서 몸통 길이의 55% 아래(겨드랑이 아래, 밑단 위) 줄에서 옷 영역의 왼쪽 끝~오른쪽 끝.
  - 소매 폭: 어깨~몸통 40% 아래 사이에서 가장 넓은 줄(소매 끝~소매 끝).
  - 기장: 어깨선부터 옷 폭이 몸판 폭의 절반 이상인 마지막 줄까지.
  모두 몸통 길이로 나눈다.

**판정** — 같은 시드의 M 대비 비율을 치수표 비율과 비교한다.
  비례도 = (측정 비율 − 1)과 (치수표 비율 − 1)의 기울기(원점을 지나는 직선). 1이면 치수표만큼 정확히 변하고, 0이면 효과 없음.
  흔들림 = 같은 사이즈에서 시드만 바꿨을 때의 차이. 사이즈 효과가 이보다 작으면 의미가 없다.
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

from body_measure import POSE_LANDMARKS  # noqa: E402
from photo_check import run_pose  # noqa: E402
from restore_original import segment  # noqa: E402

PERSON = os.path.join(ROOT, 'data', 'person', 'full_upload', 'person_team02.jpg')
CHART = os.path.join(ROOT, 'data', 'size_charts', 'musinsa_basic_tee_khaki.json')
OUT = os.path.join(ROOT, 'data', 'samples', 'results', 'size_effect')
SIZES = ('S', 'M', 'L', 'XL', 'XXL')
NAME = re.compile(r'^(?P<group>[A-Z])__team02__tee_khaki_size_(?P<size>[A-Z]+)__.*_s(?P<seed>\d+)_')
MODEL_OF_GROUP = {'N': 'FASHN', 'G': 'FitDiT'}
MAX_LUMA = 110             # 이보다 어두운 옷 픽셀 = 카키 상의. 실측(FASHN M): 상의 중앙값 67, 바지 그늘 하위 25% 126 — 140은 바지 그늘이 들어와 밑단이 내려갔다
SEARCH_DEPTH = 1.7         # 어깨선에서 몸통 길이의 이 배까지만 찾는다(허벅지 중간). XXL 기장도 1.3 안쪽
CHEST_LEVEL = 0.55
SLEEVE_BAND = 0.40


def body_frame(landmarks):
    """(어깨선 y, 몸통 길이, 몸 가운데 x) — 0~1 좌표."""
    p = {n: np.array([landmarks[i].x, landmarks[i].y]) for n, i in POSE_LANDMARKS.items()}
    shoulder = (p['left_shoulder'] + p['right_shoulder']) / 2
    hip = (p['left_hip'] + p['right_hip']) / 2
    return shoulder[1], float(np.linalg.norm(shoulder - hip)), shoulder[0]


def garment_mask(result, frame):
    """새 상의 영역: '옷' 분류 중 어두운 픽셀(카키), 허벅지 중간보다 위.

    처음엔 카키 원본 색과의 거리로 골랐는데, 모델이 그린 짙은 그림자는 빠지고 회색 바지 주름의
    그림자가 들어왔다. 카키는 어둡고(밝기 50~110) 바지·원래 크림색 티는 밝아서(150 이상) 밝기로 가른다.
    """
    classes = segment(result)
    luma = np.asarray(result.convert('L'), dtype=np.float32)
    height = result.height
    shoulder_y, torso, _cx = frame
    ys = np.arange(height)[:, None]
    mask = (classes == 4) & (luma < MAX_LUMA) & (ys < (shoulder_y + SEARCH_DEPTH * torso) * height)
    mask = Image.fromarray(mask.astype(np.uint8) * 255, 'L')
    k = max(3, result.width // 120) | 1
    # 주름 사이 구멍을 메우고 작은 점을 지운다
    mask = mask.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
    mask = mask.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))
    return np.asarray(mask) > 0


def measure(result, frame, mask_fn=None):
    """폭은 그 줄의 옷 픽셀 **개수**로 잰다 — 왼쪽 끝~오른쪽 끝으로 재면 바지 주름에 흩어진 점까지 폭이 된다.

    mask_fn(result, frame) → bool 배열: 옷을 고르는 규칙. 기본은 카키(어두운 픽셀). 다른 옷이면 바꿔 끼운다
    (유니클로 남색 반팔은 갈색 바지도 어두워서 색 규칙을 쓴다, scripts/measure_uniqlo.py).
    """
    height = result.height
    shoulder_y, torso, _cx = frame
    sy, t = shoulder_y * height, torso * height
    mask = (mask_fn or garment_mask)(result, frame)
    counts = mask.sum(axis=1)

    chest_row = int(sy + CHEST_LEVEL * t)
    chest_w = int(counts[chest_row])
    band = counts[int(sy):int(sy + SLEEVE_BAND * t)]
    sleeve_row = int(sy) + int(band.argmax())
    sleeve_w = int(band.max())
    # 가슴 높이에서 아래로 내려가며 옷 폭이 가슴 폭의 절반 밑으로 처음 떨어지는 줄 = 밑단
    hem_y = chest_row
    while hem_y + 1 < height and counts[hem_y + 1] >= 0.5 * chest_w:
        hem_y += 1
    return {'chest': chest_w / t, 'sleeve': sleeve_w / t, 'length': (hem_y - sy) / t,
            '_rows': (chest_row, sleeve_row, hem_y, int(sy)), '_mask': mask}


def slope(expected, measured):
    """원점을 지나는 직선의 기울기 — (치수표 비율 − 1) 대비 (측정 비율 − 1)."""
    e, m = np.asarray(expected) - 1, np.asarray(measured) - 1
    return float((e * m).sum() / (e * e).sum())


def draw(result, info):
    image = result.convert('RGB').copy()
    overlay = np.asarray(image).copy()
    overlay[info['_mask']] = (overlay[info['_mask']] * 0.5 + np.array([255, 60, 60]) * 0.5).astype(np.uint8)
    image = Image.fromarray(overlay)
    d = ImageDraw.Draw(image)
    chest_row, sleeve_row, hem_y, sy = info['_rows']
    for y, color in ((sy, (255, 255, 255)), (chest_row, (60, 200, 110)), (hem_y, (255, 200, 0))):
        d.line([(0, y), (image.width, y)], fill=color, width=max(1, image.width // 300))
    return image


def main(folders):
    chart = json.load(open(CHART, encoding='utf-8'))['sizes']
    expected = {s: {'chest': chart[s]['chest'] / chart['M']['chest'], 'length': chart[s]['length'] / chart['M']['length'],
                    'sleeve': chart[s]['chest'] / chart['M']['chest']} for s in SIZES}
    person = ImageOps.exif_transpose(Image.open(PERSON)).convert('RGB')
    landmarks, _ = run_pose(person)
    frame = body_frame(landmarks)

    results = {}   # (model, seed, size) -> info
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(folder, '*.png'))):
            m = NAME.match(os.path.basename(path))
            if not m or m['group'] not in MODEL_OF_GROUP:
                continue
            key = (MODEL_OF_GROUP[m['group']], int(m['seed']), m['size'])
            results[key] = (path, measure(Image.open(path), frame))
    if not results:
        sys.exit('결과 파일을 찾지 못했다')
    os.makedirs(OUT, exist_ok=True)

    rows, lines = [], []
    for model in sorted({k[0] for k in results}):
        seeds = sorted({k[1] for k in results if k[0] == model})
        lines.append(f'== {model} (시드 {seeds})')
        for metric in ('chest', 'sleeve', 'length'):
            exp, meas, noise = [], [], []
            for size in SIZES:
                vals = [results[(model, s, size)][1][metric] / results[(model, s, 'M')][1][metric]
                        for s in seeds if (model, s, size) in results and (model, s, 'M') in results]
                raw = [results[(model, s, size)][1][metric] for s in seeds if (model, s, size) in results]
                if not vals:
                    continue
                exp.append(expected[size][metric])
                meas.append(float(np.mean(vals)))
                if len(raw) > 1:
                    noise.append(abs(raw[0] - raw[1]) / np.mean(raw))
                rows.append({'model': model, 'metric': metric, 'size': size, 'expected_ratio': round(expected[size][metric], 3),
                             'measured_ratio': round(float(np.mean(vals)), 3),
                             'measured_abs': round(float(np.mean(raw)), 3)})
            k = slope(exp, meas)
            detail = '  '.join(f'{s} {e:.3f}→{m_:.3f}' for s, e, m_ in zip(SIZES, exp, meas))
            noise_txt = f' | 시드 흔들림 평균 {np.mean(noise):.1%}' if noise else ''
            lines.append(f'  {metric:7s} 비례도 {k:+.2f}  (치수표→측정, M 대비)  {detail}{noise_txt}')

        # 사이즈별 나란히 (시드마다 한 줄)
        for s in seeds:
            panels = [draw(Image.open(results[(model, s, size)][0]), results[(model, s, size)][1])
                      for size in SIZES if (model, s, size) in results]
            if panels:
                h = 720
                panels = [p.resize((round(p.width * h / p.height), h)) for p in panels]
                strip = Image.new('RGB', (sum(p.width for p in panels) + 8 * (len(panels) - 1), h), 'white')
                x = 0
                for p in panels:
                    strip.paste(p, (x, 0))
                    x += p.width + 8
                strip.save(os.path.join(OUT, f'_{model}_s{s}.jpg'), quality=90)

    with open(os.path.join(OUT, 'size_effect.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(OUT, '_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('저장:', OUT)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
