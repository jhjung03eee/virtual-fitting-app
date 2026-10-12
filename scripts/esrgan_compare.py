"""원본 되돌리기(F-25)에서 옷 자리를 단순 확대 vs Real-ESRGAN 확대로 비교한다.

F-25에서 얼굴·배경은 원본 해상도로 또렷해졌지만 옷은 FASHN 576px를 2배로 늘린 것이라 상대적으로 흐렸다.
옷 자리만 Real-ESRGAN(x4plus)으로 키운 결과로 바꿔 끼운다.

Real-ESRGAN 은 이 스크립트가 돌리지 않는다. 공식 배포판(realesrgan-ncnn-vulkan, Vulkan GPU·내장 그래픽 가능)으로
FASHN 결과 폴더를 미리 키워 두고 그 폴더를 넘긴다. 내장 그래픽(Iris Xe)은 기본 타일에서 메모리가 모자라
검은 이미지가 나왔다 → `-t 64 -j 1:1:1`, 장당 약 2분.

    realesrgan-ncnn-vulkan -i <FASHN 결과 폴더> -o <키운 폴더> -n realesrgan-x4plus -t 64 -j 1:1:1
    python scripts/esrgan_compare.py <키운 폴더>
    → data/samples/results/restore_esrgan/  (커밋 제외 — 사람 얼굴)
"""
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from restore_compare import PERSON, SCALE, column, label, row, sharpness  # noqa: E402
from restore_original import restore_original, segment  # noqa: E402

RESULTS = os.path.join(ROOT, 'data', 'samples', 'results')
OUT = os.path.join(RESULTS, 'restore_esrgan')
# (옷, 시드, FASHN 결과 위치, 종류). 오랑우탄 티는 프린트·글자가 업스케일러에서 바뀌는지 보려고 넣었다.
CASES = [
    ('shirt_blue', 42, 'fashn_baseline', 'tops'), ('shirt_blue', 123, 'fashn_baseline', 'tops'),
    ('tee_khaki', 42, 'fashn_baseline', 'tops'), ('tee_khaki', 123, 'fashn_baseline', 'tops'),
    ('pants_forest', 42, 'fashn_baseline', 'bottoms'), ('pants_forest', 123, 'fashn_baseline', 'bottoms'),
    ('tee_orangutan', 42, 'fashn_compile', 'tops'),
]
CROP = 200   # 확대 비교 칸 크기의 절반 (1152px 기준)


def garment_crops(classes, mask_small):
    """새 옷 영역의 가운데와 위쪽(칼라·프린트 윗부분) 두 곳. 1152px 좌표."""
    m = (np.asarray(mask_small) > 127) & (classes == 4)
    ys, xs = np.nonzero(m)
    cx = int(np.median(xs) * SCALE)
    top = int(np.percentile(ys, 5) * SCALE)
    mid = int(np.median(ys) * SCALE)
    return [(cx - CROP, top, cx + CROP, top + 2 * CROP), (cx - CROP, mid - CROP, cx + CROP, mid + CROP)]


def main(upscaled_dir):
    os.makedirs(OUT, exist_ok=True)
    person = ImageOps.exif_transpose(Image.open(PERSON)).convert('RGB')
    full_rows, zoom_rows, lines = [], [], []
    for key, seed, folder, category in CASES:
        path = glob.glob(os.path.join(RESULTS, folder, 'fashn_t4x2', f'*__{key}__*_s{seed}_*.png'))[0]
        up_path = os.path.join(upscaled_dir, os.path.basename(path))
        if not os.path.exists(up_path):
            print('업스케일 결과 없음, 건너뜀:', os.path.basename(path))
            continue
        result = Image.open(path).convert('RGB')
        plain, mask = restore_original(person, result, category, SCALE)
        sharp, _ = restore_original(person, result, category, SCALE, upscaled=Image.open(up_path))
        sharp.save(os.path.join(OUT, f'{key}_s{seed}.png'))

        full_rows.append(row([label(plain, f'{key} s{seed} plain'), label(sharp, 'Real-ESRGAN')]).resize(
            (plain.width + 4, plain.height // 2)))
        small_mask = mask.resize(result.size, Image.BILINEAR)
        boxes = garment_crops(segment(result), small_mask)
        zoom_rows.append(row([label(plain.crop(boxes[0]), f'{key} s{seed} plain'),
                              label(sharp.crop(boxes[0]), 'Real-ESRGAN'),
                              plain.crop(boxes[1]), sharp.crop(boxes[1])]))
        a, b = sharpness(plain, boxes[1]), sharpness(sharp, boxes[1])
        lines.append(f'{key:13s} s{seed:<4d} 옷 가운데 선명도 단순 확대 {a:7.1f} → Real-ESRGAN {b:7.1f}')
        print(lines[-1], flush=True)

    column(full_rows).save(os.path.join(OUT, '_full.jpg'), quality=90)
    column(zoom_rows).save(os.path.join(OUT, '_zoom.jpg'), quality=92)
    with open(os.path.join(OUT, '_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('저장:', OUT)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
