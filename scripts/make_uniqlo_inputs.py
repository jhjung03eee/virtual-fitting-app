"""유니클로 실측 실험(docs/PLAN.md F-31) 입력 만들기 — 모든 사진을 같은 규칙으로 크롭한다. GPU 불필요.

`data/samples/uniqlo test/`:
  - orig1~4.jpg  흰 티 · 갈색 바지 (합성에 넣을 인물 사진, 자세가 조금씩 다름)
  - M/L/XL.jpg   같은 사람이 유니클로 남색 크루넥 반팔을 사이즈별로 실제로 입은 사진 (정답)
  - cloth image.jpg  상품 사진

원본은 사람이 화면 세로의 약 60%라 촬영 가이드 기준(65%, F-12)에 걸린다. 그리고 합성 결과와 실제 착용 사진을
같은 축에서 비교하려면 사람이 화면에서 같은 크기·같은 위치에 있어야 한다. 그래서 자세 인식의 사람 영역으로
머리끝~발끝을 잡고, 그 높이 × 1.2 를 세로로 하는 3:4 상자를 몸 가운데에 맞춰 자른다.

    python scripts/make_uniqlo_inputs.py
    → data/samples/uniqlo_inputs/  (사람 얼굴이라 커밋 제외. Kaggle 비공개 데이터셋 vfa-uniqlo-test 로 올린다)
"""
import os
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'app'))

from photo_check import check_photo, run_pose  # noqa: E402

SRC = os.path.join(ROOT, 'data', 'samples', 'uniqlo test')
OUT = os.path.join(ROOT, 'data', 'samples', 'uniqlo_inputs')
MARGIN = 1.2          # 머리끝~발끝 높이의 이 배가 크롭 세로
OUT_SIZE = (1536, 2048)
# 원본 이름 → Kaggle에서 찾을 이름
PHOTOS = {
    'orig1.jpg': 'person_uniqlo_orig1.jpg', 'orig2.jpg': 'person_uniqlo_orig2.jpg',
    'orig3.jpg': 'person_uniqlo_orig3.jpg', 'orig4.jpg': 'person_uniqlo_orig4.jpg',
    'M.jpg': 'worn_uniqlo_M.jpg', 'L.jpg': 'worn_uniqlo_L.jpg', 'XL.jpg': 'worn_uniqlo_XL.jpg',
}


def person_box(image):
    """사람 영역(자세 인식의 분할 마스크)의 (왼, 위, 오른, 아래) — 원본 좌표."""
    _landmarks, seg = run_pose(image)
    mask = np.asarray(seg, dtype=np.float32).squeeze() > 0.5
    ys, xs = np.nonzero(mask)
    sx, sy = image.width / mask.shape[1], image.height / mask.shape[0]
    return xs.min() * sx, ys.min() * sy, xs.max() * sx, ys.max() * sy


def crop_person(image):
    left, top, right, bottom = person_box(image)
    h = (bottom - top) * MARGIN
    w = h * 3 / 4
    cx, cy = (left + right) / 2, (top + bottom) / 2
    box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    # 화면 밖으로 나가면 상자를 안으로 민다(크기는 유지)
    dx = max(0, -box[0]) - max(0, box[2] - image.width)
    dy = max(0, -box[1]) - max(0, box[3] - image.height)
    box = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
    return image.crop(tuple(round(v) for v in box)).resize(OUT_SIZE, Image.LANCZOS)


def main():
    os.makedirs(OUT, exist_ok=True)
    for src, dst in PHOTOS.items():
        image = ImageOps.exif_transpose(Image.open(os.path.join(SRC, src))).convert('RGB')
        cropped = crop_person(image)
        cropped.save(os.path.join(OUT, dst), quality=95)
        check = check_photo(cropped)
        print(f"{src:10s} → {dst:26s} 몸 비율 {check.metrics.get('body_fraction', 0):.2f}  "
              f"{'통과' if check.ok else 'NG: ' + ' / '.join(check.errors)}", flush=True)
    garment = ImageOps.exif_transpose(Image.open(os.path.join(SRC, 'cloth image.jpg'))).convert('RGB')
    garment.save(os.path.join(OUT, 'garment_uniqlo_navy.jpg'), quality=95)
    print('저장:', OUT)


if __name__ == '__main__':
    main()
