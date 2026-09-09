"""사용자가 올린 인물 사진을 합성에 쓸 수 있게 정리한다.

폰으로 찍은 사진을 그대로 넣으면 두 가지가 문제가 된다.

1. **EXIF 회전** — 폰은 사진을 가로로 저장하고 "세로로 보여라"를 태그로 남긴다.
   PIL은 이 태그를 적용하지 않아서 인물이 옆으로 누운 채 합성된다.
2. **인물이 화면에서 너무 작다** — 전신을 담으려다 천장과 바닥이 잔뜩 들어간다.
   768x1024로 줄이면 옷이 차지하는 픽셀이 얼마 안 남아 디테일이 뭉개진다.

MediaPipe Pose로 사람의 위치를 찾아 3:4로 알맞게 잘라준다.

    python scripts/prepare_person.py data/person -o data/person/prepared

GPU가 필요 없으므로 로컬에서 돌린다.
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

TARGET_W, TARGET_H = 768, 1024
TARGET_RATIO = TARGET_W / TARGET_H  # 0.75

# 인물 위아래로 남길 여백 (인물 키 대비 비율).
# 머리 위는 조금 더 남겨야 잘린 느낌이 안 난다. 발밑은 조금만 남긴다.
MARGIN_TOP = 0.10
MARGIN_BOTTOM = 0.06


def person_box(image):
    """MediaPipe Pose로 인물의 대략적인 경계 상자를 찾는다.

    포즈 실행은 body_measure._run_pose를 그대로 쓴다. mediapipe가 버전에 따라
    solutions API와 Tasks API로 갈리는데, 그 분기가 이미 거기 들어있다
    (0.10.35에도 mp.solutions가 없다 — docs/ENVIRONMENT.md 참고).

    반환: (left, top, right, bottom) 픽셀 좌표. 못 찾으면 None.
    """
    try:
        from body_measure import _run_pose
    except ImportError:
        return None

    landmarks, _mask = _run_pose(np.array(image.convert('RGB')))
    if not landmarks:
        return None

    width, height = image.size
    xs, ys = [], []
    for landmark in landmarks:
        # visibility가 낮은 점은 추정값이라 상자를 엉뚱하게 늘린다
        if getattr(landmark, 'visibility', 1.0) < 0.5:
            continue
        xs.append(landmark.x * width)
        ys.append(landmark.y * height)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def crop_box(image, box):
    """인물 상자를 감싸는 3:4 영역을 계산한다. 이미지 밖으로 나가지 않게 맞춘다."""
    width, height = image.size
    left, top, right, bottom = box
    person_h = bottom - top

    # 머리 위/발밑 여백을 준 세로 범위
    want_top = top - person_h * MARGIN_TOP
    want_bottom = bottom + person_h * MARGIN_BOTTOM
    want_h = want_bottom - want_top
    want_w = want_h * TARGET_RATIO

    # 사람이 옆으로 넓으면(팔을 벌린 자세) 가로를 기준으로 다시 잡는다
    person_w = right - left
    if person_w * 1.15 > want_w:
        want_w = person_w * 1.15
        want_h = want_w / TARGET_RATIO

    # 이미지보다 크면 들어갈 수 있는 최대 크기로 줄인다
    scale = min(1.0, width / want_w, height / want_h)
    want_w, want_h = want_w * scale, want_h * scale

    center_x = (left + right) / 2
    center_y = (want_top + want_bottom) / 2

    new_left = min(max(center_x - want_w / 2, 0), width - want_w)
    new_top = min(max(center_y - want_h / 2, 0), height - want_h)
    return (round(new_left), round(new_top),
            round(new_left + want_w), round(new_top + want_h))


def prepare(path, out_dir):
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    original_size = image.size

    box = person_box(image)
    if box is None:
        # 사람을 못 찾으면 가운데를 3:4로 자르기만 한다
        note = '인물 검출 실패 — 가운데 기준으로 자름'
        width, height = image.size
        want_w = min(width, height * TARGET_RATIO)
        want_h = want_w / TARGET_RATIO
        left = (width - want_w) / 2
        top = (height - want_h) / 2
        cropped = image.crop((round(left), round(top),
                              round(left + want_w), round(top + want_h)))
        coverage = None
    else:
        crop = crop_box(image, box)
        cropped = image.crop(crop)
        coverage = (box[3] - box[1]) / (crop[3] - crop[1])
        note = f'인물이 세로의 {coverage:.0%} 차지'

    result = cropped.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + '.png')
    result.save(out_path)
    print(f'  {os.path.basename(path)}: {original_size} -> {result.size}  ({note})')
    return out_path, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='인물 사진 파일 또는 폴더')
    ap.add_argument('-o', '--out', default=None,
                    help='결과 폴더 (생략 시 입력 폴더 아래 prepared/)')
    args = ap.parse_args()

    paths = []
    for item in args.inputs:
        if os.path.isdir(item):
            for ext in ('jpg', 'jpeg', 'png', 'webp'):
                paths += glob.glob(os.path.join(item, f'*.{ext}'))
        else:
            paths.append(item)
    paths = sorted(p for p in paths if os.path.isfile(p))
    if not paths:
        sys.exit('인물 사진을 찾지 못했습니다.')

    out_dir = args.out or os.path.join(os.path.dirname(paths[0]), 'prepared')
    print(f'{len(paths)}장 처리 -> {out_dir}')
    for path in paths:
        prepare(path, out_dir)


if __name__ == '__main__':
    main()
