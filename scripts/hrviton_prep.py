"""HR-VITON(변형 + GAN 방식) 비교 실험용 입력 폴더를 만든다.

diffusion(CatVTON)과 달리 HR-VITON은 옷 사진을 몸에 맞게 **변형(warping)해서 붙이고**
GAN으로 다듬는다. 옷 픽셀을 옮기므로 프린트·질감 보존에 강하다고 알려져 있지만,
학습 데이터(VITON-HD 스튜디오 상반신) 밖의 사진에서는 잘 무너진다. 우리 사진에서 실제로
어떤지 같은 인물·옷으로 비교하기 위한 입력을 만든다.

HR-VITON 테스트 로더(cp_dataset_test.py)가 요구하는 폴더와, 우리가 대신 쓰는 도구:
    image/                       768x1024 인물
    image-parse-v3/              사람 부위 파싱 0~19 (원래 CIHP_PGN) -> CatVTON의 SCHP-LIP (라벨 체계 동일)
    image-parse-agnostic-v3.2/   상의·팔을 지운 파싱 -> HR-VITON의 get_parse_agnostic.py 그대로
    openpose_json/               BODY_25 키포인트 (원래 OpenPose) -> MediaPipe Pose를 BODY_25 순서로 옮김
    openpose_img/                포즈 렌더링. 모델 입력에 안 쓰이고 시각화만 하므로 검은 이미지
    image-densepose/             DensePose 부위 색칠 (원래 detectron2 dp_segm) -> CatVTON DensePose 부위 번호를 같은 컬러맵으로
    cloth/, cloth-mask/          옷 사진(흰 배경 768x1024)과 이진 마스크

    python scripts/hrviton_prep.py --persons DIR --garments DIR --out DATAROOT/test --upper-crop NAME1,NAME2
"""
import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'app'))
from paths import VENV_PY, child_env  # noqa: E402


def _reexec_in_venv():
    if os.environ.get('VFA_NO_REEXEC') == '1':
        return
    if os.path.abspath(sys.executable) == os.path.abspath(VENV_PY):
        return
    if not os.path.exists(VENV_PY):
        sys.exit(f'Python 3.9 환경({VENV_PY})이 없습니다. scripts/setup_env.py 를 먼저 실행하세요.')
    raise SystemExit(subprocess.call(
        [VENV_PY, '-u', os.path.abspath(__file__)] + sys.argv[1:], env=child_env()))


_reexec_in_venv()

import glob  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from body_measure import POSE_LANDMARKS, _run_pose  # noqa: E402
from tryon_core import WIDTH, HEIGHT, _crop_to_aspect, _to_image  # noqa: E402

# MediaPipe 이름 -> OpenPose BODY_25 번호. HR-VITON이 실제로 쓰는 건 1~7, 9, 12 뿐이다.
BODY25_FROM_MEDIAPIPE = {
    0: 'nose', 2: 'right_shoulder', 3: 'right_elbow', 4: 'right_wrist',
    5: 'left_shoulder', 6: 'left_elbow', 7: 'left_wrist',
    9: 'right_hip', 10: 'right_knee', 11: 'right_ankle',
    12: 'left_hip', 13: 'left_knee', 14: 'left_ankle',
}
EXTRA_MEDIAPIPE = {15: 5, 16: 2, 17: 8, 18: 7}   # 눈·귀 (MediaPipe 인덱스)
DENSEPOSE_PART_LABELS = 24


def pose_body25(image):
    """768x1024 이미지 -> BODY_25 (25, 3) 픽셀 좌표. 못 찾은 점은 0 (HR-VITON 규약)."""
    small = image.copy()
    small.thumbnail((1280, 1280))
    landmarks, _mask = _run_pose(np.array(small))
    if landmarks is None:
        return None
    w, h = image.size
    points = np.zeros((25, 3), dtype=np.float64)

    def put(index, lm):
        if getattr(lm, 'visibility', 1.0) >= 0.3:
            points[index] = (lm.x * w, lm.y * h, getattr(lm, 'visibility', 1.0))

    for index, name in BODY25_FROM_MEDIAPIPE.items():
        put(index, landmarks[POSE_LANDMARKS[name]])
    for index, mp_index in EXTRA_MEDIAPIPE.items():
        put(index, landmarks[mp_index])
    for index, pair in ((1, (2, 5)), (8, (9, 12))):   # Neck, MidHip = 양쪽 중점
        a, b = points[pair[0]], points[pair[1]]
        if a[2] > 0 and b[2] > 0:
            points[index] = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2]))
    return points


def upper_body_crop(image):
    """전신 사진에서 VITON-HD처럼 머리~허벅지 윗부분 상반신을 3:4로 자른다."""
    small = image.copy()
    small.thumbnail((1280, 1280))
    landmarks, _mask = _run_pose(np.array(small))
    if landmarks is None:
        return None
    w, h = image.size

    def xy(name):
        lm = landmarks[POSE_LANDMARKS[name]]
        return np.array([lm.x * w, lm.y * h])

    shoulder = (xy('left_shoulder') + xy('right_shoulder')) / 2
    hip = (xy('left_hip') + xy('right_hip')) / 2
    torso = np.linalg.norm(hip - shoulder)
    top = xy('nose')[1] - 0.55 * torso
    bottom = hip[1] + 0.45 * torso
    box_h = bottom - top
    box_w = box_h * WIDTH / HEIGHT
    if box_w > w:
        box_w, box_h = w, w * HEIGHT / WIDTH
    left = float(np.clip(shoulder[0] - box_w / 2, 0, w - box_w))
    top = float(np.clip(top, 0, h - box_h))
    return image.crop((round(left), round(top), round(left + box_w), round(top + box_h)))


def remap_parse(lip, densepose, pose):
    """SCHP-LIP 파싱을 VITON-HD(CIHP) 규약에 맞춘다.

    - 10번: LIP은 '점프수트', CIHP는 '목·몸통 피부'. 점프수트는 상의(5)로 보내고,
      DensePose 몸통인데 LIP가 배경으로 둔 곳(드러난 목)을 10으로 칠한다.
    - 14/15 팔: HR-VITON은 14를 포즈 5·6·7(왼쪽 = 화면 오른쪽)과 짝짓는다.
      SCHP의 좌우 규약이 같은지 확신할 수 없어, 목 기준 화면 좌우로 다시 정한다.
    """
    parse = lip.copy()
    parse[parse == 10] = 5
    torso = np.isin(densepose, [1, 2])
    parse[torso & (parse == 0)] = 10
    if pose is not None and pose[1, 2] > 0:
        xs = np.arange(parse.shape[1])[None, :].repeat(parse.shape[0], axis=0)
        arm = np.isin(parse, [14, 15])
        parse[arm & (xs > pose[1, 0])] = 14
        parse[arm & (xs <= pose[1, 0])] = 15
    return parse


def colorize_densepose(parts):
    """DensePose 부위 번호(0~24) -> detectron2 dp_segm 시각화와 같은 컬러맵, 배경은 검정."""
    scaled = (parts.astype(np.float32) * 255.0 / DENSEPOSE_PART_LABELS).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_PARULA)[..., ::-1]
    color[parts == 0] = 0
    return Image.fromarray(color)


def prepare_cloth(path):
    """옷 사진 -> 흰 배경 768x1024 옷 + 이진 마스크."""
    from scipy import ndimage

    image = _to_image(path).convert('RGB')
    array = np.asarray(image, dtype=np.float64)
    h, w = array.shape[:2]
    bh, bw = max(1, h // 20), max(1, w // 20)
    corners = np.concatenate([array[:bh, :bw].reshape(-1, 3), array[:bh, -bw:].reshape(-1, 3),
                              array[-bh:, :bw].reshape(-1, 3), array[-bh:, -bw:].reshape(-1, 3)])
    background = np.median(corners, axis=0)
    mask = np.linalg.norm(array - background, axis=2) > 25
    mask = ndimage.binary_closing(mask, iterations=3)
    # 옷 안의 작은 구멍(흰 글자·단추)만 메운다. 전부 메우면 소매와 몸통 사이 배경까지
    # 옷으로 잡혀 마스크가 종 모양이 됐다(셔츠·맨투맨에서 확인).
    holes, count = ndimage.label(ndimage.binary_fill_holes(mask) & ~mask)
    if count:
        sizes = ndimage.sum(holes > 0, holes, range(1, count + 1))
        small = [i + 1 for i, s in enumerate(sizes) if s < 0.01 * mask.size]
        mask |= np.isin(holes, small)
    labels, count = ndimage.label(mask)
    if count > 1:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == (int(np.argmax(sizes)) + 1)

    white = array.copy()
    white[~mask] = 255
    ys, xs = np.nonzero(mask)
    pad = 0.06
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw_, bh_ = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    if bw_ / bh_ > WIDTH / HEIGHT:
        bh_ = bw_ * HEIGHT / WIDTH
    else:
        bw_ = bh_ * WIDTH / HEIGHT
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (round(cx - bw_ / 2), round(cy - bh_ / 2), round(cx + bw_ / 2), round(cy + bh_ / 2))

    cloth = Image.new('RGB', (box[2] - box[0], box[3] - box[1]), 'white')
    cloth.paste(Image.fromarray(white.astype(np.uint8)), (-box[0], -box[1]))
    cloth_mask = Image.new('L', cloth.size, 0)
    cloth_mask.paste(Image.fromarray((mask * 255).astype(np.uint8)), (-box[0], -box[1]))
    return (cloth.resize((WIDTH, HEIGHT), Image.LANCZOS),
            cloth_mask.resize((WIDTH, HEIGHT), Image.NEAREST))


def load_automasker():
    from huggingface_hub import snapshot_download
    from tryon_core import _ensure_repo_on_path
    _ensure_repo_on_path()
    from model.cloth_masker import AutoMasker
    repo_path = snapshot_download(repo_id='zhengchong/CatVTON')
    return AutoMasker(densepose_ckpt=os.path.join(repo_path, 'DensePose'),
                      schp_ckpt=os.path.join(repo_path, 'SCHP'), device='cuda')


def find_images(directory):
    found = []
    for ext in ('jpg', 'jpeg', 'png', 'webp'):
        found += glob.glob(os.path.join(directory, f'*.{ext}'))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--persons', required=True)
    ap.add_argument('--garments', required=True)
    ap.add_argument('--out', required=True, help='DATAROOT/test')
    ap.add_argument('--upper-crop', default='',
                    help='쉼표 목록. 파일명에 이 문자열이 든 전신 사진은 상반신 크롭본도 만든다')
    args = ap.parse_args()
    upper_names = [n for n in args.upper_crop.split(',') if n]

    folders = ['image', 'image-parse-v3', 'image-parse-agnostic-v3.2', 'openpose_json',
               'openpose_img', 'image-densepose', 'cloth', 'cloth-mask']
    for folder in folders:
        os.makedirs(os.path.join(args.out, folder), exist_ok=True)

    sys.path.insert(0, os.environ.get('HRVITON_REPO', ''))
    from get_parse_agnostic import get_im_parse_agnostic

    # 사람 사진 만들기: 가운데 3:4 (CatVTON과 같은 입력) + 필요하면 상반신 크롭
    persons = []
    for path in find_images(args.persons):
        stem = os.path.splitext(os.path.basename(path))[0]
        full = _crop_to_aspect(_to_image(path).convert('RGB'))
        persons.append((f'{stem}_full.jpg', full.resize((WIDTH, HEIGHT), Image.LANCZOS)))
        if any(n in stem for n in upper_names):
            crop = upper_body_crop(full)
            if crop is None:
                print(f'  {stem}: 포즈를 못 찾아 상반신 크롭을 건너뜀', flush=True)
            else:
                persons.append((f'{stem}_upper.jpg', crop.resize((WIDTH, HEIGHT), Image.LANCZOS)))

    automasker = load_automasker()
    kept = []
    for name, image in persons:
        pose = pose_body25(image)
        if pose is None:
            print(f'  {name}: 포즈 검출 실패 -> 제외', flush=True)
            continue
        parsed = automasker.preprocess_image(image)
        lip = np.array(parsed['schp_lip'])
        lip = lip[..., 0] if lip.ndim == 3 else lip
        densepose = np.array(parsed['densepose'])
        densepose = densepose[..., 0] if densepose.ndim == 3 else densepose
        parse = remap_parse(lip.astype(np.uint8), densepose, pose)

        stem = os.path.splitext(name)[0]
        image.save(os.path.join(args.out, 'image', name), quality=95)
        parse_image = Image.fromarray(parse.astype(np.uint8), 'L')
        parse_image.save(os.path.join(args.out, 'image-parse-v3', f'{stem}.png'))
        get_im_parse_agnostic(parse_image, pose[:, :2].copy()).save(
            os.path.join(args.out, 'image-parse-agnostic-v3.2', f'{stem}.png'))
        colorize_densepose(densepose).save(os.path.join(args.out, 'image-densepose', name), quality=95)
        Image.new('RGB', (WIDTH, HEIGHT)).save(os.path.join(args.out, 'openpose_img', f'{stem}_rendered.png'))
        with open(os.path.join(args.out, 'openpose_json', f'{stem}_keypoints.json'), 'w') as f:
            json.dump({'people': [{'pose_keypoints_2d': pose.reshape(-1).tolist()}]}, f)
        print(f'  {name}: 준비 완료 (파싱 라벨 {sorted(np.unique(parse).tolist())})', flush=True)
        kept.append(name)

    garments = []
    for path in find_images(args.garments):
        name = os.path.splitext(os.path.basename(path))[0] + '.jpg'
        cloth, cloth_mask = prepare_cloth(path)
        cloth.save(os.path.join(args.out, 'cloth', name), quality=95)
        cloth_mask.save(os.path.join(args.out, 'cloth-mask', name), quality=100)
        garments.append(name)
        print(f'  옷 {name}: 마스크 {np.asarray(cloth_mask).mean() / 255:.0%}', flush=True)

    # 테스트 로더는 'paired' 설정용으로 cloth/<인물 파일명> 도 연다. 실제로는 unpaired만 쓰므로 자리만 채운다.
    for name in kept:
        for folder in ('cloth', 'cloth-mask'):
            src = os.path.join(args.out, folder, garments[0])
            Image.open(src).save(os.path.join(args.out, folder, name), quality=95)

    pairs_path = os.path.join(os.path.dirname(args.out.rstrip('/\\')), 'test_pairs.txt')
    with open(pairs_path, 'w') as f:
        for person in kept:
            for garment in garments:
                f.write(f'{person} {garment}\n')
    print(f'인물 {len(kept)} × 옷 {len(garments)} = {len(kept) * len(garments)}쌍 -> {pairs_path}', flush=True)


if __name__ == '__main__':
    main()
