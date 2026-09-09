"""실제 쇼핑몰 옷으로 추론 설정을 검증한다.

가속 설정을 데모 이미지로만 고르면 안 된다는 것을 이 스크립트로 확인했다.
CatVTON 데모 카디건 한 벌에서는 CFG를 꺼도 멀쩡해 보였지만, 실제 상품으로 돌려보니
사진 프린트 티셔츠가 통째로 뭉개졌다. 그래서 CFG 끄기는 폐기하고,
지금은 **스텝 수만** 촘촘히 비교한다 (docs/PLAN.md 속도 최적화 절 참고).

각 옷마다 설정별로 합성해 옷 사진과 함께 한 장에 나란히 붙인다.
SSIM은 **옷 마스크 안쪽만** 잰다 — 전체 이미지로 재면 인물과 배경이 화면의
대부분이라 옷이 붕괴해도 값이 거의 안 움직여서 지표 구실을 못 한다.

실행:
    python scripts/real_garment_check.py --garments data/samples
venv39가 아니면 자동으로 재실행한다. 중단 후 다시 돌리면 끝난 건은 건너뛴다.
"""
import argparse
import csv
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'app'))
from paths import VENV_PY, OUT_ROOT, PLATFORM, child_env  # noqa: E402


def _reexec_in_venv():
    """venv39 인터프리터가 아니면 그걸로 다시 실행한다."""
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

import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_grid import build_grid  # noqa: E402

from tryon_core import try_on, load_models, REPO_DIR  # noqa: E402

GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

OUT_DIR = os.path.join(OUT_ROOT, 'real')
CSV_PATH = os.path.join(OUT_DIR, 'results.csv')
FIELDS = ['garment', 'cloth_type', 'config', 'scheduler', 'steps', 'guidance',
          'mask_s', 'diffusion_s', 'total_s', 'ssim_vs_ref', 'ssim_garment',
          'out_path', 'platform', 'gpu']

# (표시명, 스케줄러, 스텝, guidance). 각 묶음의 첫 항목이 SSIM 기준점이다.
CONFIG_SETS = {
    # 스텝을 얼마나 줄일 수 있는지 (속도 관점)
    'steps': [
        ('기준 30스텝', 'dpm', 30, 2.5, True, None),
        ('8스텝 (현재 기본값)', 'dpm', 8, 2.5, True, None),
        ('6스텝', 'dpm', 6, 2.5, True, None),
        ('5스텝', 'dpm', 5, 2.5, True, None),
        ('4스텝', 'dpm', 4, 2.5, True, None),
    ],
    # 이 모델로 낼 수 있는 최선이 어디인지 (품질 관점)
    'quality': [
        ('원본 설정 DDIM 30 g2.5', 'ddim', 30, 2.5, True, None),
        ('DPM 30 g2.5', 'dpm', 30, 2.5, True, None),
        ('DPM 30 g5.0', 'dpm', 30, 5.0, True, None),
        ('DPM 30 g7.5', 'dpm', 30, 7.5, True, None),
        ('DPM 8 g5.0', 'dpm', 8, 5.0, True, None),
        ('DPM 8 g2.5 (현재 기본값)', 'dpm', 8, 2.5, True, None),
    ],
    # 마스크 밖을 원본으로 되돌리면(composite) guidance를 올려도 얼굴·배경이
    # 안 망가지는지. 망가지지 않는다면 옷 반영을 강하게 줄 수 있게 된다.
    'composite': [
        ('합성없음 g2.5 (지금까지)', 'dpm', 8, 2.5, False, None),
        ('합성 g2.5', 'dpm', 8, 2.5, True, None),
        ('합성없음 g5.0', 'dpm', 8, 5.0, False, None),
        ('합성 g5.0', 'dpm', 8, 5.0, True, None),
        ('합성 g7.5', 'dpm', 8, 7.5, True, None),
        ('합성 30스텝 g5.0', 'dpm', 30, 5.0, True, None),
    ],
    # 프린트가 나오는지가 시드에 따라 갈리는지. 지금까지 seed=42로 고정해와서
    # '이 인물에서는 안 된다'가 사실은 '이 시드에서는 안 된다'일 수 있다.
    # 설정은 모두 같고 시드만 바꾼다.
    # GitHub 예시처럼 안 나오는 이유를 가른다. 저장소 데모 앱의 기본값은
    # DDIM 50스텝이고 마스크 밖을 되돌리지 않는다. 우리는 DPM 8스텝 + 합성이다.
    # 스텝 차이 때문인지, 합성 때문인지, 아니면 입력 사진 자체가 학습 분포
    # (VITON-HD/DressCode 스튜디오 촬영) 밖이라 그런지 본다.
    'repo': [
        ('저장소 그대로 DDIM 50 합성없음', 'ddim', 50, 2.5, False, None),
        ('DDIM 50 + 합성', 'ddim', 50, 2.5, True, None),
        ('DPM 50 + 합성', 'dpm', 50, 2.5, True, None),
        ('DPM 30 + 합성', 'dpm', 30, 2.5, True, None),
        ('DPM 8 + 합성 (현재)', 'dpm', 8, 2.5, True, None),
        ('DPM 50 g5.0 + 합성', 'dpm', 50, 5.0, True, None),
    ],
    'seeds': [(f'seed {v}', 'dpm', 8, 2.5, True, v) for v in (42, 1, 7, 123, 2024, 31337)],
}
CONFIGS = CONFIG_SETS['steps']


# 파일명으로 상/하의를 가른다. 옷 종류를 틀리면 엉뚱한 부위에 합성된다
# (상의 이미지에 cloth_type='lower'를 주면 회색 트레이닝 바지가 나온다).
LOWER_HINTS = ('pants', 'bottom', 'skirt', 'denim', 'trouser', '바지', '하의')


def cloth_type_of(path):
    name = os.path.basename(path).lower()
    return 'lower' if any(h in name for h in LOWER_HINTS) else 'upper'


def find_garments(directory):
    """치수표 스크린샷(*_size.*)은 빼고 옷 사진만 고른다."""
    found = []
    for ext in ('jpg', 'jpeg', 'png', 'webp'):
        found += glob.glob(os.path.join(directory, f'*.{ext}'))
    return sorted(p for p in found
                  if not os.path.splitext(os.path.basename(p))[0].endswith('_size'))


def ssim_against(reference_path, path, mask=None):
    """기준 이미지와 얼마나 같은 그림인지.

    mask를 주면 **그 안쪽만** 잰다. 전체 이미지로 재면 인물과 배경이 화면의
    대부분이라 옷이 완전히 망가져도 값이 거의 안 움직인다 — 실제로 붕괴한
    티셔츠가 0.950, 멀쩡한 맨투맨이 0.953으로 구별되지 않았다.
    옷 재현을 재려면 옷 영역만 봐야 한다.

    반환: (전체 SSIM, 옷 영역 SSIM). skimage가 없으면 (None, None).
    """
    try:
        import numpy as np
        from skimage.metrics import structural_similarity
    except ImportError:
        return None, None
    a = np.array(Image.open(reference_path).convert('L'), dtype=np.float64)
    b = np.array(Image.open(path).convert('L'), dtype=np.float64)
    if a.shape != b.shape:
        return None, None

    score, smap = structural_similarity(a, b, data_range=255.0, full=True)
    full_score = round(float(score), 4)
    if mask is None:
        return full_score, None

    selected = np.array(mask.convert('L')) > 127
    if not selected.any():
        return full_score, None
    return full_score, round(float(smap[selected].mean()), 4)


def garment_mask(person_path, cloth_type):
    """옷 영역 마스크. SSIM을 옷 안쪽으로 한정하는 데 쓴다.

    같은 인물·같은 부위면 결과가 같으므로 부위별로 한 번만 만든다.
    """
    _pipeline, automasker, _proc, _device = load_models()
    from utils import resize_and_crop
    person = resize_and_crop(Image.open(person_path).convert('RGB'), (768, 1024))
    return automasker(person, cloth_type)['mask']


def append_row(row):
    is_new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--garments', default=os.path.join(REPO_ROOT, 'data', 'samples'),
                    help='옷 사진이 든 폴더 (치수표 *_size.* 는 자동 제외)')
    ap.add_argument('--person', default=None, help='인물 사진 (생략 시 저장소 데모)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--config-set', default='steps', choices=sorted(CONFIG_SETS),
                    help="steps=스텝을 얼마나 줄일 수 있나 / quality=이 모델의 천장이 어디인가")
    args = ap.parse_args()

    global CONFIGS
    CONFIGS = CONFIG_SETS[args.config_set]

    garments = find_garments(args.garments)
    if not garments:
        sys.exit(f'옷 사진을 찾지 못했습니다: {args.garments}')

    person = args.person or sorted(
        glob.glob(os.path.join(REPO_DIR, 'resource/demo/example/person/men/*')))[0]

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'옷 {len(garments)}벌 × 설정 {len(CONFIGS)}가지 = {len(garments) * len(CONFIGS)}건',
          flush=True)
    print(f'인물: {person}', flush=True)
    print(f'환경: {PLATFORM} / {GPU_NAME}\n', flush=True)

    load_models()
    print('모델 로딩 완료\n', flush=True)

    masks = {}  # 부위별 옷 마스크 (같은 인물이므로 한 번만 만든다)

    for garment in garments:
        name = os.path.splitext(os.path.basename(garment))[0]
        cloth_type = cloth_type_of(garment)
        print(f'--- {name} ({cloth_type}) ---', flush=True)

        if cloth_type not in masks:
            masks[cloth_type] = garment_mask(person, cloth_type)
        mask = masks[cloth_type]

        results, reference_path = [], None
        for label, scheduler, steps, guidance, composite, seed in CONFIGS:
            seed = args.seed if seed is None else seed
            slug = f'{name}_{scheduler}{steps}_g{guidance:g}_s{seed}'
            if not composite:
                slug += '_nocomp'
            out_path = os.path.join(OUT_DIR, f'{slug}.png')

            if os.path.exists(out_path):
                print(f'  건너뜀 (이미 있음): {label}', flush=True)
                seconds = None
            else:
                result, _mask, timing = try_on(
                    person=person, garment=garment, cloth_type=cloth_type,
                    steps=steps, guidance_scale=guidance, seed=seed,
                    scheduler=scheduler, return_timing=True, composite=composite,
                )
                result.save(out_path)
                seconds = timing['total_s']
                print(f'  {label}: {seconds:.1f}초', flush=True)

            if reference_path is None:
                reference_path = out_path
                ssim_full, ssim_garment = 1.0, 1.0
            else:
                ssim_full, ssim_garment = ssim_against(reference_path, out_path, mask)
                print(f'    SSIM 전체 {ssim_full}  옷영역 {ssim_garment}', flush=True)

            results.append((label, out_path, seconds, ssim_garment))
            if seconds is not None:
                append_row({
                    'garment': name, 'cloth_type': cloth_type, 'config': label,
                    'scheduler': scheduler, 'steps': steps, 'guidance': guidance,
                    'mask_s': timing['mask_s'], 'diffusion_s': timing['diffusion_s'],
                    'total_s': seconds, 'ssim_vs_ref': ssim_full,
                    'ssim_garment': ssim_garment, 'out_path': out_path,
                    'platform': PLATFORM, 'gpu': GPU_NAME,
                })

        grid = build_grid(
            garment, results, os.path.join(OUT_DIR, f'compare_{name}.png'),
            caption=f'{name} · 인물 {os.path.basename(person)} · {PLATFORM} {GPU_NAME}')
        print(f'  그리드 -> {grid}\n', flush=True)

    print('=== 결과 ===', flush=True)
    for f in sorted(os.listdir(OUT_DIR)):
        p = os.path.join(OUT_DIR, f)
        print(f'{os.path.getsize(p) / 1024:8.0f} KB  {p}', flush=True)


if __name__ == '__main__':
    main()
