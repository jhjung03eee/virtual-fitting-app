"""실제 쇼핑몰 옷으로 CFG/스텝 설정을 검증한다.

지금 기본값(DPM++ 4스텝 + CFG 끔)은 데모 카디건 **한 벌**로만 확인했다.
CFG를 끄면 옷 반영 강도가 약해질 수 있는데, 무늬가 복잡하거나 색이 흐린 옷에서
어떤지 모르는 상태다. 실제 상품 사진으로 그 단서를 닫는 것이 목적이다.

각 옷마다 5가지 설정으로 합성해 한 장에 나란히 붙인다:
    DPM++ 30 + CFG on   품질 기준점 (같은 샘플러라 스텝/CFG 효과만 분리된다)
    DPM++ 8  + CFG on
    DPM++ 8  + CFG off
    DPM++ 4  + CFG on
    DPM++ 4  + CFG off  현재 기본값

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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from tryon_core import try_on, load_models, REPO_DIR  # noqa: E402

GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

OUT_DIR = os.path.join(OUT_ROOT, 'real')
CSV_PATH = os.path.join(OUT_DIR, 'results.csv')
FIELDS = ['garment', 'cloth_type', 'config', 'scheduler', 'steps', 'guidance',
          'mask_s', 'diffusion_s', 'total_s', 'ssim_vs_ref', 'out_path',
          'platform', 'gpu']

# (표시명, 스케줄러, 스텝, guidance). 첫 항목이 SSIM 기준점이다.
CONFIGS = [
    ('기준 30스텝 CFG켬', 'dpm', 30, 2.5),
    ('8스텝 CFG켬', 'dpm', 8, 2.5),
    ('8스텝 CFG끔', 'dpm', 8, 1.0),
    ('4스텝 CFG켬', 'dpm', 4, 2.5),
    ('4스텝 CFG끔 (현재 기본값)', 'dpm', 4, 1.0),
]

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


def _pil_font(size):
    """PIL 기본 폰트는 한글을 못 그린다(두부 처리). 설치된 TTF를 직접 연다."""
    from matplotlib import font_manager
    for name in ('NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 'AppleGothic',
                 'Noto Sans CJK KR', 'DejaVu Sans'):
        try:
            return ImageFont.truetype(
                font_manager.findfont(name, fallback_to_default=False), size)
        except Exception:
            continue
    return ImageFont.load_default()


def ssim_against(reference_path, path):
    """기준 이미지와 얼마나 같은 그림인지. skimage가 없으면 None."""
    try:
        import numpy as np
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    a = np.array(Image.open(reference_path).convert('L'), dtype=np.float64)
    b = np.array(Image.open(path).convert('L'), dtype=np.float64)
    if a.shape != b.shape:
        return None
    return round(float(structural_similarity(a, b, data_range=255.0)), 4)


def append_row(row):
    is_new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def build_grid(garment_path, person_path, results, out_path):
    """옷 사진 + 설정별 결과를 한 장에 나란히 붙인다.

    표로 숫자만 보면 '옷이 제대로 반영됐는지'는 알 수 없다. 눈으로 비교하는 게
    이 실험의 목적이므로 그리드가 본체다.
    """
    cell_w, cell_h = 288, 384
    pad, header, caption = 12, 34, 40
    columns = [('옷 사진', garment_path, None, None)] + [
        (label, path, seconds, ssim) for label, path, seconds, ssim in results
    ]

    width = pad + len(columns) * (cell_w + pad)
    height = header + cell_h + caption + pad * 2
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    font_title = _pil_font(16)
    font_note = _pil_font(14)

    for index, (label, path, seconds, ssim) in enumerate(columns):
        x = pad + index * (cell_w + pad)
        thumb = Image.open(path).convert('RGB')
        thumb.thumbnail((cell_w, cell_h))
        offset = x + (cell_w - thumb.width) // 2
        canvas.paste(thumb, (offset, header + pad))
        draw.text((x, pad), label, fill='black', font=font_title)
        if seconds is not None:
            note = f'{seconds:.1f}초'
            if ssim is not None:
                note += f'   SSIM {ssim:.3f}'
            draw.text((x, header + pad + cell_h + 6), note, fill='#444', font=font_note)

    name = os.path.splitext(os.path.basename(garment_path))[0]
    draw.text((pad, height - 22),
              f'{name} · 인물 {os.path.basename(person_path)} · {PLATFORM} {GPU_NAME}',
              fill='#888', font=font_note)
    canvas.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--garments', default=os.path.join(REPO_ROOT, 'data', 'samples'),
                    help='옷 사진이 든 폴더 (치수표 *_size.* 는 자동 제외)')
    ap.add_argument('--person', default=None, help='인물 사진 (생략 시 저장소 데모)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

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

    for garment in garments:
        name = os.path.splitext(os.path.basename(garment))[0]
        cloth_type = cloth_type_of(garment)
        print(f'--- {name} ({cloth_type}) ---', flush=True)

        results, reference_path = [], None
        for label, scheduler, steps, guidance in CONFIGS:
            slug = f'{name}_{scheduler}{steps}_g{guidance:g}'
            out_path = os.path.join(OUT_DIR, f'{slug}.png')

            if os.path.exists(out_path):
                print(f'  건너뜀 (이미 있음): {label}', flush=True)
                seconds = None
            else:
                result, _mask, timing = try_on(
                    person=person, garment=garment, cloth_type=cloth_type,
                    steps=steps, guidance_scale=guidance, seed=args.seed,
                    scheduler=scheduler, return_timing=True,
                )
                result.save(out_path)
                seconds = timing['total_s']
                print(f'  {label}: {seconds:.1f}초', flush=True)

            if reference_path is None:
                reference_path = out_path
                ssim = 1.0
            else:
                ssim = ssim_against(reference_path, out_path)

            results.append((label, out_path, seconds, ssim))
            if seconds is not None:
                append_row({
                    'garment': name, 'cloth_type': cloth_type, 'config': label,
                    'scheduler': scheduler, 'steps': steps, 'guidance': guidance,
                    'mask_s': timing['mask_s'], 'diffusion_s': timing['diffusion_s'],
                    'total_s': seconds, 'ssim_vs_ref': ssim, 'out_path': out_path,
                    'platform': PLATFORM, 'gpu': GPU_NAME,
                })

        grid = build_grid(garment, person, results,
                          os.path.join(OUT_DIR, f'compare_{name}.png'))
        print(f'  그리드 -> {grid}\n', flush=True)

    print('=== 결과 ===', flush=True)
    for f in sorted(os.listdir(OUT_DIR)):
        p = os.path.join(OUT_DIR, f)
        print(f'{os.path.getsize(p) / 1024:8.0f} KB  {p}', flush=True)


if __name__ == '__main__':
    main()
