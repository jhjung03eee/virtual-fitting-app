"""실제 인물 사진으로 합성해 실패 케이스를 모은다 (docs/PLAN.md 항목 C).

지금까지의 테스트는 전부 CatVTON 데모 이미지였다. 스튜디오에서 찍은 정면 정자세라
**실제로 사람들이 올릴 사진과 다르다.** 계획서의 "우려되는 점"에 적은 팔 겹침·특이
포즈·복잡한 배경이 미검증 상태다.

인물 × 옷 조합을 기본 설정으로 전부 돌리고, 인물마다 한 장에 붙여 비교한다.
합성 결과와 함께 **마스크도 저장한다** — 결과가 이상할 때 원인이 마스크인지
확산인지 구별하려면 마스크를 봐야 한다.

    python scripts/real_person_check.py --persons data/person/prepared \\
        --garments data/samples
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_grid import build_grid  # noqa: E402
from real_garment_check import cloth_type_of, find_garments  # noqa: E402
from tryon_core import try_on, load_models, DEFAULT_STEPS, DEFAULT_SCHEDULER  # noqa: E402

GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

OUT_DIR = os.path.join(OUT_ROOT, 'real_person')
CSV_PATH = os.path.join(OUT_DIR, 'results.csv')
FIELDS = ['person', 'garment', 'cloth_type', 'steps', 'seed', 'scheduler',
          'mask_s', 'diffusion_s', 'total_s', 'out_path', 'mask_path',
          'platform', 'gpu']


def find_images(directory):
    found = []
    for ext in ('jpg', 'jpeg', 'png', 'webp'):
        found += glob.glob(os.path.join(directory, f'*.{ext}'))
    return sorted(found)


def append_row(row):
    is_new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--persons', required=True, help='인물 사진 폴더')
    ap.add_argument('--garments', required=True, help='옷 사진 폴더')
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--seeds', default=None,
                    help='쉼표로 구분한 시드 목록 (예: 42,123,7). 생략하면 --seed 하나')
    args = ap.parse_args()

    persons = find_images(args.persons)
    garments = find_garments(args.garments)
    if not persons:
        sys.exit(f'인물 사진을 찾지 못했습니다: {args.persons}')
    if not garments:
        sys.exit(f'옷 사진을 찾지 못했습니다: {args.garments}')

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'인물 {len(persons)}명 × 옷 {len(garments)}벌 = '
          f'{len(persons) * len(garments)}건, {args.steps}스텝', flush=True)
    print(f'환경: {PLATFORM} / {GPU_NAME}\n', flush=True)

    load_models()
    print('모델 로딩 완료\n', flush=True)

    # 결과가 시드에 따라 크게 흔들린다(같은 설정에서 프린트가 나왔다 말았다 했다).
    # 한 장만 보고 판단하면 운을 품질로 착각하므로 시드 여러 개를 함께 돌린다.
    seeds = [int(s) for s in str(args.seeds or args.seed).split(',') if s.strip()]

    for person in persons:
        person_name = os.path.splitext(os.path.basename(person))[0]
        print(f'--- {person_name} ---', flush=True)

        results, mask_results = [], []
        for garment in garments:
            garment_name = os.path.splitext(os.path.basename(garment))[0]
            cloth_type = cloth_type_of(garment)

            for seed in seeds:
                result, mask_vis, timing = try_on(
                    person=person, garment=garment, cloth_type=cloth_type,
                    steps=args.steps, seed=seed, return_timing=True)

                stem = f'{person_name}__{garment_name}_s{seed}'
                out_path = os.path.join(OUT_DIR, f'{stem}.png')
                mask_path = os.path.join(OUT_DIR, f'{stem}_mask.png')
                result.save(out_path)
                mask_vis.save(mask_path)

                label = f'{garment_name} s{seed}'
                results.append((label, out_path, timing['total_s'], None))
                if seed == seeds[0]:
                    # 마스크는 시드와 무관하므로 첫 시드 것만 붙인다
                    mask_results.append((garment_name, mask_path, None, None))
                print(f'  {garment_name} ({cloth_type}) seed {seed}: '
                      f'{timing["total_s"]:.1f}초', flush=True)

                append_row({
                    'person': person_name, 'garment': garment_name,
                    'cloth_type': cloth_type, 'steps': args.steps, 'seed': seed,
                    'scheduler': DEFAULT_SCHEDULER, 'mask_s': timing['mask_s'],
                    'diffusion_s': timing['diffusion_s'], 'total_s': timing['total_s'],
                    'out_path': out_path, 'mask_path': mask_path,
                    'platform': PLATFORM, 'gpu': GPU_NAME,
                })

        caption = f'{person_name} · {args.steps}스텝 · 시드 {seeds} · {PLATFORM} {GPU_NAME}'
        grid = build_grid(person, results,
                          os.path.join(OUT_DIR, f'compare_{person_name}.png'),
                          caption=caption)
        # 마스크는 따로 붙인다. 결과가 이상할 때 원인이 마스크인지 봐야 한다.
        mask_grid = build_grid(person, mask_results,
                               os.path.join(OUT_DIR, f'mask_{person_name}.png'),
                               caption=f'{caption} — 자동 생성 마스크')
        print(f'  결과 -> {grid}\n  마스크 -> {mask_grid}\n', flush=True)

    print('=== 결과 ===', flush=True)
    for f in sorted(os.listdir(OUT_DIR)):
        p = os.path.join(OUT_DIR, f)
        print(f'{os.path.getsize(p) / 1024:8.0f} KB  {p}', flush=True)


if __name__ == '__main__':
    main()
