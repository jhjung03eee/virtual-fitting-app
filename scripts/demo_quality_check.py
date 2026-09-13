"""CatVTON 저장소 데모 데이터로 현재 설정의 품질 천장을 잰다.

우리 옷·사진으로는 합격률이 낮았다. 원인이 파이프라인인지 입력인지 가르려면
**모델이 원래 잘하는 입력**에서의 결과가 기준점으로 있어야 한다. 저장소 데모 데이터는
인물·옷 사진 모두 학습 데이터(VITON-HD/DressCode)와 같은 스튜디오 스타일이다.

초기 벤치마크(docs/TEST_RESULTS.md 품질 시나리오)와 **같은 케이스**를 쓰되,
그때와 달리 현재 기본 설정(DPM++ 8스텝, composite, guidance 2.5)으로 돌리고,
원본 옷을 옆에 붙여 판정 기준(색조·형태·얼룩·실루엣·보존)으로 볼 수 있게 한다.
의도적 실패 케이스(상의 이미지를 하의로 지정)는 뺀다.

    python scripts/demo_quality_check.py --seeds 42,123
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
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_grid import find_korean_font, to_english  # noqa: E402
from tryon_core import try_on, load_models, REPO_DIR, DEFAULT_STEPS  # noqa: E402

GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
OUT_DIR = os.path.join(OUT_ROOT, 'demo_quality')
DEMO = os.path.join(REPO_DIR, 'resource/demo/example')


def _demo(pattern):
    return sorted(glob.glob(os.path.join(DEMO, pattern)))


def build_cases():
    """초기 벤치마크 build_quality_suite 와 같은 조합 (실패 케이스 제외)."""
    persons = _demo('person/men/*') + _demo('person/women/*')
    uppers = _demo('condition/upper/*')
    overalls = _demo('condition/overall/*')
    if not persons or not uppers:
        sys.exit(f'데모 이미지를 못 찾음. DEMO={DEMO}')

    cases = []
    for p in persons[:4]:
        cases.append((p, uppers[0], 'upper', '인물 변화 / 프린트 카디건'))
    for g in uppers:
        cases.append((persons[0], g, 'upper', '상의 변화'))
    for g in overalls[:2]:
        cases.append((persons[0], g, 'lower', '하의'))
        cases.append((persons[0], g, 'overall', '원피스'))
    return [(f'q{i:02d}',) + c for i, c in enumerate(cases)]


def build_sheet(rows, seeds, out_path):
    """케이스마다 한 줄: 인물 | 원본 옷 | 시드별 결과."""
    cell_w, cell_h, pad, head = 180, 240, 8, 26
    font, has_korean = find_korean_font(14)
    fix = (lambda s: s) if has_korean else to_english
    cols = 2 + len(seeds)
    width = pad + cols * (cell_w + pad)
    height = head + len(rows) * (cell_h + head + pad)
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)

    for r, (case_id, person, garment, cloth_type, note, results) in enumerate(rows):
        y = head + r * (cell_h + head + pad)
        draw.text((pad, y - 2), fix(f'{case_id} {cloth_type} · {note}'), fill='black', font=font)
        images = [person, garment] + results
        labels = ['인물', '원본 옷'] + [f'seed {s}' for s in seeds]
        for c, (path, label) in enumerate(zip(images, labels)):
            thumb = Image.open(path).convert('RGB')
            thumb.thumbnail((cell_w, cell_h))
            x = pad + c * (cell_w + pad)
            canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 18))
            draw.text((x, y + 18 + cell_h - 16), fix(label), fill='#666', font=font)
    canvas.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,123')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    cases = build_cases()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'케이스 {len(cases)}건 × 시드 {len(seeds)}개, {DEFAULT_STEPS}스텝', flush=True)
    print(f'환경: {PLATFORM} / {GPU_NAME}\n', flush=True)
    load_models()

    csv_path = os.path.join(OUT_DIR, 'results.csv')
    rows = []
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'person', 'garment', 'cloth_type', 'seed', 'total_s', 'out_path'])
        for case_id, person, garment, cloth_type, note in cases:
            results = []
            for seed in seeds:
                result, _mask, timing = try_on(
                    person=person, garment=garment, cloth_type=cloth_type,
                    seed=seed, return_timing=True)
                out_path = os.path.join(OUT_DIR, f'{case_id}_{cloth_type}_s{seed}.png')
                result.save(out_path)
                results.append(out_path)
                writer.writerow([case_id, os.path.basename(person), os.path.basename(garment),
                                 cloth_type, seed, timing['total_s'], out_path])
                print(f'  {case_id} {cloth_type} seed {seed}: {timing["total_s"]:.1f}초', flush=True)
            rows.append((case_id, person, garment, cloth_type, note, results))

    half = (len(rows) + 1) // 2
    for i, part in enumerate((rows[:half], rows[half:])):
        sheet = build_sheet(part, seeds, os.path.join(OUT_DIR, f'sheet_{i + 1}.png'))
        print('비교표 ->', sheet, flush=True)


if __name__ == '__main__':
    main()
