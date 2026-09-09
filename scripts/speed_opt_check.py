"""스텝당 비용을 줄이는 최적화를 재고, 결과가 정말 같은지 검증한다.

스텝을 줄이는 가속은 8스텝에서 한계에 부딪혔다(6스텝이면 사진 프린트가 사라진다).
남은 길은 **샘플링 수학은 그대로 두고 스텝당 비용만 줄이는 것**이다.
이 방향은 원리상 품질 손실이 없어야 하는데, "없어야 한다"와 "없다"는 다르므로
기준 설정과의 옷 영역 SSIM으로 확인한다. 1.000에서 멀어지면 무손실이 아니다.

이미 적용돼 있어 손댈 수 없는 것 (CatVTON 원본 확인):
  - SDPA attention (AttnProcessor2_0), fp16, torch.no_grad, TF32

재는 것:
  - channels_last  : NHWC 메모리 배치
  - torch.compile  : 커널 융합. 첫 호출에서 컴파일하므로 워밍업을 빼고 잰다

실행:
    python scripts/speed_opt_check.py --garments data/samples
"""
import argparse
import csv
import gc
import os
import subprocess
import sys
import traceback

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
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tryon_core  # noqa: E402
from tryon_core import try_on, load_models, REPO_DIR, DEFAULT_STEPS  # noqa: E402
from real_garment_check import cloth_type_of, find_garments, garment_mask, ssim_against  # noqa: E402

GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

OUT_DIR = os.path.join(OUT_ROOT, 'speed_opt')
CSV_PATH = os.path.join(OUT_DIR, 'results.csv')
FIELDS = ['variant', 'garment', 'steps', 'mask_s', 'diffusion_s', 'total_s',
          'ssim_garment_vs_base', 'load_s', 'warmup_s', 'status',
          'platform', 'gpu']

# (표시명, compile_model, channels_last). 첫 항목이 기준이다.
VARIANTS = [
    ('기준 (현재 설정)', False, False),
    ('channels_last', False, True),
    ('torch.compile', True, False),
    ('compile + channels_last', True, True),
    # T4는 지속 부하에서 클럭이 떨어져 뒤에 잰 설정이 불리해진다. 처음과 같은
    # 설정을 마지막에 한 번 더 재서, 위 차이가 최적화 때문인지 드리프트 때문인지
    # 구별한다 (docs/ENVIRONMENT.md #13).
    ('기준 재측정 (드리프트 확인)', False, False),
]


def compile_probe():
    """torch.compile이 이 GPU에서 실제로 동작하는지 확인한다.

    Inductor 백엔드는 Triton으로 커널을 만드는데, Triton은 **compute capability
    7.0 이상**을 요구한다. P100은 6.0이라 조건에 못 미치는데, 이때 torch.compile은
    에러를 내는 대신 조용히 eager로 되돌아간다. 그러면 "컴파일했는데 안 빨라졌다"와
    "애초에 컴파일이 안 됐다"가 로그상 구별되지 않는다. 이 둘은 결론이 정반대다.
    """
    if not torch.cuda.is_available():
        return 'CUDA 없음'
    major, minor = torch.cuda.get_device_capability()
    line = f'compute capability {major}.{minor}'
    if (major, minor) < (7, 0):
        return f'{line} — Triton은 7.0 이상 필요. torch.compile이 eager로 되돌아간다'

    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        compiled = torch.compile(lambda x: (x * 2).relu(), backend='inductor')
        compiled(torch.randn(8, 8, device='cuda', dtype=torch.float16))
        return f'{line} — inductor 동작 확인'
    except Exception as e:
        return f'{line} — inductor 실패: {type(e).__name__}: {e}'


def free_models():
    """다음 설정을 로드하기 전에 GPU 메모리를 비운다."""
    tryon_core._models = None
    tryon_core._models_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def append_row(row):
    is_new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument('--garments', default=os.path.join(REPO_ROOT, 'data', 'samples'))
    ap.add_argument('--person', default=None)
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    garments = find_garments(args.garments)
    if not garments:
        sys.exit(f'옷 사진을 찾지 못했습니다: {args.garments}')
    person = args.person or sorted(
        glob.glob(os.path.join(REPO_DIR, 'resource/demo/example/person/men/*')))[0]

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'설정 {len(VARIANTS)}가지 × 옷 {len(garments)}벌, {args.steps}스텝', flush=True)
    print(f'환경: {PLATFORM} / {GPU_NAME}', flush=True)
    print(f'torch {torch.__version__} / {compile_probe()}\n', flush=True)

    masks, baseline_paths = {}, {}

    for variant, compile_model, channels_last in VARIANTS:
        print(f'=== {variant} ===', flush=True)
        free_models()

        started = time.perf_counter()
        try:
            load_models(compile_model=compile_model, channels_last=channels_last)
        except Exception:
            print(f'  로드 실패 — 건너뜁니다\n{traceback.format_exc()}', flush=True)
            append_row({'variant': variant, 'garment': '', 'steps': args.steps,
                        'status': 'load_failed', 'platform': PLATFORM, 'gpu': GPU_NAME})
            continue
        load_s = round(time.perf_counter() - started, 1)
        print(f'  로드 {load_s}초', flush=True)

        # torch.compile은 첫 호출에서 컴파일한다. 이걸 재면 측정이 무의미하다.
        started = time.perf_counter()
        try:
            try_on(person=person, garment=garments[0],
                   cloth_type=cloth_type_of(garments[0]),
                   steps=args.steps, seed=args.seed)
        except Exception:
            print(f'  워밍업 실패 — 건너뜁니다\n{traceback.format_exc()}', flush=True)
            append_row({'variant': variant, 'garment': '', 'steps': args.steps,
                        'load_s': load_s, 'status': 'warmup_failed',
                        'platform': PLATFORM, 'gpu': GPU_NAME})
            continue
        warmup_s = round(time.perf_counter() - started, 1)
        print(f'  워밍업 {warmup_s}초 (측정에서 제외)', flush=True)

        for garment in garments:
            name = os.path.splitext(os.path.basename(garment))[0]
            cloth_type = cloth_type_of(garment)
            if cloth_type not in masks:
                masks[cloth_type] = garment_mask(person, cloth_type)

            result, _mask_vis, timing = try_on(
                person=person, garment=garment, cloth_type=cloth_type,
                steps=args.steps, seed=args.seed, return_timing=True)

            slug = variant.split()[0].replace('(', '').replace('+', 'and')
            out_path = os.path.join(OUT_DIR, f'{name}_{slug}.png')
            result.save(out_path)

            if variant == VARIANTS[0][0]:
                baseline_paths[name] = out_path
                ssim = 1.0
            else:
                _full, ssim = ssim_against(baseline_paths[name], out_path,
                                           masks[cloth_type])

            print(f'  {name}: 확산 {timing["diffusion_s"]:.1f}초 '
                  f'(전체 {timing["total_s"]:.1f}초)  기준과 옷 SSIM {ssim}', flush=True)
            append_row({
                'variant': variant, 'garment': name, 'steps': args.steps,
                'mask_s': timing['mask_s'], 'diffusion_s': timing['diffusion_s'],
                'total_s': timing['total_s'], 'ssim_garment_vs_base': ssim,
                'load_s': load_s, 'warmup_s': warmup_s, 'status': 'ok',
                'platform': PLATFORM, 'gpu': GPU_NAME,
            })
        print('', flush=True)

    print('=== 요약 ===', flush=True)
    with open(CSV_PATH, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['status'] == 'ok']
    base_mean = None
    for variant, _c, _cl in VARIANTS:
        mine = [float(r['diffusion_s']) for r in rows if r['variant'] == variant]
        if not mine:
            print(f'{variant:24} — 실패', flush=True)
            continue
        mean = sum(mine) / len(mine)
        if base_mean is None:
            base_mean = mean
        ssims = [float(r['ssim_garment_vs_base']) for r in rows if r['variant'] == variant]
        print(f'{variant:26} 확산 {mean:6.1f}초  '
              f'{base_mean / mean:5.2f}배  기준과 옷 SSIM {min(ssims):.4f}', flush=True)

    first = [float(r['diffusion_s']) for r in rows if r['variant'] == VARIANTS[0][0]]
    last = [float(r['diffusion_s']) for r in rows if r['variant'] == VARIANTS[-1][0]]
    if first and last:
        drift = (sum(last) / len(last)) - (sum(first) / len(first))
        print('', flush=True)
        print(f'같은 설정의 처음/마지막 차이(드리프트): {drift:+.1f}초', flush=True)
        print('이 값보다 작은 차이는 최적화 효과로 볼 수 없다.', flush=True)


if __name__ == '__main__':
    main()
