"""테스트 시나리오 자동 실행 — 발표용 결과(품질/속도) 수집.

Colab에서:
    !cd /content/vfa && python scripts/benchmark.py --suite all

venv39가 아닌 인터프리터로 실행되면 자동으로 venv39로 자기 자신을 재실행한다.
중단됐다 다시 돌리면 이미 끝난 건은 건너뛴다(재개 가능).
"""
import argparse
import csv
import os
import subprocess
import sys

VENV_PY = '/content/venv39/bin/python'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _reexec_in_venv():
    """venv39 인터프리터가 아니면 그걸로 다시 실행한다."""
    if os.path.abspath(sys.executable) == os.path.abspath(VENV_PY):
        return
    if not os.path.exists(VENV_PY):
        sys.exit(
            'Python 3.9 환경(/content/venv39)이 없습니다.\n'
            '    !python /content/vfa/scripts/setup_py39_and_run.py\n'
            '를 먼저 실행하세요.'
        )
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    raise SystemExit(subprocess.call([VENV_PY, '-u', os.path.abspath(__file__)] + sys.argv[1:], env=env))


_reexec_in_venv()

sys.path.insert(0, os.path.join(REPO_ROOT, 'app'))
import glob  # noqa: E402
from tryon_core import try_on, load_models, REPO_DIR  # noqa: E402

OUT_ROOT = os.path.join(REPO_ROOT, 'outputs', 'bench')
CSV_PATH = os.path.join(OUT_ROOT, 'results.csv')
FIELDS = ['suite', 'case_id', 'person', 'garment', 'cloth_type', 'scheduler',
          'steps', 'guidance', 'seed', 'mask_s', 'diffusion_s', 'total_s', 'out_path']

DEMO = os.path.join(REPO_DIR, 'resource/demo/example')


def _demo(pattern):
    return sorted(glob.glob(os.path.join(DEMO, pattern)))


def build_quality_suite():
    """어떤 입력에서 품질이 무너지는지 보는 시나리오. 설정은 고정(DDIM 30, g=2.5)."""
    persons = _demo('person/men/*') + _demo('person/women/*')
    uppers = _demo('condition/upper/*')
    overalls = _demo('condition/overall/*')
    if not persons or not uppers:
        sys.exit(f'데모 이미지를 못 찾음. DEMO={DEMO}')

    cases = []

    # 1) 인물 바꿔가며 같은 옷 — 인물 의존성
    for p in persons[:4]:
        cases.append(dict(person=p, garment=uppers[0], cloth_type='upper',
                          note='인물 변화 / 프린트 있는 카디건'))

    # 2) 옷 바꿔가며 같은 인물 — 옷 종류/프린트 의존성
    for g in uppers:
        cases.append(dict(person=persons[0], garment=g, cloth_type='upper',
                          note='상의 변화'))

    # 3) 하의 / 원피스 카테고리
    for g in overalls[:2]:
        cases.append(dict(person=persons[0], garment=g, cloth_type='lower',
                          note='하의'))
        cases.append(dict(person=persons[0], garment=g, cloth_type='overall',
                          note='원피스(상하 일체)'))

    # 4) 의도적 실패 케이스 — 상의 이미지를 하의로 지정
    cases.append(dict(person=persons[0], garment=uppers[0], cloth_type='lower',
                      note='FAIL CASE: 상의 이미지를 하의로 지정'))

    out = []
    for i, c in enumerate(cases):
        out.append(dict(suite='quality', case_id=f'q{i:02d}', scheduler='ddim',
                        steps=30, guidance=2.5, seed=42, **c))
    return out


def build_speed_suite():
    """속도/품질 트레이드오프. 인물x옷 2조합 고정, 설정을 바꿔가며 측정."""
    persons = _demo('person/men/*')
    uppers = _demo('condition/upper/*')
    combos = [(persons[0], uppers[0])]
    if len(persons) > 1 and len(uppers) > 1:
        combos.append((persons[1], uppers[1]))

    settings = [('ddim', 30, 2.5)]  # 베이스라인 (현재 70초)
    for sched in ('ddim', 'dpm'):
        for steps in (4, 8, 15, 30):
            for guidance in (2.5, 1.0):
                if (sched, steps, guidance) == ('ddim', 30, 2.5):
                    continue
                settings.append((sched, steps, guidance))

    out = []
    i = 0
    for person, garment in combos:
        for sched, steps, guidance in settings:
            out.append(dict(suite='speed', case_id=f's{i:02d}', person=person, garment=garment,
                            cloth_type='upper', scheduler=sched, steps=steps,
                            guidance=guidance, seed=42,
                            note=f'{sched} {steps}steps g={guidance}'))
            i += 1
    return out


def already_done(case):
    return os.path.exists(case_out_path(case))


def case_out_path(case):
    return os.path.join(OUT_ROOT, case['suite'], f"{case['case_id']}.png")


def append_row(row):
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ['note'])
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', choices=['quality', 'speed', 'all'], default='all')
    ap.add_argument('--limit', type=int, default=0, help='앞에서 N건만 실행 (0이면 전체)')
    a = ap.parse_args()

    cases = []
    if a.suite in ('quality', 'all'):
        cases += build_quality_suite()
    if a.suite in ('speed', 'all'):
        cases += build_speed_suite()
    if a.limit:
        cases = cases[:a.limit]

    todo = [c for c in cases if not already_done(c)]
    print(f'총 {len(cases)}건 중 {len(todo)}건 실행 (나머지는 이미 완료)', flush=True)
    if not todo:
        return

    os.makedirs(OUT_ROOT, exist_ok=True)
    load_models()
    print('모델 로딩 완료', flush=True)

    for n, c in enumerate(todo, 1):
        out_path = case_out_path(c)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(f'[{n}/{len(todo)}] {c["case_id"]} {c["note"]}', flush=True)

        result, mask_vis, timing = try_on(
            person=c['person'], garment=c['garment'], cloth_type=c['cloth_type'],
            steps=c['steps'], guidance_scale=c['guidance'], seed=c['seed'],
            scheduler=c['scheduler'], return_timing=True,
        )
        result.save(out_path)
        if c['suite'] == 'quality':
            mask_vis.save(out_path.replace('.png', '_mask.png'))

        row = {k: c.get(k) for k in ('suite', 'case_id', 'person', 'garment', 'cloth_type',
                                     'scheduler', 'steps', 'guidance', 'seed', 'note')}
        row.update(timing)
        row['out_path'] = os.path.relpath(out_path, REPO_ROOT)
        append_row(row)
        print(f'    {timing["total_s"]}s (mask {timing["mask_s"]}s / diffusion {timing["diffusion_s"]}s)',
              flush=True)

    print(f'\n완료. CSV: {CSV_PATH}', flush=True)


if __name__ == '__main__':
    main()
