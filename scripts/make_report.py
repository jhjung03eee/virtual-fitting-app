"""벤치마크 결과를 발표용 리포트로 정리.

    !cd /content/vfa && python scripts/make_report.py

생성물:
    docs/TEST_RESULTS.md               시나리오 표 + 속도/품질 표
    outputs/report/quality_grid.png    품질 시나리오 컨택트시트
    outputs/report/speed_quality.png   지연시간 vs 품질 산점도
"""
import csv
import os
import subprocess
import sys

VENV_PY = '/content/venv39/bin/python'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _reexec_in_venv():
    if os.path.abspath(sys.executable) == os.path.abspath(VENV_PY):
        return
    if not os.path.exists(VENV_PY):
        sys.exit('venv39가 없습니다. setup_py39_and_run.py 를 먼저 실행하세요.')
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    raise SystemExit(
        subprocess.call([VENV_PY, '-u', os.path.abspath(__file__)] + sys.argv[1:], env=env)
    )


_reexec_in_venv()

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from skimage.metrics import structural_similarity as ssim_fn  # noqa: E402

BENCH_DIR = os.path.join(REPO_ROOT, 'outputs', 'bench')
CSV_PATH = os.path.join(BENCH_DIR, 'results.csv')
REPORT_DIR = os.path.join(REPO_ROOT, 'outputs', 'report')
DOC_PATH = os.path.join(REPO_ROOT, 'docs', 'TEST_RESULTS.md')

# dataviz 스킬의 검증 통과 팔레트 (light 모드, 카테고리 슬롯 1·2)
SERIES = {'ddim': '#2a78d6', 'dpm': '#eb6834'}
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
SURFACE = '#fcfcfb'
GRID = '#d9d8d4'


def load_rows():
    if not os.path.exists(CSV_PATH):
        sys.exit(f'{CSV_PATH} 가 없습니다. benchmark.py 를 먼저 실행하세요.')
    with open(CSV_PATH, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['steps'] = int(r['steps'])
        r['guidance'] = float(r['guidance'])
        for k in ('mask_s', 'diffusion_s', 'total_s'):
            r[k] = float(r[k])
    dedup = {}
    for r in rows:  # 같은 case_id가 여러 번 있으면 마지막 것만
        dedup[(r['suite'], r['case_id'])] = r
    return list(dedup.values())


def abs_out(r):
    return os.path.join(REPO_ROOT, r['out_path'])


def ssim_vs(baseline_path, path):
    a = np.asarray(Image.open(baseline_path).convert('L'), dtype=np.float64)
    b = np.asarray(Image.open(path).convert('L'), dtype=np.float64)
    return float(ssim_fn(a, b, data_range=255))


def make_quality_grid(rows):
    """품질 시나리오 결과를 격자 이미지로 합성."""
    qs = [r for r in rows if r['suite'] == 'quality' and os.path.exists(abs_out(r))]
    if not qs:
        return None
    qs.sort(key=lambda r: r['case_id'])

    cols = 4
    cell_w, cell_h, label_h, pad = 220, 293, 34, 8
    rows_n = (len(qs) + cols - 1) // cols
    width = cols * (cell_w + pad) + pad
    height = rows_n * (cell_h + label_h + pad) + pad

    sheet = Image.new('RGB', (width, height), SURFACE)
    draw = ImageDraw.Draw(sheet)

    for i, r in enumerate(qs):
        cx, cy = i % cols, i // cols
        x = pad + cx * (cell_w + pad)
        y = pad + cy * (cell_h + label_h + pad)
        img = Image.open(abs_out(r)).convert('RGB').resize((cell_w, cell_h), Image.LANCZOS)
        sheet.paste(img, (x, y))
        draw.text((x, y + cell_h + 4), f"{r['case_id']}  {r['cloth_type']}", fill=TEXT_PRIMARY)
        draw.text((x, y + cell_h + 18), r.get('note', '')[:38], fill=TEXT_SECONDARY)

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, 'quality_grid.png')
    sheet.save(out)
    return out


def make_speed_chart(rows):
    """지연시간 vs 품질(SSIM) 산점도. 스케줄러로 색, CFG 여부는 채움으로 구분."""
    sp = [r for r in rows if r['suite'] == 'speed' and os.path.exists(abs_out(r))]
    if not sp:
        return None, []

    baselines = {}
    for r in sp:
        if r['scheduler'] == 'ddim' and r['steps'] == 30 and r['guidance'] == 2.5:
            baselines[(r['person'], r['garment'])] = abs_out(r)

    scored = []
    for r in sp:
        key = (r['person'], r['garment'])
        if key not in baselines:
            continue
        r = dict(r)
        r['ssim'] = ssim_vs(baselines[key], abs_out(r))
        scored.append(r)
    if not scored:
        return None, []

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for sched, color in SERIES.items():
        pts = [r for r in scored if r['scheduler'] == sched]
        if not pts:
            continue
        for cfg_on, style in (
            (True, dict(facecolors=color, edgecolors=color)),
            (False, dict(facecolors='none', edgecolors=color)),
        ):
            sub = [r for r in pts if (r['guidance'] > 1.0) == cfg_on]
            if not sub:
                continue
            name = 'DDIM' if sched == 'ddim' else 'DPM++'
            ax.scatter(
                [r['total_s'] for r in sub], [r['ssim'] for r in sub],
                s=70, linewidths=2, zorder=3,
                label=f"{name} · CFG {'on' if cfg_on else 'off'}", **style
            )

    seen = set()
    for r in scored:  # 점 위 숫자는 값이 아니라 설정(스텝 수)이라 정보량이 있다
        k = (round(r['total_s']), round(r['ssim'], 3))
        if k in seen:
            continue
        seen.add(k)
        ax.annotate(
            str(r['steps']), (r['total_s'], r['ssim']),
            textcoords='offset points', xytext=(0, 9),
            ha='center', fontsize=8, color=TEXT_SECONDARY,
        )

    ax.set_xlabel('한 장 생성 시간 (초)', color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel('베이스라인 대비 SSIM', color=TEXT_SECONDARY, fontsize=10)
    ax.set_title('속도 / 품질 트레이드오프  (점 위 숫자 = 추론 스텝)',
                 color=TEXT_PRIMARY, fontsize=12, pad=12, loc='left')
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    leg = ax.legend(frameon=False, fontsize=9, loc='lower right')
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, 'speed_quality.png')
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out, scored


def _rel(path):
    return os.path.relpath(path, os.path.dirname(DOC_PATH)).replace(os.sep, '/')


def write_doc(rows, grid_path, chart_path, scored):
    qs = sorted([r for r in rows if r['suite'] == 'quality'], key=lambda r: r['case_id'])
    lines = [
        '# 테스트 결과',
        '',
        'CatVTON 기반 가상 피팅의 품질 한계와 속도/품질 트레이드오프를 측정한 결과.',
        '',
        '실행: `python scripts/benchmark.py --suite all` → `python scripts/make_report.py`',
        '',
        '테스트 이미지는 재현성을 위해 CatVTON 저장소의 데모 이미지를 사용했다.',
        '',
        '## 1. 품질 시나리오',
        '',
    ]
    if grid_path:
        lines += [f'![품질 시나리오]({_rel(grid_path)})', '']
    lines += ['| ID | 옷 종류 | 시나리오 | 시간(초) |', '|---|---|---|---|']
    for r in qs:
        lines.append(f"| {r['case_id']} | {r['cloth_type']} | {r.get('note', '')} | {r['total_s']:.1f} |")

    lines += ['', '## 2. 속도 / 품질 트레이드오프', '']
    if chart_path:
        lines += [f'![속도 품질]({_rel(chart_path)})', '']
    if scored:
        base = [r for r in scored
                if r['scheduler'] == 'ddim' and r['steps'] == 30 and r['guidance'] == 2.5]
        base_t = base[0]['total_s'] if base else None
        lines += ['| 스케줄러 | 스텝 | CFG | 시간(초) | 배속 | SSIM |', '|---|---|---|---|---|---|']
        agg = {}
        for r in scored:
            agg.setdefault((r['scheduler'], r['steps'], r['guidance']), []).append(r)
        for k in sorted(agg, key=lambda k: (k[0], -k[1], -k[2])):
            g = agg[k]
            t = sum(x['total_s'] for x in g) / len(g)
            s = sum(x['ssim'] for x in g) / len(g)
            speedup = f'{base_t / t:.1f}x' if base_t else '-'
            name = 'DDIM' if k[0] == 'ddim' else 'DPM++'
            lines.append(f"| {name} | {k[1]} | {'on' if k[2] > 1.0 else 'off'} | "
                         f'{t:.1f} | {speedup} | {s:.3f} |')
        lines += [
            '',
            'SSIM은 베이스라인(DDIM 30스텝, CFG on) 결과와의 구조적 유사도다. 1.0에 가까울수록',
            '베이스라인과 같은 그림이라는 뜻이며, 베이스라인 자체의 품질이 좋다는 의미는 아니다.',
        ]

    lines += [
        '',
        '## 3. 측정 환경',
        '',
        '- Colab T4 (16GB), Python 3.9 venv, torch 2.1.2+cu121, fp16',
        '- 해상도 768×1024, AutoMasker(DensePose+SCHP) 자동 마스크',
        '- 시간은 마스크 생성 + 확산 추론 합계이며 모델 로딩(약 40초)은 제외',
    ]

    os.makedirs(os.path.dirname(DOC_PATH), exist_ok=True)
    with open(DOC_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return DOC_PATH


def main():
    rows = load_rows()
    print(f'{len(rows)}건 로드', flush=True)
    grid = make_quality_grid(rows)
    print('격자 이미지:', grid, flush=True)
    chart, scored = make_speed_chart(rows)
    print('산점도:', chart, flush=True)
    print('리포트:', write_doc(rows, grid, chart, scored), flush=True)


if __name__ == '__main__':
    main()
