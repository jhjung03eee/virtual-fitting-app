"""벤치마크 결과를 발표용 리포트로 정리.

    python scripts/make_report.py

생성물:
    docs/TEST_RESULTS.md               시나리오 표 + 속도/품질 표
    outputs/report/quality_grid.png    품질 시나리오 컨택트시트
    outputs/report/speed_quality.png   지연시간 vs 품질 산점도
"""
import argparse
import csv
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'app'))
from paths import VENV_PY, OUT_ROOT, child_env, describe  # noqa: E402


def _reexec_in_venv():
    # 이미 적절한 파이썬으로 실행 중이면(예: 리포트 전용 커널) 재실행을 건너뛴다
    if os.environ.get('VFA_NO_REEXEC') == '1':
        return
    if os.path.abspath(sys.executable) == os.path.abspath(VENV_PY):
        return
    if not os.path.exists(VENV_PY):
        sys.exit(f'venv39가 없습니다({VENV_PY}). setup_env.py 를 먼저 실행하세요.')
    env = child_env()
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

_ap = argparse.ArgumentParser()
_ap.add_argument('--bench-dir', default=os.path.join(OUT_ROOT, 'bench'),
                 help='benchmark.py 결과 폴더 (results.csv가 있는 곳)')
_ap.add_argument('--out-dir', default=os.path.join(OUT_ROOT, 'report'))
_args = _ap.parse_args()

BENCH_DIR = _args.bench_dir
CSV_PATH = os.path.join(BENCH_DIR, 'results.csv')
REPORT_DIR = _args.out_dir
# 리포트는 결과물과 같은 폴더에 쓴다. Kaggle에서는 /kaggle/working 아래여야
# 커널 output으로 받아올 수 있고, 이미지 상대경로도 같은 폴더라 그대로 맞는다.
DOC_PATH = os.path.join(REPORT_DIR, 'TEST_RESULTS.md')

def _setup_font():
    """한글 폰트가 있으면 쓰고, 없으면 라벨을 영어로 돌린다.

    Kaggle/Colab 기본 이미지에는 한글 폰트가 없어서 그냥 두면 축·범례가
    두부(□)로 깨진다. 발표 자료로 못 쓰므로 반드시 확인해야 한다.
    """
    from matplotlib import font_manager
    wanted = ('NanumGothic', 'Nanum Gothic', 'Malgun Gothic', 'AppleGothic',
              'NanumBarunGothic', 'Noto Sans CJK KR', 'Noto Sans KR')
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in wanted:
        if name in available:
            matplotlib.rcParams['font.family'] = name
            matplotlib.rcParams['axes.unicode_minus'] = False
            return True
    matplotlib.rcParams['axes.unicode_minus'] = False
    return False


HAS_KO_FONT = _setup_font()
LABELS = {
    True: dict(x='한 장 생성 시간 (초)', y='베이스라인 대비 SSIM',
               title='속도 / 품질 트레이드오프  (점 위 숫자 = 추론 스텝)',
               cfg_on='CFG 켬', cfg_off='CFG 끔'),
    False: dict(x='Latency per image (s)', y='SSIM vs baseline',
                title='Speed / quality trade-off  (number = inference steps)',
                cfg_on='CFG on', cfg_off='CFG off'),
}[HAS_KO_FONT]

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
    """CSV의 out_path는 생성 당시 절대경로다. 다른 환경에서 리포트를 다시 만들 때는
    그 경로가 없으므로 bench-dir 기준으로 재구성한다."""
    q = r['out_path']
    if os.path.exists(q):
        return q
    cand = os.path.join(BENCH_DIR, r['suite'], f"{r['case_id']}.png")
    if os.path.exists(cand):
        return cand
    return q if os.path.isabs(q) else os.path.join(REPO_ROOT, q)


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

    # 기준 1: 전체 베이스라인 (DDIM 30, CFG on) - 현재 운영 설정 대비 얼마나 달라지는가
    # 기준 2: 같은 샘플러의 30스텝 - 스텝을 줄여서 생긴 열화만 분리해서 본다
    #   (샘플러가 다르면 같은 스텝이어도 그림이 달라지므로, 기준 1만 보면
    #    DPM++가 '항상 0.93쯤'으로 눌려 보여 스텝 감소의 영향을 읽을 수 없다)
    baselines, sampler_refs = {}, {}
    for r in sp:
        key = (r['person'], r['garment'])
        if r['steps'] == 30 and r['guidance'] == 2.5:
            sampler_refs[(key, r['scheduler'])] = abs_out(r)
            if r['scheduler'] == 'ddim':
                baselines[key] = abs_out(r)

    scored = []
    for r in sp:
        key = (r['person'], r['garment'])
        if key not in baselines:
            continue
        r = dict(r)
        r['ssim'] = ssim_vs(baselines[key], abs_out(r))
        ref = sampler_refs.get((key, r['scheduler']))
        r['ssim_self'] = ssim_vs(ref, abs_out(r)) if ref else None
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
            cfg_txt = LABELS['cfg_on'] if cfg_on else LABELS['cfg_off']
            ax.scatter(
                [r['total_s'] for r in sub], [r['ssim'] for r in sub],
                s=70, linewidths=2, zorder=3,
                label=f'{name} · {cfg_txt}', **style
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

    ax.set_xlabel(LABELS['x'], color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(LABELS['y'], color=TEXT_SECONDARY, fontsize=10)
    ax.set_title(LABELS['title'], color=TEXT_PRIMARY, fontsize=12, pad=12, loc='left')
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
        lines += ['| 스케줄러 | 스텝 | CFG | 시간(초) | 배속 | SSIM vs 베이스라인 | SSIM vs 같은 샘플러 30스텝 |',
                  '|---|---|---|---|---|---|---|']
        agg = {}
        for r in scored:
            agg.setdefault((r['scheduler'], r['steps'], r['guidance']), []).append(r)
        for k in sorted(agg, key=lambda k: (k[0], -k[1], -k[2])):
            g = agg[k]
            t = sum(x['total_s'] for x in g) / len(g)
            s = sum(x['ssim'] for x in g) / len(g)
            speedup = f'{base_t / t:.1f}x' if base_t else '-'
            name = 'DDIM' if k[0] == 'ddim' else 'DPM++'
            selfs = [x['ssim_self'] for x in g if x.get('ssim_self') is not None]
            self_txt = f'{sum(selfs) / len(selfs):.3f}' if selfs else '-'
            lines.append(f"| {name} | {k[1]} | {'on' if k[2] > 1.0 else 'off'} | "
                         f'{t:.1f} | {speedup} | {s:.3f} | {self_txt} |')
        lines += [
            '',
            'SSIM은 "같은 그림인가"를 재는 지표이지 "잘 만들었는가"가 아니다. 두 기준을 함께 본다.',
            '',
            '- **vs 베이스라인**: 현재 운영 설정(DDIM 30스텝, CFG on) 대비 결과가 얼마나 달라지는가.',
            '  샘플러를 바꾸면 같은 스텝이어도 그림이 달라지므로 DPM++는 이 값이 구조적으로 낮게 나온다.',
            '- **vs 같은 샘플러 30스텝**: 스텝을 줄여서 생긴 열화만 분리한 값. 가속의 대가를 보려면 이쪽을 본다.',
        ]

    gpus = sorted({r.get('gpu') for r in rows if r.get('gpu')})
    plats = sorted({r.get('platform') for r in rows if r.get('platform')})
    hw = ', '.join(gpus) if gpus else '기록 없음'
    plat = ', '.join(plats) if plats else '기록 없음'
    lines += [
        '',
        '## 3. 측정 환경',
        '',
        f'- 실행 플랫폼: {plat} / GPU: {hw}',
        '- Python 3.9 venv, torch 2.1.2+cu121, fp16',
        '- 해상도 768×1024, AutoMasker(DensePose+SCHP) 자동 마스크',
        '- 시간은 마스크 생성 + 확산 추론 합계이며 모델 로딩(약 40초)은 제외',
        '- GPU에 따라 절대 시간은 달라진다 (같은 설정에서 T4 약 73초, P100 약 103초).',
        '  배속과 SSIM은 같은 GPU 안에서의 상대 비교이므로 영향받지 않는다.',
    ]

    os.makedirs(os.path.dirname(DOC_PATH), exist_ok=True)
    with open(DOC_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return DOC_PATH


def main():
    rows = load_rows()
    print(describe(), flush=True)
    print('한글 폰트:', '사용' if HAS_KO_FONT else '없음 -> 차트 라벨을 영어로', flush=True)
    print(f'{len(rows)}건 로드', flush=True)
    grid = make_quality_grid(rows)
    print('격자 이미지:', grid, flush=True)
    chart, scored = make_speed_chart(rows)
    print('산점도:', chart, flush=True)
    print('리포트:', write_doc(rows, grid, chart, scored), flush=True)


if __name__ == '__main__':
    main()
