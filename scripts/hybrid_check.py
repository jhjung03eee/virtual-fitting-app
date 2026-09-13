"""하이브리드(변형 초안 + CatVTON) vs CatVTON 단독 비교.

HR-VITON 커널이 만든 입력(image/, cloth/)과 변형 결과(output_warp/, output_region/)를 읽는다.
인물 사진은 HR-VITON 입력과 **같은 768x1024 이미지**를 써야 변형된 옷 위치가 맞는다.

    python scripts/hybrid_check.py --data DATAROOT/test --warp OUTPUT_DIR --out DIR \\
        --persons kakao_front_upper,demo_person0_full --garments sweatshirt_text,demo_cardigan \\
        --strengths 0.5,0.7,0.9 --seeds 42,123 --steps 30
"""
import argparse
import csv
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

from PIL import Image, ImageDraw  # noqa: E402

from tryon_core import load_models, try_on  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--warp', required=True, help='HR-VITON output_dir (옆에 _warp, _region 폴더가 있음)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--persons', required=True)
    ap.add_argument('--garments', required=True)
    ap.add_argument('--strengths', default='0.5,0.7,0.9')
    ap.add_argument('--seeds', default='42,123')
    ap.add_argument('--steps', type=int, default=30)
    args = ap.parse_args()

    persons = args.persons.split(',')
    garments = args.garments.split(',')
    strengths = [float(s) for s in args.strengths.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    os.makedirs(args.out, exist_ok=True)
    load_models()

    rows = []
    with open(os.path.join(args.out, 'results.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['person', 'garment', 'seed', 'config', 'total_s', 'path'])
        for person in persons:
            person_img = Image.open(os.path.join(args.data, 'image', f'{person}.jpg')).convert('RGB')
            for garment in garments:
                cloth = Image.open(os.path.join(args.data, 'cloth', f'{garment}.jpg')).convert('RGB')
                name = f'{person}_{garment}.png'
                warped = Image.open(os.path.join(args.warp + '_warp', name)).convert('RGB')
                region = Image.open(os.path.join(args.warp + '_region', name)).convert('L')
                hrviton = Image.open(os.path.join(args.warp, name)).convert('RGB')
                # 초안(CatVTON 마스크로 자르기 전) — 결과가 이상하면 초안 탓인지 봐야 한다
                from hybrid_tryon import make_draft
                draft = make_draft(person_img, Image.new('L', person_img.size, 255), warped, region)
                draft.save(os.path.join(args.out, f'{person}__{garment}__draft.png'))
                for seed in seeds:
                    cells = [('원본 옷', cloth), ('HR-VITON', hrviton), ('변형 초안', draft)]
                    configs = [('catvton', None, 1.0)] + [(f'hybrid{s:g}', (warped, region), s) for s in strengths]
                    for config, warp, strength in configs:
                        result, _mask, timing = try_on(
                            person_img, cloth, cloth_type='upper', steps=args.steps, seed=seed,
                            scheduler='ddim', return_timing=True, warp=warp, strength=strength)
                        path = os.path.join(args.out, f'{person}__{garment}__{config}_s{seed}.png')
                        result.save(path)
                        writer.writerow([person, garment, seed, config, timing['total_s'], path])
                        label = 'CatVTON 단독' if warp is None else f'하이브리드 {strength:g}'
                        cells.append((f'{label} ({timing["total_s"]:.0f}초)', result))
                        print(f'  {person} {garment} s{seed} {config}: {timing["total_s"]:.1f}초', flush=True)
                    rows.append((f'{person} · {garment} · s{seed}', cells))

    cell_w, cell_h, head = 260, 360, 24
    cols = max(len(c) for _, c in rows)
    sheet = Image.new('RGB', (cols * (cell_w + 8) + 8, len(rows) * (cell_h + head * 2) + 8), 'white')
    draw = ImageDraw.Draw(sheet)
    for r, (title, cells) in enumerate(rows):
        y = 8 + r * (cell_h + head * 2)
        draw.text((8, y), title, fill='black')
        for c, (label, image) in enumerate(cells):
            thumb = image.copy()
            thumb.thumbnail((cell_w, cell_h))
            x = 8 + c * (cell_w + 8)
            sheet.paste(thumb, (x, y + head))
            draw.text((x, y + head + cell_h + 2), label, fill='#444')
    sheet.save(os.path.join(args.out, 'sheet.jpg'), quality=90)
    print('비교표 ->', os.path.join(args.out, 'sheet.jpg'), flush=True)


if __name__ == '__main__':
    main()
