"""설정별 결과를 한 장에 나란히 붙인 비교 그리드를 만든다.

**torch도 GPU도 쓰지 않는다.** 합성이 끝난 뒤 이미지 파일만 읽어 붙이므로,
Kaggle에서 받은 결과로 로컬에서 그리드를 다시 만들 수 있다
(라벨이 깨졌을 때 재실행 없이 고치려면 이 분리가 필요하다).

    python scripts/compare_grid.py data/samples/results

한글 폰트가 없는 환경(Kaggle/Colab 기본 이미지)에서는 라벨이 두부(□)로 깨진다.
그런 경우 조용히 깨진 그림을 내놓지 말고 **영어 라벨로 바꿔서** 읽을 수 있게 한다.
"""
import argparse
import csv
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

KO_FONTS = ('NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 'AppleGothic',
            'NanumSquare', 'Noto Sans CJK KR', 'Noto Sans KR')

# 한글 폰트가 없을 때 쓸 대체 표기. 라벨이 고정된 소수의 문구라 이 방식으로 충분하다.
EN_FALLBACK = (
    ('옷 사진', 'garment'),
    ('기준 ', 'reference '),
    ('현재 기본값', 'current default'),
    ('스텝', ' steps'),
    ('옷 SSIM', 'garment SSIM'),
    ('CFG켬', 'CFG on'),
    ('CFG끔', 'CFG off'),
    ('인물', 'person'),
    ('초', 's'),
)


def find_korean_font(size):
    """한글을 그릴 수 있는 폰트를 찾는다. 없으면 (None, False)."""
    try:
        from matplotlib import font_manager
    except ImportError:
        return ImageFont.load_default(), False

    for name in KO_FONTS:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        try:
            return ImageFont.truetype(path, size), True
        except Exception:
            continue

    # 한글은 못 그려도 영어는 제대로 그려야 한다
    for name in ('DejaVu Sans', 'Liberation Sans', 'Arial'):
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            return ImageFont.truetype(path, size), False
        except Exception:
            continue
    return ImageFont.load_default(), False


def to_english(text):
    """한글 폰트가 없을 때 라벨을 읽을 수 있는 영어로 바꾼다."""
    for korean, english in EN_FALLBACK:
        text = text.replace(korean, english)
    # 남은 한글이 있으면 두부가 되므로 지운다
    return re.sub(r'[가-힣]+', '', text).strip()


def build_grid(garment_path, results, out_path, caption='', ssim_label='옷 SSIM'):
    """옷 사진 + 설정별 결과를 한 장에 나란히 붙인다.

    results: [(라벨, 이미지경로, 초 또는 None, SSIM 또는 None), ...]
    ssim_label: 캡션에 적을 지표 이름. 옷 마스크 안쪽만 잰 값인지 전체 이미지
        값인지 반드시 구분해서 적어야 한다 — 전체 SSIM은 옷이 붕괴해도 거의
        움직이지 않아서, 같은 이름으로 적으면 읽는 사람을 오도한다.

    표로 숫자만 보면 '옷이 제대로 반영됐는지'는 알 수 없다. 눈으로 비교하는 것이
    목적이므로 그리드가 본체다.
    """
    cell_w, cell_h = 288, 384
    pad, header, caption_h = 12, 34, 40

    font_title, has_korean = find_korean_font(16)
    font_note, _ = find_korean_font(14)
    fix = (lambda s: s) if has_korean else to_english

    columns = [(fix('옷 사진'), garment_path, None, None)]
    columns += [(fix(label), path, seconds, ssim) for label, path, seconds, ssim in results]

    width = pad + len(columns) * (cell_w + pad)
    height = header + cell_h + caption_h + pad * 2
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)

    for index, (label, path, seconds, ssim) in enumerate(columns):
        x = pad + index * (cell_w + pad)
        thumb = Image.open(path).convert('RGB')
        thumb.thumbnail((cell_w, cell_h))
        canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, header + pad))
        draw.text((x, pad), label, fill='black', font=font_title)
        if seconds is not None:
            note = f'{seconds:.1f}' + fix('초')
            if ssim is not None:
                note += f'   {fix(ssim_label)} {ssim:.3f}'
            draw.text((x, header + pad + cell_h + 6), note, fill='#444', font=font_note)

    if caption:
        draw.text((pad, height - 22), fix(caption), fill='#888', font=font_note)
    canvas.save(out_path)
    return out_path


def rebuild_from_csv(directory, garment_dir=None):
    """results.csv와 결과 이미지로 그리드를 다시 만든다."""
    csv_path = os.path.join(directory, 'results.csv')
    if not os.path.exists(csv_path):
        sys.exit(f'results.csv 가 없습니다: {csv_path}')

    with open(csv_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    by_garment = {}
    for row in rows:
        by_garment.setdefault(row['garment'], []).append(row)

    made = []
    for garment, entries in sorted(by_garment.items()):
        garment_photo = None
        for folder in (garment_dir, os.path.dirname(directory.rstrip('/\\')), directory):
            if not folder:
                continue
            for ext in ('jpg', 'jpeg', 'png', 'webp'):
                candidate = os.path.join(folder, f'{garment}.{ext}')
                if os.path.exists(candidate):
                    garment_photo = candidate
                    break
            if garment_photo:
                break
        if not garment_photo:
            print(f'  건너뜀 — 옷 사진을 찾지 못했습니다: {garment}')
            continue

        results = []
        ssim_label = '옷 SSIM'
        for row in entries:
            # CSV의 out_path는 Kaggle 경로다. 파일명만 떼어 로컬에서 찾는다.
            local = os.path.join(directory, os.path.basename(row['out_path']))
            if not os.path.exists(local):
                print(f'  건너뜀 — 결과 이미지 없음: {local}')
                continue
            ssim = row.get('ssim_garment') or ''
            if not ssim:
                # 옷 영역 SSIM이 없는 옛 결과다. 전체 이미지 SSIM을 쓰되
                # 라벨로 그 사실을 밝힌다.
                ssim = row.get('ssim_vs_ref') or ''
                ssim_label = '전체 SSIM(옷 붕괴 감지 못함)'
            results.append((
                row['config'], local,
                float(row['total_s']) if row.get('total_s') else None,
                float(ssim) if ssim else None,
            ))

        if not results:
            continue
        out = build_grid(
            garment_photo, results, os.path.join(directory, f'compare_{garment}.png'),
            caption=f"{garment} · {entries[0].get('gpu', '')}",
            ssim_label=ssim_label,
        )
        made.append(out)
        print(f'  {out}')
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', help='결과 이미지와 results.csv 가 있는 폴더')
    ap.add_argument('--garment-dir', default=None,
                    help='옷 원본 사진이 있는 폴더 (생략 시 상위 폴더에서 찾음)')
    args = ap.parse_args()

    _font, has_korean = find_korean_font(16)
    print('한글 폰트:', '있음' if has_korean else '없음 → 영어 라벨로 대체')
    made = rebuild_from_csv(args.directory, args.garment_dir)
    print(f'{len(made)}장 생성')


if __name__ == '__main__':
    main()
