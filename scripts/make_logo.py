"""FITCHECK 로고를 PNG로 생성한다.

마크는 **밑변에 줄자 눈금이 새겨진 옷걸이**다. 옷(가상 피팅)과 치수(사이즈 추천)를
한 그림에 담는다. 워드마크는 Gill Sans 대문자에 자간을 넓게 주고, 끝에 굵은 초록 체크를 붙인다.

마크는 가는 선이라 작게 쓰면 눈금이 사라진다. 그래서 푸터처럼 작은 자리에는
워드마크 단독(fitcheck_wordmark_*)을 쓰고, 마크가 들어간 조합은 표지·마지막 장처럼
크게 놓이는 자리에만 쓴다.

PIL의 선·원호는 안티에일리어싱이 없어서 **4배로 그린 뒤 축소**한다.

    python scripts/make_logo.py

산출물은 docs/brand/ 아래. 발표 자료(scripts/build_proposal_deck.py)가 가져다 쓴다.
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'docs', 'brand')

WORDMARK_FONT = r'C:\Windows\Fonts\GIL_____.TTF'  # Gill Sans MT
SS = 4  # 슈퍼샘플링 배율

WHITE = (255, 255, 255, 255)
BLACK = (17, 17, 17, 255)
DIM = (90, 90, 94, 255)  # 검정 배경 슬라이드 푸터용 (그림 투명도는 PPT에서 다루기 번거롭다)
CLEAR = (0, 0, 0, 0)
# 체크 초록은 배경마다 톤을 달리한다: 검정 위엔 밝게, 흰 위엔 대비를 위해 어둡게, 흐린 푸터엔 채도를 낮춰
GREEN_ON_DARK = (48, 209, 88, 255)
GREEN_ON_LIGHT = (36, 138, 61, 255)
GREEN_DIM = (40, 104, 60, 255)

MARK_ASPECT = 0.72     # 마크 높이 / 너비
STROKE = 0.026         # 너비 대비 선 두께
BAR_Y = 0.90           # 밑변 높이 (마크 높이 대비)
NECK = (0.5, 0.34)
HOOK_C = (0.5, 0.175)
HOOK_R = 0.09          # 마크 높이 대비
HOOK_START = 205       # 고리 끝 각도 (0=3시, 시계방향). 작을수록 고리가 닫혀 반지처럼 보인다
TICKS = 11             # 5칸마다 긴 눈금
TICK_SPAN = (0.25, 0.75)
WORD_TRACKING = 0.38   # em
CHECK_WIDTH = 1.15     # 체크 폭 / 대문자 높이
CHECK_GAP = 0.62       # K와 체크 사이 / 대문자 높이. 글자 사이 자간(0.38em ≈ 0.56)과 맞춘다
CHECK_STROKE = 0.20    # 체크 선 두께 / 대문자 높이 (Gill Sans 획의 약 두 배 — 포인트로 읽히게)
CHECK_PTS = ((0.0, 0.52), (0.33, 1.0), (1.0, 0.0))


def _dot(d, x, y, w, color):
    d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=color)


def render_mark(width, color):
    mw, mh = width * SS, int(width * MARK_ASPECT) * SS
    img = Image.new('RGBA', (mw, mh), CLEAR)
    d = ImageDraw.Draw(img)
    s = STROKE * mw
    P = lambda x, y: (x * mw, y * mh)

    # 고리: 목에서 올라가 위를 넘어 왼쪽으로 내려오다 열린 채 끝나는 모양
    cx, cy = P(*HOOK_C)
    r = HOOK_R * mh
    d.arc((cx - r, cy - r, cx + r, cy + r), HOOK_START, 450, fill=color, width=int(round(s)))
    # PIL 원호는 두께를 경계 상자 안쪽으로 채우므로, 끝단 원은 선의 중심 반지름에 둔다
    rc = r - s / 2
    a = math.radians(HOOK_START)
    _dot(d, cx + rc * math.cos(a), cy + rc * math.sin(a), s, color)
    d.line([(cx, cy + rc), P(*NECK)], fill=color, width=int(round(s)))
    _dot(d, cx, cy + rc, s, color)  # 원호 끝과 직선이 만나는 곳의 턱을 메운다

    left, right = P(0.04, BAR_Y), P(0.96, BAR_Y)
    neck = P(*NECK)
    d.line([left, neck, right, left], fill=color, width=int(round(s)), joint='curve')
    for pt in (left, right, neck):  # 다각형 시작점은 joint가 안 먹어서 모서리를 따로 둥글린다
        _dot(d, *pt, s, color)

    # 줄자 눈금
    x0, x1 = TICK_SPAN
    for i in range(TICKS):
        x = x0 + (x1 - x0) * i / (TICKS - 1)
        length = 0.095 if i % 5 == 0 else 0.05
        d.line([P(x, BAR_Y), P(x, BAR_Y - length)], fill=color, width=int(round(s * 0.6)))

    return img.resize((width, int(width * MARK_ASPECT)), Image.LANCZOS)


def render_wordmark(cap_px, color, check_color):
    """FITCHECK 글자 뒤에 초록 체크를 붙인 워드마크. 체크 높이는 대문자 높이에 맞춘다."""
    size = int(cap_px * SS / 0.68)  # Gill Sans 대문자 높이는 em의 약 0.68
    font = ImageFont.truetype(WORDMARK_FONT, size)
    tracking = size * WORD_TRACKING
    text = 'FITCHECK'
    width = int(sum(font.getlength(ch) + tracking for ch in text) + size)

    img = Image.new('RGBA', (width, int(size * 1.6)), CLEAR)
    d = ImageDraw.Draw(img)
    x = size * 0.25
    for ch in text:
        d.text((x, size * 0.25), ch, font=font, fill=color)
        x += font.getlength(ch) + tracking
    word = img.crop(img.getbbox())

    cap = word.height  # FITCHECK엔 내림획이 없어 잘라낸 높이가 곧 대문자 높이
    check_w = int(cap * CHECK_WIDTH)
    gap = int(cap * CHECK_GAP)
    out = Image.new('RGBA', (word.width + gap + check_w, cap), CLEAR)
    out.alpha_composite(word, (0, 0))
    d = ImageDraw.Draw(out)
    s = cap * CHECK_STROKE
    ox = word.width + gap
    # 선 두께만큼 안쪽으로 들여 그려야 둥근 끝이 캔버스 밖으로 잘리지 않는다
    pts = [(ox + s / 2 + (check_w - s) * px, s / 2 + (cap - s) * py) for px, py in CHECK_PTS]
    d.line(pts, fill=check_color, width=int(round(s)), joint='curve')
    for px, py in pts:
        _dot(d, px, py, s, check_color)

    return out.resize((max(1, out.width // SS), max(1, out.height // SS)), Image.LANCZOS)


def _paste(canvas, img, left, top):
    canvas.alpha_composite(img, (int(round(left)), int(round(top))))


def build_set(color, check_color, suffix):
    mark_w = 640
    mark = render_mark(mark_w, color)
    mark.save(os.path.join(OUT_DIR, f'fitcheck_mark_{suffix}.png'))

    # 세로형: 마크 아래 워드마크. 워드마크 폭을 마크 밑변 폭에 가깝게 맞춘다
    word = render_wordmark(int(mark_w * 0.085), color, check_color)
    gap = int(mark_w * 0.13)
    width = max(mark.width, word.width)
    canvas = Image.new('RGBA', (width, mark.height + gap + word.height), CLEAR)
    _paste(canvas, mark, (width - mark.width) / 2, 0)
    _paste(canvas, word, (width - word.width) / 2, mark.height + gap)
    canvas.save(os.path.join(OUT_DIR, f'fitcheck_lockup_v_{suffix}.png'))

    # 가로형: 마크 오른쪽에 워드마크
    word_h = render_wordmark(int(mark.height * 0.2), color, check_color)
    gap_h = int(mark.height * 0.32)
    canvas = Image.new('RGBA', (mark.width + gap_h + word_h.width, mark.height), CLEAR)
    _paste(canvas, mark, 0, 0)
    _paste(canvas, word_h, mark.width + gap_h, mark.height * 0.62 - word_h.height / 2)
    canvas.crop(canvas.getbbox()).save(os.path.join(OUT_DIR, f'fitcheck_lockup_h_{suffix}.png'))

    render_wordmark(120, color, check_color).save(
        os.path.join(OUT_DIR, f'fitcheck_wordmark_{suffix}.png'))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for color, check_color, suffix in ((WHITE, GREEN_ON_DARK, 'white'),
                                       (BLACK, GREEN_ON_LIGHT, 'black'),
                                       (DIM, GREEN_DIM, 'dim')):
        build_set(color, check_color, suffix)
    print(f'로고 생성 완료 -> {OUT_DIR}')


if __name__ == '__main__':
    main()
