"""직접 고친 발표 자료에서 구현 01·02·03 슬라이드의 화면과 '모델 후보' 슬라이드만 바꾼다.

build_proposal_deck.py로 다시 만들면 PowerPoint에서 손으로 고친 내용이 전부 날아가므로,
기존 파일을 열어 해당 슬라이드의 그림·패널·표·쪽번호만 고친다.

모델 비교표의 연도·발표처·사양은 각 모델 공식 저장소/블로그 기준이다. IDM-VTON은 공식
사양이 없어 사용자 보고치(Hugging Face 토론)를 쓰고 표에 그렇게 표시한다.

    python scripts/update_impl_slides.py [원본.pptx] [저장.pptx]

인자를 생략하면 docs/fitcheck_proposal_v2.pptx를 제자리에서 고친다(PowerPoint가 열고
있으면 저장이 막히므로 먼저 닫을 것). 화면 이미지는 docs/mockups/screen_*.png.
"""
import os
import sys

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_proposal_deck import (  # noqa: E402  디자인 토큰·도구를 발표 자료와 공유
    COL_W, FG, FG_DIM, FG_SUB, MOCKUP_DIR, ROOT, SLIDE_W, SURFACE, M,
    F_LIGHT, F_MED, F_REG, F_THIN,
    _emu, _lines, _picture, _rect, _rule, _textbox, _vrule,
)

# 네 모델을 같은 기준으로 공정하게 보여 준다(특정 모델 강조 없음). HR-VITON만 방식이 GAN이라
# 이름 아래 방식 표기와 구분선으로 나눈다.
MODEL_COLUMNS = [('CatVTON', 'diffusion'), ('IDM-VTON', 'diffusion'),
                 ('FASHN VTON v1.5', 'diffusion'), ('HR-VITON', 'GAN')]
MODEL_ROWS = [
    ('공개 연도', ['2024', '2024', '2026', '2022']),
    ('발표처', ['ICLR 2025', 'ECCV 2024', '논문 준비 중', 'ECCV 2022']),
    ('특징', ['가벼운 diffusion 모델', '대형 모델(SDXL) 기반', '사진을 압축 없이 생성', '옷 사진을 변형해 합성']),
    ('장점', ['적은 메모리로 실행', '옷 디테일 보존 우수', '상업적 사용 가능', '빠르고 가벼움']),
    ('단점', ['작은 프린트가 흐려짐', '무겁고 느림', '결과 해상도가 낮음', '흰 배경 사진 위주']),
]
MODEL_SUB = '같은 인물 사진 · 같은 옷으로 네 모델을 비교한다  ·  연도 · 발표처는 각 모델 공식 자료 기준'
MODEL_NOTES = (
    '[약 50초]\n\n'
    '합성 모델은 네 후보를 같은 조건에서 비교해 고릅니다. 세 개는 diffusion 모델이고, 마지막 '
    'HR-VITON은 비교를 위해 넣은 GAN 모델입니다.\n\n'
    'CatVTON은 2024년에 공개돼 ICLR에 채택됐고, 가벼워서 적은 메모리로 돌아가지만 작은 프린트가 '
    '흐려질 수 있습니다. IDM-VTON은 ECCV 2024 모델로 옷 디테일을 잘 살리지만 무겁습니다.\n\n'
    'FASHN VTON은 올해 공개돼 상업적으로 쓸 수 있지만 결과 해상도가 낮고, HR-VITON은 빠르지만 '
    '흰 배경 사진 위주로 학습됐습니다.'
)

PANEL_TOP = Inches(0.95)
PANEL_H = Inches(5.55)
PANEL_RIGHT = SLIDE_W - M


def _find_slide(prs, eyebrow_prefix):
    # 상단 eyebrow만 본다. Appendix에도 '모델 후보 — CatVTON' 같은 줄이 있어 본문까지 보면 오인한다
    for index, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if shape.has_text_frame and shape.top < Inches(1.0) \
                    and shape.text_frame.text.startswith(eyebrow_prefix):
                return index, slide
    raise SystemExit(f'슬라이드를 찾지 못했다: {eyebrow_prefix}')


def _remove(shape):
    shape._element.getparent().remove(shape._element)


def _clear_visuals(slide):
    """오른쪽의 기존 패널·화면 그림을 지운다 (하단 워드마크 그림은 남긴다)."""
    for shape in list(slide.shapes):
        is_panel = shape.shape_type == 1 and shape.left >= Inches(6)
        is_screen = shape.shape_type == 13 and shape.height > Inches(1)
        if is_panel or is_screen:
            _remove(shape)


def _set_page_number(slide, number):
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip().isdigit() and shape.top > Inches(6.5):
            run = shape.text_frame.paragraphs[0].runs[0]
            run.text = f'{number:02d}'


def _fit_text_column(slide, width):
    """패널이 넓어진 슬라이드에서 왼쪽 글 상자가 패널 밑으로 들어가지 않게 폭을 줄인다."""
    for shape in slide.shapes:
        if shape.has_text_frame and shape.top > Inches(1.5) and shape.left < Inches(1) \
                and shape.width > width:
            shape.width = _emu(width)


def _panel(slide, left):
    return _rect(slide, left, PANEL_TOP, PANEL_RIGHT - left, PANEL_H, SURFACE, rounded=0.05)


def _one_screen(slide, image):
    left = Inches(8.1)
    _panel(slide, left)
    img_h = Inches(4.85)
    pic = _picture(slide, os.path.join(MOCKUP_DIR, image), Inches(0),
                   _emu(PANEL_TOP + (PANEL_H - img_h) / 2), img_h)
    pic.left = _emu(left + (PANEL_RIGHT - left - pic.width) / 2)


def _two_screens(slide, images, labels):
    left = Inches(6.85)
    _panel(slide, left)
    img_h = Inches(4.45)
    gap = Inches(0.55)
    top = PANEL_TOP + Inches(0.3)
    pics = [_picture(slide, os.path.join(MOCKUP_DIR, img), Inches(0), top, img_h) for img in images]
    total = pics[0].width + gap + pics[1].width
    x = _emu(left + (PANEL_RIGHT - left - total) / 2)
    for pic, label in zip(pics, labels):
        pic.left = x
        tf = _textbox(slide, x, top + img_h + Inches(0.12), pic.width, Inches(0.3),
                      align=PP_ALIGN.CENTER)
        _lines(tf, [label], 11, font=F_REG, color=FG_SUB, align=PP_ALIGN.CENTER)
        x = _emu(x + pic.width + gap)
    arrow_x = pics[0].left + pics[0].width
    tf = _textbox(slide, arrow_x, top + img_h / 2 - Inches(0.2), gap, Inches(0.4),
                  align=PP_ALIGN.CENTER)
    _lines(tf, ['→'], 18, font=F_LIGHT, color=FG_DIM, align=PP_ALIGN.CENTER)


def _model_table(slide):
    """eyebrow·워드마크·쪽번호만 남기고 모델 4종 비교표로 다시 채운다."""
    for shape in list(slide.shapes):
        keep = (shape.top < Inches(1.0)                                   # eyebrow
                or (shape.shape_type == 13 and shape.height < Inches(0.3))  # 하단 워드마크
                or (shape.has_text_frame and shape.top > Inches(6.5)))    # 쪽번호
        if not keep:
            _remove(shape)

    tf = _textbox(slide, M, Inches(1.4), COL_W, Inches(0.7))
    _lines(tf, ['네 후보를 같은 조건에서 비교한다'], 32, font=F_THIN)
    tf = _textbox(slide, M, Inches(2.12), COL_W, Inches(0.35))
    _lines(tf, [MODEL_SUB], 13, font=F_LIGHT, color=FG_SUB)

    label_w = Inches(1.45)
    col_w = _emu((COL_W - label_w) / len(MODEL_COLUMNS))
    top = Inches(2.7)
    _rule(slide, M, top, COL_W)

    head_h = Inches(0.78)
    for c, (name, method) in enumerate(MODEL_COLUMNS):
        left = M + label_w + col_w * c
        tf = _textbox(slide, left, top + Inches(0.12), col_w - Inches(0.15), Inches(0.35))
        _lines(tf, [name], 16, font=F_REG, color=FG)
        tf = _textbox(slide, left, top + Inches(0.46), col_w - Inches(0.15), Inches(0.28))
        _lines(tf, [method], 11, font=F_MED, color=FG_DIM if method == 'GAN' else FG_SUB, tracking=0.6)
    y = top + head_h
    _rule(slide, M, y, COL_W)

    row_h = Inches(0.56)
    for label, cells in MODEL_ROWS:
        tf = _textbox(slide, M, y + Inches(0.15), label_w - Inches(0.1), Inches(0.3))
        _lines(tf, [label], 12, font=F_MED, color=FG_DIM, tracking=0.4)
        for c, text in enumerate(cells):
            tf = _textbox(slide, M + label_w + col_w * c, y + Inches(0.14), col_w - Inches(0.15),
                          row_h - Inches(0.14))
            _lines(tf, [text], 13, font=F_LIGHT, color=FG)
        y += row_h
        _rule(slide, M, y, COL_W)

    # diffusion 세 모델과 GAN 모델(HR-VITON) 사이 구분선
    gan_x = M + label_w + col_w * 3 - Inches(0.12)
    _vrule(slide, gan_x, top + Inches(0.1), y - top - Inches(0.1), color=FG_DIM, weight=Pt(1))

    slide.notes_slide.notes_text_frame.text = MODEL_NOTES


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    models_only = '--models-only' in sys.argv  # 구현 슬라이드 제목을 손으로 바꾼 파일에서는 모델 표만 고친다
    src = args[0] if args else os.path.join(ROOT, 'docs', 'fitcheck_proposal_v2.pptx')
    dst = args[1] if len(args) > 1 else src
    prs = Presentation(src)

    if models_only:
        number, slide = _find_slide(prs, '모델 후보')
        _model_table(slide)
        _set_page_number(slide, number)
        prs.save(dst)
        print(f'모델 후보 표 수정 완료 -> {dst}')
        return

    number, slide = _find_slide(prs, '구현  01')
    _clear_visuals(slide)
    _one_screen(slide, 'screen_upload.png')
    _set_page_number(slide, number)

    number, slide = _find_slide(prs, '구현  02')
    _clear_visuals(slide)
    _fit_text_column(slide, Inches(5.6))
    _two_screens(slide, ['screen_measure.png', 'screen_result.png'], ['치수 입력', '추천 결과'])
    _set_page_number(slide, number)

    number, slide = _find_slide(prs, '구현  03')
    _clear_visuals(slide)
    _one_screen(slide, 'screen_compare.png')
    _set_page_number(slide, number)

    number, slide = _find_slide(prs, '모델 후보')
    _model_table(slide)
    _set_page_number(slide, number)

    prs.save(dst)
    print(f'구현 슬라이드 3장 + 모델 후보 표 수정 완료 -> {dst}')


if __name__ == '__main__':
    main()
