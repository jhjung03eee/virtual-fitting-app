"""계획서 발표(9월, 과제제안) PPT를 생성한다.

**결과/실험 데이터를 넣지 않는다.** 이 발표는 제안이지 중간보고가 아니므로
"왜 이 과제를 하는지 · 어떻게 구현할 계획인지 · 무엇을 기대하는지"만 담는다.
진행 상황·벤치마크·실패 사례는 11월 중간발표용 자료에서 다룬다.

슬라이드 내용은 이 파일 안 SLIDES 리스트로 정의한다. 발표 팀명/조원은
표지 슬라이드의 TEAM_INFO를 채워 넣는다.

    python scripts/build_proposal_deck.py
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCKUP_DIR = os.path.join(ROOT, 'docs', 'mockups')
OUT_PATH = os.path.join(ROOT, 'docs', 'proposal_presentation.pptx')

FONT = '맑은 고딕'
INK = RGBColor(0x11, 0x11, 0x11)
SUBINK = RGBColor(0x76, 0x76, 0x76)
FAINT = RGBColor(0xD4, 0xD4, 0xD4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.75)

# TODO: 발표 전 채워 넣기
TEAM_INFO = "성균관대학교 소프트웨어학과 · 종합설계프로젝트 · [팀명 / 조원 이름]"

SLIDES = [
    {
        'type': 'title',
        'eyebrow': 'SIZE THEN BUY',
        'title': '사이즈 정보를 반영한\n가상 피팅 앱',
        'subtitle': '계획서 발표 · 과제 제안',
    },
    {
        'type': 'bullets',
        'title': '배경 및 문제의식',
        'bullets': [
            '온라인 의류 구매는 직접 입어볼 수 없어 사이즈·핏에 대한 불확실성이 크다',
            '이로 인한 반품·교환은 소비자와 판매 기업 모두에게 상당한 경제적 손실로 이어진다',
            '기존 가상 피팅 서비스는 "입어본 모습"만 보여줄 뿐, 그 사이즈가 나에게 맞는지는 알려주지 않는다',
            '판매자가 제공하는 치수표(예: 가슴단면 52cm)와 내가 아는 내 몸 치수(가슴둘레 96cm)는 기준 자체가 달라 직접 비교하기 어렵다',
        ],
    },
    {
        'type': 'bullets',
        'title': '과제 목적',
        'bullets': [
            '가상 피팅으로 핏감을 예측하고, 신체 정보로 사이즈까지 추천하는 end-to-end 옷 추천 서비스',
            '사용자 정보(신체 수치·사진)와 옷 정보(디자인 이미지·치수표)만 있으면 전체 파이프라인이 동작하도록 설계',
            '"입어본 모습"과 "맞는 사이즈"를 하나의 흐름으로 연결해 제공한다',
            '별도의 실측 장비나 추가 촬영 없이, 기존에 판매자가 이미 갖고 있는 정보만으로 구현 가능',
        ],
    },
    {
        'type': 'bullets',
        'title': '기대 효과',
        'bullets': [
            '구매 전 "맞는 사이즈로 입은 모습"을 확인해 구매 확신을 높인다',
            '사이즈 미스매치로 인한 반품·교환 — 소비자의 불편과 기업의 물류 비용을 함께 줄인다',
            '기존 가상 피팅 앱과의 차별점: 착용 모습 + 사이즈 근거를 함께 제시',
            '치수 정보가 없어 온라인 구매를 꺼리던 사용자층 공략',
        ],
    },
    {
        'type': 'bullets_image',
        'title': '구현 방법 (1) — 가상 피팅',
        'image': 'mockup_upload.png',
        'caption': 'UI 목업 · 사진 업로드 화면',
        'bullets': [
            '사용자 전신 사진 + 옷 사진을 입력받는다',
            'diffusion 기반 이미지 합성으로 옷을 입은 모습을 생성한다',
            '검증된 오픈소스 가상 피팅 모델을 기반으로 구현할 계획',
            '상의 / 하의 / 원피스 등 카테고리별로 합성을 지원한다',
        ],
    },
    {
        'type': 'bullets_image',
        'title': '구현 방법 (2) — 사이즈 추천',
        'image': 'mockup_measure.png',
        'caption': 'UI 목업 · 치수 입력 화면',
        'bullets': [
            '사용자가 아는 신체 치수(키·어깨너비·가슴둘레·허리둘레 등)만 입력해도 동작',
            '치수표의 단면 표기와 사용자의 둘레 표기를 자동으로 맞춰 비교한다',
            '옷 치수와 신체 치수의 여유분을 계산해 S/M/L 핏을 추천한다',
            '입력하지 않은 항목은 판정에서 제외하고 사용자에게 알린다',
        ],
    },
    {
        'type': 'bullets_image',
        'title': '구현 방법 (3) — 두 기능의 통합',
        'image': 'mockup_compare.png',
        'caption': 'UI 목업 · 사이즈별 비교 화면',
        'bullets': [
            '추천된 사이즈들의 착용 모습을 나란히 비교해서 보여준다',
            '여유분 값을 착용 실루엣에 반영해 "레귤러/루즈" 같은 핏 차이를 시각적으로 표현한다',
            '가상 피팅과 사이즈 추천이 따로 동작하지 않고 하나의 흐름으로 연결되는 것이 이 과제의 핵심 제안이다',
        ],
    },
    {
        'type': 'bullets',
        'title': '시스템 구조 (개략)',
        'bullets': [
            '모바일 앱 UI — 사진 업로드, 치수 입력, 결과 비교',
            '가상 피팅 추론 — diffusion 모델을 이용한 이미지 합성 서버',
            '사이즈 계산 로직 — 둘레·단면 변환, 여유분 계산 (GPU 불필요, 독립적으로 검증 가능한 모듈로 설계)',
            'Python 기반 백엔드로 구현하고, 세 요소를 순차적으로 통합한다',
        ],
    },
    {
        'type': 'schedule',
        'title': '세부 일정',
        'rows': [
            ('9월', '모델·관련 연구 조사, 계획서 제출'),
            ('10월', '체형 분석 · 여유분 계산 로직 구현, 가상 피팅 모델 연동'),
            ('11월 초', '중간발표'),
            ('11월 ~ 12월 초', '사이즈별 핏 반영 개선, 앱 UI 통합'),
            ('12월', '최종발표'),
        ],
    },
    {
        'type': 'closing',
        'title': '감사합니다',
        'subtitle': 'Q & A',
    },
]


def _set_font(run, size, bold=False, color=INK, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def _textbox(slide, left, top, width, height, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    return box, tf


def _rule(slide, left, top, width, color=INK, weight=Pt(1)):
    line = slide.shapes.add_connector(1, left, top, left + width, top)
    line.line.color.rgb = color
    line.line.width = weight


def _header(slide, title):
    box, tf = _textbox(slide, MARGIN, Inches(0.55), SLIDE_W - 2 * MARGIN, Inches(0.9))
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    _set_font(run, 28, bold=True)
    _rule(slide, MARGIN, Inches(1.35), SLIDE_W - 2 * MARGIN, color=FAINT, weight=Pt(1))


def _bullets_block(slide, left, top, width, height, items, size=17):
    box, tf = _textbox(slide, left, top, width, height)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(14)
        run = p.add_run()
        run.text = '·  ' + item
        _set_font(run, size, color=INK)


def build_title_slide(slide, spec):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE

    _, tf = _textbox(slide, MARGIN, Inches(2.5), SLIDE_W - 2 * MARGIN, Inches(0.4))
    run = tf.paragraphs[0].add_run()
    run.text = spec['eyebrow']
    _set_font(run, 12, bold=True, color=SUBINK)
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT

    _, tf = _textbox(slide, MARGIN, Inches(2.95), SLIDE_W - 2 * MARGIN, Inches(2.0))
    for i, line in enumerate(spec['title'].split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = line
        _set_font(run, 40, bold=True)

    _rule(slide, MARGIN, Inches(4.9), Inches(2.2), color=INK, weight=Pt(2))

    _, tf = _textbox(slide, MARGIN, Inches(5.1), SLIDE_W - 2 * MARGIN, Inches(0.5))
    run = tf.paragraphs[0].add_run()
    run.text = spec['subtitle']
    _set_font(run, 16, color=SUBINK)

    _, tf = _textbox(slide, MARGIN, SLIDE_H - Inches(0.9), SLIDE_W - 2 * MARGIN, Inches(0.5))
    run = tf.paragraphs[0].add_run()
    run.text = TEAM_INFO
    _set_font(run, 12, color=SUBINK)


def build_bullets_slide(slide, spec):
    _header(slide, spec['title'])
    _bullets_block(slide, MARGIN, Inches(1.75), SLIDE_W - 2 * MARGIN, SLIDE_H - Inches(2.2),
                    spec['bullets'], size=18)


def build_bullets_image_slide(slide, spec):
    _header(slide, spec['title'])

    text_width = Inches(7.6)
    _bullets_block(slide, MARGIN, Inches(1.9), text_width, SLIDE_H - Inches(2.3),
                    spec['bullets'], size=16)

    img_path = os.path.join(MOCKUP_DIR, spec['image'])
    img_height = Inches(5.1)
    img_top = Inches(1.75)
    pic = slide.shapes.add_picture(img_path, Inches(0), img_top, height=img_height)
    pic.left = SLIDE_W - MARGIN - pic.width
    pic.top = img_top
    # add a thin frame around the mockup
    pic.line.color.rgb = FAINT
    pic.line.width = Pt(1)

    _, tf = _textbox(slide, pic.left, img_top + img_height + Inches(0.08), pic.width, Inches(0.4))
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = spec['caption']
    _set_font(run, 11, color=SUBINK, italic=True)


def build_schedule_slide(slide, spec):
    _header(slide, spec['title'])

    rows = spec['rows']
    n = len(rows) + 1
    table_top = Inches(1.9)
    table_height = Inches(4.6)
    table_width = SLIDE_W - 2 * MARGIN
    gfx = slide.shapes.add_table(n, 2, MARGIN, table_top, table_width, table_height)
    table = gfx.table
    table.columns[0].width = Inches(2.6)
    table.columns[1].width = table_width - Inches(2.6)

    header_cells = ('시기', '내용')
    for c, text in enumerate(header_cells):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = INK
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        run = p.add_run()
        run.text = text
        _set_font(run, 14, bold=True, color=WHITE)

    for r, (when, what) in enumerate(rows, start=1):
        for c, text in enumerate((when, what)):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = text
            _set_font(run, 14, bold=(c == 0), color=INK)


def build_closing_slide(slide, spec):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = INK

    _, tf = _textbox(slide, MARGIN, Inches(3.0), SLIDE_W - 2 * MARGIN, Inches(1.2),
                      anchor=MSO_ANCHOR.MIDDLE)
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    run = tf.paragraphs[0].add_run()
    run.text = spec['title']
    _set_font(run, 40, bold=True, color=WHITE)

    _, tf = _textbox(slide, MARGIN, Inches(4.1), SLIDE_W - 2 * MARGIN, Inches(0.6))
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    run = tf.paragraphs[0].add_run()
    run.text = spec['subtitle']
    _set_font(run, 16, color=RGBColor(0xC4, 0xC4, 0xC4))


BUILDERS = {
    'title': build_title_slide,
    'bullets': build_bullets_slide,
    'bullets_image': build_bullets_image_slide,
    'schedule': build_schedule_slide,
    'closing': build_closing_slide,
}


def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank_layout = prs.slide_layouts[6]

    for spec in SLIDES:
        slide = prs.slides.add_slide(blank_layout)
        BUILDERS[spec['type']](slide, spec)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    prs.save(OUT_PATH)
    print(f'{len(SLIDES)}장 생성 완료 -> {OUT_PATH}')


if __name__ == '__main__':
    main()
