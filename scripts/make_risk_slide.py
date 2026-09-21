"""'우려되는 점과 대응' 단독 슬라이드를 만든다.

발표 본 파일은 사용자가 PowerPoint에서 직접 고치고 있어서 건드리지 않고, 같은 디자인의
한 장짜리 파일로 만들어 복사해 넣게 한다. 우려 네 가지는 docs/PLAN.md와 앱 화면 설계에서
이미 확인된 문제만 고른다(결과 편차=시드 불안정, 사진 조건, 치수표 형식, 생성 시간).
치수표는 Vision 모델이 읽으므로, 형식 차이는 예외 규칙을 시스템 프롬프트에 추가해 흡수한다.

    python scripts/make_risk_slide.py [저장.pptx]
"""
import os
import sys

from pptx import Presentation
from pptx.util import Inches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_proposal_deck import (  # noqa: E402  디자인 토큰·도구를 발표 자료와 공유
    BG, COL_W, FG, FG_SUB, GREEN, ROOT, SLIDE_H, SLIDE_W, SURFACE, M,
    F_LIGHT, F_MED, F_REG, F_THIN,
    _emu, _eyebrow, _lines, _picture, _rect, _textbox, _timed_notes,
)

OUT_PATH = os.path.join(ROOT, 'docs', 'slide_risks.pptx')

# (우려, 이유 한 줄, 대응) — 사용자가 정한 세 가지. 3번 대응은 앱 설계(여러 장 중 선택)와
# PLAN.md 판정 기준에 맞춰 채웠다.
RISKS = [
    ('사진 찍는 방식에 따라 결과가 달라진다', '자세 · 배경 · 조명에 따라 합성 품질이 크게 바뀐다',
     '촬영 가이드로 원하는 포즈를 지시'),
    ('이미지 생성에 시간이 오래 걸린다', '품질 좋은 모델일수록 한 장에 수십 초 이상 걸린다',
     '더 좋은 GPU 채택 · 모델 경량화 추진'),
    ('결과 품질을 일정하게 유지하기 어렵다', '같은 입력으로도 결과가 매번 조금씩 달라진다',
     '기준으로 점검하고, 여러 장 중 최선을 선택'),
]
NOTES = (
    '마지막으로 우려되는 점과 대응입니다.\n\n'
    '첫째, 사진을 어떻게 찍느냐에 따라 모델의 결과가 크게 달라집니다. 그래서 촬영 가이드로 '
    '사용자에게 원하는 포즈를 지시합니다.\n\n'
    '둘째, 이미지를 만드는 데 시간이 오래 걸립니다. 더 좋은 GPU를 쓰거나 모델을 가볍게 만드는 '
    '방향으로 추진하겠습니다.\n\n'
    '셋째, 결과 품질을 일정하게 유지하기 어렵습니다. 정해 둔 기준으로 결과를 점검하고, 여러 장 중 '
    '가장 좋은 결과를 고르는 방식으로 대응합니다.'
)


def build(slide):
    _eyebrow(slide, '우려되는 점')
    tf = _textbox(slide, M, Inches(1.35), COL_W, Inches(0.7))
    _lines(tf, ['예상되는 어려움과 대응'], 32, font=F_THIN)

    # 가로로 긴 카드 세 줄: 왼쪽 우려(+이유), 오른쪽 대응
    card_h = Inches(1.22)
    row_gap = Inches(0.22)
    top = Inches(2.4)
    pad = Inches(0.4)
    answer_x = M + Inches(6.4)

    for i, (risk, why, answer) in enumerate(RISKS):
        y = _emu(top + i * (card_h + row_gap))
        _rect(slide, M, y, COL_W, card_h, SURFACE, rounded=0.1)

        tf = _textbox(slide, M + pad, y + Inches(0.3), Inches(0.5), Inches(0.35))
        _lines(tf, [f'{i + 1:02d}'], 13, font=F_MED, color=FG_SUB)
        tf = _textbox(slide, M + pad + Inches(0.55), y + Inches(0.26), Inches(5.3), Inches(0.4))
        _lines(tf, [risk], 17, font=F_REG, color=FG)
        tf = _textbox(slide, M + pad + Inches(0.55), y + Inches(0.72), Inches(5.3), Inches(0.35))
        _lines(tf, [why], 12.5, font=F_LIGHT, color=FG_SUB)

        tf = _textbox(slide, answer_x, y + Inches(0.3), Inches(0.6), Inches(0.35))
        _lines(tf, ['대응'], 13, font=F_MED, color=GREEN)
        tf = _textbox(slide, answer_x + Inches(0.7), y + Inches(0.28), M + COL_W - answer_x - Inches(0.7) - pad,
                      Inches(0.7))
        _lines(tf, [answer], 15, font=F_REG, color=FG, spacing=1.2)

    _picture(slide, os.path.join(ROOT, 'docs', 'brand', 'fitcheck_wordmark_dim.png'), M,
             SLIDE_H - Inches(0.67), Inches(0.1))


def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    build(slide)
    _, seconds = _timed_notes(NOTES, 0.0)
    slide.notes_slide.notes_text_frame.text = f'[약 {round(seconds / 5) * 5}초]\n\n{NOTES}'
    out = sys.argv[1] if len(sys.argv) > 1 else OUT_PATH
    prs.save(out)
    print(f'슬라이드 생성 완료 (대본 약 {int(seconds)}초) -> {out}')


if __name__ == '__main__':
    main()
