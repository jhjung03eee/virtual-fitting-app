"""데모 화면 스타일 (gradio_app.py 에서 씀). gradio를 임포트만 하고 GPU·모델은 쓰지 않는다.

색·모서리는 CSS 변수(--fc-*) 한 벌로 정하고, gradio 테마 변수는 전부 그 변수를 가리키게 한다.
그래서 후보(PALETTES)를 바꿔도 테마·CSS를 따로 고칠 필요가 없다.
    FITCHECK_THEME=mono | store | dark   (기본 mono)
"""
import os

import gradio as gr

PALETTES = {
    # 흰 바탕 · 검은 글자 · 가는 선 · 모서리 없음. 잡지 같은 여백
    'mono': dict(bg='#FFFFFF', surface='#FFFFFF', soft='#F6F6F6', ink='#111111', sub='#8A8A8A',
                 line='#E5E5E5', accent='#111111', accent_ink='#FFFFFF', point='#111111', point_ink='#FFFFFF',
                 header_bg='#FFFFFF', header_ink='#111111', header_sub='#8A8A8A',
                 radius='0px', radius_sm='0px', panel_pad='0px', panel_bg='transparent',
                 ok='#0A8F4E', warn='#B86E00', err='#E0262B'),
    # 회색 바탕 위 흰 카드 · 검은 상단 바 · 추천 사이즈만 파란색
    'store': dict(bg='#F1F1F1', surface='#FFFFFF', soft='#F7F7F7', ink='#000000', sub='#8E8E8E',
                  line='#E3E3E3', accent='#000000', accent_ink='#FFFFFF', point='#0078FF', point_ink='#FFFFFF',
                  header_bg='#000000', header_ink='#FFFFFF', header_sub='#A0A0A0',
                  radius='4px', radius_sm='4px', panel_pad='24px', panel_bg='#FFFFFF',
                  ok='#0A8F4E', warn='#B86E00', err='#E0262B'),
    # 발표 자료와 같은 검은 바탕 · 초록 포인트
    'dark': dict(bg='#000000', surface='#141416', soft='#1C1C1E', ink='#F5F5F7', sub='#98989E',
                 line='#2C2C2E', accent='#30D158', accent_ink='#000000', point='#30D158', point_ink='#000000',
                 header_bg='#000000', header_ink='#FFFFFF', header_sub='#98989E',
                 radius='14px', radius_sm='10px', panel_pad='24px', panel_bg='#0B0B0C',
                 ok='#30D158', warn='#FFC800', err='#FF6B6B'),
}
THEME_NAME = os.environ.get('FITCHECK_THEME', 'mono')
PALETTE = PALETTES.get(THEME_NAME, PALETTES['mono'])

FONT = ['Pretendard Variable', 'Pretendard', 'Noto Sans KR', 'Apple SD Gothic Neo', 'sans-serif']
HEAD = ('<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/'
        'dist/web/variable/pretendardvariable-dynamic-subset.min.css">')
# 시스템이 다크 모드면 gradio가 어두운 색으로 바꾼다. 색은 팔레트가 정하므로 끈다
FORCE_LIGHT_JS = "() => { document.body.classList.remove('dark'); document.documentElement.classList.remove('dark'); }"


def _v(name):
    return f'var(--fc-{name})'


def _theme():
    theme = gr.themes.Base(font=FONT, radius_size=gr.themes.sizes.radius_none)
    values = dict(
        body_background_fill=_v('bg'), body_text_color=_v('ink'), body_text_color_subdued=_v('sub'),
        background_fill_primary=_v('surface'), background_fill_secondary=_v('soft'),
        border_color_primary=_v('line'), border_color_accent=_v('ink'),
        color_accent=_v('point'), color_accent_soft=_v('soft'), link_text_color=_v('ink'),
        block_background_fill=_v('surface'), block_border_color=_v('line'), block_border_width='1px',
        block_label_background_fill=_v('surface'), block_label_text_color=_v('sub'),
        block_label_border_color=_v('line'), block_title_text_color=_v('ink'), block_info_text_color=_v('sub'),
        block_radius=_v('radius'), block_shadow='none', panel_background_fill=_v('surface'),
        panel_border_color=_v('line'), shadow_drop='none', shadow_drop_lg='none',
        input_background_fill=_v('surface'), input_border_color=_v('line'), input_border_color_focus=_v('ink'),
        input_radius=_v('radius_sm'), input_shadow='none', input_shadow_focus='none',
        button_primary_background_fill=_v('accent'), button_primary_background_fill_hover=_v('accent'),
        button_primary_text_color=_v('accent_ink'), button_primary_border_color=_v('accent'),
        button_secondary_background_fill=_v('surface'), button_secondary_background_fill_hover=_v('soft'),
        button_secondary_text_color=_v('ink'), button_secondary_border_color=_v('ink'),
        button_large_radius=_v('radius_sm'), button_medium_radius=_v('radius_sm'), button_small_radius=_v('radius_sm'),
        checkbox_background_color_selected=_v('accent'), checkbox_border_color_selected=_v('accent'),
        checkbox_label_background_fill=_v('surface'), checkbox_label_background_fill_hover=_v('soft'),
        checkbox_label_background_fill_selected=_v('accent'), checkbox_label_text_color_selected=_v('accent_ink'),
        checkbox_label_border_color=_v('line'), checkbox_label_border_color_selected=_v('accent'),
        slider_color=_v('accent'),
    )
    values = {k: v for k, v in values.items() if hasattr(theme, k)}  # gradio 버전마다 이름이 조금 다르다
    return theme.set(**values, **{k + '_dark': v for k, v in values.items() if hasattr(theme, k + '_dark')})


def _palette_css():
    return ':root, .dark {' + ''.join(f'--fc-{k}:{v};' for k, v in PALETTE.items()) + '}'


CSS = _palette_css() + """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 0 28px 72px !important;
  background: var(--fc-bg) !important; }
footer { display: none !important; }
.gradio-container, .gradio-container button, .gradio-container input, .gradio-container textarea {
  font-family: 'Pretendard Variable', Pretendard, 'Noto Sans KR', sans-serif !important; letter-spacing: -0.01em; }

/* 상단 바 + 첫 화면 문구 */
.fc-header { display: flex; align-items: center; justify-content: space-between; background: var(--fc-header_bg);
  color: var(--fc-header_ink); margin: 0 -28px; padding: 18px 28px; border-bottom: 1px solid var(--fc-line); }
.fc-logo { font-size: 15px; font-weight: 700; letter-spacing: .34em; }
.fc-logo b { color: var(--fc-point); font-weight: 700; letter-spacing: 0; }
.fc-nav { display: flex; gap: 22px; font-size: 13px; color: var(--fc-header_sub); }
.fc-nav span:first-child { color: var(--fc-header_ink); font-weight: 600; }
.fc-hero { padding: 52px 0 40px; }
.fc-eyebrow { margin: 0 0 12px; font-size: 11px; font-weight: 600; letter-spacing: .2em; text-transform: uppercase;
  color: var(--fc-sub); }
.fc-hero h1 { margin: 0; font-size: clamp(30px, 4.2vw, 46px); line-height: 1.22; font-weight: 700;
  letter-spacing: -0.035em; color: var(--fc-ink); }
.fc-hero .fc-sub { margin: 16px 0 0; font-size: 15px; color: var(--fc-sub); }
.fc-steps { display: flex; gap: 0; margin-top: 28px; border-top: 1px solid var(--fc-line); }
.fc-steps div { flex: 1; padding: 14px 0 0; font-size: 13px; color: var(--fc-sub); }
.fc-steps b { display: block; font-size: 11px; letter-spacing: .14em; color: var(--fc-ink); margin-bottom: 4px; }

/* 입력 · 결과 영역 */
.fc-main { gap: 40px !important; align-items: flex-start !important; }
.fc-panel { background: var(--fc-panel_bg); padding: var(--fc-panel_pad) !important; border-radius: var(--fc-radius);
  gap: 14px !important; }
.fc-result { position: sticky; top: 16px; background: var(--fc-panel_bg); padding: var(--fc-panel_pad) !important;
  border-radius: var(--fc-radius); gap: 14px !important; }
.fc-sec { display: flex; align-items: baseline; gap: 12px; margin-top: 18px; padding-top: 14px;
  border-top: 2px solid var(--fc-ink); }
.fc-sec .fc-num { font-size: 12px; font-weight: 700; letter-spacing: .08em; color: var(--fc-point); }
.fc-sec h2 { margin: 0; font-size: 19px; font-weight: 700; letter-spacing: -0.02em; color: var(--fc-ink); }
.fc-sec p { margin: 0 0 0 auto; font-size: 12px; color: var(--fc-sub); }
.fc-time { align-self: center; padding: 3px 9px; font-size: 12px; font-weight: 700; line-height: 1.4;
  font-variant-numeric: tabular-nums; color: var(--fc-point_ink); background: var(--fc-point);
  border-radius: var(--fc-radius_sm); }
.fc-help { margin: 0; font-size: 12px; line-height: 1.6; color: var(--fc-sub); }
.fc-info p { margin: 0; font-size: 12px; color: var(--fc-sub); }

/* 검사 안내 */
.fc-note { padding: 11px 14px; border-left: 3px solid var(--fc-line); background: var(--fc-soft);
  border-radius: var(--fc-radius_sm); font-size: 13px; line-height: 1.6; color: var(--fc-ink); }
.fc-note p { margin: 0; }
.fc-note__time { float: right; margin-left: 12px; font-size: 12px; font-weight: 700; color: var(--fc-ink);
  font-variant-numeric: tabular-nums; }
.fc-note--ok { border-left-color: var(--fc-ok); }
.fc-note--warn { border-left-color: var(--fc-warn); }
.fc-note--err { border-left-color: var(--fc-err); }
.fc-note--hint { color: var(--fc-sub); }

/* gradio 기본 글자색(prose)이 상단 바 색을 덮지 않게 */
.fc-header .fc-logo { color: var(--fc-header_ink) !important; }
.fc-header .fc-nav span { color: var(--fc-header_sub) !important; }
.fc-header .fc-nav span:first-child { color: var(--fc-header_ink) !important; }

/* 버튼 */
.fc-actions { gap: 8px !important; margin-top: 6px; }
.fc-actions button { min-height: 56px; font-size: 15px !important; font-weight: 700 !important; }
.fc-cta { font-size: 16px !important; }
.fc-actions button.secondary { border: 1px solid var(--fc-ink) !important; }

/* 사이즈 추천 카드 */
.fc-size { padding: 24px; border: 1px solid var(--fc-line); border-radius: var(--fc-radius); background: var(--fc-surface);
  color: var(--fc-ink); }
.fc-size__title { margin: 0 0 6px; font-size: 17px; font-weight: 700; }
.fc-size__msg { margin: 0; font-size: 13px; line-height: 1.6; color: var(--fc-sub); }
.fc-size__head { display: flex; align-items: baseline; gap: 14px; }
.fc-size__big { font-size: 52px; font-weight: 800; line-height: 1; letter-spacing: -0.04em; }
.fc-size__verdict { font-size: 16px; font-weight: 700; color: var(--fc-point); }
.fc-size__verdict.is-no { color: var(--fc-err); }
.fc-size__src { margin: 10px 0 18px; font-size: 12px; color: var(--fc-sub); }
.fc-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.fc-chip { min-width: 58px; padding: 11px 14px; text-align: center; font-size: 14px; font-weight: 600;
  border: 1px solid var(--fc-line); border-radius: var(--fc-radius_sm); background: var(--fc-surface); color: var(--fc-ink); }
.fc-chip.is-pick { background: var(--fc-point); border-color: var(--fc-point); color: var(--fc-point_ink); }
.fc-chip.is-no { color: var(--fc-sub); text-decoration: line-through; }
.fc-chip.is-unknown { color: var(--fc-sub); border-style: dashed; }
.fc-size table.fc-table, .fc-size .fc-table tr { border: none !important; }
.fc-size table.fc-table { width: 100%; margin: 0; border-collapse: collapse; font-size: 13px; border: none; }
.fc-size .fc-table th { padding: 8px 6px; text-align: left; font-size: 12px; font-weight: 500; color: var(--fc-sub);
  background: transparent; border: none !important; border-bottom: 1px solid var(--fc-ink) !important; }
.fc-size .fc-table td { padding: 11px 6px; border: none !important; border-bottom: 1px solid var(--fc-line) !important;
  background: transparent; color: var(--fc-ink); font-variant-numeric: tabular-nums; }
.fc-size .fc-table tr.is-pick td { font-weight: 700; background: var(--fc-soft); }
.fc-size__name { font-weight: 700; }
.fc-tone { margin-left: 6px; font-size: 11px; font-weight: 500; }
.fc-tone::before { content: ''; display: inline-block; width: 6px; height: 6px; margin-right: 4px; border-radius: 50%;
  background: currentColor; vertical-align: 1px; }
.tone-ok { color: var(--fc-ok); } .tone-warn { color: var(--fc-warn); } .tone-err { color: var(--fc-err); }
.tone-mute { color: var(--fc-sub); }
.fc-size__legend { margin: 12px 0 0; font-size: 11px; color: var(--fc-sub); }
.fc-size__notes { margin: 6px 0 0; padding: 0; list-style: none; font-size: 11px; line-height: 1.6; color: var(--fc-sub); }

@media (max-width: 900px) {
  .fc-main { flex-direction: column !important; }
  .fc-result { position: static; }
  .fc-panel, .fc-result { padding: min(var(--fc-panel_pad), 14px) !important; }
  .gradio-container { padding: 0 16px 56px !important; }
  .fc-header { margin: 0 -16px; padding: 16px; }
  .fc-steps, .fc-nav { display: none !important; }
  .fc-size { padding: 18px; }
  .fc-size .fc-table th, .fc-size .fc-table td { white-space: nowrap; padding-left: 4px; padding-right: 4px; }
  .fc-tone { display: block; margin: 2px 0 0; }
}
"""

HEADER_HTML = """
<header class="fc-header">
  <div class="fc-logo">FITCHECK<b>✓</b></div>
  <nav class="fc-nav"><span>가상 피팅</span><span>사이즈 추천</span><span>SKKU 종합설계</span></nav>
</header>
<section class="fc-hero">
  <p class="fc-eyebrow">Virtual fitting · Size</p>
  <h1>입어 보기 전에,<br>어울리는지 그리고 맞는지까지</h1>
  <p class="fc-sub">내 사진과 옷 사진으로 착용 모습을 보고, 내 치수로 사이즈를 추천받아요.</p>
  <div class="fc-steps">
    <div><b>01</b>사진 검사</div><div><b>02</b>AI 합성</div><div><b>03</b>얼굴·배경 되돌리기</div><div><b>04</b>사이즈 추천</div>
  </div>
</section>
"""


def section(num, title, hint='', elapsed=None):
    """섹션 제목 줄. elapsed(예: '2.1s')를 주면 오른쪽 끝에 걸린 시간 배지를 붙인다."""
    badge = f'<span class="fc-time">{elapsed}</span>' if elapsed else ''
    return (f'<div class="fc-sec"><span class="fc-num">{num}</span><h2>{title}</h2>'
            f'<p>{hint}</p>{badge}</div>')


STYLE = dict(theme=_theme(), css=CSS, head=HEAD, js=FORCE_LIGHT_JS)
