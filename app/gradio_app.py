"""가상 피팅 웹 데모. 합성 엔진은 FASHN VTON v1.5 (app/fashn_core.py).

기본 파이썬(3.10 이상)에서 실행 — CatVTON용 3.9 venv가 아니다:
    python app/gradio_app.py
FASHN 저장소·가중치 위치는 app/paths.py (FASHN_REPO, FASHN_WEIGHTS 환경변수로 바꿀 수 있음).
화면 스타일은 app/ui_style.py (FITCHECK_THEME 환경변수로 후보를 바꿔 볼 수 있음).
"""
import html
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import gradio_client.utils as _gcu

# --- gradio_client 버그 우회 ---------------------------------------------
# pydantic이 만든 JSON 스키마에는 `additionalProperties: true` 처럼 값이 dict가 아니라
# bool인 경우가 있는데, gradio_client가 이를 dict로 가정하고 `"const" in schema` 를 해서
#   TypeError: argument of type 'bool' is not iterable
# 로 터진다. gradio의 메인 라우트가 show_api 설정과 무관하게 api_info()를 호출하므로
# 페이지 자체가 안 열린다. bool 스키마를 Any로 처리하도록 감싼다.
_orig_json_schema_to_python_type = _gcu._json_schema_to_python_type
_orig_get_type = _gcu.get_type


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return 'Any'
    return _orig_json_schema_to_python_type(schema, defs)


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return 'Any'
    return _orig_get_type(schema)


_gcu._json_schema_to_python_type = _safe_json_schema_to_python_type
_gcu.get_type = _safe_get_type
# -------------------------------------------------------------------------

from fashn_core import (try_on, load_models, warmup, CATEGORY_BY_CLOTH_TYPE,
                        DEFAULT_STEPS, DEFAULT_GUIDANCE, DEFAULT_SEED)
from body_profile import BodyProfile
from garment_check import check_garment
from photo_check import check_photo, run_pose
from photo_guide import draw_guide
from restore_original import restore_original
from size_fit import SizeChart, ko, recommend
from ui_style import HEADER_HTML, STYLE, section

# 이너·아우터는 FASHN에서 상의와 같은 처리라(fashn_core.CATEGORY_BY_CLOTH_TYPE) 화면에서는 뺐다
CLOTH_LABELS = {
    '상의': 'upper',
    '하의': 'lower',
    '원피스': 'overall',
}


CHART_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'size_charts')


def load_chart(filename):
    with open(os.path.join(CHART_DIR, filename), encoding='utf-8') as f:
        return SizeChart.from_dict(json.load(f))


def list_charts(category=None):
    """치수표 목록. category를 주면 그 부위의 치수표만 돌려준다.

    상의를 입히면서 하의 치수표를 고르면 공통 항목이 없어 추천이 안 나온다.
    선택 자체를 막는 게 낫다.
    """
    if not os.path.isdir(CHART_DIR):
        return []
    files = sorted(f for f in os.listdir(CHART_DIR) if f.endswith('.json'))
    if category is None:
        return files
    out = []
    for f in files:
        try:
            if load_chart(f).category == category:
                out.append(f)
        except Exception:
            continue
    return out


def charts_for_cloth(cloth_label):
    """옷 종류 라디오 선택에 맞는 치수표로 드롭다운을 갱신한다."""
    cloth_type = CLOTH_LABELS.get(cloth_label, 'upper')
    category = 'lower' if cloth_type == 'lower' else 'upper'
    files = list_charts(category)
    return gr.update(choices=files, value=files[0] if files else None)


def _rich(text):
    """안내 문구의 **굵게** 만 HTML로 바꾼다. 나머지는 이스케이프."""
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html.escape(text))


def _note(level, lines, elapsed=None):
    """검사 결과 안내 상자. level: ok · warn · err · hint. elapsed(초)를 주면 오른쪽 위에 '검사 3.0s'."""
    body = ''.join(f'<p>{_rich(line)}</p>' for line in lines)
    took = f'<span class="fc-note__time">검사 {secs(elapsed)}</span>' if elapsed is not None else ''
    return f'<div class="fc-note fc-note--{level}">{took}{body}</div>'


def _recommend(chart_file, height, shoulder, chest, waist):
    """(치수표, 추천, None) 또는 (None, None, (제목, [문구]))."""
    if not chart_file:
        return None, None, (None, ['치수표가 없어 사이즈 추천을 건너뜁니다.'])
    try:
        profile = BodyProfile.from_inputs(
            height_cm=height, shoulder_cm=shoulder, chest_circumference_cm=chest,
            waist_circumference_cm=waist,
        )
    except ValueError as e:
        return None, None, ('사이즈 추천 불가', [str(e)])

    problems = profile.validate()
    if problems:
        return None, None, ('입력값을 확인해주세요', problems)

    body = profile.to_chart_dimensions()
    if not body:
        return None, None, (None, ['신체 치수를 하나 이상 입력하면 사이즈를 추천합니다.'])

    chart = load_chart(chart_file)
    rec = recommend(chart, body)
    if rec.best is None and not set(chart.dimensions()) & set(body):
        return None, None, ('사이즈 추천 불가', [
            f'{chart.name}({chart.category})와 입력한 치수가 맞지 않습니다. '
            '옷 종류에 맞는 치수표를 선택했는지 확인해주세요.'])
    if rec.best is None:
        return None, None, ('사이즈 추천 불가', [' '.join(rec.notes)])
    return chart, rec, None


# 여유분 판정(size_fit.FIT_BANDS 라벨) → 표의 색 점
FIT_TONE = {'작음': 'err', '타이트': 'warn', '레귤러': 'ok', '루즈': 'mute', '오버핏': 'mute'}


def size_advice(chart_file, height, shoulder, chest, waist) -> str:
    """입력한 신체 치수와 옷 치수표를 비교해 사이즈 추천 카드(HTML)를 만든다."""
    chart, rec, error = _recommend(chart_file, height, shoulder, chest, waist)
    head = '<p class="fc-eyebrow">Size</p>'
    if error:
        title, lines = error
        title_html = f'<h3 class="fc-size__title">{html.escape(title)}</h3>' if title else ''
        return (f'<div class="fc-size">{head}{title_html}'
                + ''.join(f'<p class="fc-size__msg">{html.escape(line)}</p>' for line in lines) + '</div>')

    best = rec.best
    by_size = {fit.size: fit for fit in rec.ranked}
    if best.wearable:
        labels = {d.dimension: d.label for d in best.dimensions}
        verdict = f"{labels.get('chest') or labels.get('waist') or '레귤러'} 핏"
    else:
        verdict = ('비교 가능한 사이즈는 모두 몸보다 작아요' if any(
            not fit.scorable for fit in rec.ranked) else '모든 사이즈가 몸보다 작아요')
    chips = ''.join(
        f'<span class="fc-chip{" is-pick" if size == best.size else ""}'
        f'{" is-unknown" if not by_size[size].scorable else " is-no" if not by_size[size].wearable else ""}'
        f'">{html.escape(size)}</span>'
        for size in chart.sizes)

    dims = [d.dimension for d in best.dimensions]
    has_length = any('length' in row for row in chart.sizes.values())
    header = ''.join(f'<th>{ko(d)}</th>' for d in dims) + ('<th>총장</th>' if has_length else '')
    rows = []
    for size in chart.sizes:
        fit = by_size[size]
        fit_dims = {d.dimension: d for d in fit.dimensions}
        cells = ''.join(
            f'<td>{d.ease_cm:+.1f}<span class="fc-tone tone-{FIT_TONE.get(d.label, "mute")}">'
            f'{html.escape(d.label)}</span></td>' if (d := fit_dims.get(dimension)) else '<td>-</td>'
            for dimension in dims)
        if has_length:
            length = chart.sizes[size].get('length')
            cells += f'<td>{length:g}</td>' if length is not None else '<td>-</td>'
        pick = ' class="is-pick"' if size == best.size else ''
        name = html.escape(size) + (' <small>(판단 불가)</small>' if not fit.scorable else '')
        rows.append(f'<tr{pick}><td class="fc-size__name">{name}</td>{cells}</tr>')
    notes = ''.join(f'<li>{html.escape(note)}</li>' for note in rec.notes)
    return (
        f'<div class="fc-size">{head}'
        f'<div class="fc-size__head"><span class="fc-size__big">{html.escape(best.size)}</span>'
        f'<span class="fc-size__verdict{"" if best.wearable else " is-no"}">{verdict}</span></div>'
        f'<p class="fc-size__src">{html.escape(chart.name)} · 레귤러핏 옷 기준</p>'
        f'<div class="fc-chips">{chips}</div>'
        f'<table class="fc-table"><thead><tr><th>사이즈</th>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        '<p class="fc-size__legend">표의 숫자 = 옷 단면 − 몸 단면 (cm). + 는 여유, − 는 옷이 작음</p>'
        f'<ul class="fc-size__notes">{notes}</ul></div>'
    )


# 섹션 제목 줄 (번호 → 제목, 안내). 오른쪽 끝에 그 단계에 걸린 시간을 배지로 붙인다
SECTIONS = {
    '01': ('내 사진', '정면 · 전신 · 3:4 세로 사진'),
    '02': ('옷', '배경이 깔끔한 상품 사진일수록 좋아요'),
    '03': ('내 치수', '아는 항목만 · 둘레로 입력'),
    '04': ('피팅 결과', '한 장 약 50초 · T4 GPU'),
}


def secs(seconds):
    """걸린 시간 표기: <0.01s · 0.03s · 2.1s · 52s"""
    if seconds < 0.01:
        return '<0.01s'
    if seconds < 1:
        return f'{seconds:.2f}s'
    return f'{seconds:.1f}s' if seconds < 10 else f'{seconds:.0f}s'


def sec(num, seconds=None):
    title, hint = SECTIONS[num]
    return section(num, title, hint, secs(seconds) if seconds is not None else None)


def size_only(chart_file, height, shoulder, chest, waist):
    """합성 없이 사이즈 추천만 바로 보여준다 (합성은 한 장에 약 50초)."""
    start = time.perf_counter()
    advice = size_advice(chart_file, height, shoulder, chest, waist)
    return advice, sec('03', time.perf_counter() - start)


GARMENT_PHOTO_LABELS = {
    '상품 사진 (옷만 펴 놓고 찍음)': 'flat-lay',
    '착용 사진 (사람이 입고 찍음)': 'model',
}

PHOTO_HINT = _note('hint', ['올리는 순간 자세·거리를 검사해요. 정면 · 팔은 몸에서 주먹 하나 · 머리부터 발끝까지 화면 가득'])


def photo_status(person):
    """인물 사진을 올린 순간 자세·거리·역광을 검사해 안내한다 (app/photo_check.py).

    합성 한 장에 2분이 걸리므로, 규칙에 안 맞는 사진은 올리는 순간 알려주는 편이 낫다.
    포즈 인식(mediapipe)이 없거나 실패하면 photo_check 가 경고만 남기고 넘어간다.
    """
    if person is None:
        return PHOTO_HINT, None, sec('01')
    start = time.perf_counter()
    # 포즈를 한 번만 돌려 검사와 그림에 같이 쓴다. 따로 돌리면 느리고 둘이 어긋날 수 있다
    try:
        landmarks, seg_mask = run_pose(person)
    except Exception:  # mediapipe 미설치 등 — check_photo 가 스스로 경고를 남긴다
        landmarks = seg_mask = None
    check = check_photo(person, landmarks, seg_mask)
    guide = draw_guide(person, landmarks, check) if landmarks else None
    took = time.perf_counter() - start
    header = sec('01', took)
    if check.ok and not check.warnings:
        return _note('ok', [check.message()], took), guide, header
    return _note('err' if check.errors else 'warn', check.message().splitlines(), took), guide, header


def garment_status(garment, garment_photo_label):
    """옷 사진을 올리거나 사진 종류를 바꾸면 검사해 안내한다 (app/garment_check.py).

    경고만 하고 합성은 막지 않는다. 기준은 FASHN 실험 결과에 맞췄다(docs/PLAN.md F-23).
    """
    if garment is None:
        return '', sec('02')
    start = time.perf_counter()
    check = check_garment(garment, photo_type=GARMENT_PHOTO_LABELS.get(garment_photo_label, 'flat-lay'))
    took = time.perf_counter() - start
    return (_note('ok' if check.suitable else 'warn', check.message().splitlines(), took),
            sec('02', took))


def postprocess(person, result, cloth_type):
    """옷 자리만 합성 결과, 얼굴·손·배경은 원본으로 (app/restore_original.py, docs/PLAN.md F-25).

    반환: (이미지, 안내 문구). 할 수 없으면 합성 결과를 그대로 돌려준다.
    """
    from PIL import ImageOps
    person = ImageOps.exif_transpose(person).convert('RGB')
    # FASHN 결과는 3:4. 원본 비율이 다르면 옷 자리가 어긋나므로 건너뛴다 (폰 세로 사진은 3:4)
    if abs(person.width / person.height - result.width / result.height) > 0.02:
        return result, '후처리 건너뜀 — 인물 사진이 3:4 비율이 아님'
    start = time.perf_counter()
    restored, mask = restore_original(person, result, CATEGORY_BY_CLOTH_TYPE[cloth_type])
    if mask is None:
        return result, '후처리 건너뜀 — 신체 부위 구분 모델(mediapipe)을 못 씀'
    return restored, f'후처리 {secs(time.perf_counter() - start)}'


def run(person, garment, cloth_label, garment_photo_label,
        chart_file, height, shoulder, chest, waist, progress=gr.Progress()):
    """합성 → 후처리 → 사이즈 추천. 설정은 엔진 기본값으로 고정한다(스텝 25 · guidance 2.5 · 시드 42).

    고급 설정(스텝·시드·후보 장수·후처리 끄기·사진 검사 무시)과 후처리 전후 비교는 2026-11-02에 뺐다.
    시드를 고정하면 같은 사진·옷은 늘 같은 결과라 시연 백업 이미지와 일치한다.
    """
    if person is None or garment is None:
        raise gr.Error('인물 사진과 옷 사진을 모두 올려주세요.')

    check = check_photo(person)
    if not check.ok:
        raise gr.Error(check.message() + '\n사진 검사 안내대로 다시 찍어 올려 주세요.')

    start = time.perf_counter()
    advice = size_advice(chart_file, height, shoulder, chest, waist)
    size_header = sec('03', time.perf_counter() - start)

    cloth_type = CLOTH_LABELS[cloth_label]
    progress(0.0, desc='합성 중 (약 50초)')
    start = time.perf_counter()
    result, timing = try_on(
        person=person,
        garment=garment,
        cloth_type=cloth_type,
        garment_photo_type=GARMENT_PHOTO_LABELS[garment_photo_label],
        steps=DEFAULT_STEPS,
        guidance_scale=DEFAULT_GUIDANCE,
        seed=DEFAULT_SEED,
        return_timing=True,
    )
    progress(0.95, desc='얼굴·배경 되돌리는 중')
    restored, note = postprocess(person, result, cloth_type)
    info = f"합성 {secs(timing['total_s'])} · {note}"
    return restored, advice, info, size_header, sec('04', time.perf_counter() - start)


# gradio 6부터 theme·css·head 는 Blocks() 가 아니라 launch() 인자다 (Colab은 최신을 설치한다)
GRADIO6 = int(gr.__version__.split('.')[0]) >= 6

with gr.Blocks(title='FITCHECK — 가상 피팅', **({} if GRADIO6 else STYLE)) as demo:
    gr.HTML(HEADER_HTML, padding=False)

    with gr.Row(equal_height=False, elem_classes='fc-main'):
        with gr.Column(scale=5, elem_classes='fc-panel'):
            # 섹션 제목 줄도 출력이다 — 각 단계가 끝나면 걸린 시간 배지를 붙여 다시 그린다
            sec1_out = gr.HTML(sec('01'), padding=False)
            with gr.Row():
                person_in = gr.Image(label='인물 사진', type='pil', height=360)
                # 초록 = 통과, 빨강 = 다시 찍기. 노란 막대 = 필요한 코~발목 길이, 그 옆 막대 = 사진 속 길이 (app/photo_guide.py)
                photo_guide_out = gr.Image(label='사진 검사 · 초록 통과 · 빨강 다시 찍기',
                                           type='pil', height=360, interactive=False)
            photo_status_out = gr.HTML(PHOTO_HINT, padding=False)

            sec2_out = gr.HTML(sec('02'), padding=False)
            with gr.Row():
                garment_in = gr.Image(label='옷 사진', type='pil', height=360)
                with gr.Column():
                    cloth_in = gr.Radio(
                        choices=list(CLOTH_LABELS.keys()),
                        value='상의',
                        label='옷 종류',
                    )
                    garment_photo_in = gr.Radio(
                        choices=list(GARMENT_PHOTO_LABELS.keys()),
                        value='상품 사진 (옷만 펴 놓고 찍음)',
                        label='옷 사진 종류',
                    )
                    garment_status_out = gr.HTML('', padding=False)

            sec3_out = gr.HTML(sec('03'), padding=False)
            with gr.Row():
                height_in = gr.Number(value=175, label='키 (cm)')
                shoulder_in = gr.Number(value=None, label='어깨너비 (cm)', placeholder='모르면 비워 두기')
            with gr.Row():
                chest_in = gr.Number(value=None, label='가슴둘레 (cm)', placeholder='모르면 비워 두기')
                waist_in = gr.Number(value=None, label='허리둘레 (cm)', placeholder='모르면 비워 두기')
            _initial = list_charts('upper')
            chart_in = gr.Dropdown(
                choices=_initial, value=(_initial or [None])[0],
                label='옷 치수표 (옷 종류에 맞춰 바뀜)',
            )
            gr.HTML('<p class="fc-help">둘레로 넣으면 치수표의 단면과 자동으로 맞춰요 (가슴둘레 96 → 단면 48). '
                    '레귤러핏 옷 기준이라, 오버핏으로 나온 옷은 크게 추천될 수 있어요.</p>', padding=False)

            with gr.Row(elem_classes='fc-actions'):
                size_btn = gr.Button('사이즈만 보기', variant='secondary', size='lg', scale=2)
                run_btn = gr.Button('피팅 해보기', variant='primary', size='lg', scale=3, elem_classes='fc-cta')

        with gr.Column(scale=6, elem_classes='fc-result'):
            sec4_out = gr.HTML(sec('04'), padding=False)
            result_out = gr.Image(label='피팅 결과', show_label=False, type='pil', height=640, interactive=False)
            info_out = gr.Markdown(elem_classes='fc-info')
            size_out = gr.HTML(size_advice(None, None, None, None, None), padding=False)

    # 옷 종류를 바꾸면 그에 맞는 치수표만 남긴다
    cloth_in.change(fn=charts_for_cloth, inputs=[cloth_in], outputs=[chart_in])
    # 사진을 올리는 순간 검사한다 (합성 한 장에 2분이라 미리 걸러야 한다)
    person_in.change(fn=photo_status, inputs=[person_in],
                     outputs=[photo_status_out, photo_guide_out, sec1_out])
    # 옷 사진도 올리는 순간 검사한다. 사진 종류(상품/착용)에 따라 기준이 달라 둘 다 연결한다
    garment_in.change(fn=garment_status, inputs=[garment_in, garment_photo_in],
                      outputs=[garment_status_out, sec2_out])
    garment_photo_in.change(fn=garment_status, inputs=[garment_in, garment_photo_in],
                            outputs=[garment_status_out, sec2_out])

    run_btn.click(
        fn=run,
        inputs=[person_in, garment_in, cloth_in, garment_photo_in,
                chart_in, height_in, shoulder_in, chest_in, waist_in],
        outputs=[result_out, size_out, info_out, sec3_out, sec4_out],
    )
    size_btn.click(
        fn=size_only,
        inputs=[chart_in, height_in, shoulder_in, chest_in, waist_in],
        outputs=[size_out, sec3_out],
    )


if __name__ == '__main__':
    print('모델 로딩 중... (최초 1회)', flush=True)
    load_models()
    # 첫 사용자가 컴파일 시간(약 90초)을 떠안지 않게 미리 한 장 돌려 둔다
    print('준비 운동 중... (torch.compile, 약 1~2분)', flush=True)
    print('준비 운동 완료:', warmup(), '초', flush=True)
    print('로딩 완료. Gradio 실행합니다.', flush=True)
    # show_api=False: API 스키마 생성 경로(gradio_client)가 pydantic 버전에 따라 터지는 걸 회피.
    #   단 gradio 6에서 이 인자가 없어져 TypeError가 난다(Colab 2026-09-16) → 빼고 다시 띄운다.
    # prevent_thread_lock=True로 먼저 URL을 받아 직접 flush 출력한다.
    # (gradio 자체 print는 flush를 안 해서 파이프로 실행하면 링크가 안 보인다)
    options = dict(share=True, show_error=True, prevent_thread_lock=True, **(STYLE if GRADIO6 else {}))
    try:
        _, local_url, share_url = demo.queue().launch(show_api=False, **options)
    except TypeError:
        _, local_url, share_url = demo.queue().launch(**options)
    print('=' * 60, flush=True)
    print('로컬 주소 :', local_url, flush=True)
    print('공개 링크 :', share_url or '(생성 실패 — 아래 안내 참고)', flush=True)
    print('=' * 60, flush=True)
    if not share_url:
        print(
            'share 링크를 못 만들었습니다. Colab이라면 아래 셀로 접속하세요:\n'
            '    from google.colab.output import eval_js\n'
            '    print(eval_js("google.colab.kernel.proxyPort(7860)"))',
            flush=True,
        )
    demo.block_thread()
