"""가상 피팅 웹 데모. 합성 엔진은 FASHN VTON v1.5 (app/fashn_core.py).

기본 파이썬(3.10 이상)에서 실행 — CatVTON용 3.9 venv가 아니다:
    python app/gradio_app.py
FASHN 저장소·가중치 위치는 app/paths.py (FASHN_REPO, FASHN_WEIGHTS 환경변수로 바꿀 수 있음).
"""
import glob
import json
import os
import sys

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

from fashn_core import (try_on, load_models, FASHN_REPO,
                        DEFAULT_STEPS, DEFAULT_GUIDANCE, DEFAULT_SEED)
from body_profile import BodyProfile
from size_fit import SizeChart, recommend

CLOTH_LABELS = {
    '상의': 'upper',
    '하의': 'lower',
    '원피스 (상하 일체)': 'overall',
    '이너': 'inner',
    '아우터': 'outer',
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


def size_advice(chart_file, height, shoulder, chest, waist, hip) -> str:
    """입력한 신체 치수와 옷 치수표를 비교해 사이즈 추천 문구를 만든다."""
    if not chart_file:
        return '치수표가 없어 사이즈 추천을 건너뜁니다.'
    try:
        profile = BodyProfile.from_inputs(
            height_cm=height, shoulder_cm=shoulder, chest_circumference_cm=chest,
            waist_circumference_cm=waist, hip_circumference_cm=hip,
        )
    except ValueError as e:
        return f'**사이즈 추천 불가** — {e}'

    problems = profile.validate()
    if problems:
        return '**입력값을 확인해주세요**\n\n' + '\n'.join(f'- {p}' for p in problems)

    body = profile.to_chart_dimensions()
    if not body:
        return '신체 치수를 하나 이상 입력하면 사이즈를 추천합니다.'

    chart = load_chart(chart_file)

    rec = recommend(chart, body)
    if rec.best is None and not set(chart.dimensions()) & set(body):
        return ('**사이즈 추천 불가** — '
                f'`{chart.name}`({chart.category})와 입력한 치수가 맞지 않습니다. '
                '옷 종류에 맞는 치수표를 선택했는지 확인해주세요.')
    if rec.best is None:
        return '**사이즈 추천 불가** — ' + ' '.join(rec.notes)

    lines = [f'### {rec.message}', '', f'**{chart.name}** 기준', '',
             '| 사이즈 | 항목별 여유분 | 입을 수 있음 |', '|---|---|---|']
    for fit in rec.ranked:
        detail = ' / '.join(f'{d.dimension} {d.ease_cm:+.1f}cm ({d.label})'
                            for d in fit.dimensions)
        lines.append(f"| {fit.size} | {detail} | {'O' if fit.wearable else 'X'} |")
    for note in rec.notes:
        lines.append(f'\n> {note}')
    return '\n'.join(lines)


GARMENT_PHOTO_LABELS = {
    '상품 사진 (옷만 펴 놓고 찍음)': 'flat-lay',
    '착용 사진 (사람이 입고 찍음)': 'model',
}


def run(person, garment, cloth_label, garment_photo_label, steps, guidance_scale, seed,
        chart_file, height, shoulder, chest, waist, hip):
    if person is None or garment is None:
        raise gr.Error('인물 사진과 옷 사진을 모두 올려주세요.')

    advice = size_advice(chart_file, height, shoulder, chest, waist, hip)

    result, timing = try_on(
        person=person,
        garment=garment,
        cloth_type=CLOTH_LABELS[cloth_label],
        garment_photo_type=GARMENT_PHOTO_LABELS[garment_photo_label],
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        return_timing=True,
    )
    info = f"합성 {timing['total_s']:.0f}초 · 시드 {timing['seed']} · {timing['dtype']}"
    return result, advice, info


def _examples():
    """FASHN 저장소 예시 이미지로 예시 조합을 만든다. 없으면 빈 리스트."""
    person = glob.glob(os.path.join(FASHN_REPO, 'examples/data/model.*'))
    garment = glob.glob(os.path.join(FASHN_REPO, 'examples/data/garment.*'))
    if not person or not garment:
        return []
    return [[person[0], garment[0], '상의', '착용 사진 (사람이 입고 찍음)']]


with gr.Blocks(title='사이즈 반영 가상 피팅') as demo:
    gr.Markdown(
        '# 가상 피팅 데모\n'
        '인물 사진과 옷 사진을 올리고 옷 종류를 고르면 합성 결과가 나옵니다 (FASHN VTON v1.5).\n\n'
        '- 인물은 **정면, 머리부터 발끝까지 화면을 꽉 채운** 사진일수록 결과가 좋습니다\n'
        '- T4 GPU 기준 한 장에 약 2분 (50스텝, 품질 우선). 빠르게 보려면 고급 설정에서 스텝을 30으로'
    )

    with gr.Row():
        with gr.Column():
            person_in = gr.Image(label='인물 사진', type='pil', height=400)
            garment_in = gr.Image(label='옷 사진', type='pil', height=400)
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
            with gr.Accordion('내 신체 치수 (사이즈 추천용)', open=True):
                gr.Markdown(
                    '아는 항목만 넣어도 됩니다. **둘레**로 입력하면 치수표의 단면과 '
                    '자동으로 맞춰 계산합니다 (가슴둘레 96 → 가슴단면 48).'
                )
                height_in = gr.Number(value=175, label='키 (cm)')
                shoulder_in = gr.Number(value=None, label='어깨너비 (cm, 폭)')
                chest_in = gr.Number(value=None, label='가슴둘레 (cm)')
                waist_in = gr.Number(value=None, label='허리둘레 (cm)')
                hip_in = gr.Number(value=None, label='엉덩이둘레 (cm)')
                _initial = list_charts('upper')
                chart_in = gr.Dropdown(
                    choices=_initial, value=(_initial or [None])[0],
                    label='옷 치수표 (선택한 옷 종류에 맞춰 자동 변경)',
                )

            with gr.Accordion('고급 설정', open=False):
                steps_in = gr.Slider(
                    20, 50, value=DEFAULT_STEPS, step=1,
                    label='추론 스텝 (50 기본, 30이면 약 1.7배 빠름)',
                )
                guidance_in = gr.Slider(
                    1.0, 4.0, value=DEFAULT_GUIDANCE, step=0.1,
                    label='guidance scale (2.5 기본. 낮추면 글자 프린트가 휜다)',
                )
                seed_in = gr.Number(value=DEFAULT_SEED, precision=0,
                                    label='시드 (-1이면 매번 랜덤. 결과가 어색하면 바꿔서 다시)')
            run_btn = gr.Button('피팅 해보기', variant='primary')

        with gr.Column():
            result_out = gr.Image(label='합성 결과', height=520)
            info_out = gr.Markdown()
            size_out = gr.Markdown(label='사이즈 추천')

    # 옷 종류를 바꾸면 그에 맞는 치수표만 남긴다
    cloth_in.change(fn=charts_for_cloth, inputs=[cloth_in], outputs=[chart_in])

    run_btn.click(
        fn=run,
        inputs=[person_in, garment_in, cloth_in, garment_photo_in, steps_in, guidance_in, seed_in,
                chart_in, height_in, shoulder_in, chest_in, waist_in, hip_in],
        outputs=[result_out, size_out, info_out],
    )

    examples = _examples()
    if examples:
        gr.Examples(examples=examples, inputs=[person_in, garment_in, cloth_in, garment_photo_in], label='예시')


if __name__ == '__main__':
    print('모델 로딩 중... (최초 1회)', flush=True)
    load_models()
    print('로딩 완료. Gradio 실행합니다.', flush=True)
    # show_api=False: API 스키마 생성 경로(gradio_client)가 pydantic 버전에 따라 터지는 걸 회피
    # prevent_thread_lock=True로 먼저 URL을 받아 직접 flush 출력한다.
    # (gradio 자체 print는 flush를 안 해서 파이프로 실행하면 링크가 안 보인다)
    _, local_url, share_url = demo.queue().launch(
        share=True, show_error=True, show_api=False, prevent_thread_lock=True
    )
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
