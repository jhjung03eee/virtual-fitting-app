"""가상 피팅 웹 데모.

Python 3.9 venv에서 실행:
    MPLBACKEND=Agg /content/venv39/bin/python app/gradio_app.py
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from tryon_core import try_on, load_models, REPO_DIR

CLOTH_LABELS = {
    '상의': 'upper',
    '하의': 'lower',
    '원피스 (상하 일체)': 'overall',
    '이너': 'inner',
    '아우터': 'outer',
}


def run(person, garment, cloth_label, steps, guidance_scale, seed):
    if person is None or garment is None:
        raise gr.Error('인물 사진과 옷 사진을 모두 올려주세요.')

    result, mask_vis = try_on(
        person=person,
        garment=garment,
        cloth_type=CLOTH_LABELS[cloth_label],
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
    )
    return result, mask_vis


def _examples():
    """저장소 데모 이미지로 예시 조합을 만든다. 없으면 빈 리스트."""
    persons = sorted(glob.glob(os.path.join(REPO_DIR, 'resource/demo/example/person/men/*')))
    uppers = sorted(glob.glob(os.path.join(REPO_DIR, 'resource/demo/example/condition/upper/*')))
    overalls = sorted(glob.glob(os.path.join(REPO_DIR, 'resource/demo/example/condition/overall/*')))
    if not persons:
        return []

    rows = []
    if uppers:
        rows.append([persons[0], uppers[0], '상의'])
    if overalls:
        rows.append([persons[0], overalls[0], '하의'])
    if len(persons) > 1 and uppers:
        rows.append([persons[1], uppers[1 % len(uppers)], '상의'])
    return rows


with gr.Blocks(title='사이즈 반영 가상 피팅') as demo:
    gr.Markdown(
        '# 가상 피팅 데모\n'
        '인물 사진과 옷 사진을 올리고 옷 종류를 고르면 합성 결과가 나옵니다. '
        '옷 영역 마스크는 자동으로 잡습니다 (DensePose + SCHP).\n\n'
        '- 인물은 **정면 전신, 정자세** 사진일수록 결과가 좋습니다\n'
        '- T4 GPU 기준 한 장에 약 70초 걸립니다'
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
            with gr.Accordion('고급 설정', open=False):
                steps_in = gr.Slider(10, 50, value=30, step=1, label='추론 스텝 (높을수록 품질↑ 속도↓)')
                guidance_in = gr.Slider(1.0, 7.5, value=2.5, step=0.1, label='guidance scale')
                seed_in = gr.Number(value=42, precision=0, label='시드 (-1이면 매번 랜덤)')
            run_btn = gr.Button('피팅 해보기', variant='primary')

        with gr.Column():
            result_out = gr.Image(label='합성 결과', height=520)
            with gr.Accordion('자동 생성된 마스크 (결과가 이상할 때 확인용)', open=False):
                mask_out = gr.Image(label='마스크', height=400)

    run_btn.click(
        fn=run,
        inputs=[person_in, garment_in, cloth_in, steps_in, guidance_in, seed_in],
        outputs=[result_out, mask_out],
    )

    examples = _examples()
    if examples:
        gr.Examples(examples=examples, inputs=[person_in, garment_in, cloth_in], label='예시')


if __name__ == '__main__':
    print('모델 로딩 중... (최초 1회, 약 40초)', flush=True)
    load_models()
    print('로딩 완료. Gradio 실행합니다.', flush=True)
    demo.queue().launch(share=True, show_error=True)
