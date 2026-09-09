"""CatVTON 추론 로직 — Gradio 앱과 벤치마크가 공유하는 단일 소스.

반드시 Python 3.9 venv에서 실행할 것 (detectron2 .so가 cp39 전용).
자세한 이유는 docs/ENVIRONMENT.md 참고.
"""
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CATVTON_REPO as REPO_DIR  # noqa: E402

BASE_MODEL = 'runwayml/stable-diffusion-inpainting'
CATVTON_REPO_ID = 'zhengchong/CatVTON'

WIDTH, HEIGHT = 768, 1024
CLOTH_TYPES = ['upper', 'lower', 'overall', 'inner', 'outer']
SCHEDULERS = ['ddim', 'dpm']

_models = None


def _ensure_repo_on_path():
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    os.chdir(REPO_DIR)


def load_models(device='cuda', mixed_precision='fp16'):
    """파이프라인과 AutoMasker를 1회만 로드하고 캐시한다. T4 기준 약 40초."""
    global _models
    if _models is not None:
        return _models

    _ensure_repo_on_path()

    from huggingface_hub import snapshot_download
    from diffusers.image_processor import VaeImageProcessor
    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker
    from utils import init_weight_dtype

    repo_path = snapshot_download(repo_id=CATVTON_REPO_ID)

    pipeline = CatVTONPipeline(
        base_ckpt=BASE_MODEL,
        attn_ckpt=repo_path,
        attn_ckpt_version='mix',
        weight_dtype=init_weight_dtype(mixed_precision),
        use_tf32=True,
        device=device,
        # 신버전 transformers와 safety checker API가 안 맞음 (docs/ENVIRONMENT.md #5)
        skip_safety_check=True,
    )
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, 'DensePose'),
        schp_ckpt=os.path.join(repo_path, 'SCHP'),
        device=device,
    )
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )

    _models = (pipeline, automasker, mask_processor, device)
    return _models


def _to_image(x):
    return Image.open(x) if isinstance(x, str) else x


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# 기본값 근거 (docs/TEST_RESULTS.md 벤치마크 45건 + 육안 비교):
#   - DDIM은 스텝을 줄이면 형체가 무너진다(4스텝 SSIM 0.895). 스텝만 줄이는 게 아니라
#     샘플러를 DPM++로 바꾸는 것이 핵심이다.
#   - DPM++는 4스텝에서도 30스텝과 실루엣·질감·색이 거의 같다.
#   - CFG는 guidance_scale > 1 일 때만 배치를 2배로 만든다. 끄면 정확히 2배 빨라진다.
# 합쳐서 T4 기준 약 7초 (DDIM 30스텝 73초 대비 10배).
#
# 주의: CFG를 끄면 옷 반영 강도가 약해질 수 있다. 검증한 건 색과 형태가 뚜렷한 옷
# 한 벌뿐이므로, 무늬가 복잡하거나 색이 흐린 옷에서 반영이 약하면
# guidance_scale=2.5로 올리거나 steps를 8로 늘려서 쓴다.
DEFAULT_SCHEDULER = 'dpm'
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 1.0  # 1.0 이하 = CFG 끔


def try_on(person, garment, cloth_type='upper', steps=DEFAULT_STEPS,
           guidance_scale=DEFAULT_GUIDANCE, seed=42, scheduler=DEFAULT_SCHEDULER,
           eta=1.0, return_timing=False):
    """인물 사진에 옷을 합성한다.

    person/garment: 파일 경로 또는 PIL.Image
    cloth_type: CLOTH_TYPES 중 하나
    scheduler: 'ddim'(기본) 또는 'dpm'(DPMSolverMultistep, 적은 스텝에서 유리)
    eta: DDIM 확률성. 1.0이면 DDPM에 가깝고 0.0이면 결정적. DPM++에서는 무시된다.
        (repo 기본값이 1.0이라 그대로 둔다)
    guidance_scale: 1.0 이하면 CFG가 꺼져 배치가 절반 -> 약 2배 빠름

    반환: (result, mask_vis) 또는 return_timing=True면 (result, mask_vis, timing dict)
    """
    if cloth_type not in CLOTH_TYPES:
        raise ValueError(f'cloth_type must be one of {CLOTH_TYPES}, got {cloth_type!r}')
    if scheduler not in SCHEDULERS:
        raise ValueError(f'scheduler must be one of {SCHEDULERS}, got {scheduler!r}')

    pipeline, automasker, mask_processor, device = load_models()

    from model.cloth_masker import vis_mask
    from utils import resize_and_crop, resize_and_padding

    person = resize_and_crop(_to_image(person).convert('RGB'), (WIDTH, HEIGHT))
    garment = resize_and_padding(_to_image(garment).convert('RGB'), (WIDTH, HEIGHT))

    _sync()
    t0 = time.perf_counter()
    mask = automasker(person, cloth_type)['mask']
    mask = mask_processor.blur(mask, blur_factor=9)
    _sync()
    t1 = time.perf_counter()

    original_scheduler = pipeline.noise_scheduler
    if scheduler == 'dpm':
        from diffusers import DPMSolverMultistepScheduler
        pipeline.noise_scheduler = DPMSolverMultistepScheduler.from_config(
            original_scheduler.config
        )

    try:
        generator = torch.Generator(device=device).manual_seed(seed) if seed != -1 else None
        result = pipeline(
            image=person,
            condition_image=garment,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            eta=eta,
        )[0]
        _sync()
    finally:
        pipeline.noise_scheduler = original_scheduler

    t2 = time.perf_counter()

    mask_vis = vis_mask(person, mask)
    if return_timing:
        return result, mask_vis, {
            'mask_s': round(t1 - t0, 2),
            'diffusion_s': round(t2 - t1, 2),
            'total_s': round(t2 - t0, 2),
        }
    return result, mask_vis
