"""CatVTON 추론 로직 — Gradio 앱과 배치 워커가 공유하는 단일 소스.

반드시 Python 3.9 venv에서 실행할 것 (detectron2 .so가 cp39 전용).
자세한 이유는 docs/ENVIRONMENT.md 참고.
"""
import os
import sys

import torch
from PIL import Image

REPO_DIR = os.environ.get('CATVTON_REPO', '/content/scratch/CatVTON')
BASE_MODEL = 'runwayml/stable-diffusion-inpainting'
CATVTON_REPO_ID = 'zhengchong/CatVTON'

WIDTH, HEIGHT = 768, 1024
CLOTH_TYPES = ['upper', 'lower', 'overall', 'inner', 'outer']

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


def try_on(person, garment, cloth_type='upper', steps=30, guidance_scale=2.5, seed=42):
    """인물 사진에 옷을 합성한다.

    person/garment: 파일 경로 또는 PIL.Image
    cloth_type: CLOTH_TYPES 중 하나
    반환: (합성 결과, 마스크를 얹은 인물 이미지)
    """
    if cloth_type not in CLOTH_TYPES:
        raise ValueError(f'cloth_type must be one of {CLOTH_TYPES}, got {cloth_type!r}')

    pipeline, automasker, mask_processor, device = load_models()

    from model.cloth_masker import vis_mask
    from utils import resize_and_crop, resize_and_padding

    person = resize_and_crop(_to_image(person).convert('RGB'), (WIDTH, HEIGHT))
    garment = resize_and_padding(_to_image(garment).convert('RGB'), (WIDTH, HEIGHT))

    mask = automasker(person, cloth_type)['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    generator = torch.Generator(device=device).manual_seed(seed) if seed != -1 else None
    result = pipeline(
        image=person,
        condition_image=garment,
        mask=mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )[0]

    return result, vis_mask(person, mask)
