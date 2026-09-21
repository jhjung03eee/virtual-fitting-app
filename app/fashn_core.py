"""FASHN VTON v1.5 추론 로직 — 앱의 주력 합성 엔진 (2026-09-15 결정, docs/PLAN.md F-16 결정).

CatVTON(`tryon_core.py`)은 Python 3.9 venv 전용이지만 FASHN은 **Python 3.10 이상**이 필요하다
(pyproject requires-python >=3.10). 두 엔진은 한 프로세스에서 같이 못 쓴다.
앱은 FASHN을 쓰고, `tryon_core.py` 는 벤치마크·과거 실험 재현용으로 남긴다.

설치 (Kaggle/Colab 기본 파이썬):
    git clone https://github.com/fashn-AI/fashn-vton-1.5.git <FASHN_REPO>
    pip install -e <FASHN_REPO>
    python <FASHN_REPO>/scripts/download_weights.py --weights-dir <FASHN_WEIGHTS>

torch를 임포트하지 않고도 상수·옷 종류 변환을 쓸 수 있게, 무거운 임포트는 함수 안에 둔다.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FASHN_REPO, FASHN_WEIGHTS  # noqa: E402

# 기본값 근거:
#   - guidance 2.5: 글자 프린트가 가장 또렷했다. guidance가 스텝보다 중요(F-11). FASHN 저장소 기본은 1.5.
#   - 스텝 30: 50스텝과 결과가 사실상 같은데 1.7배 빠르다(F-20, T4 118초 → 68초).
#     같은 인물·옷 3벌·시드 2개로 비교했을 때 평균 픽셀 차이 0.2~0.3, 프린트를 확대해도 구별 불가.
#     CatVTON에서 스텝을 줄였다가 질감이 무너진 적이 있으므로(F-7) 더 줄이려면 반드시 눈으로 확인할 것.
#   - 정밀도: T4는 fp16이 fp32보다 4배 빠르고(50스텝 117초 vs 472초) 결과는 같다(F-13·F-14).
#     FASHN은 T4·P100에서도 bf16을 고르는데, 이 GPU들은 bf16을 흉내만 내서 가장 느리다(T4 430초/30스텝).
#     bf16을 제대로 지원하는 GPU(compute capability 8 이상: L4·A100·H100)에서는 bf16을 그대로 쓴다.
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 2.5
DEFAULT_SEED = 42

# 앱의 옷 종류 → FASHN category. FASHN은 이너/아우터 구분이 없어 상의로 넣는다.
CATEGORY_BY_CLOTH_TYPE = {
    'upper': 'tops',
    'inner': 'tops',
    'outer': 'tops',
    'lower': 'bottoms',
    'overall': 'one-pieces',
}
CLOTH_TYPES = list(CATEGORY_BY_CLOTH_TYPE)
# 'flat-lay': 바닥에 펴 놓고 찍은 상품 사진, 'model': 사람이 입고 찍은 사진
GARMENT_PHOTO_TYPES = ['flat-lay', 'model']

_pipeline = None


def choose_dtype(torch):
    """GPU에 맞는 정밀도. 근거는 위 기본값 주석."""
    if not torch.cuda.is_available():
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def load_models(weights_dir=None):
    """파이프라인을 1회만 로드하고 캐시한다."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    src = os.path.join(FASHN_REPO, 'src')
    if os.path.isdir(src) and src not in sys.path:
        # editable 설치 직후 같은 프로세스에서는 패키지를 못 찾는 경우가 있었다(kaggle_fashn 커밋 e36b8ba)
        sys.path.insert(0, src)
    from fashn_vton import TryOnPipeline

    pipeline = TryOnPipeline(weights_dir=weights_dir or FASHN_WEIGHTS)
    dtype = choose_dtype(torch)
    if pipeline.inference_dtype != dtype:
        # kaggle_fashn_speed·kaggle_fashn_t4x2에서 검증한 방식
        pipeline.inference_dtype = dtype
        pipeline.tryon_model.to(dtype=dtype)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _pipeline = pipeline
    return pipeline


def _to_rgb(x):
    """경로나 PIL 이미지를 EXIF 회전을 적용한 RGB로. 폰 사진은 태그 없이 쓰면 누운 채 합성된다(tryon_core._to_image)."""
    from PIL import Image, ImageOps
    image = Image.open(x) if isinstance(x, str) else x
    return ImageOps.exif_transpose(image).convert('RGB')


def try_on(person, garment, cloth_type='upper', steps=DEFAULT_STEPS, guidance_scale=DEFAULT_GUIDANCE,
           seed=DEFAULT_SEED, garment_photo_type='flat-lay', return_timing=False):
    """인물 사진에 옷을 입힌다.

    person/garment: 파일 경로 또는 PIL.Image
    cloth_type: CLOTH_TYPES 중 하나 (앱의 옷 종류 이름, tryon_core와 같음)
    seed: -1이면 매번 다른 결과
    garment_photo_type: 'flat-lay'(상품 사진) 또는 'model'(착용 사진)

    FASHN은 마스크 없이 사진 전체를 576x768로 다시 그린다. 원래 사진 비율로 되돌려 준다.
    반환: result 또는 return_timing=True면 (result, timing dict)
    """
    if cloth_type not in CATEGORY_BY_CLOTH_TYPE:
        raise ValueError(f'cloth_type must be one of {CLOTH_TYPES}, got {cloth_type!r}')
    if garment_photo_type not in GARMENT_PHOTO_TYPES:
        raise ValueError(f'garment_photo_type must be one of {GARMENT_PHOTO_TYPES}, got {garment_photo_type!r}')
    if seed == -1:
        import random
        seed = random.randint(0, 2 ** 31 - 1)

    pipeline = load_models()
    start = time.perf_counter()
    output = pipeline(
        person_image=_to_rgb(person),
        garment_image=_to_rgb(garment),
        category=CATEGORY_BY_CLOTH_TYPE[cloth_type],
        garment_photo_type=garment_photo_type,
        num_timesteps=int(steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
    )
    result = output.images[0]
    if not return_timing:
        return result
    return result, {'total_s': round(time.perf_counter() - start, 2), 'seed': int(seed),
                    'dtype': str(pipeline.inference_dtype).replace('torch.', '')}
