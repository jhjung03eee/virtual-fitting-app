"""CatVTON 추론 로직 — Gradio 앱과 벤치마크가 공유하는 단일 소스.

반드시 Python 3.9 venv에서 실행할 것 (detectron2 .so가 cp39 전용).
자세한 이유는 docs/ENVIRONMENT.md 참고.
"""
import os
import sys
import time

import torch
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CATVTON_REPO as REPO_DIR  # noqa: E402

BASE_MODEL = 'runwayml/stable-diffusion-inpainting'
CATVTON_REPO_ID = 'zhengchong/CatVTON'

WIDTH, HEIGHT = 768, 1024
CLOTH_TYPES = ['upper', 'lower', 'overall', 'inner', 'outer']
SCHEDULERS = ['ddim', 'dpm']

_models = None
_models_key = None


def _ensure_repo_on_path():
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    os.chdir(REPO_DIR)


def load_models(device='cuda', mixed_precision='fp16',
                compile_model=False, channels_last=False):
    """파이프라인과 AutoMasker를 1회만 로드하고 캐시한다. T4 기준 약 40초.

    compile_model / channels_last 는 **스텝당 비용**을 줄이는 옵션이다.
    샘플링 수학을 건드리지 않으므로 결과가 달라지지 않아야 한다 —
    스텝을 줄이는 것과 달리 품질 손실이 원리상 없다.

      compile_model: torch.compile 로 UNet/VAE 커널을 융합한다. 첫 호출에서
        컴파일하느라 수 분이 걸리므로, 서버처럼 한 번 띄워두고 쓰는 경우에만 이득이다.
      channels_last: NHWC 메모리 배치. conv 위주 모델에서 텐서코어 활용이 좋아진다.

    설정이 바뀌면 다시 로드한다 (같은 프로세스에서 여러 설정을 비교할 수 있어야 한다).
    """
    global _models, _models_key
    key = (device, mixed_precision, compile_model, channels_last)
    if _models is not None and _models_key == key:
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
        compile=compile_model,
    )
    if channels_last:
        pipeline.unet = pipeline.unet.to(memory_format=torch.channels_last)
        pipeline.vae = pipeline.vae.to(memory_format=torch.channels_last)
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, 'DensePose'),
        schp_ckpt=os.path.join(repo_path, 'SCHP'),
        device=device,
    )
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )

    _models = (pipeline, automasker, mask_processor, device)
    _models_key = key
    return _models


def _to_image(x):
    """경로나 PIL 이미지를 받아 **EXIF 회전을 적용한** 이미지로 돌려준다.

    폰으로 찍은 사진은 센서 방향 그대로 가로로 저장하고 EXIF orientation 태그로
    "세로로 보여라"라고 표시한다. PIL의 Image.open()은 이 태그를 적용하지 않으므로
    그냥 쓰면 **인물이 옆으로 누운 채 합성된다.** 사용자가 올리는 사진은 대부분
    폰 사진이라 반드시 처리해야 한다.
    """
    image = Image.open(x) if isinstance(x, str) else x
    return ImageOps.exif_transpose(image)


def _garment_pixels(garment):
    """옷 사진에서 옷에 해당하는 픽셀만 (N,3) 배열로. 흰 배경은 뺀다."""
    import numpy as np
    array = np.asarray(garment.convert('RGB'), dtype=np.float64)
    foreground = array.sum(axis=2) < 720  # 거의 흰색인 배경 제외
    if foreground.mean() < 0.05:          # 옷이 온통 흰색이면 전부 쓴다
        foreground = np.ones(array.shape[:2], dtype=bool)
    return array[foreground]


def match_garment_color(result, person, mask, garment, strength=1.0):
    """생성된 옷 영역의 **색조**를 원본 옷 사진에 맞춘다.

    확산 설정을 아무리 만져도 하의 색이 원본과 달랐다 — 다크 네이비 코듀로이가
    밝은 워싱 데님으로 나온다. 그런데 **정답 색은 우리가 알고 있다.** 옷 사진이
    입력으로 들어와 있기 때문이다. 그러면 생성 결과를 그 색에 맞추면 된다.

    LAB 색공간에서 중앙값과 산포를 맞춘다. 평균/표준편차 대신 **중앙값과 MAD**를
    쓰는 이유는, 마스크 안에 피부(반팔의 팔)나 그림자가 섞여도 통계가 덜 끌려가기
    때문이다. 밝기(L)의 산포는 건드리지 않는다 — 주름과 음영이 거기 들어있어서
    스케일을 바꾸면 옷이 평평해진다. **색조(a·b)만 옮기고 명암은 유지한다.**

    strength: 0이면 그대로, 1이면 완전히 맞춘다.
    """
    import numpy as np
    try:
        from skimage.color import rgb2lab, lab2rgb
    except ImportError:
        return result  # skimage 없으면 조용히 건너뛴다

    selected = np.asarray(mask.convert('L')) > 127
    if not selected.any():
        return result

    generated = np.asarray(result.convert('RGB'), dtype=np.float64) / 255.0
    lab = rgb2lab(generated)

    target_lab = rgb2lab(_garment_pixels(garment).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)

    def robust(values):
        median = np.median(values, axis=0)
        spread = np.median(np.abs(values - median), axis=0) + 1e-6
        return median, spread

    source_median, source_spread = robust(lab[selected])
    target_median, target_spread = robust(target_lab)

    adjusted = lab[selected].copy()
    # L: 중앙값만 옮기고 산포는 유지 (주름·음영 보존)
    adjusted[:, 0] += (target_median[0] - source_median[0]) * strength
    # a, b: 중앙값과 산포를 모두 맞춘다 (색조가 목표)
    for channel in (1, 2):
        scaled = ((adjusted[:, channel] - source_median[channel])
                  * (target_spread[channel] / source_spread[channel])
                  + target_median[channel])
        adjusted[:, channel] += (scaled - adjusted[:, channel]) * strength

    lab[selected] = adjusted
    corrected = np.clip(lab2rgb(lab) * 255.0, 0, 255).astype('uint8')
    return Image.fromarray(corrected)


def person_area_mask(parsed, grow_px=0):
    """SCHP 파싱 결과에서 인물 영역만 뽑는다. 흰색(255)이 사람.

    ATR/LIP 두 파싱 모두 **0번 라벨이 배경**이므로 0이 아닌 곳이 사람이다.
    둘을 OR로 합치면 한쪽이 놓친 부분(머리카락 끝, 신발 등)을 서로 메운다.
    AutoMasker가 이미 돌린 파싱을 재사용하므로 추가 비용이 없다.
    """
    import numpy as np
    from PIL import ImageFilter

    def flatten(image):
        array = np.squeeze(np.array(image))
        return array[..., 0] if array.ndim == 3 else array

    atr = flatten(parsed['schp_atr'])
    lip = flatten(parsed['schp_lip'])
    area = (atr != 0)
    if lip.shape == atr.shape:
        area |= (lip != 0)

    # 파싱이 실패하면 인물 영역이 터무니없이 작거나 커진다. 그 마스크로
    # 배경을 지우면 사람이 잘려나가거나 배경이 그대로 남는다. 그럴 바에는
    # 배경 정규화를 건너뛰는 게 낫다.
    fraction = float(area.mean())
    if not 0.10 <= fraction <= 0.90:
        return None

    mask = Image.fromarray((area * 255).astype('uint8'), mode='L')

    # SCHP 실루엣이 몸에 딱 붙어 있으면, 팔처럼 가느다란 부위 바로 옆까지
    # 흰 배경이 닿는다. 그러면 모델이 그 흰색을 옷 안으로 끌고 들어와
    # 소매에 흰 얼룩이 생긴다(실제로 긴팔 셔츠에서 관찰됨).
    # 인물 영역을 몇 픽셀 넓혀 흰색을 옷에서 떼어놓는다.
    if grow_px > 0:
        size = grow_px * 2 + 1
        mask = mask.filter(ImageFilter.MaxFilter(size if size % 2 else size + 1))

    # 실루엣 경계가 계단처럼 되지 않게 아주 약하게만 흐린다
    return mask.filter(ImageFilter.GaussianBlur(2))


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# 기본값 근거 (docs/TEST_RESULTS.md 벤치마크 45건 + 실제 상품 3벌 검증):
#   - DDIM은 스텝을 줄이면 형체가 무너진다(4스텝 SSIM 0.895). 스텝만 줄이는 게 아니라
#     샘플러를 DPM++로 바꾸는 것이 핵심이다.
#   - DPM++ 8스텝은 30스텝과 눈으로 구별하기 어려우면서 3.5배 빠르다.
#
# CFG를 끄면(guidance_scale=1.0) 배치가 절반이 되어 정확히 2배 빨라지지만,
# **옷이 무너진다.** 실제 상품으로 확인한 결과:
#   - 무지 코듀로이 바지: 멀쩡함
#   - 큰 영문 텍스트 맨투맨: 글자가 흐려짐
#   - 사진 프린트 티셔츠: 옷이 통째로 뭉개져 알아볼 수 없는 무늬가 됨
# 데모 카디건 한 벌로만 보고 CFG를 껐던 것이 잘못이었다. CFG는 켜 둔다.
#
# 스텝도 4로 줄이면 사진 프린트가 작은 얼룩으로 쪼그라든다. 8스텝은 유지된다.
DEFAULT_SCHEDULER = 'dpm'
DEFAULT_STEPS = 8
DEFAULT_GUIDANCE = 2.5  # 1.0 이하면 CFG가 꺼져 2배 빨라지지만 옷이 무너진다

# 부위별 guidance. 실제 사진 실험에서 상의와 하의가 갈렸다.
#   상의: 2.5가 최선. 5.0으로 올리면 옷 가장자리가 지저분해지고 프린트가 깨진다.
#   하의: 2.5에서 **색이 아예 틀렸다** — 네이비 코듀로이가 크림색으로 나왔고
#         인물 3명 모두 그랬다. 5.0으로 올리니 실제 데님 색이 나왔다.
# 근거가 하의 한 벌(인물 3명)뿐이므로, 다른 바지에서 이상하면 다시 볼 것.
DEFAULT_GUIDANCE_BY_TYPE = {'lower': 5.0}


def default_guidance(cloth_type):
    return DEFAULT_GUIDANCE_BY_TYPE.get(cloth_type, DEFAULT_GUIDANCE)


def try_on(person, garment, cloth_type='upper', steps=DEFAULT_STEPS,
           guidance_scale=None, seed=42, scheduler=DEFAULT_SCHEDULER,
           eta=1.0, return_timing=False, composite=True,
           normalize_background=False, background_grow_px=0, color_match=0.0):
    """인물 사진에 옷을 합성한다.

    person/garment: 파일 경로 또는 PIL.Image
    cloth_type: CLOTH_TYPES 중 하나
    scheduler: 'ddim'(기본) 또는 'dpm'(DPMSolverMultistep, 적은 스텝에서 유리)
    eta: DDIM 확률성. 1.0이면 DDPM에 가깝고 0.0이면 결정적. DPM++에서는 무시된다.
        (repo 기본값이 1.0이라 그대로 둔다)
    guidance_scale: None이면 부위별 기본값(상의 2.5 / 하의 5.0).
        1.0 이하면 CFG가 꺼져 배치가 절반 -> 약 2배 빠르지만 옷이 무너진다
    composite: 마스크 밖을 원본 사진으로 되돌린다. 아래 주석 참고
    normalize_background: 인물만 오려 흰 배경에 올린 뒤 합성한다(배경 정규화).
        학습 데이터(VITON-HD/DressCode)가 전부 흰 배경 스튜디오 촬영이라,
        복도 같은 배경이 들어간 사진은 분포 밖이 되어 성공률이 떨어진다.
        효과가 옷에 따라 갈린다. 사진 프린트 티셔츠는 성공률이 오르지만
        (3개 시드 중 1개 -> 3개), **긴팔 셔츠는 소매에 흰 얼룩이 생겨 나빠진다.**
        팔이 몸통에서 떨어져 있으면 팔 옆의 흰 배경이 소매 안으로 번진다.
        그래서 기본값은 꺼 두고, background_grow_px 로 실루엣을 넓혀
        흰색을 옷에서 떼어놓는 방법을 검증 중이다.
        인물 파싱이 실패하면 자동으로 건너뛴다.
    background_grow_px: 배경 정규화 시 인물 영역을 몇 픽셀 넓힐지
    color_match: 0~1. 생성된 옷 영역의 색조를 원본 옷 사진에 맞춘다(후처리).
        확산 설정으로 해결되지 않던 색조 차이를 직접 잡는다. 1이면 완전히 맞춘다

    반환: (result, mask_vis) 또는 return_timing=True면 (result, mask_vis, timing dict)
    """
    if cloth_type not in CLOTH_TYPES:
        raise ValueError(f'cloth_type must be one of {CLOTH_TYPES}, got {cloth_type!r}')
    if scheduler not in SCHEDULERS:
        raise ValueError(f'scheduler must be one of {SCHEDULERS}, got {scheduler!r}')
    if guidance_scale is None:
        guidance_scale = default_guidance(cloth_type)

    pipeline, automasker, mask_processor, device = load_models()

    from model.cloth_masker import vis_mask
    from utils import resize_and_crop, resize_and_padding

    person = resize_and_crop(_to_image(person).convert('RGB'), (WIDTH, HEIGHT))
    garment = resize_and_padding(_to_image(garment).convert('RGB'), (WIDTH, HEIGHT))

    _sync()
    t0 = time.perf_counter()
    # AutoMasker.__call__ 대신 파싱을 직접 받는다. 배경 정규화에도 같은 파싱이
    # 필요한데, __call__을 쓰면 densepose+SCHP를 두 번 돌리게 된다.
    parsed = automasker.preprocess_image(person)
    mask = automasker.cloth_agnostic_mask(
        parsed['densepose'], parsed['schp_lip'], parsed['schp_atr'], part=cloth_type)
    mask = mask_processor.blur(mask, blur_factor=9)

    model_input = person
    person_area = None
    if normalize_background:
        # 학습 데이터(VITON-HD/DressCode)는 전부 흰 배경 스튜디오 촬영이다.
        # 복도·패턴 바닥이 들어간 사진은 분포 밖이라 성공률이 떨어진다.
        # 인물만 오려 흰 배경에 올려서 입력을 학습 분포 쪽으로 민다.
        # 원래 배경은 마지막 합성에서 되돌아온다(composite가 원본 person을 쓴다).
        person_area = person_area_mask(parsed, grow_px=background_grow_px)
        if person_area is not None:
            model_input = Image.composite(
                person, Image.new('RGB', person.size, 'white'), person_area)
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
            image=model_input,
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

    if color_match > 0:
        # 확산 설정으로는 옷 색조를 맞추지 못했다(다크 네이비 -> 밝은 워싱).
        # 정답 색은 입력 옷 사진에 있으므로 후처리로 맞춘다. 합성 전에 적용해
        # 마스크 경계에서 피부까지 물들지 않게 한다.
        if result.size != person.size:
            result = result.resize(person.size, Image.LANCZOS)
        result = match_garment_color(result, person, mask, garment, strength=color_match)

    if composite:
        # CatVTON 파이프라인은 latent 전체를 디코딩해 돌려준다. 즉 **마스크 밖도
        # 다시 생성된다.** 얼굴·손·배경이 VAE를 왕복하며 뭉개지고, guidance를
        # 올리면 그 열화가 눈에 띄게 커진다 — 실제 사진에서 g5.0 이상이면 얼굴이
        # 일그러지고 배경이 물감처럼 번졌다.
        #
        # 인페인팅에서는 마스크 밖을 원본으로 되돌리는 것이 표준이다. 그렇게 하면
        # 바뀌는 곳이 옷 영역뿐이므로, 옷 반영을 강하게 주면서도 얼굴과 배경을
        # 지킬 수 있다. 마스크가 blur되어 있어 경계는 자연스럽게 섞인다.
        if result.size != person.size:
            result = result.resize(person.size, Image.LANCZOS)

        blend = mask.convert('L')
        if person_area is not None:
            # 옷 마스크는 blur 때문에 인물 실루엣보다 바깥으로 번져 있다.
            # 흰 배경으로 생성했으므로 그 띠에는 흰색이 들어있고, 그대로
            # 합성하면 인물 주위에 흰 후광이 남는다. 인물 영역과 교집합을
            # 취해 실루엣 안쪽만 바꾼다.
            from PIL import ImageChops
            blend = ImageChops.multiply(blend, person_area)
        result = Image.composite(result, person, blend)

    mask_vis = vis_mask(person, mask)
    if return_timing:
        return result, mask_vis, {
            'mask_s': round(t1 - t0, 2),
            'diffusion_s': round(t2 - t1, 2),
            'total_s': round(t2 - t0, 2),
        }
    return result, mask_vis
