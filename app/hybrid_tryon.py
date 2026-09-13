"""하이브리드 합성: 변형(warping) 초안에서 출발하는 CatVTON.

변형+GAN(HR-VITON)과 diffusion(CatVTON)을 같은 사진에 돌려보니(docs/PLAN.md F-8)
  - HR-VITON의 **변형된 옷 자체는 색·글자 배치가 정확**했지만, 소매가 날개처럼 늘어났고
    마지막 GAN 생성 단계에서 색이 어두워지고 배경이 번졌다.
  - CatVTON은 자연스럽지만 옷을 '다시 그리기' 때문에 글자·프린트가 원본과 달라진다.

그래서 GAN 생성 단계는 버리고, **변형된 옷을 사람 사진에 붙인 초안**을 만든 뒤
CatVTON이 완전한 노이즈가 아니라 그 초안에 노이즈를 일부만 섞은 지점에서 출발해 다듬게 한다
(SDEdit 방식). strength=1이면 초안을 무시하는 원래 CatVTON과 같고, 낮을수록 초안을 따른다.

초안의 옷은 HR-VITON이 예측한 상의 영역 ∧ CatVTON 마스크 안쪽만 쓴다 — 날개처럼 늘어난
소매는 예측 상의 영역 밖이라 여기서 잘린다.

CatVTON 파이프라인 __call__ 을 그대로 따라가되 시작 latent와 timestep만 바꾼다.
DDIM 전용이다(DPM++ 멀티스텝은 중간 timestep에서 시작하는 상태 초기화가 다르다).
"""
import numpy as np
import torch
from PIL import Image, ImageFilter


def make_draft(person, mask, warped_cloth, warp_region, feather=3):
    """사람 사진 위에 변형된 옷을 붙인 초안. 모두 같은 크기의 PIL 이미지."""
    region = np.asarray(warp_region.convert('L'), dtype=np.float32) / 255.0
    inside = np.asarray(mask.convert('L'), dtype=np.float32) / 255.0
    alpha = Image.fromarray((np.clip(region * inside, 0, 1) * 255).astype(np.uint8), 'L')
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(warped_cloth.convert('RGB'), person.convert('RGB'), alpha)


@torch.no_grad()
def run_from_draft(pipeline, image, condition_image, mask, draft, strength=0.7,
                   num_inference_steps=50, guidance_scale=2.5, generator=None, eta=1.0,
                   height=1024, width=768):
    """CatVTONPipeline.__call__ 과 같지만 draft 에서 출발한다."""
    from utils import compute_vae_encodings, numpy_to_pil, prepare_image, prepare_mask_image

    if not 0.0 < strength <= 1.0:
        raise ValueError(f'strength must be in (0, 1], got {strength}')
    concat_dim = -2
    image, condition_image, mask = pipeline.check_inputs(image, condition_image, mask, width, height)
    draft = draft.resize((width, height), Image.LANCZOS)
    device, dtype = pipeline.device, pipeline.weight_dtype

    image = prepare_image(image).to(device, dtype=dtype)
    condition_image = prepare_image(condition_image).to(device, dtype=dtype)
    draft = prepare_image(draft).to(device, dtype=dtype)
    mask = prepare_mask_image(mask).to(device, dtype=dtype)

    masked_latent = compute_vae_encodings(image * (mask < 0.5), pipeline.vae)
    condition_latent = compute_vae_encodings(condition_image, pipeline.vae)
    draft_latent = compute_vae_encodings(draft, pipeline.vae)
    mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode='nearest')

    masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
    mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
    # 원래는 여기서 순수 노이즈. 하이브리드는 [초안 ; 옷] latent 에 해당 시점만큼 노이즈를 섞는다.
    clean_concat = torch.cat([draft_latent, condition_latent], dim=concat_dim)

    scheduler = pipeline.noise_scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    start = min(int(round(num_inference_steps * (1.0 - strength))), num_inference_steps - 1)
    timesteps = scheduler.timesteps[start:]
    noise = torch.randn(clean_concat.shape, generator=generator, device=device, dtype=dtype)
    if strength >= 1.0:
        latents = noise * scheduler.init_noise_sigma
    else:
        latents = scheduler.add_noise(clean_concat, noise, timesteps[:1])

    do_cfg = guidance_scale > 1.0
    if do_cfg:
        masked_latent_concat = torch.cat([
            torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
            masked_latent_concat,
        ])
        mask_latent_concat = torch.cat([mask_latent_concat] * 2)

    extra = pipeline.prepare_extra_step_kwargs(generator, eta)
    for t in timesteps:
        model_input = torch.cat([latents] * 2) if do_cfg else latents
        model_input = scheduler.scale_model_input(model_input, t)
        model_input = torch.cat([model_input, mask_latent_concat, masked_latent_concat], dim=1)
        noise_pred = pipeline.unet(model_input, t.to(device), encoder_hidden_states=None, return_dict=False)[0]
        if do_cfg:
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (cond - uncond)
        latents = scheduler.step(noise_pred, t, latents, **extra).prev_sample

    latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
    decoded = pipeline.vae.decode((latents / pipeline.vae.config.scaling_factor).to(device, dtype=dtype)).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    return numpy_to_pil(decoded)[0]
