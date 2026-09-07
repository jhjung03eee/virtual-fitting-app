import os
import glob
import sys

print('python:', sys.version)

os.system('nvidia-smi')

import torch
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())

WORK_DIR = '/kaggle/working'  # 결과 이미지만 여기 저장 (kernel output으로 잡힘)
SCRATCH_DIR = '/kaggle/tmp'   # 저장소 clone은 여기에 (output에 안 잡혀서 다운로드가 안 무거워짐)
os.makedirs(SCRATCH_DIR, exist_ok=True)
REPO_DIR = os.path.join(SCRATCH_DIR, 'CatVTON')
os.chdir(SCRATCH_DIR)

if not os.path.exists(REPO_DIR):
    os.system('git clone -q https://github.com/Zheng-Chong/CatVTON.git')
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print('cwd:', os.getcwd())

# Kaggle 베이스 이미지에 이미 최신 accelerate/transformers/huggingface_hub가 깔려 있고
# 서로 버전이 맞물려 있어서, 여기서 구버전으로 강제로 낮추면(예: huggingface_hub==0.23.4)
# 오히려 그 위에 얹힌 transformers/accelerate가 깨진다 (is_offline_mode ImportError 등).
# diffusers만 없을 수 있으므로 그것만 확인 설치하고, 나머지는 이미지에 있는 버전을 그대로 쓴다.
os.system('pip install -q --no-deps diffusers==0.29.2')

from huggingface_hub import snapshot_download

BASE_MODEL = 'runwayml/stable-diffusion-inpainting'
repo_path = snapshot_download(repo_id='zhengchong/CatVTON')
print('CatVTON weights:', repo_path)

from diffusers.image_processor import VaeImageProcessor
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
from PIL import Image, ImageDraw

# GPU가 잡혀도 이 torch 빌드가 그 compute capability를 지원 안 할 수 있음
# (예: Kaggle P100은 sm_60인데 최신 torch 휠은 Pascal 지원을 뺀 경우가 있음).
# 그런 경우 자동으로 CPU로 내려가서 최소 동작 확인만 한다.
DEVICE = 'cpu'
if torch.cuda.is_available():
    try:
        major, minor = torch.cuda.get_device_capability()
        arch_list = torch.cuda.get_arch_list()  # e.g. ['sm_70', 'sm_75', ...]
        cap_str = f'sm_{major}{minor}'
        if any(a.endswith(f'{major}{minor}') for a in arch_list) or major >= 7:
            DEVICE = 'cuda'
        else:
            print(f'GPU capability {cap_str} not in supported arch list {arch_list} -> falling back to CPU')
    except Exception as e:
        print('GPU capability check failed, falling back to CPU:', e)

print('using device:', DEVICE)

WIDTH, HEIGHT = 768, 1024
MIXED_PRECISION = 'no' if DEVICE == 'cpu' else 'fp16'  # Kaggle T4/P100은 bf16 미지원, CPU는 fp32만
STEPS = 30 if DEVICE == 'cuda' else 4  # CPU면 시간이 오래 걸리므로 step을 크게 줄여 동작만 확인

pipeline = CatVTONPipeline(
    base_ckpt=BASE_MODEL,
    attn_ckpt=repo_path,
    attn_ckpt_version='mix',
    weight_dtype=init_weight_dtype(MIXED_PRECISION),
    use_tf32=True,
    device=DEVICE,
    skip_safety_check=True,  # NSFW 세이프티체커가 신버전 transformers와 API 안 맞아서 스킵
)
print('pipeline ready')


def make_rect_mask(person_img, cloth_type='upper'):
    """AutoMasker(DensePose/SCHP)는 detectron2가 Python 3.9 전용 컴파일 확장이라
    이 환경에서 못 씀. 1차 스모크 테스트용으로 대략적인 사각형 마스크로 대체."""
    w, h = person_img.size
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    boxes = {
        'upper': (0.15, 0.10, 0.85, 0.55),
        'inner': (0.15, 0.10, 0.85, 0.55),
        'outer': (0.12, 0.08, 0.88, 0.60),
        'lower': (0.15, 0.45, 0.85, 0.95),
        'overall': (0.10, 0.08, 0.90, 0.95),
    }
    l, t, r, b = boxes.get(cloth_type, boxes['upper'])
    draw.rectangle((l * w, t * h, r * w, b * h), fill=255)
    return mask


def try_on(person, garment, cloth_type='upper', steps=STEPS, guidance_scale=2.5, seed=42):
    if isinstance(person, str):
        person = Image.open(person)
    if isinstance(garment, str):
        garment = Image.open(garment)
    person = resize_and_crop(person.convert('RGB'), (WIDTH, HEIGHT))
    garment = resize_and_padding(garment.convert('RGB'), (WIDTH, HEIGHT))

    mask = make_rect_mask(person, cloth_type)

    generator = torch.Generator(device=DEVICE).manual_seed(seed) if seed != -1 else None
    result = pipeline(
        image=person,
        condition_image=garment,
        mask=mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )[0]
    return result, person, garment, mask


out_dir = os.path.join(WORK_DIR, 'outputs')
os.makedirs(out_dir, exist_ok=True)

person_candidates = sorted(glob.glob('resource/demo/example/person/men/*'))
garment_candidates = sorted(glob.glob('resource/demo/example/condition/upper/*'))
print('persons found:', person_candidates)
print('garments found:', garment_candidates)

person_path = person_candidates[0]
garment_path = garment_candidates[0]

result, person, garment, mask = try_on(person_path, garment_path, cloth_type='upper')
result.save(os.path.join(out_dir, 'result_00.png'))
person.save(os.path.join(out_dir, 'person_00.png'))
garment.save(os.path.join(out_dir, 'garment_00.png'))
mask.save(os.path.join(out_dir, 'mask_00.png'))

print('DONE. saved to', out_dir)
print(os.listdir(out_dir))
