import os, sys, glob
print('python:', sys.version)
os.system('nvidia-smi')
import torch
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())

WORK_DIR = '/content'
SCRATCH_DIR = '/content/scratch'
os.makedirs(SCRATCH_DIR, exist_ok=True)
REPO_DIR = os.path.join(SCRATCH_DIR, 'CatVTON')
os.chdir(SCRATCH_DIR)
if not os.path.exists(REPO_DIR):
    os.system('git clone -q https://github.com/Zheng-Chong/CatVTON.git')
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print('cwd:', os.getcwd())

os.system('pip install -q --no-deps diffusers==0.29.2')

from huggingface_hub import snapshot_download
BASE_MODEL = 'runwayml/stable-diffusion-inpainting'
repo_path = snapshot_download(repo_id='zhengchong/CatVTON')
print('CatVTON weights:', repo_path)

from diffusers.image_processor import VaeImageProcessor
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
from PIL import Image, ImageDraw

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WIDTH, HEIGHT = 768, 1024
MIXED_PRECISION = 'fp16' if DEVICE == 'cuda' else 'no'
STEPS = 30

print('using device:', DEVICE)

pipeline = CatVTONPipeline(
    base_ckpt=BASE_MODEL,
    attn_ckpt=repo_path,
    attn_ckpt_version='mix',
    weight_dtype=init_weight_dtype(MIXED_PRECISION),
    use_tf32=True,
    device=DEVICE,
    skip_safety_check=True,
)
print('pipeline ready')

def make_rect_mask(person_img, cloth_type='upper'):
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

from IPython.display import display
display(result)
