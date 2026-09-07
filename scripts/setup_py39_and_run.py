import os, sys, subprocess, textwrap, time

t0 = time.time()
def sh(cmd):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tail = (r.stdout or '')[-1500:] + (r.stderr or '')[-1500:]
    print(tail, flush=True)
    print(f'   -> exit {r.returncode}  ({time.time()-t0:.0f}s elapsed)', flush=True)
    return r.returncode

REPO_DIR = '/content/scratch/CatVTON'
VENV = '/content/venv39'
PY = VENV + '/bin/python'

os.makedirs('/content/scratch', exist_ok=True)
if not os.path.exists(REPO_DIR):
    sh('cd /content/scratch && git clone -q https://github.com/Zheng-Chong/CatVTON.git')

# 1) repo가 요구하는 Python 3.9 설치 (detectron2 .so가 cp39 전용이라 필수)
if not os.path.exists('/usr/bin/python3.9'):
    sh('add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1')
    sh('apt-get install -y -qq python3.9 python3.9-venv python3.9-dev > /dev/null 2>&1')
print('python3.9 present:', os.path.exists('/usr/bin/python3.9'), flush=True)

# 2) 격리된 venv 구성
if not os.path.exists(PY):
    sh('python3.9 -m venv ' + VENV)
    sh(PY + ' -m pip install -q --upgrade pip')

# 3) repo가 고정한 torch 2.1.2 (cu121; T4=sm_75 지원됨)
sh(PY + ' -m pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121')

# 4) 나머지 의존성 + detectron2 런타임 의존성(fvcore 등, repo requirements에 빠져있음)
sh(PY + ' -m pip install -q accelerate==0.31.0 diffusers==0.29.2 huggingface_hub==0.23.4 '
   'transformers==4.27.3 numpy==1.26.4 opencv-python==4.10.0.84 pillow==10.3.0 PyYAML==6.0.1 '
   'scipy==1.13.1 scikit-image==0.24.0 tqdm==4.66.4 matplotlib==3.9.1 '
   'fvcore iopath pycocotools omegaconf hydra-core termcolor yacs tabulate cloudpickle')

# 5) 워커 스크립트 작성 (AutoMasker 원본 그대로 사용)
worker = textwrap.dedent('''
    import os, sys, glob, torch
    REPO_DIR = '/content/scratch/CatVTON'
    os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)
    print('python:', sys.version)
    print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| arch:', torch.cuda.get_arch_list()[:5])

    from huggingface_hub import snapshot_download
    repo_path = snapshot_download(repo_id='zhengchong/CatVTON')
    print('weights:', repo_path)

    from diffusers.image_processor import VaeImageProcessor
    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker, vis_mask
    from utils import init_weight_dtype, resize_and_crop, resize_and_padding
    from PIL import Image

    WIDTH, HEIGHT = 768, 1024
    pipeline = CatVTONPipeline(
        base_ckpt='runwayml/stable-diffusion-inpainting',
        attn_ckpt=repo_path,
        attn_ckpt_version='mix',
        weight_dtype=init_weight_dtype('fp16'),
        use_tf32=True,
        device='cuda',
        skip_safety_check=True,
    )
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, 'DensePose'),
        schp_ckpt=os.path.join(repo_path, 'SCHP'),
        device='cuda',
    )
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False,
                                       do_binarize=True, do_convert_grayscale=True)
    print('pipeline + automasker ready', flush=True)

    person_path = sorted(glob.glob('resource/demo/example/person/men/*'))[0]
    garment_path = sorted(glob.glob('resource/demo/example/condition/upper/*'))[0]
    print('person:', person_path, '| garment:', garment_path)

    person = resize_and_crop(Image.open(person_path).convert('RGB'), (WIDTH, HEIGHT))
    garment = resize_and_padding(Image.open(garment_path).convert('RGB'), (WIDTH, HEIGHT))
    mask = automasker(person, 'upper')['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    result = pipeline(
        image=person, condition_image=garment, mask=mask,
        num_inference_steps=30, guidance_scale=2.5,
        generator=torch.Generator(device='cuda').manual_seed(42),
    )[0]

    out = '/content/outputs39'
    os.makedirs(out, exist_ok=True)
    result.save(out + '/result.png')
    person.save(out + '/person.png')
    garment.save(out + '/garment.png')
    vis_mask(person, mask).save(out + '/mask.png')
    print('WORKER DONE ->', out, flush=True)
''')
with open('/content/worker39.py', 'w') as f:
    f.write(worker)

# 6) py3.9 환경에서 실행
print('=== running worker in python3.9 venv ===', flush=True)
rc = subprocess.run([PY, '/content/worker39.py'], capture_output=True, text=True)
print(rc.stdout[-4000:], flush=True)
if rc.returncode != 0:
    print('--- STDERR ---', flush=True)
    print(rc.stderr[-4000:], flush=True)
print('exit code:', rc.returncode, f'| total {time.time()-t0:.0f}s', flush=True)

if os.path.exists('/content/outputs39/result.png'):
    from IPython.display import display
    from PIL import Image as PILImage
    display(PILImage.open('/content/outputs39/mask.png'))
    display(PILImage.open('/content/outputs39/result.png'))
