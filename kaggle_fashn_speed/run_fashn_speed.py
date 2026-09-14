"""Kaggle 커널 진입점 — FASHN VTON v1.5 속도 원인 가르기 (한 장 기준).

F-10에서 P100 장당 177초·메모리 15.5GB였다. 로그상 원인 후보:
  1. P100은 bfloat16을 흉내만 내는데 FASHN이 is_bf16_supported()=True 를 보고 bfloat16을 골랐다
  2. 포즈 인식(DWPose, onnxruntime)이 CUDA 연결 실패로 CPU에서 돌았다
설정별로 같은 입력 한 장을 돌려 시간·최대 메모리·결과 이미지를 남긴다.
GPU 종류는 `kaggle kernels push --accelerator NvidiaTeslaT4` 처럼 올릴 때 고른다.
50스텝(현재 기본값, F-11)은 P100에서 30스텝으로 추정만 했으므로 여기서 직접 잰다.
"""
import os
import subprocess
import sys
import time

WORK = '/kaggle/tmp/fashn'
REPO = os.path.join(WORK, 'fashn-vton-1.5')
WEIGHTS = os.path.join(WORK, 'weights')
OUT = '/kaggle/working/fashn_speed'
PERSON = 'kakao_front_upper.jpg'
GARMENT = 'garment_tee_khaki.jpg'


def run(cmd):
    print('\n>>>', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {cmd}')


os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

import torch  # noqa: E402

capability = torch.cuda.get_device_capability(0)
if f'sm_{capability[0]}{capability[1]}' not in torch.cuda.get_arch_list():
    run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.5.1', 'torchvision==0.20.1',
         '--index-url', 'https://download.pytorch.org/whl/cu121'])
    os.execv(sys.executable, [sys.executable] + sys.argv)

gpu = torch.cuda.get_device_name(0)
print('GPU:', gpu, capability, 'torch', torch.__version__, flush=True)

if not os.path.exists(REPO):
    run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/fashn-AI/fashn-vton-1.5.git', REPO])
run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPO])
# onnxruntime-gpu는 CUDA/cuDNN 라이브러리를 찾지 못하면 조용히 CPU로 떨어진다(F-10 로그).
# torch가 설치한 nvidia 라이브러리를 먼저 올려 두면 찾는다(onnxruntime 1.21+ preload_dlls).
run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'onnxruntime-gpu>=1.21'])
if not os.path.exists(os.path.join(WEIGHTS, 'model.safetensors')):
    run([sys.executable, os.path.join(REPO, 'scripts', 'download_weights.py'), '--weights-dir', WEIGHTS])

sys.path.insert(0, os.path.join(REPO, 'src'))
import onnxruntime as ort  # noqa: E402
if hasattr(ort, 'preload_dlls'):
    ort.preload_dlls()
print('onnxruntime', ort.__version__, 'providers', ort.get_available_providers(), flush=True)

from PIL import Image, ImageOps  # noqa: E402
from fashn_vton import TryOnPipeline  # noqa: E402

person_path = garment_path = None
for root, _dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == PERSON and root.endswith('inputs/image'):
            person_path = os.path.join(root, f)
        elif f == GARMENT:
            garment_path = os.path.join(root, f)
if not person_path or not garment_path:
    sys.exit(f'입력을 찾지 못했습니다: person={person_path}, garment={garment_path}')
person = ImageOps.exif_transpose(Image.open(person_path)).convert('RGB')
garment = ImageOps.exif_transpose(Image.open(garment_path)).convert('RGB')

pipeline = TryOnPipeline(weights_dir=WEIGHTS)
print('기본 dtype:', pipeline.inference_dtype, '| 입력 크기', pipeline.tryon_model.input_shape, flush=True)

# 포즈 인식만 따로 시간 재기
import numpy as np  # noqa: E402
arr = np.array(person)[..., ::-1]
pipeline.pose_model(arr)   # 워밍업
t = time.time()
for _ in range(3):
    pipeline.pose_model(arr)
pose_s = (time.time() - t) / 3
print(f'포즈 인식 1회 {pose_s:.2f}초', flush=True)

CONFIGS = [
    ('bf16_30', torch.bfloat16, 30),
    ('fp32_30', torch.float32, 30),
    ('fp16_30', torch.float16, 30),
    ('fp32_20', torch.float32, 20),
    ('fp16_20', torch.float16, 20),
    ('fp32_50', torch.float32, 50),
    ('fp16_50', torch.float16, 50),
]

tag = gpu.split()[-1].replace('-', '').lower()   # 예: p10016gb, t4
with open(os.path.join(OUT, f'speed_{tag}.csv'), 'w', encoding='utf-8') as log:
    log.write(f'gpu,config,steps,seconds,peak_mem_gb,pose_s,nan\n')
    for name, dtype, steps in CONFIGS:
        pipeline.inference_dtype = dtype
        pipeline.tryon_model.to(dtype=dtype)
        torch.cuda.empty_cache()
        try:
            # 워밍업 1회(커널 선택·캐시) 후 측정
            pipeline(person_image=person, garment_image=garment, category='tops',
                     garment_photo_type='flat-lay', num_timesteps=3, seed=42)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.time()
            result = pipeline(person_image=person, garment_image=garment, category='tops',
                              garment_photo_type='flat-lay', num_timesteps=steps, seed=42)
            torch.cuda.synchronize()
            seconds = time.time() - start
            peak = torch.cuda.max_memory_allocated() / 1e9
            image = result.images[0]
            nan = int(np.asarray(image).std() < 2)   # float16 오버플로면 한 색으로 뭉개진다
            image.save(os.path.join(OUT, f'{tag}_{name}.png'))
            log.write(f'{gpu},{name},{steps},{seconds:.1f},{peak:.1f},{pose_s:.2f},{nan}\n')
            print(f'  {name}: {seconds:.1f}초, 최대 메모리 {peak:.1f}GB, 이미지 이상={nan}', flush=True)
        except Exception as e:   # 메모리 부족 등은 기록하고 다음 설정으로
            log.write(f'{gpu},{name},{steps},ERROR,,{pose_s:.2f},{type(e).__name__}\n')
            print(f'  {name}: 실패 {type(e).__name__}: {str(e)[:200]}', flush=True)
        log.flush()
print('완료', flush=True)
