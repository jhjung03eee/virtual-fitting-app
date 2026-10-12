"""Kaggle 커널 진입점 — FASHN에 8비트 양자화(torchao)를 걸면 빨라지는가, 품질은 유지되는가.

배경(docs/PLAN.md F-13·F-20~F-22): T4에서 fp16 + 25스텝 + torch.compile 로 한 장 약 50초까지 줄였다.
남은 카드 중 하나가 양자화다. 기대와 위험:
  - int8 weight-only: 가중치만 8비트로 저장하고 계산은 fp16으로 되돌린다 → 메모리는 줄지만 **느려질 수 있다**
  - int8 dynamic (W8A8): 가중치·활성값 모두 8비트로 계산 → T4의 int8 회로를 써서 **빨라질 수 있다**
  - 어느 쪽이든 글자·프린트 같은 섬세한 부분이 먼저 깨질 위험이 있어 품질을 같이 본다

**공정한 비교를 위해 GPU 한 장에서 설정을 순서대로 돌린다.** (F-22에서 두 GPU에 나눠 재다가
GPU 성능 차이와 설정 차이가 섞여 결론을 못 냈다.)
양자화는 모델을 되돌릴 수 없게 바꾸므로 설정마다 워커 프로세스를 새로 띄운다.

올릴 때: PYTHONUTF8=1 kaggle kernels push -p kaggle_fashn_quant --accelerator NvidiaTeslaT4
"""
import csv
import json
import os
import subprocess
import sys
import time

WORK = '/kaggle/tmp/fashn'
REPO = os.path.join(WORK, 'fashn-vton-1.5')
WEIGHTS = os.path.join(WORK, 'weights')
OUT = '/kaggle/working/fashn_quant'
FIELDS = ['quant', 'person', 'garment', 'steps', 'guidance', 'seed', 'seconds', 'peak_mem_gb', 'broken', 'file']

PERSON = 'person_team02.jpg'
GARMENTS = {'shirt_blue': 'garment_shirt_blue.jpg',
            'tee_khaki': 'garment_tee_khaki.jpg',
            'tee_orangutan': 'tshirts.jpg'}
STEPS, GUIDANCE, SEED = 25, 2.5, 42          # 앱 기본값 (F-22)
QUANTS = ['none', 'int8_weight', 'int8_dynamic']
COMPILE = True                                # 세 설정 모두 같은 조건으로


def run(cmd):
    print('\n>>>', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {cmd}')


def worker(quant):
    sys.path.insert(0, os.path.join(REPO, 'src'))
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from fashn_vton import TryOnPipeline

    tag = f'[{quant}]'
    with open(os.path.join(WORK, 'inputs.json'), encoding='utf-8') as f:
        paths = json.load(f)

    t0 = time.time()
    pipeline = TryOnPipeline(weights_dir=WEIGHTS)
    pipeline.inference_dtype = torch.float16
    pipeline.tryon_model.to(dtype=torch.float16)
    load_s = time.time() - t0

    if quant != 'none':
        from torchao.quantization import (int8_dynamic_activation_int8_weight,
                                          int8_weight_only, quantize_)
        recipe = int8_weight_only() if quant == 'int8_weight' else int8_dynamic_activation_int8_weight()
        t0 = time.time()
        quantize_(pipeline.tryon_model, recipe)
        print(f'{tag} 양자화 {time.time() - t0:.1f}초', flush=True)

    if COMPILE:
        pipeline.tryon_model.forward_for_cfg = torch.compile(pipeline.tryon_model.forward_for_cfg)

    person = ImageOps.exif_transpose(Image.open(paths['person'])).convert('RGB')
    garments = {k: ImageOps.exif_transpose(Image.open(v)).convert('RGB') for k, v in paths['garments'].items()}

    # 컴파일·커널 선택을 끝내고 측정한다 (버리는 1회)
    t0 = time.time()
    pipeline(person_image=person, garment_image=garments['tee_khaki'], category='tops',
             garment_photo_type='flat-lay', num_timesteps=3, guidance_scale=GUIDANCE, seed=0)
    print(f'{tag} 모델 로딩 {load_s:.1f}초, 워밍업 {time.time() - t0:.1f}초', flush=True)

    rows = []
    for name, garment in garments.items():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        try:
            result = pipeline(person_image=person, garment_image=garment, category='tops',
                              garment_photo_type='flat-lay', num_timesteps=STEPS,
                              guidance_scale=GUIDANCE, seed=SEED)
            torch.cuda.synchronize()
            seconds = f'{time.time() - t0:.1f}'
            image = result.images[0]
            filename = f'{quant}__{name}.png'
            image.save(os.path.join(OUT, filename))
            broken = int(np.asarray(image).std() < 2)
        except Exception as e:                    # 한 설정이 실패해도 나머지는 계속
            seconds, filename, broken = f'ERROR {type(e).__name__}', '', ''
            print(f'{tag} 실패 {name}: {str(e)[:300]}', flush=True)
        peak = torch.cuda.max_memory_allocated() / 1e9
        rows.append(dict(zip(FIELDS, [quant, 'team02', name, STEPS, GUIDANCE, SEED,
                                      seconds, f'{peak:.1f}', broken, filename])))
        print(f'{tag} {name}: {seconds}초, 최대 메모리 {peak:.1f}GB', flush=True)

    with open(os.path.join(OUT, f'results_{quant}.csv'), 'w', encoding='utf-8', newline='') as log:
        writer = csv.DictWriter(log, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{tag} 완료', flush=True)


if len(sys.argv) == 3 and sys.argv[1] == '--worker':
    worker(sys.argv[2])
    sys.exit(0)

# ---- 부모: 설치 → 입력 정리 → 설정마다 워커를 순서대로 ----
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

import torch  # noqa: E402

capability = torch.cuda.get_device_capability(0)
if f'sm_{capability[0]}{capability[1]}' not in torch.cuda.get_arch_list():
    run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.5.1', 'torchvision==0.20.1',
         '--index-url', 'https://download.pytorch.org/whl/cu121'])
    os.execv(sys.executable, [sys.executable] + sys.argv)
print('GPU:', torch.cuda.get_device_name(0), capability, 'torch', torch.__version__, flush=True)

if not os.path.exists(REPO):
    run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/fashn-AI/fashn-vton-1.5.git', REPO])
run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPO])
run([sys.executable, '-m', 'pip', 'install', '-q', 'torchao'])
if not os.path.exists(os.path.join(WEIGHTS, 'model.safetensors')):
    run([sys.executable, os.path.join(REPO, 'scripts', 'download_weights.py'), '--weights-dir', WEIGHTS])

found = {}
for root, _dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == PERSON or f in GARMENTS.values():
            found[f] = os.path.join(root, f)
missing = [f for f in [PERSON] + list(GARMENTS.values()) if f not in found]
if missing:
    sys.exit(f'입력 누락: {missing}')
with open(os.path.join(WORK, 'inputs.json'), 'w', encoding='utf-8') as f:
    json.dump({'person': found[PERSON],
               'garments': {k: found[v] for k, v in GARMENTS.items()}}, f, ensure_ascii=False)

# 같은 GPU(0번)에서 설정을 하나씩. 양자화는 되돌릴 수 없어 프로세스를 새로 띄운다
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
start = time.time()
for quant in QUANTS:
    code = subprocess.call([sys.executable, os.path.abspath(sys.argv[0]), '--worker', quant], env=env)
    print(f'--- {quant} 종료 코드 {code}', flush=True)

rows = []
for quant in QUANTS:
    path = os.path.join(OUT, f'results_{quant}.csv')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            rows += list(csv.DictReader(f))
with open(os.path.join(OUT, 'results.csv'), 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
print(f'전체 경과 {time.time() - start:.0f}초, 결과 {len(rows)}줄', flush=True)
print('완료', flush=True)
