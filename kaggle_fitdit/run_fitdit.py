"""Kaggle 커널 진입점 — FitDiT를 FASHN(F-14)과 같은 인물·옷으로 돌린다. T4×2 + fp16.

FitDiT(arXiv 2411.10499, CC BY-NC-SA — 비상업 용도라 문제없음): 질감·프린트 디테일과 고해상도(기본 1152x1536)에 강점.
SD3 기반 트랜스포머 두 개(옷용·합성용) + CLIP 이미지 인코더 두 개. 저자 코드 gradio_sd3.py의 FitDiTGenerator를 그대로 쓴다.

구조는 kaggle_fashn_t4x2/와 같다: 설치·다운로드는 부모가 1번, GPU마다 워커 프로세스.
- FitDiT도 기본이 bf16이라 T4에서 느리다(F-13) → fp16 지정.
- 가중치만 약 12~13GB(추정). 1152x1536에서 T4 16GB가 모자라면 그 워커만 CPU offload로 바꿔 다시 시도한다.
- 워커가 모델을 CPU 메모리에 올렸다가 GPU로 옮기므로, 둘이 동시에 올리면 Kaggle RAM(약 30GB)이 빠듯하다 → 한 워커가 로딩을 끝낸 뒤 다음을 띄운다.
- 실측(v2~v3): 1152x1536은 T4에 offload로만 들어가고(장당 약 2분), offload 워커는 RAM 때문에 1개만 가능.

올릴 때: PYTHONUTF8=1 kaggle kernels push -p kaggle_fitdit --accelerator NvidiaTeslaT4
"""
import csv
import json
import os
import subprocess
import sys
import time

WORK = '/kaggle/tmp/fitdit'
REPO = os.path.join(WORK, 'FitDiT')
WEIGHTS = os.path.join(WORK, 'weights')
INPUTS = os.path.join(WORK, 'inputs')
OUT = '/kaggle/working/fitdit'
os.environ.setdefault('HF_HOME', os.path.join(WORK, 'hf'))   # CLIP 인코더 캐시. 워커도 같은 경로를 본다
FIELDS = ['group', 'person', 'garment', 'category', 'steps', 'guidance', 'seed', 'resolution',
          'gpu', 'offload', 'seconds', 'broken', 'file']
CATEGORY = {'tops': 'Upper-body', 'bottoms': 'Lower-body', 'dresses': 'Dresses'}
OFFLOAD = True   # T4 16GB + 1152x1536. 메모리가 넉넉한 GPU(A100 등)에서는 False

# (묶음, 인물, 옷, category, 스텝, guidance, 시드, 해상도) — FASHN F-14와 같은 인물·옷·시드
# 스텝은 저자 UI 최대치 30(기본 20), guidance는 저자 기본 2. 품질 우선이라 해상도는 저자 기본 1152x1536.
JOBS = []
for person, garment, category in (('kakao_front_upper', 'sweatshirt_text', 'tops'),
                                  ('kakao_front_upper', 'tee_orangutan', 'tops'),
                                  ('kakao_front_upper', 'shirt_blue', 'tops'),
                                  ('demo_person1_full', 'pants_corduroy', 'bottoms')):
    # v2에서 시드 123 4장, v3에서 시드 42 중 오랑우탄·코듀로이 2장 성공(data/samples/results/fitdit_v2, _v3).
    # v4는 v3 GPU0에서 못 만든 시드 42 두 장만 돌린다
    if garment in ('sweatshirt_text', 'shirt_blue'):
        JOBS.append(('E', person, garment, category, 30, 2.0, 42, '1152x1536'))


def run(cmd):
    print('\n>>>', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {cmd}')


def job_name(job):
    group, person, garment, _category, steps, guidance, seed, resolution = job
    return f'{group}__{person}__{garment}__st{steps}_g{guidance:g}_s{seed}_{resolution}.png'


def worker(index):
    """CUDA_VISIBLE_DEVICES로 GPU 한 장만 보이는 상태에서 받은 작업을 돌린다."""
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    # 저자 파일이 맨 위에서 gradio를 import하지만 UI는 안 쓴다. Kaggle의 gradio 5.50은
    # huggingface_hub 0.26과 안 맞으므로 진짜 gradio를 불러오지 않고 빈 모듈로 막는다
    import types
    sys.modules['gradio'] = types.ModuleType('gradio')
    import numpy as np
    import torch
    from PIL import Image
    import onnxruntime as ort
    # 포즈·부위 인식(onnx)이 CUDA 라이브러리를 못 찾으면 CPU로 떨어진다(F-10). torch가 깐 라이브러리를 먼저 올린다
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    from gradio_sd3 import FitDiTGenerator

    tag = f'[GPU{index}]'
    with open(os.path.join(WORK, f'jobs_{index}.json'), encoding='utf-8') as f:
        spec = json.load(f)
    jobs = [tuple(j) for j in spec['jobs']]

    t0 = time.time()
    # v2: 가중치만 12.2GB라 1152x1536은 T4 16GB에 안 들어갔다. 메모리 부족 뒤 offload로 바꾸는 방식은 GPU1(OutOfMemoryError)에선
    # 됐지만 GPU0은 같은 부족이 AcceleratorError로 올라와 못 잡았고 이후 CUDA 상태가 망가져 4장 모두 실패 → 처음부터 offload.
    # offload로도 장당 약 2분(30스텝)이었다.
    generator = FitDiTGenerator(WEIGHTS, offload=OFFLOAD, device='cuda:0', with_fp16=True)
    print(f'{tag} {torch.cuda.get_device_name(0)} 모델 로딩 {time.time() - t0:.1f}초, '
          f'GPU 메모리 {torch.cuda.memory_allocated() / 1e9:.1f}GB, 작업 {len(jobs)}장', flush=True)
    open(os.path.join(WORK, f'ready_{index}'), 'w').close()   # 부모가 다음 워커를 띄워도 된다는 신호

    offload = OFFLOAD
    masks = {}
    csv_path = os.path.join(OUT, f'results_gpu{index}.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as log:
        writer = csv.DictWriter(log, fieldnames=FIELDS)
        writer.writeheader()
        for done, job in enumerate(jobs, 1):
            group, person_name, garment_name, category, steps, guidance, seed, resolution = job
            person_path = spec['persons'][person_name]
            garment_path = spec['garments'][garment_name]
            name = job_name(job)
            row = dict(zip(FIELDS, [group, person_name, garment_name, category, steps, guidance, seed, resolution, index]))
            start = time.time()
            try:
                key = (person_name, category)
                if key not in masks:
                    # 저자 UI의 Step1(마스크) — 사람·부위가 같으면 시드가 달라도 같다
                    mask, pose = generator.generate_mask(person_path, CATEGORY[category], 0, 0, 0, 0)
                    masks[key] = (mask, np.array(pose))
                    Image.fromarray(mask['composite']).convert('RGB').save(
                        os.path.join(OUT, f'_mask__{person_name}__{category}.png'))
                mask, pose = masks[key]
                for attempt in (1, 2):
                    try:
                        images = generator.process(person_path, garment_path, mask, pose, steps, guidance,
                                                   seed, 1, resolution)
                        break
                    except torch.cuda.OutOfMemoryError:
                        if attempt == 2 or offload:
                            raise
                        # 1152x1536이 T4 16GB에 안 들어가면 그 워커만 offload로 바꾼다(저자 --offload와 같은 호출)
                        print(f'{tag} 메모리 부족 → CPU offload로 바꿔 다시 시도', flush=True)
                        torch.cuda.empty_cache()
                        generator.pipeline.enable_model_cpu_offload()
                        offload = True
                        start = time.time()
                image = images[0]
                image.save(os.path.join(OUT, name))
                row['seconds'] = f'{time.time() - start:.1f}'
                row['broken'] = int(np.asarray(image).std() < 2)   # fp16 오버플로면 한 색으로 뭉개진다
                row['file'] = name
            except Exception as e:   # 한 장 실패로 전체를 멈추지 않는다
                row['seconds'] = f'ERROR {type(e).__name__}'
                print(f'{tag} 실패 {name}: {str(e)[:300]}', flush=True)
            row['offload'] = int(offload)
            writer.writerow(row)
            log.flush()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f'{tag} [{done}/{len(jobs)}] {name}: {row["seconds"]}초, 최대 메모리 {peak:.1f}GB, offload={int(offload)}',
                  flush=True)
    print(f'{tag} 완료', flush=True)


if len(sys.argv) == 3 and sys.argv[1] == '--worker':
    worker(int(sys.argv[2]))
    sys.exit(0)

# ---- 부모: 설치 1회 → 가중치 → 입력 정리 → GPU마다 워커 ----
for d in (WORK, INPUTS, OUT):
    os.makedirs(d, exist_ok=True)

import torch  # noqa: E402

capability = torch.cuda.get_device_capability(0)
if f'sm_{capability[0]}{capability[1]}' not in torch.cuda.get_arch_list():
    run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.5.1', 'torchvision==0.20.1',
         '--index-url', 'https://download.pytorch.org/whl/cu121'])
    os.execv(sys.executable, [sys.executable] + sys.argv)
gpu_count = torch.cuda.device_count()
print(f'GPU 개수: {gpu_count}', [torch.cuda.get_device_name(i) for i in range(gpu_count)],
      'torch', torch.__version__, flush=True)

if not os.path.exists(REPO):
    run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/BoyuanJiang/FitDiT.git', REPO])
# 저자 requirements.txt 버전 중 코드가 직접 기대는 diffusers·transformers만 맞춘다. torch·numpy는 Kaggle 기본.
# v1 실패: accelerate를 0.31로 내렸더니 Kaggle에 깔린 peft가 새 accelerate 함수를 못 찾아 diffusers import가 죽었다.
#   → accelerate는 Kaggle 것을 쓰고, diffusers가 peft를 불러오지 않게 peft를 지운다(LoRA 안 씀).
# onnxruntime-gpu 최신판은 CUDA 13용이라 torch(cu128) 환경에서 libcublas.so.13을 못 찾았다 → CUDA 12용 1.22.
run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers==0.31.0', 'transformers==4.39.3',
     'huggingface_hub==0.26.5', 'einops==0.7.0', 'scikit-image', 'onnxruntime-gpu==1.22.0'])
run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'peft'])
# 워커 둘을 띄우기 전에 import만 먼저 확인한다(실패하면 여기서 바로 원인이 보인다)
run([sys.executable, '-c', f'import sys, types; sys.path.insert(0, {REPO!r}); '
     "sys.modules['gradio'] = types.ModuleType('gradio'); import gradio_sd3; print('FitDiT import OK')"])

from huggingface_hub import snapshot_download  # noqa: E402

t0 = time.time()
snapshot_download('BoyuanJiang/FitDiT', local_dir=WEIGHTS, allow_patterns=['*.json', '*.safetensors', '*.bin', '*.onnx'])
# 워커 둘이 동시에 받지 않게 CLIP 인코더도 여기서 캐시에 받아 둔다
for repo in ('openai/clip-vit-large-patch14', 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'):
    snapshot_download(repo, allow_patterns=['*.json', 'model.safetensors'])
print(f'가중치 다운로드 {time.time() - t0:.0f}초', flush=True)

from PIL import Image, ImageOps  # noqa: E402

found_p, found_g = {}, {}
for root, _dirs, files in os.walk('/kaggle/input'):
    for f in files:
        path = os.path.join(root, f)
        if root.endswith('inputs/image') and f in ('demo_person1_full.jpg', 'kakao_front_upper.jpg'):
            found_p[os.path.splitext(f)[0]] = path
        elif f == 'garment_shirt_blue.jpg':
            found_g['shirt_blue'] = path
        elif f == 'garment_tee_khaki.jpg':
            found_g['tee_khaki'] = path
        elif f == 'shirts.jpg':
            found_g['sweatshirt_text'] = path
        elif f == 'pants.jpg':
            found_g['pants_corduroy'] = path
        elif f == 'tshirts.jpg':
            found_g['tee_orangutan'] = path
print('인물:', sorted(found_p), '\n옷:', sorted(found_g), flush=True)
need_p = {j[1] for j in JOBS}
need_g = {j[2] for j in JOBS}
if not need_p <= set(found_p) or not need_g <= set(found_g):
    sys.exit(f'입력 누락: 인물 {need_p - set(found_p)}, 옷 {need_g - set(found_g)}')

# 저자 코드는 파일 경로를 Image.open만 한다 → EXIF 회전·RGB 변환을 미리 해서 넘긴다
persons, garments = {}, {}
for table, found in ((persons, found_p), (garments, found_g)):
    for key in (need_p if table is persons else need_g):
        target = os.path.join(INPUTS, f'{key}.png')
        ImageOps.exif_transpose(Image.open(found[key])).convert('RGB').save(target)
        table[key] = target
        print(f'  {key}: {Image.open(target).size}', flush=True)

# v3: offload 워커 둘을 동시에 띄우자 GPU0 워커가 RAM 부족으로 강제 종료(-9)됐다.
# offload는 모델 약 12GB를 CPU RAM에 상주시키므로 Kaggle RAM(약 30GB)에 둘은 안 들어간다 → offload면 워커 1개
workers = 1 if OFFLOAD else max(gpu_count, 1)
for i in range(workers):
    with open(os.path.join(WORK, f'jobs_{i}.json'), 'w', encoding='utf-8') as f:
        json.dump({'jobs': JOBS[i::workers], 'persons': persons, 'garments': garments}, f, ensure_ascii=False)
    ready = os.path.join(WORK, f'ready_{i}')
    if os.path.exists(ready):
        os.remove(ready)

start = time.time()
procs = []
for i in range(workers):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
    procs.append(subprocess.Popen([sys.executable, os.path.abspath(sys.argv[0]), '--worker', str(i)], env=env))
    # 로딩이 끝날 때까지 기다린다(RAM). 워커가 로딩 중에 죽으면 기다리지 않는다
    while not os.path.exists(os.path.join(WORK, f'ready_{i}')) and procs[-1].poll() is None:
        time.sleep(5)
codes = [p.wait() for p in procs]
elapsed = time.time() - start

rows = []
for i in range(workers):
    path = os.path.join(OUT, f'results_gpu{i}.csv')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            rows += list(csv.DictReader(f))
with open(os.path.join(OUT, 'results.csv'), 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: r['file'] or ''))

ok = [r for r in rows if not r['seconds'].startswith('ERROR')]
print(f'워커 종료 코드 {codes} | 성공 {len(ok)}/{len(JOBS)}장, 이상 {sum(int(r["broken"]) for r in ok)}장, '
      f'offload {sum(int(r["offload"]) for r in ok)}장', flush=True)
print(f'전체 경과 {elapsed:.0f}초 (모델 로딩 포함) | 장당 시간 합 {sum(float(r["seconds"]) for r in ok):.0f}초', flush=True)
if any(codes):
    sys.exit('워커 중 비정상 종료가 있었다')
print('완료', flush=True)
