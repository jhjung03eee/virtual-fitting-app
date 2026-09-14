"""Kaggle 커널 진입점 — FASHN VTON v1.5를 T4 + fp16 + GPU 2장 동시로 돌린다.

F-13: T4는 fp16이 fp32보다 4배 빠르고(50스텝 117초) 결과는 같았다(티셔츠 한 벌·시드 하나).
FASHN은 T4에서도 bf16(430초)을 고르므로 dtype을 직접 지정한다.
Kaggle T4는 보통 2장이라, GPU마다 워커 프로세스를 하나씩 띄워 작업을 나눠 돌린다.

다른 실험에 쓸 때는 맨 위 JOBS만 바꾼다. 한 줄 = 한 장.
올릴 때: PYTHONUTF8=1 kaggle kernels push -p kaggle_fashn_t4x2 --accelerator NvidiaTeslaT4

지금 JOBS: 글자·사진 프린트·질감 옷에서도 fp16이 fp32와 같은가 (50스텝, guidance 2.5 = 현재 기본값).
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
OUT = '/kaggle/working/fashn_t4x2'
FIELDS = ['group', 'person', 'garment', 'category', 'steps', 'guidance', 'seed', 'dtype',
          'gpu', 'seconds', 'broken', 'file']

# (묶음, 인물, 옷, category, 스텝, guidance, 시드, dtype)
JOBS = []
for person, garment, category in (('kakao_front_upper', 'sweatshirt_text', 'tops'),
                                  ('kakao_front_upper', 'tee_orangutan', 'tops'),
                                  ('kakao_front_upper', 'shirt_blue', 'tops'),
                                  ('demo_person1_full', 'pants_corduroy', 'bottoms')):
    for seed in (42, 123):
        for dtype in ('fp32', 'fp16'):
            JOBS.append(('D', person, garment, category, 50, 2.5, seed, dtype))


def run(cmd):
    print('\n>>>', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {cmd}')


def job_name(job):
    group, person, garment, _category, steps, guidance, seed, dtype = job
    return f'{group}__{person}__{garment}__st{steps}_g{guidance:g}_s{seed}_{dtype}.png'


def worker(index):
    """CUDA_VISIBLE_DEVICES로 GPU 한 장만 보이는 상태에서 받은 작업을 돌린다."""
    sys.path.insert(0, os.path.join(REPO, 'src'))
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from fashn_vton import TryOnPipeline

    tag = f'[GPU{index}]'
    with open(os.path.join(WORK, f'jobs_{index}.json'), encoding='utf-8') as f:
        spec = json.load(f)
    dtypes = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}
    # 같은 dtype끼리 붙여 모델 변환 횟수를 줄인다
    jobs = sorted((tuple(j) for j in spec['jobs']), key=lambda j: j[7])

    t0 = time.time()
    pipeline = TryOnPipeline(weights_dir=WEIGHTS)
    print(f'{tag} {torch.cuda.get_device_name(0)} 모델 로딩 {time.time() - t0:.1f}초, 작업 {len(jobs)}장', flush=True)

    current = None
    csv_path = os.path.join(OUT, f'results_gpu{index}.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as log:
        writer = csv.DictWriter(log, fieldnames=FIELDS)
        writer.writeheader()
        for done, job in enumerate(jobs, 1):
            group, person_name, garment_name, category, steps, guidance, seed, dtype = job
            if dtype != current:
                # kaggle_fashn_speed에서 검증한 방식
                pipeline.inference_dtype = dtypes[dtype]
                pipeline.tryon_model.to(dtype=dtypes[dtype])
                torch.cuda.empty_cache()
                current = dtype
            person = ImageOps.exif_transpose(Image.open(spec['persons'][person_name])).convert('RGB')
            garment = ImageOps.exif_transpose(Image.open(spec['garments'][garment_name])).convert('RGB')
            name = job_name(job)
            row = dict(zip(FIELDS, [group, person_name, garment_name, category, steps, guidance, seed, dtype, index]))
            start = time.time()
            try:
                result = pipeline(person_image=person, garment_image=garment, category=category,
                                  garment_photo_type='flat-lay', num_timesteps=steps,
                                  guidance_scale=guidance, seed=seed)
                image = result.images[0]
                image.save(os.path.join(OUT, name))
                row['seconds'] = f'{time.time() - start:.1f}'
                row['broken'] = int(np.asarray(image).std() < 2)   # fp16 오버플로면 한 색으로 뭉개진다
                row['file'] = name
            except Exception as e:   # 한 장 실패로 전체를 멈추지 않는다
                row['seconds'] = f'ERROR {type(e).__name__}'
                print(f'{tag} 실패 {name}: {str(e)[:200]}', flush=True)
            writer.writerow(row)
            log.flush()
            print(f'{tag} [{done}/{len(jobs)}] {name}: {row["seconds"]}초', flush=True)
    print(f'{tag} 완료', flush=True)


if len(sys.argv) == 3 and sys.argv[1] == '--worker':
    worker(int(sys.argv[2]))
    sys.exit(0)

# ---- 부모: 설치 1회 → 입력 찾기 → GPU마다 워커 ----
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

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
    run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/fashn-AI/fashn-vton-1.5.git', REPO])
run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPO])
if not os.path.exists(os.path.join(WEIGHTS, 'model.safetensors')):
    run([sys.executable, os.path.join(REPO, 'scripts', 'download_weights.py'), '--weights-dir', WEIGHTS])

persons, garments = {}, {}
for root, _dirs, files in os.walk('/kaggle/input'):
    for f in files:
        path = os.path.join(root, f)
        if f == 'KakaoTalk_20260909_190109015_02.jpg':
            persons['kakao_front_original'] = path
        elif root.endswith('inputs/image') and f in ('demo_person1_full.jpg', 'kakao_front_upper.jpg'):
            persons[os.path.splitext(f)[0]] = path
        elif f == 'garment_tee_khaki.jpg':
            garments['tee_khaki'] = path
        elif f == 'garment_shirt_blue.jpg':
            garments['shirt_blue'] = path
        elif f == 'garment_pants_forest.jpg':
            garments['pants_forest'] = path
        elif f == 'shirts.jpg':
            garments['sweatshirt_text'] = path
        elif f == 'pants.jpg':
            garments['pants_corduroy'] = path
        elif f == 'tshirts.jpg':
            garments['tee_orangutan'] = path
print('인물:', sorted(persons), '\n옷:', sorted(garments), flush=True)
need_p = {j[1] for j in JOBS}
need_g = {j[2] for j in JOBS}
if not need_p <= set(persons) or not need_g <= set(garments):
    sys.exit(f'입력 누락: 인물 {need_p - set(persons)}, 옷 {need_g - set(garments)}')

# 느린 작업(fp32)이 한쪽에 몰리지 않게 dtype 순으로 세운 뒤 번갈아 나눈다
workers = max(gpu_count, 1)
ordered = sorted(JOBS, key=lambda j: j[7])
for i in range(workers):
    with open(os.path.join(WORK, f'jobs_{i}.json'), 'w', encoding='utf-8') as f:
        json.dump({'jobs': ordered[i::workers], 'persons': persons, 'garments': garments}, f, ensure_ascii=False)

start = time.time()
procs = []
for i in range(workers):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
    # __file__ 대신 argv[0]: 위 os.execv 재시작이 이미 Kaggle에서 이 경로로 동작했다
    procs.append(subprocess.Popen([sys.executable, os.path.abspath(sys.argv[0]), '--worker', str(i)], env=env))
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
sum_seconds = sum(float(r['seconds']) for r in ok)
print(f'워커 종료 코드 {codes} | 성공 {len(ok)}/{len(JOBS)}장, 이상 {sum(int(r["broken"]) for r in ok)}장', flush=True)
print(f'전체 경과 {elapsed:.0f}초 (모델 로딩 포함) | 장당 시간 합 {sum_seconds:.0f}초', flush=True)
if any(codes):
    sys.exit('워커 중 비정상 종료가 있었다')
print('완료', flush=True)
