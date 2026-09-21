"""Kaggle 커널 템플릿 — 새 가상 피팅 모델을 팀 공통 조건으로 돌린다 (docs/MODEL_STUDY.md).

쓰는 법:
  1. 이 폴더를 kaggle_<모델이름>/ 으로 복사하고 파일 이름을 run_<모델이름>.py 로 바꾼다
  2. kernel-metadata.json 의 id·title·code_file 을 바꾼다
  3. 아래 TODO 세 곳만 채운다 — 설치 / 모델 로딩 / 추론 호출
  4. PYTHONUTF8=1 kaggle kernels push -p kaggle_<모델이름> --accelerator NvidiaTeslaT4

측정·CSV 저장·입력 찾기·워밍업은 이미 들어 있다. 바꾸지 말 것:
  같은 인물 1명 × 옷 4벌 × 시드 2개 = 8장. 조건이 다르면 다른 팀원 결과와 비교가 안 된다.
"""
import csv
import os
import subprocess
import sys
import time

MODEL_NAME = 'template'          # TODO: 모델 이름 (결과 파일 이름에 쓰인다)
WORK = f'/kaggle/tmp/{MODEL_NAME}'
OUT = f'/kaggle/working/{MODEL_NAME}'

# 팀 공통 입력 (docs/MODEL_STUDY.md). 파일 이름은 Kaggle 데이터셋 안의 이름이다
PERSON = 'person_team02.jpg'
GARMENTS = {
    'shirt_blue': ('garment_shirt_blue.jpg', 'tops'),
    'tee_khaki': ('garment_tee_khaki.jpg', 'tops'),
    'tee_orangutan': ('tshirts.jpg', 'tops'),
    'pants_corduroy': ('pants.jpg', 'bottoms'),   # 하의를 지원하지 않는 모델이면 이 줄을 지운다
}
SEEDS = (42, 123)
FIELDS = ['model', 'garment', 'category', 'seed', 'seconds', 'peak_mem_gb', 'broken', 'file']


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
    # Kaggle 기본 torch가 이 GPU를 지원하지 않는 빌드일 때만
    run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.5.1', 'torchvision==0.20.1',
         '--index-url', 'https://download.pytorch.org/whl/cu121'])
    os.execv(sys.executable, [sys.executable] + sys.argv)
print(f'GPU: {torch.cuda.get_device_name(0)} {capability}, torch {torch.__version__}', flush=True)

# ---------------------------------------------------------------------------
# TODO 1. 설치와 가중치 내려받기
#   - 저장소 clone 과 가중치는 /kaggle/tmp (WORK) 안에. /kaggle/working 에 두면 output이 수천 개가 된다
#   - 패키지는 필요한 것만 최소로 고정한다. 구버전을 깔면 Kaggle 기본 패키지와 충돌한다
# 예:
#   REPO = os.path.join(WORK, 'SomeVTON')
#   if not os.path.exists(REPO):
#       run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/…', REPO])
#   run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPO])
#   run([sys.executable, os.path.join(REPO, 'scripts', 'download_weights.py'), '--out', WEIGHTS])
# ---------------------------------------------------------------------------

t_install = time.time()
# (여기에 설치 코드)
print(f'설치·다운로드 {time.time() - t_install:.0f}초', flush=True)

# 입력 찾기 — Kaggle 데이터셋 어디에 있든 파일 이름으로 찾는다
wanted = {PERSON} | {f for f, _ in GARMENTS.values()}
found = {}
for root, _dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f in wanted:
            found[f] = os.path.join(root, f)
missing = sorted(wanted - set(found))
if missing:
    sys.exit(f'입력 누락: {missing} — 커널 설정에서 데이터셋 3개를 붙였는지 확인할 것')
print('입력 확인:', sorted(found), flush=True)

from PIL import Image, ImageOps  # noqa: E402

# 폰 사진은 EXIF 회전을 적용하지 않으면 누운 채로 합성된다
person_image = ImageOps.exif_transpose(Image.open(found[PERSON])).convert('RGB')
garment_images = {k: ImageOps.exif_transpose(Image.open(found[f])).convert('RGB')
                  for k, (f, _c) in GARMENTS.items()}

# ---------------------------------------------------------------------------
# TODO 2. 모델 로딩
#   - **정밀도를 직접 지정할 것.** T4·P100은 bf16을 흉내만 내서 가장 느리다.
#     fp16이 fp32보다 4배 빠르고 결과는 같았다 (docs/PLAN.md F-13·F-14)
#   - 메모리가 모자라면 CPU offload. 단 offload는 CPU RAM을 12GB 이상 먹는다 (F-15)
# 예:
#   model = SomePipeline.from_pretrained(WEIGHTS, torch_dtype=torch.float16).to('cuda')
# ---------------------------------------------------------------------------

t_load = time.time()
model = None   # (여기에 로딩 코드)
load_s = time.time() - t_load
print(f'모델 로딩 {load_s:.1f}초, GPU 메모리 {torch.cuda.memory_allocated() / 1e9:.1f}GB', flush=True)


def generate(garment_key, seed):
    """한 장 생성. 반환: PIL 이미지.

    TODO 3. 맡은 모델의 추론 호출로 바꾼다.
      - 옷 종류는 GARMENTS[garment_key][1] ('tops' 또는 'bottoms') — 모델이 쓰는 이름으로 바꿀 것
      - 시드를 반드시 반영할 것. 시드가 안 먹으면 두 장이 똑같이 나와 비교가 무의미해진다
    """
    _category = GARMENTS[garment_key][1]
    raise NotImplementedError('generate() 를 채울 것')


# 워밍업 1장 — 첫 호출은 커널 컴파일·캐시 때문에 느리다. 측정에서 뺀다
first = next(iter(GARMENTS))
t0 = time.time()
try:
    generate(first, 0)
    print(f'워밍업 {time.time() - t0:.1f}초', flush=True)
except NotImplementedError:
    sys.exit('generate() 를 먼저 채우세요 (TODO 3)')

rows = []
total = len(GARMENTS) * len(SEEDS)
done = 0
for key in GARMENTS:
    for seed in SEEDS:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        filename, broken = '', ''
        try:
            image = generate(key, seed)
            torch.cuda.synchronize()
            seconds = f'{time.time() - t0:.1f}'
            filename = f'{MODEL_NAME}__{key}__s{seed}.png'
            image.save(os.path.join(OUT, filename))
            import numpy as np
            broken = int(np.asarray(image).std() < 2)   # 한 색으로 뭉개졌는지
        except Exception as e:      # 한 장 실패로 전체를 멈추지 않는다
            seconds = f'ERROR {type(e).__name__}'
            print(f'  실패 {key} s{seed}: {str(e)[:300]}', flush=True)
        peak = torch.cuda.max_memory_allocated() / 1e9
        rows.append(dict(zip(FIELDS, [MODEL_NAME, key, GARMENTS[key][1], seed,
                                      seconds, f'{peak:.1f}', broken, filename])))
        done += 1
        print(f'  [{done}/{total}] {key} s{seed}: {seconds}초, 최대 메모리 {peak:.1f}GB', flush=True)

with open(os.path.join(OUT, 'results.csv'), 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

ok = [r for r in rows if not r['seconds'].startswith('ERROR')]
if ok:
    times = [float(r['seconds']) for r in ok]
    print(f'성공 {len(ok)}/{total}장 | 장당 평균 {sum(times) / len(times):.1f}초 '
          f'(최소 {min(times):.1f} 최대 {max(times):.1f}) | 모델 로딩 {load_s:.1f}초', flush=True)
print('완료 — 결과를 docs/MODEL_STUDY.md 의 "제출물" 형식으로 정리할 것', flush=True)
