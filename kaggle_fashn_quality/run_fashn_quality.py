"""Kaggle 커널 진입점 — FASHN VTON v1.5 품질 한계 확인 (속도 무관, 품질 우선).

F-10에서 FASHN이 색·실루엣·자세 대응 모두 CatVTON보다 좋았다. 주력 모델로 정하기 전에:
  A. 스텝 30/50 × guidance 1.5/2.5 — 질감·글자·원본 충실도가 더 오르는지, 색이 과해지지 않는지
  B. 하의(바지) — 상의만 시험했었다. 올리브 바지(성공 이력), 네이비 코듀로이(CatVTON 실패 이력)
  C. 시드 추가 — 원본에 없는 디테일(셔츠 어깨 덮개)이 몇 번에 한 번 생기는지
설치 방식은 F-10에서 정상 결과가 나온 그대로 둔다(포즈 인식이 CPU로 돌아도 품질 영향 없음).
"""
import os
import subprocess
import sys
import time

WORK = '/kaggle/tmp/fashn'
REPO = os.path.join(WORK, 'fashn-vton-1.5')
WEIGHTS = os.path.join(WORK, 'weights')
OUT = '/kaggle/working/fashn_quality'

# (묶음, 인물, 옷, category, 스텝, guidance, 시드들)
RUNS = []
for garment in ('sweatshirt_text', 'shirt_blue'):
    for steps in (30, 50):
        for guidance in (1.5, 2.5):
            RUNS.append(('A', 'kakao_front_upper', garment, 'tops', steps, guidance, (42, 123)))
for person in ('kakao_front_original', 'demo_person1_full'):
    for garment in ('pants_forest', 'pants_corduroy'):
        RUNS.append(('B', person, garment, 'bottoms', 50, 1.5, (42, 123)))
for garment in ('shirt_blue', 'tee_khaki'):
    RUNS.append(('C', 'kakao_front_upper', garment, 'tops', 50, 1.5, (7, 777, 2024, 31415)))


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
print('GPU:', torch.cuda.get_device_name(0), 'torch', torch.__version__, flush=True)

if not os.path.exists(REPO):
    run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/fashn-AI/fashn-vton-1.5.git', REPO])
run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPO])
if not os.path.exists(os.path.join(WEIGHTS, 'model.safetensors')):
    run([sys.executable, os.path.join(REPO, 'scripts', 'download_weights.py'), '--weights-dir', WEIGHTS])
sys.path.insert(0, os.path.join(REPO, 'src'))

from PIL import Image, ImageOps  # noqa: E402

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
print('인물:', sorted(persons), '\n옷:', sorted(garments), flush=True)
need_p = {r[1] for r in RUNS}
need_g = {r[2] for r in RUNS}
if not need_p <= set(persons) or not need_g <= set(garments):
    sys.exit(f'입력 누락: 인물 {need_p - set(persons)}, 옷 {need_g - set(garments)}')

from fashn_vton import TryOnPipeline  # noqa: E402

pipeline = TryOnPipeline(weights_dir=WEIGHTS)
print('dtype', pipeline.inference_dtype, '입력 크기', pipeline.tryon_model.input_shape, flush=True)

total = sum(len(r[6]) for r in RUNS)
done = 0
csv_path = os.path.join(OUT, 'results.csv')
with open(csv_path, 'w', encoding='utf-8') as log:
    log.write('group,person,garment,category,steps,guidance,seed,seconds,file\n')
    for group, person_name, garment_name, category, steps, guidance, seeds in RUNS:
        person = ImageOps.exif_transpose(Image.open(persons[person_name])).convert('RGB')
        garment = ImageOps.exif_transpose(Image.open(garments[garment_name])).convert('RGB')
        for seed in seeds:
            name = f'{group}__{person_name}__{garment_name}__st{steps}_g{guidance:g}_s{seed}.png'
            start = time.time()
            try:
                result = pipeline(person_image=person, garment_image=garment, category=category,
                                  garment_photo_type='flat-lay', num_timesteps=steps,
                                  guidance_scale=guidance, seed=seed)
                result.images[0].save(os.path.join(OUT, name))
                seconds = f'{time.time() - start:.1f}'
            except Exception as e:   # 한 장 실패로 전체를 멈추지 않는다
                seconds = f'ERROR {type(e).__name__}'
                print(f'  실패 {name}: {str(e)[:200]}', flush=True)
            done += 1
            log.write(f'{group},{person_name},{garment_name},{category},{steps},{guidance},{seed},{seconds},{name}\n')
            log.flush()
            print(f'  [{done}/{total}] {name}: {seconds}초', flush=True)
print('완료', flush=True)
