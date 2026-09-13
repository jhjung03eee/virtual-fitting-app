"""Kaggle 커널 진입점 — FASHN VTON v1.5 (오픈소스, Apache-2.0)를 우리 사진·옷에 돌린다.

FASHN VTON v1.5는 픽셀 공간에서 직접 생성하는 MMDiT(9.7억 파라미터) 가상 피팅 모델이다.
CatVTON(SD1.5)은 VAE로 8배 압축한 latent에서 그려 미세 질감이 뭉개질 수 있는데, 이 모델은
압축 단계가 없고 마스크 없이(segmentation-free) 입힌다. 포즈(DWPose)·파싱은 파이프라인이 자체 처리한다.

비교 조건은 HR-VITON·하이브리드 실험과 같게 맞춘다 (vfa-hrviton 커널 출력의 768x1024 인물 이미지 재사용).
    인물: 데모 남·여, 카톡 정자세 상반신 크롭, 카톡 정자세 원본(전신)
    옷:   카키 반팔, 블루 셔츠, 글자 맨투맨(제품 사진), 스누피 카디건 — 전부 flat-lay, category=tops
    시드 42·123, 30스텝(권장 기본)
"""
import os
import subprocess
import sys
import time

WORK = '/kaggle/tmp/fashn'
REPO = os.path.join(WORK, 'fashn-vton-1.5')
WEIGHTS = os.path.join(WORK, 'weights')
OUT = '/kaggle/working/fashn'
SEEDS = [42, 123]
STEPS = 30


def run(cmd, **kwargs):
    print('\n>>>', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd, **kwargs)
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {cmd}')


os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

import torch  # noqa: E402

name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print('GPU:', name, capability, 'torch', torch.__version__, 'arch', torch.cuda.get_arch_list(), flush=True)
if f'sm_{capability[0]}{capability[1]}' not in torch.cuda.get_arch_list():
    # 최신 Kaggle torch 빌드는 P100(sm_60)을 빼는 경우가 있다. 지원하는 마지막 계열로 내린다.
    print('이 GPU를 지원하지 않는 torch 빌드 -> torch 2.5.1 (cu121) 설치', flush=True)
    run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.5.1', 'torchvision==0.20.1',
         '--index-url', 'https://download.pytorch.org/whl/cu121'])
    os.execv(sys.executable, [sys.executable] + sys.argv)   # 새 torch로 다시 시작

if not os.path.exists(REPO):
    run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/fashn-AI/fashn-vton-1.5.git', REPO])
run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPO])
if not os.path.exists(os.path.join(WEIGHTS, 'model.safetensors')):
    run([sys.executable, os.path.join(REPO, 'scripts', 'download_weights.py'), '--weights-dir', WEIGHTS])

from PIL import Image, ImageOps  # noqa: E402

persons, garments = {}, {}
for root, _dirs, files in os.walk('/kaggle/input'):
    for f in files:
        path = os.path.join(root, f)
        if f == 'KakaoTalk_20260909_190109015_02.jpg':
            persons['kakao_front_original'] = path
        elif root.endswith('inputs/image') and f in ('demo_person0_full.jpg', 'demo_person1_full.jpg', 'kakao_front_upper.jpg'):
            persons[os.path.splitext(f)[0]] = path
        elif f == 'garment_tee_khaki.jpg':
            garments['tee_khaki'] = path
        elif f == 'garment_shirt_blue.jpg':
            garments['shirt_blue'] = path
        elif f == 'shirts.jpg':
            garments['sweatshirt_text'] = path
        elif root.endswith('inputs/cloth') and f == 'demo_cardigan.jpg':
            garments['demo_cardigan'] = path
print('인물:', sorted(persons), '\n옷:', sorted(garments), flush=True)
if len(persons) != 4 or len(garments) != 4:
    tree = [f'  {r}: {sorted(fs)[:5]}' for r, _d, fs in os.walk('/kaggle/input')]
    sys.exit('입력 개수가 예상(인물 4, 옷 4)과 다릅니다.\n' + '\n'.join(tree[:30]))

from fashn_vton import TryOnPipeline  # noqa: E402

t0 = time.time()
pipeline = TryOnPipeline(weights_dir=WEIGHTS)
print(f'모델 로딩 {time.time() - t0:.1f}초, dtype {pipeline.inference_dtype}, '
      f'입력 크기 {pipeline.tryon_model.input_shape}', flush=True)

with open(os.path.join(OUT, 'results.csv'), 'w', encoding='utf-8') as log:
    log.write('person,garment,seed,seconds,width,height,peak_mem_gb\n')
    for person_name, person_path in sorted(persons.items()):
        person = ImageOps.exif_transpose(Image.open(person_path)).convert('RGB')
        for garment_name, garment_path in sorted(garments.items()):
            garment = ImageOps.exif_transpose(Image.open(garment_path)).convert('RGB')
            for seed in SEEDS:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.time()
                result = pipeline(person_image=person, garment_image=garment, category='tops',
                                  garment_photo_type='flat-lay', num_timesteps=STEPS, seed=seed)
                torch.cuda.synchronize()
                seconds = time.time() - start
                image = result.images[0]
                image.save(os.path.join(OUT, f'{person_name}__{garment_name}_s{seed}.png'))
                peak = torch.cuda.max_memory_allocated() / 1e9
                log.write(f'{person_name},{garment_name},{seed},{seconds:.1f},{image.width},{image.height},{peak:.1f}\n')
                log.flush()
                print(f'  {person_name} × {garment_name} s{seed}: {seconds:.1f}초, '
                      f'{image.size}, 최대 메모리 {peak:.1f}GB', flush=True)

print('완료', flush=True)
