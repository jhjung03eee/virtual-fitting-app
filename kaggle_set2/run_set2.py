"""Kaggle 커널 진입점 — 합성이 잘 될 조건으로 고른 옷 3벌을 검증한다.

지금까지 실험으로 정리된 조건(단색, 프린트·특수 질감 없음, 중간 톤, 제품 단독 컷)에
맞춰 무신사에서 고른 반팔·셔츠·바지를, 샘플 모델(CatVTON 데모 인물)과 본인 사진
두 명에게 입힌다. 시드에 따라 결과가 흔들리므로 시드 3개를 함께 본다.

데이터셋 파일은 접두어로 구분한다.
    garment_*.jpg  옷 사진
    person_*.png   인물 사진
"""
import os
import shutil
import subprocess
import sys

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'
WORK = '/kaggle/tmp/set2'
SEEDS = '42,123,7'

os.makedirs('/kaggle/tmp', exist_ok=True)
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)
print('repo:', REPO_DIR, flush=True)

source = None
for root, _dirs, files in os.walk('/kaggle/input'):
    if any(f.startswith('garment_') for f in files):
        source = root
        break
if source is None:
    tree = [f'  {r}: {sorted(f)[:6]}' for r, _d, f in os.walk('/kaggle/input')]
    sys.exit('vfa-set2 데이터셋을 찾지 못했습니다.\n' + '\n'.join(tree[:20]))

garments_dir = os.path.join(WORK, 'garments')
persons_dir = os.path.join(WORK, 'persons')
os.makedirs(garments_dir, exist_ok=True)
os.makedirs(persons_dir, exist_ok=True)
for name in sorted(os.listdir(source)):
    # 접두어를 떼서 복사한다. 옷 종류는 파일명으로 판별하므로(pants -> 하의) 이름이 중요하다.
    if name.startswith('garment_'):
        shutil.copy(os.path.join(source, name), os.path.join(garments_dir, name[len('garment_'):]))
    elif name.startswith('person_'):
        shutil.copy(os.path.join(source, name), os.path.join(persons_dir, name[len('person_'):]))
print('옷  :', sorted(os.listdir(garments_dir)), flush=True)
print('인물:', sorted(os.listdir(persons_dir)), flush=True)


def run(*args):
    print('\n>>>', ' '.join(args), flush=True)
    rc = subprocess.call([sys.executable, '-u'] + list(args))
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {args}')


subprocess.call('apt-get -qq install -y fonts-nanum > /dev/null 2>&1', shell=True)
run(os.path.join(REPO_DIR, 'scripts', 'setup_env.py'))
CHECK = os.path.join(REPO_DIR, 'scripts', 'real_person_check.py')
BASE = ['--persons', persons_dir, '--garments', garments_dir]

# 1차 결과(v1)에서 나온 문제를 하나씩 가른다.
# (1) 반팔: 사진에서 지운 카드지갑이 손에 들린 채 되살아났다. 흔적까지 지운 사진으로 다시.
run(CHECK, *BASE, '--only', 'tee', '--seeds', SEEDS, '--tag', 'clean')
# (2) 바지: 위장무늬처럼 얼룩졌다. 하의 기본값 guidance 5.0은 코듀로이 한 벌로 정한 값이라
#     이 바지에서도 맞는지 모른다. 2.5와 비교하고, 색 보정 유무도 함께 본다.
run(CHECK, *BASE, '--only', 'pants', '--seeds', '42,123', '--guidance', '2.5', '--tag', 'g2.5')
run(CHECK, *BASE, '--only', 'pants', '--seeds', '42,123', '--guidance', '2.5',
    '--color-match', '1.0', '--tag', 'g2.5_color')
run(CHECK, *BASE, '--only', 'pants', '--seeds', '42,123', '--guidance', '5.0',
    '--color-match', '1.0', '--tag', 'g5_color')
# (3) 셔츠: 형태는 좋은데 원본(스틸 블루)보다 연하다. 색 보정이 맞춰주는지.
run(CHECK, *BASE, '--only', 'shirt', '--seeds', '42,123', '--color-match', '1.0', '--tag', 'color')

print('\n=== /kaggle/working 결과물 ===', flush=True)
for root, _dirs, files in os.walk('/kaggle/working'):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f'{os.path.getsize(p) / 1024:8.0f} KB  {p}', flush=True)
