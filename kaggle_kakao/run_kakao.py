"""Kaggle 커널 진입점 — 카톡으로 받은 인물 사진 3장에 새로 고른 옷 3벌을 입힌다.

인물 사진(정자세 / 팔 벌림 / 엄지척)은 vfa-person-photos 데이터셋에 전처리된 채로 있고,
옷(카키 반팔 / 블루 셔츠 / 올리브 바지)은 vfa-set2 데이터셋에 garment_ 접두어로 있다.
현재 기본 설정(DPM++ 8스텝, composite, guidance 2.5)으로 시드 42/123을 본다.
"""
import os
import shutil
import subprocess
import sys

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'
WORK = '/kaggle/tmp/kakao'

os.makedirs('/kaggle/tmp', exist_ok=True)
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)

garments_dir = os.path.join(WORK, 'garments')
persons_dir = os.path.join(WORK, 'persons')
os.makedirs(garments_dir, exist_ok=True)
os.makedirs(persons_dir, exist_ok=True)

for root, _dirs, files in os.walk('/kaggle/input'):
    for name in files:
        path = os.path.join(root, name)
        if name.startswith('garment_'):
            shutil.copy(path, os.path.join(garments_dir, name[len('garment_'):]))
        elif name.startswith('KakaoTalk_') and name.lower().endswith('.png'):
            shutil.copy(path, os.path.join(persons_dir, name))

print('옷  :', sorted(os.listdir(garments_dir)), flush=True)
print('인물:', sorted(os.listdir(persons_dir)), flush=True)
if len(os.listdir(garments_dir)) != 3 or len(os.listdir(persons_dir)) != 3:
    tree = [f'  {r}: {sorted(f)[:6]}' for r, _d, f in os.walk('/kaggle/input')]
    sys.exit('입력 개수가 예상(옷 3, 인물 3)과 다릅니다.\n' + '\n'.join(tree[:20]))


def run(*args):
    print('\n>>>', ' '.join(args), flush=True)
    rc = subprocess.call([sys.executable, '-u'] + list(args))
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {args}')


subprocess.call('apt-get -qq install -y fonts-nanum > /dev/null 2>&1', shell=True)
run(os.path.join(REPO_DIR, 'scripts', 'setup_env.py'))
run(os.path.join(REPO_DIR, 'scripts', 'real_person_check.py'),
    '--persons', persons_dir, '--garments', garments_dir, '--seeds', '42,123', '--tag', 'kakao')
