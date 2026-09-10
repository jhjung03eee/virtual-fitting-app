"""Kaggle 커널 진입점 — 품질 관련 설정 비교.

기본은 composite 묶음: 마스크 밖을 원본으로 되돌리면 guidance를 올려도
얼굴·배경이 버티는지 본다. VFA_CONFIG_SET 으로 quality/steps 로 바꿀 수 있다.

실제 인물 사진에서 하의 색이 틀리고 프린트가 사라졌다. 원인이
(a) 속도를 위해 깎은 설정인지 (b) 모델 자체의 한계인지 갈라야
다음 결정(설정을 되돌린다 vs 모델을 바꾼다)이 선다.

스텝과 guidance를 최대로 올려서 이 모델로 낼 수 있는 최선을 본다.
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'

os.makedirs('/kaggle/tmp', exist_ok=True)
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)
print('repo:', REPO_DIR, flush=True)


def find_dataset(keyword):
    for root, _dirs, files in os.walk('/kaggle/input'):
        if keyword in os.path.basename(root).lower() and any(
                f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            return root
    return None


persons_dir = find_dataset('person')
garments = find_dataset('garment')
if not persons_dir or not garments:
    listing = []
    for root, _dirs, files in os.walk('/kaggle/input'):
        listing.append(f'  {root}: {sorted(files)[:5]}')
    sys.exit('입력 데이터셋을 찾지 못했습니다.\n' + '\n'.join(listing[:20]))

# 정자세 사진(_02)을 쓴다. 자세가 가장 평범해서 설정 차이만 드러난다.
candidates = sorted(f for f in os.listdir(persons_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')))
person = os.path.join(persons_dir, next(
    (f for f in candidates if f.endswith('_02.png')), candidates[0]))
print('인물:', person, flush=True)
print('옷  :', garments, flush=True)


def run(*args):
    print('\n>>>', ' '.join(args), flush=True)
    rc = subprocess.call([sys.executable, '-u'] + list(args))
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {args}')


subprocess.call('apt-get -qq install -y fonts-nanum > /dev/null 2>&1', shell=True)
run(os.path.join(REPO_DIR, 'scripts', 'setup_env.py'))
run(os.path.join(REPO_DIR, 'scripts', 'real_garment_check.py'),
    '--garments', garments, '--person', person, '--config-set', os.environ.get('VFA_CONFIG_SET', 'background'))

print('\n=== /kaggle/working 결과물 ===', flush=True)
for root, _dirs, files in os.walk('/kaggle/working'):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f'{os.path.getsize(p) / 1024:8.0f} KB  {p}', flush=True)
