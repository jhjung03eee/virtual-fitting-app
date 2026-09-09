"""Kaggle 커널 진입점 — 실제 인물 사진으로 합성해 실패 케이스를 모은다.

지금까지 테스트는 전부 CatVTON 데모 이미지(스튜디오 정면 정자세)였다.
실제 사용자 사진에서 팔 겹침·복잡한 배경이 어떻게 나오는지 본다.

인물 사진과 옷 사진은 각각 다른 비공개 데이터셋에 있다. 한 폴더에 섞으면
스크립트가 인물 사진을 옷으로 오인한다.
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
    """이름에 keyword가 든 입력 폴더를 찾는다."""
    for root, _dirs, files in os.walk('/kaggle/input'):
        if keyword in os.path.basename(root).lower() and any(
                f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            return root
    return None


persons = find_dataset('person')
garments = find_dataset('garment')
if not persons or not garments:
    listing = []
    for root, _dirs, files in os.walk('/kaggle/input'):
        listing.append(f'  {root}: {sorted(files)[:5]}')
    sys.exit('입력 데이터셋을 찾지 못했습니다.\n' + '\n'.join(listing[:20]))
print('인물:', persons, sorted(os.listdir(persons)), flush=True)
print('옷  :', garments, sorted(os.listdir(garments)), flush=True)


def run(*args):
    print('\n>>>', ' '.join(args), flush=True)
    rc = subprocess.call([sys.executable, '-u'] + list(args))
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {args}')


subprocess.call('apt-get -qq install -y fonts-nanum > /dev/null 2>&1', shell=True)
run(os.path.join(REPO_DIR, 'scripts', 'setup_env.py'))
run(os.path.join(REPO_DIR, 'scripts', 'real_person_check.py'),
    '--persons', persons, '--garments', garments)

print('\n=== /kaggle/working 결과물 ===', flush=True)
for root, _dirs, files in os.walk('/kaggle/working'):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f'{os.path.getsize(p) / 1024:8.0f} KB  {p}', flush=True)
