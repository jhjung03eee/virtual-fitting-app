"""Kaggle 커널 진입점 — 실제 쇼핑몰 옷으로 CFG/스텝 설정을 검증한다.

옷 사진은 저작권 때문에 저장소에 커밋하지 않으므로, 비공개 Kaggle 데이터셋
(jefferyjung/vfa-real-garments)으로 올려서 /kaggle/input 으로 받는다.

로컬에서:
    kaggle kernels push -p kaggle_real/
    kaggle kernels status jefferyjung/vfa-real-garment-check
    kaggle kernels output jefferyjung/vfa-real-garment-check -p <dir>

clone과 venv는 /kaggle/tmp에 두고 결과만 /kaggle/working에 남긴다.
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'
GARMENTS = '/kaggle/input/vfa-real-garments'

os.makedirs('/kaggle/tmp', exist_ok=True)
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)
print('repo:', REPO_DIR, flush=True)

if not os.path.isdir(GARMENTS):
    # 데이터셋이 /kaggle/input 바로 아래가 아니라 몇 단계 안쪽에 붙기도 한다
    # (실제로 /kaggle/input/datasets/... 로 들어왔다). 이미지가 들어 있는
    # 폴더를 재귀로 찾는다.
    found = []
    for root, _dirs, files in os.walk('/kaggle/input'):
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            found.append(root)
    if not found:
        tree = []
        for root, _dirs, files in os.walk('/kaggle/input'):
            tree.append(f'  {root}: {sorted(files)[:5]}')
        sys.exit(f'옷 사진 데이터셋이 없습니다: {GARMENTS}\n'
                 + ('\n'.join(tree[:20]) or '  /kaggle/input 이 비어 있습니다') + '\n'
                 '커널 설정에서 vfa-real-garments 데이터셋을 추가했는지 확인하세요.')
    GARMENTS = sorted(found, key=len)[0]
    print('데이터셋을 다른 경로에서 찾았습니다:', GARMENTS, flush=True)
print('옷 사진:', sorted(os.listdir(GARMENTS)), flush=True)


def run(*args):
    print('\n>>>', ' '.join(args), flush=True)
    rc = subprocess.call([sys.executable, '-u'] + list(args))
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {args}')


run(os.path.join(REPO_DIR, 'scripts', 'setup_env.py'))
run(os.path.join(REPO_DIR, 'scripts', 'real_garment_check.py'), '--garments', GARMENTS)

print('\n=== /kaggle/working 결과물 ===', flush=True)
for root, _dirs, files in os.walk('/kaggle/working'):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f'{os.path.getsize(p) / 1024:8.0f} KB  {p}', flush=True)
