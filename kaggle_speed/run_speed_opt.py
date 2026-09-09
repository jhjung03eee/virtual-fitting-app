"""Kaggle 커널 진입점 — 스텝당 비용 최적화 측정.

스텝을 줄이는 가속은 8스텝에서 한계에 부딪혔다. 여기서는 샘플링 수학을 그대로 둔 채
channels_last / torch.compile 로 스텝당 비용만 줄여보고, 결과가 정말 같은지 본다.

로컬에서:
    kaggle kernels push -p kaggle_speed/
    kaggle kernels status jefferyjung/vfa-speed-opt
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

# 옷 사진은 저작권상 저장소에 없다. 비공개 데이터셋에서 찾는다.
garments = None
for root, _dirs, files in os.walk('/kaggle/input'):
    if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
        garments = root
        break
if garments is None:
    sys.exit('옷 사진 데이터셋을 찾지 못했습니다. vfa-real-garments 를 커널에 추가하세요.')
print('옷 사진:', garments, sorted(os.listdir(garments)), flush=True)


def run(*args):
    print('\n>>>', ' '.join(args), flush=True)
    rc = subprocess.call([sys.executable, '-u'] + list(args))
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {args}')


subprocess.call('apt-get -qq install -y fonts-nanum > /dev/null 2>&1', shell=True)
run(os.path.join(REPO_DIR, 'scripts', 'setup_env.py'))
run(os.path.join(REPO_DIR, 'scripts', 'speed_opt_check.py'), '--garments', garments)

print('\n=== /kaggle/working 결과물 ===', flush=True)
for root, _dirs, files in os.walk('/kaggle/working'):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f'{os.path.getsize(p) / 1024:8.0f} KB  {p}', flush=True)
