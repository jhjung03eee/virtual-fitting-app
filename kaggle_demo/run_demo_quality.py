"""Kaggle 커널 진입점 — 저장소 데모 데이터로 현재 설정의 품질 천장을 잰다.

데모 이미지는 CatVTON 저장소에 들어 있으므로 별도 데이터셋이 필요 없다.
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'

os.makedirs('/kaggle/tmp', exist_ok=True)
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)


def run(*args):
    print('\n>>>', ' '.join(args), flush=True)
    rc = subprocess.call([sys.executable, '-u'] + list(args))
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {args}')


subprocess.call('apt-get -qq install -y fonts-nanum > /dev/null 2>&1', shell=True)
run(os.path.join(REPO_DIR, 'scripts', 'setup_env.py'))
run(os.path.join(REPO_DIR, 'scripts', 'demo_quality_check.py'), '--seeds', '42,123')
