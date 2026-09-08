"""이전 벤치마크 커널의 출력을 입력으로 받아 리포트만 다시 생성한다 (GPU 불필요).

kernel_sources로 jefferyjung/vfa-benchmark 를 마운트하면
그 커널의 /kaggle/working 산출물이 /kaggle/input/vfa-benchmark 아래에 붙는다.
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'
BENCH_DIR = '/kaggle/input/vfa-benchmark/bench'

os.makedirs('/kaggle/tmp', exist_ok=True)
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)

if not os.path.isdir(BENCH_DIR):
    print('입력 폴더 목록:', flush=True)
    for root, dirs, files in os.walk('/kaggle/input'):
        print(root, dirs[:5], files[:5], flush=True)
    sys.exit(f'{BENCH_DIR} 가 없습니다.')

# 리포트 생성에 필요한 것만 (GPU/torch 불필요)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'scikit-image', 'matplotlib', 'pillow', 'numpy'], check=True)

env = dict(os.environ, VFA_NO_REEXEC='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
rc = subprocess.call([sys.executable, '-u',
                      os.path.join(REPO_DIR, 'scripts', 'make_report.py'),
                      '--bench-dir', BENCH_DIR,
                      '--out-dir', '/kaggle/working/report'], env=env)
sys.exit(rc)
