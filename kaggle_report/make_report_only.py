"""이전 벤치마크 커널의 출력을 입력으로 받아 리포트만 다시 생성한다 (GPU 불필요).

kernel_sources로 jefferyjung/vfa-benchmark 를 마운트하면
그 커널의 /kaggle/working 산출물이 /kaggle/input/vfa-benchmark 아래에 붙는다.
"""
import os
import subprocess
import sys

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'

def find_bench_dir():
    """kernel_sources 마운트 위치가 버전에 따라 달라서 results.csv를 직접 찾는다.
    (실측: /kaggle/input/notebooks/<user>/<kernel>/bench)"""
    for root, _dirs, files in os.walk('/kaggle/input'):
        if 'results.csv' in files:
            return root
    return None


os.makedirs('/kaggle/tmp', exist_ok=True)
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)

BENCH_DIR = find_bench_dir()
if not BENCH_DIR:
    print('입력 폴더 목록:', flush=True)
    for root, dirs, files in os.walk('/kaggle/input'):
        print(root, dirs[:5], files[:5], flush=True)
    sys.exit('results.csv 를 /kaggle/input 아래에서 찾지 못했습니다.')
print('bench dir:', BENCH_DIR, flush=True)

# 한글 폰트가 없으면 차트 축·범례가 두부(□)로 깨진다
subprocess.run('apt-get install -y -qq fonts-nanum > /dev/null 2>&1', shell=True)
subprocess.run('fc-cache -f > /dev/null 2>&1', shell=True)
for cache in ('/root/.cache/matplotlib', os.path.expanduser('~/.cache/matplotlib')):
    subprocess.run(f'rm -rf {cache}', shell=True)

# 리포트 생성에 필요한 것만 (GPU/torch 불필요)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'scikit-image', 'matplotlib', 'pillow', 'numpy'], check=True)

env = dict(os.environ, VFA_NO_REEXEC='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
rc = subprocess.call([sys.executable, '-u',
                      os.path.join(REPO_DIR, 'scripts', 'make_report.py'),
                      '--bench-dir', BENCH_DIR,
                      '--out-dir', '/kaggle/working/report'], env=env)
sys.exit(rc)
