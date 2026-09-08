"""Kaggle 커널 진입점 — 저장소를 받아 환경 구성 후 벤치마크와 리포트를 실행한다.

로컬에서:
    kaggle kernels push -p kaggle/
    kaggle kernels status jefferyjung/vfa-benchmark
    kaggle kernels output jefferyjung/vfa-benchmark -p <dir> --file-pattern "report/.*"

clone과 venv는 /kaggle/tmp에 두고 결과만 /kaggle/working에 남긴다
(working 아래는 전부 커널 output으로 잡혀서 받아올 때 수천 개 파일이 된다).
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


def run(*args):
    print('\n>>>', ' '.join(args), flush=True)
    rc = subprocess.call([sys.executable, '-u'] + list(args))
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {args}')


run(os.path.join(REPO_DIR, 'scripts', 'setup_env.py'))
run(os.path.join(REPO_DIR, 'scripts', 'benchmark.py'), '--suite', 'all')
run(os.path.join(REPO_DIR, 'scripts', 'make_report.py'))

print('\n=== /kaggle/working 결과물 ===', flush=True)
for root, _dirs, files in os.walk('/kaggle/working'):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f'{os.path.getsize(p) / 1024:8.0f} KB  {p}', flush=True)
