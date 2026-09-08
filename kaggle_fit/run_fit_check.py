"""체형 추정 + 사이즈 추천을 실제 사진으로 검증한다 (GPU 불필요).

mediapipe만 있으면 되므로 CatVTON venv 전체를 만들지 않고
Kaggle 기본 파이썬에 mediapipe만 설치해서 돌린다.
"""
import glob
import os
import subprocess
import sys

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'
CATVTON_DIR = '/kaggle/tmp/scratch/CatVTON'

os.makedirs('/kaggle/tmp/scratch', exist_ok=True)
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)
if not os.path.exists(CATVTON_DIR):
    # 데모 인물 이미지만 필요해서 얕게 받는다
    subprocess.run(['git', 'clone', '-q', '--depth', '1',
                    'https://github.com/Zheng-Chong/CatVTON.git', CATVTON_DIR], check=True)

print('mediapipe 설치 중...', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mediapipe'], check=True)
import mediapipe as mp  # noqa: E402
print('mediapipe', mp.__version__, flush=True)

sys.path.insert(0, os.path.join(REPO_DIR, 'app'))
os.environ['CATVTON_REPO'] = CATVTON_DIR
os.environ['VFA_NO_REEXEC'] = '1'

print('\n=== 단위 테스트 ===', flush=True)
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'],
               cwd=REPO_DIR, check=True)

persons = sorted(glob.glob(os.path.join(CATVTON_DIR, 'resource/demo/example/person/*/*')))
print(f'\n=== 실제 사진 {len(persons)}장 검증 ===', flush=True)

script = os.path.join(REPO_DIR, 'scripts', 'fit_check.py')
for p in persons[:6]:
    print('\n' + '=' * 70, flush=True)
    subprocess.run([sys.executable, '-u', script, '--person', p, '--height', '175'],
                   cwd=REPO_DIR, env=dict(os.environ))
