"""Kaggle 커널 진입점 — 합성된 옷 질감이 흐릿한 원인을 가른다.

카톡 사진 결과를 보고 "잘 입혀지는데 질감이 흐릿하다"는 지적이 나왔다. 후보는 둘이다.

1. 해상도: 3024x4032 전신 사진을 768x1024에 통째로 넣어 옷이 폭 200px 남짓이 된다.
   -> 줌(옷 영역만 원본에서 잘라 합성 후 원본에 붙이기)
2. 샘플러: 저장소 데모 결과에서 DPM++ 8/30/50스텝은 똑같이 무르고 DDIM 50만 선명했다.
   -> DDIM 30스텝

기준(DPM++ 8스텝, 전처리 사진)은 kaggle_kakao 결과를 그대로 쓴다.
인물 원본은 vfa-person-full(jpg), 전처리본은 vfa-person-photos(png)에 있다.
"""
import os
import shutil
import subprocess
import sys

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'
WORK = '/kaggle/tmp/texture'

os.makedirs('/kaggle/tmp', exist_ok=True)
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)

garments_dir = os.path.join(WORK, 'garments')
full_dir = os.path.join(WORK, 'persons_full')
prepared_dir = os.path.join(WORK, 'persons_prepared')
for d in (garments_dir, full_dir, prepared_dir):
    os.makedirs(d, exist_ok=True)

for root, _dirs, files in os.walk('/kaggle/input'):
    for name in files:
        path = os.path.join(root, name)
        if name.startswith('garment_'):
            shutil.copy(path, os.path.join(garments_dir, name[len('garment_'):]))
        elif name.startswith('KakaoTalk_') and name.lower().endswith('.jpg'):
            shutil.copy(path, os.path.join(full_dir, name))
        elif name.startswith('KakaoTalk_') and name.lower().endswith('.png'):
            shutil.copy(path, os.path.join(prepared_dir, name))

counts = {d: len(os.listdir(d)) for d in (garments_dir, full_dir, prepared_dir)}
print(counts, flush=True)
if any(n != 3 for n in counts.values()):
    tree = [f'  {r}: {sorted(f)[:6]}' for r, _d, f in os.walk('/kaggle/input')]
    sys.exit('입력 개수가 예상(각 3)과 다릅니다.\n' + '\n'.join(tree[:20]))


def run(*args):
    print('\n>>>', ' '.join(args), flush=True)
    rc = subprocess.call([sys.executable, '-u'] + list(args))
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {args}')


subprocess.call('apt-get -qq install -y fonts-nanum > /dev/null 2>&1', shell=True)
run(os.path.join(REPO_DIR, 'scripts', 'setup_env.py'))
CHECK = os.path.join(REPO_DIR, 'scripts', 'real_person_check.py')
FULL = ['--persons', full_dir, '--garments', garments_dir, '--seeds', '42']

# 싼 것부터: 줌만 켠 것 -> 줌 + DDIM -> DDIM만
run(CHECK, *FULL, '--zoom', '--tag', 'zoom_dpm8')
run(CHECK, *FULL, '--zoom', '--scheduler', 'ddim', '--steps', '30', '--tag', 'zoom_ddim30')
run(CHECK, '--persons', prepared_dir, '--garments', garments_dir, '--seeds', '42',
    '--scheduler', 'ddim', '--steps', '30', '--tag', 'ddim30')
