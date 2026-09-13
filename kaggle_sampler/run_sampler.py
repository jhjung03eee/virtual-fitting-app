"""Kaggle 커널 진입점 — 흐림의 원인이 속도 최적화(DPM++ 8스텝)인지 확정한다.

저장소 기본값은 DDIM 50스텝인데 우리는 SSIM·썸네일 비교로 "8스텝과 구별 안 됨"이라 보고
DPM++ 8스텝으로 바꿨다. SSIM은 미세 질감을 못 본다. 원본 해상도로 다시 보니
같은 맨투맨이 DDIM 50에서는 글자·시보리가 또렷하고 DPM++ 8에서는 번졌다.

  1. 저장소 데모 인물 2명 × 데모 상의 2벌 (깃헙 예시와 같은 조건)
  2. 카톡 정자세 원본 사진 × 상의 3벌(글자 맨투맨, 카키 반팔, 블루 셔츠), 줌 적용
각각 DPM++ 8 / DDIM 50 / DDIM 30, 시드 42·123.
"""
import glob
import os
import shutil
import subprocess
import sys

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'
WORK = '/kaggle/tmp/sampler'
SEEDS = '42,123'
KAKAO_PERSON = 'KakaoTalk_20260909_190109015_02.jpg'   # 정자세 (따봉·팔 벌림 제외)
KAKAO_GARMENTS = {'garment_tee_khaki.jpg': 'tee_khaki.jpg',
                  'garment_shirt_blue.jpg': 'shirt_blue.jpg',
                  'shirts.jpg': 'sweatshirt_text.jpg'}

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

dirs = {name: os.path.join(WORK, name) for name in
        ('demo_persons', 'demo_garments', 'kakao_persons', 'kakao_garments')}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

for root, _dirs, files in os.walk('/kaggle/input'):
    for name in files:
        path = os.path.join(root, name)
        if name == KAKAO_PERSON:
            shutil.copy(path, os.path.join(dirs['kakao_persons'], name))
        elif name in KAKAO_GARMENTS:
            shutil.copy(path, os.path.join(dirs['kakao_garments'], KAKAO_GARMENTS[name]))

# 데모 이미지는 setup_env 가 받아둔 CatVTON 저장소 안에 있다
demo = next((r for r, _d, _f in os.walk('/kaggle') if r.endswith('resource/demo/example')), None)
if demo is None:
    sys.exit('CatVTON 데모 폴더를 찾지 못했습니다.')
persons = sorted(glob.glob(f'{demo}/person/men/*'))[:1] + sorted(glob.glob(f'{demo}/person/women/*'))[:1]
uppers = sorted(glob.glob(f'{demo}/condition/upper/*'))[:2]
for i, p in enumerate(persons):
    shutil.copy(p, os.path.join(dirs['demo_persons'], f'demo_person{i}{os.path.splitext(p)[1]}'))
for i, g in enumerate(uppers):
    shutil.copy(g, os.path.join(dirs['demo_garments'], f'demo_upper{i}{os.path.splitext(g)[1]}'))

for d in dirs.values():
    print(os.path.basename(d), sorted(os.listdir(d)), flush=True)
if len(os.listdir(dirs['kakao_persons'])) != 1 or len(os.listdir(dirs['kakao_garments'])) != 3:
    sys.exit('카톡 입력이 예상(인물 1, 옷 3)과 다릅니다.')

CHECK = os.path.join(REPO_DIR, 'scripts', 'real_person_check.py')
CONFIGS = [('dpm8', ['--scheduler', 'dpm', '--steps', '8']),
           ('ddim50', ['--scheduler', 'ddim', '--steps', '50']),
           ('ddim30', ['--scheduler', 'ddim', '--steps', '30'])]

# 설정마다 데모와 카톡을 같이 돌려, 중간에 끊겨도 앞 설정은 두 조건 모두 남게 한다
for tag, options in CONFIGS:
    run(CHECK, '--persons', dirs['demo_persons'], '--garments', dirs['demo_garments'],
        '--seeds', SEEDS, '--tag', f'demo_{tag}', *options)
    # 줌은 옷이 작게 찍힌 전신 사진에서만 켜진다(데모 상반신 사진은 자동으로 건너뜀)
    run(CHECK, '--persons', dirs['kakao_persons'], '--garments', dirs['kakao_garments'],
        '--seeds', SEEDS, '--zoom', '--tag', f'kakao_zoom_{tag}', *options)
