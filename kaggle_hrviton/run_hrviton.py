"""Kaggle 커널 진입점 — 변형(warping)+GAN 방식 HR-VITON을 우리 사진·옷에 돌려 CatVTON과 비교한다.

diffusion 말고 다른 방식도 되는지 확인하는 비교 실험이다. HR-VITON은 CatVTON과 같은 768x1024이라
해상도 조건이 같다. 원래 전처리 도구(OpenPose, CIHP_PGN)는 쓰지 않고 우리가 가진 것으로 대신한다
(scripts/hrviton_prep.py 설명 참고). 그래서 결과가 나쁘면 모델 한계인지 전처리 차이인지를
중간 산출물(HR-VITON grid 이미지)로 같이 확인해야 한다.

인물: 데모 남·여 2명, 카톡 정자세(전신 그대로 / 상반신 크롭)
옷:   카키 반팔, 블루 셔츠, 글자 맨투맨, 데모 스누피 카디건
"""
import glob
import os
import shutil
import subprocess
import sys
import time

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'
HRV_URL = 'https://github.com/sangyun884/HR-VITON.git'
HRV_DIR = '/kaggle/tmp/HR-VITON'
WORK = '/kaggle/tmp/hrv_inputs'
DATAROOT = '/kaggle/tmp/hrv_data'
OUT = '/kaggle/working/hrviton'
# README 의 Google Drive 체크포인트 (condition generator, image generator)
CHECKPOINTS = {'mtviton.pth': '1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ',
               'gen.pth': '1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy'}
KAKAO_PERSON = 'KakaoTalk_20260909_190109015_02.jpg'
GARMENTS = {'garment_tee_khaki.jpg': 'tee_khaki.jpg',
            'garment_shirt_blue.jpg': 'shirt_blue.jpg',
            'shirts.jpg': 'sweatshirt_text.jpg'}

os.makedirs('/kaggle/tmp', exist_ok=True)
for url, path in ((REPO_URL, REPO_DIR), (HRV_URL, HRV_DIR)):
    if not os.path.exists(path):
        subprocess.run(['git', 'clone', '-q', '--depth', '1', url, path], check=True)


def run(cmd, **kwargs):
    print('\n>>>', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd, **kwargs)
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {cmd}')


run([sys.executable, '-u', os.path.join(REPO_DIR, 'scripts', 'setup_env.py')])
sys.path.insert(0, os.path.join(REPO_DIR, 'app'))
from paths import VENV_PY  # noqa: E402

# 포즈(MediaPipe)·로그(tensorboardX)·체크포인트 다운로드(gdown)는 venv에 없다.
# numpy 를 같이 고정해야 한다 — v1에서 mediapipe가 numpy 2.x를 끌어와 torch 2.1.2가
# "Numpy is not available"로 죽었다.
run([VENV_PY, '-m', 'pip', 'install', '-q', 'mediapipe==0.10.14', 'numpy==1.26.4', 'tensorboardX', 'gdown'])
run([VENV_PY, '-c', 'import numpy, torch, mediapipe; print("numpy", numpy.__version__, "torch", torch.__version__); '
     'torch.from_numpy(numpy.zeros(1))'])

# HR-VITON 은 2022년 코드라 최신 라이브러리와 두 군데 안 맞는다.
#  - np.float / np.int 는 numpy 1.24 에서 제거됨
#  - torchgeometry 는 더 이상 유지보수되지 않음 -> 쓰는 건 GaussianBlur 하나라 torchvision 으로 대체
for py in glob.glob(os.path.join(HRV_DIR, '*.py')):
    with open(py, encoding='utf-8') as f:
        text = f.read()
    patched = (text.replace('astype(np.float)', 'astype(float)')
                   .replace('astype(np.int)', 'astype(int)'))
    if patched != text:
        with open(py, 'w', encoding='utf-8') as f:
            f.write(patched)
with open(os.path.join(HRV_DIR, 'torchgeometry.py'), 'w', encoding='utf-8') as f:
    f.write('import types\nimport torchvision\n'
            'image = types.SimpleNamespace(GaussianBlur=lambda k, s: '
            'torchvision.transforms.GaussianBlur(k, s))\n')

weights = os.path.join(HRV_DIR, 'eval_models', 'weights', 'v0.1')
os.makedirs(weights, exist_ok=True)
for name, file_id in CHECKPOINTS.items():
    target = os.path.join(weights, name)
    if not os.path.exists(target):
        run([VENV_PY, '-m', 'gdown', file_id, '-O', target])
    print(name, f'{os.path.getsize(target) / 1e6:.0f} MB', flush=True)

persons_dir = os.path.join(WORK, 'persons')
garments_dir = os.path.join(WORK, 'garments')
os.makedirs(persons_dir, exist_ok=True)
os.makedirs(garments_dir, exist_ok=True)
for root, _dirs, files in os.walk('/kaggle/input'):
    for name in files:
        if name == KAKAO_PERSON:
            shutil.copy(os.path.join(root, name), os.path.join(persons_dir, 'kakao_front.jpg'))
        elif name in GARMENTS:
            shutil.copy(os.path.join(root, name), os.path.join(garments_dir, GARMENTS[name]))

demo = next((r for r, _d, _f in os.walk('/kaggle') if r.endswith('resource/demo/example')), None)
if demo is None:
    sys.exit('CatVTON 데모 폴더를 찾지 못했습니다.')
for i, p in enumerate(sorted(glob.glob(f'{demo}/person/men/*'))[:1] + sorted(glob.glob(f'{demo}/person/women/*'))[:1]):
    shutil.copy(p, os.path.join(persons_dir, f'demo_person{i}{os.path.splitext(p)[1]}'))
shutil.copy(sorted(glob.glob(f'{demo}/condition/upper/*'))[0],
            os.path.join(garments_dir, 'demo_cardigan' + os.path.splitext(sorted(glob.glob(f'{demo}/condition/upper/*'))[0])[1]))
print('인물:', sorted(os.listdir(persons_dir)), '\n옷:', sorted(os.listdir(garments_dir)), flush=True)
if len(os.listdir(garments_dir)) != 4 or not os.path.exists(os.path.join(persons_dir, 'kakao_front.jpg')):
    sys.exit('입력이 예상(옷 4벌, 카톡 정자세 1장)과 다릅니다.')

env = dict(os.environ, HRVITON_REPO=HRV_DIR)
run([VENV_PY, '-u', os.path.join(REPO_DIR, 'scripts', 'hrviton_prep.py'),
     '--persons', persons_dir, '--garments', garments_dir,
     '--out', os.path.join(DATAROOT, 'test'), '--upper-crop', 'kakao'], env=env)

os.makedirs(OUT, exist_ok=True)
start = time.time()
run([VENV_PY, '-u', os.path.join(HRV_DIR, 'test_generator.py'),
     '--occlusion', '--cuda', 'True', '--gpu_ids', '0', '-j', '2',
     '--test_name', 'vfa', '--dataroot', DATAROOT, '--datamode', 'test',
     '--data_list', 'test_pairs.txt', '--datasetting', 'unpaired',
     '--tocg_checkpoint', os.path.join(weights, 'mtviton.pth'),
     '--gen_checkpoint', os.path.join(weights, 'gen.pth'),
     '--output_dir', os.path.join(OUT, 'output')], cwd=OUT)
print(f'HR-VITON 추론 전체 {time.time() - start:.1f}초 (모델 로딩 포함)', flush=True)

# 입력 자체도 남긴다 — 결과가 나쁠 때 전처리 탓인지 가르려면 파싱·DensePose·포즈를 봐야 한다
for folder in ('image', 'image-parse-v3', 'image-parse-agnostic-v3.2', 'image-densepose', 'cloth', 'cloth-mask', 'openpose_json'):
    shutil.copytree(os.path.join(DATAROOT, 'test', folder), os.path.join(OUT, 'inputs', folder), dirs_exist_ok=True)

for root, _dirs, files in os.walk(OUT):
    print(root, len(files), 'files', flush=True)
