"""Kaggle 커널 진입점 — 하이브리드(HR-VITON 변형 초안 + CatVTON) vs CatVTON 단독.

kaggle_hrviton 과 같은 순서로 HR-VITON을 돌리되, 변형된 옷(warped cloth)과 예측 상의 영역을
따로 저장하도록 test_generator.py 를 고친다. 그 다음 scripts/hybrid_check.py 가
CatVTON 단독과 하이브리드(strength 0.5/0.7/0.9)를 같은 시드로 비교한다. docs/PLAN.md F-9.
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
# 하이브리드용: 변형된 옷과 예측 상의 영역(7클래스 파싱의 2번)을 따로 저장한다
generator_py = os.path.join(HRV_DIR, 'test_generator.py')
with open(generator_py, encoding='utf-8') as f:
    text = f.read()
anchor = '            save_images(output, unpaired_names, output_dir)\n'
if anchor not in text:
    sys.exit('test_generator.py 저장 위치를 찾지 못했습니다 (HR-VITON 코드가 바뀜).')
text = text.replace(anchor, anchor +
                    "            os.makedirs(output_dir + '_warp', exist_ok=True)\n"
                    "            os.makedirs(output_dir + '_region', exist_ok=True)\n"
                    "            save_images(warped_cloth, unpaired_names, output_dir + '_warp')\n"
                    "            save_images(parse[:, 2:3] * 2 - 1, unpaired_names, output_dir + '_region')\n")
with open(generator_py, 'w', encoding='utf-8') as f:
    f.write(text)

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

for sub in ('output_warp', 'output_region'):
    print(sub, sorted(os.listdir(os.path.join(OUT, sub)))[:3], flush=True)

# 하이브리드 비교. 전신 사진은 HR-VITON 변형이 무너졌으므로(F-8) 상반신 크롭과 데모 남성만 쓴다.
# 스텝은 DDIM 30 — 비교 조건 11개 × 170초는 커널 시간을 넘으므로 상대 비교용으로 줄였다.
run([VENV_PY, '-u', os.path.join(REPO_DIR, 'scripts', 'hybrid_check.py'),
     '--data', os.path.join(DATAROOT, 'test'), '--warp', os.path.join(OUT, 'output'),
     '--out', '/kaggle/working/hybrid',
     '--persons', 'kakao_front_upper,demo_person0_full',
     '--garments', 'sweatshirt_text,demo_cardigan,tee_khaki',
     '--strengths', '0.5,0.7,0.9', '--seeds', '42,123', '--steps', '30'])
