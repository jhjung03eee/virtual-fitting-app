"""Kaggle 커널 진입점 — 앱 코드(app/fashn_core.py + gradio_app.py)가 실제 GPU에서 도는지 확인한다.

지금까지 FASHN은 커널 안에서 파이프라인을 직접 호출해 검증했다(F-10~F-14). 앱에 연결한 뒤에는
**앱 코드 경로로도 같은 결과가 나오는지**를 따로 봐야 한다. 확인 항목:
  1. fashn_core.load_models() 가 GPU에 맞는 정밀도를 고르는가 (T4 → fp16)
  2. fashn_core.try_on() 결과가 F-14 커널 결과와 같은가 (같은 인물·옷·시드·설정)
  3. gradio_app.run() 이 합성 + 사이즈 추천까지 한 번에 도는가 (배선)
  4. Gradio 서버가 실제로 뜨고 페이지가 응답하는가
앱 코드는 GitHub에서 받는다(kaggle_ui/ui_smoke.py와 같은 방식).
"""
import os
import subprocess
import sys
import time

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
WORK = '/kaggle/tmp/appcheck'
APP_REPO = os.path.join(WORK, 'repo')
FASHN_REPO = os.path.join(WORK, 'fashn-vton-1.5')
FASHN_WEIGHTS = os.path.join(WORK, 'fashn-weights')
OUT = '/kaggle/working/app_check'
# app/paths.py 가 읽는다. 앱이 저장소·가중치를 어디서 찾을지 알려준다
os.environ['FASHN_REPO'] = FASHN_REPO
os.environ['FASHN_WEIGHTS'] = FASHN_WEIGHTS
os.environ['HF_HOME'] = os.path.join(WORK, 'hf')   # 사람 파싱 가중치 캐시(약 244MB)

# F-14와 같은 조건. 같은 이미지가 나와야 앱 경로가 맞다
PERSON = 'kakao_front_upper.jpg'
GARMENT = 'shirts.jpg'          # 글자 맨투맨
CLOTH_TYPE = 'upper'
SEED = 42


def run(cmd):
    print('\n>>>', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f'실패 (exit {rc}): {cmd}')


os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

import torch  # noqa: E402

capability = torch.cuda.get_device_capability(0)
if f'sm_{capability[0]}{capability[1]}' not in torch.cuda.get_arch_list():
    run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.5.1', 'torchvision==0.20.1',
         '--index-url', 'https://download.pytorch.org/whl/cu121'])
    os.execv(sys.executable, [sys.executable] + sys.argv)
print('GPU:', torch.cuda.get_device_name(0), capability, 'torch', torch.__version__, flush=True)

if not os.path.exists(APP_REPO):
    run(['git', 'clone', '-q', REPO_URL, APP_REPO])
if not os.path.exists(FASHN_REPO):
    run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/fashn-AI/fashn-vton-1.5.git', FASHN_REPO])
run([sys.executable, '-m', 'pip', 'install', '-q', '-e', FASHN_REPO])
if not os.path.exists(os.path.join(FASHN_WEIGHTS, 'model.safetensors')):
    run([sys.executable, os.path.join(FASHN_REPO, 'scripts', 'download_weights.py'),
         '--weights-dir', FASHN_WEIGHTS])

person_path = garment_path = None
for root, _dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == PERSON and root.endswith('inputs/image'):
            person_path = os.path.join(root, f)
        elif f == GARMENT:
            garment_path = os.path.join(root, f)
if not person_path or not garment_path:
    sys.exit(f'입력을 찾지 못했습니다: person={person_path}, garment={garment_path}')

sys.path.insert(0, os.path.join(APP_REPO, 'app'))
sys.path.insert(0, os.path.join(FASHN_REPO, 'src'))

# --- 1. 엔진 로딩과 정밀도 ------------------------------------------------
import fashn_core  # noqa: E402

t0 = time.time()
pipeline = fashn_core.load_models()
print(f'[1] 모델 로딩 {time.time() - t0:.1f}초, 정밀도 {pipeline.inference_dtype}, '
      f'입력 크기 {pipeline.tryon_model.input_shape}', flush=True)
expected = 'torch.float16' if capability[0] < 8 else 'torch.bfloat16'
print(f'    기대한 정밀도 {expected} → {"맞음" if str(pipeline.inference_dtype) == expected else "다름!"}', flush=True)

# --- 2. try_on 결과가 F-14 커널 결과와 같은가 ------------------------------
result, timing = fashn_core.try_on(person_path, garment_path, cloth_type=CLOTH_TYPE,
                                   seed=SEED, return_timing=True)
name = f'app__{os.path.splitext(PERSON)[0]}__{os.path.splitext(GARMENT)[0]}__s{SEED}.png'
result.save(os.path.join(OUT, name))
print(f'[2] try_on 성공: {result.size}, {timing}', flush=True)
print(f'    저장 {name} — 로컬에서 F-14 결과(D__kakao_front_upper__sweatshirt_text__st50_g2.5_s42_fp16.png)와 비교할 것',
      flush=True)

# --- 3. gradio_app.run() 배선 ---------------------------------------------
import gradio_app  # noqa: E402

charts = gradio_app.list_charts('upper')
print('[3] 상의 치수표:', charts, flush=True)
out_image, advice, info = gradio_app.run(
    person_path, garment_path, '상의', '상품 사진 (옷만 펴 놓고 찍음)',
    fashn_core.DEFAULT_STEPS, fashn_core.DEFAULT_GUIDANCE, SEED,
    charts[0] if charts else None, 175, None, 96, 80, None)
out_image.save(os.path.join(OUT, 'app__gradio_run.png'))
print('    합성 결과:', out_image.size, '|', info, flush=True)
print('    사이즈 추천:\n' + '\n'.join('      ' + line for line in advice.splitlines()[:8]), flush=True)

# --- 4. Gradio 서버가 실제로 뜨는가 ----------------------------------------
import urllib.request  # noqa: E402

_app, local_url, _share = gradio_app.demo.queue().launch(
    share=False, show_error=True, show_api=False, prevent_thread_lock=True)
try:
    with urllib.request.urlopen(local_url, timeout=30) as response:
        body = response.read(2000).decode('utf-8', 'replace')
    print(f'[4] Gradio 응답 {response.status}, 페이지 {len(body)}바이트, '
          f'제목 포함={"사이즈" in body or "gradio" in body.lower()}', flush=True)
finally:
    gradio_app.demo.close()
print('완료', flush=True)
