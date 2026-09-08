"""UI가 실제로 뜨고 응답하는지 검증한다 (GPU·모델 로딩 없이).

배치 커널에서는 사람이 브라우저로 접속할 수 없으므로,
서버를 띄운 뒤 로컬에서 HTTP 요청을 보내 확인하는 방식으로 검증한다.

확인 항목
1. gradio_app 모듈이 import 되는가 (문법·의존성)
2. Blocks가 구성되는가 (컴포넌트 정의 오류 없음)
3. 서버가 실제로 뜨고 200을 돌려주는가
4. 사이즈 추천 로직이 UI 함수 경로로 동작하는가
"""
import os
import subprocess
import sys
import types
import urllib.request

REPO_URL = 'https://github.com/jhjung03eee/virtual-fitting-app.git'
REPO_DIR = '/kaggle/tmp/vfa'

os.makedirs('/kaggle/tmp', exist_ok=True)
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)

# 배포 환경(venv39)과 같은 버전 조합으로 맞춘다.
# Kaggle 기본 huggingface_hub는 1.x라 HfFolder가 없어서 gradio 4.44.1 import가 깨진다.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'gradio==4.44.1', 'huggingface_hub==0.23.4'], check=True)

# CatVTON 파이프라인은 GPU와 4GB 가중치가 필요하다. UI 검증에는 불필요하므로
# tryon_core를 가짜 모듈로 바꿔치기해서 무거운 import를 피한다.
fake = types.ModuleType('tryon_core')
fake.REPO_DIR = '/kaggle/tmp/none'
fake.load_models = lambda *a, **k: None


def _fake_try_on(**kwargs):
    from PIL import Image
    return Image.new('RGB', (76, 102), 'gray'), Image.new('RGB', (76, 102), 'black')


fake.try_on = _fake_try_on
sys.modules['tryon_core'] = fake

sys.path.insert(0, os.path.join(REPO_DIR, 'app'))
os.chdir(REPO_DIR)

print('=== 1. import ===', flush=True)
import gradio_app  # noqa: E402
print('OK — gradio_app import 성공', flush=True)

print('\n=== 2. 치수표 목록 ===', flush=True)
charts = gradio_app.list_charts()
print('charts:', charts, flush=True)
assert charts, '치수표를 못 찾았습니다'

print('\n=== 3. 사이즈 추천 (UI 함수 경로) ===', flush=True)
cases = [
    ('평균 체형', dict(height=175, shoulder=45, chest=96, waist=80, hip=94)),
    ('가슴만 입력', dict(height=175, shoulder=None, chest=86, waist=None, hip=None)),
    ('단위 오타', dict(height=175, shoulder=None, chest=9.6, waist=None, hip=None)),
    ('치수 미입력', dict(height=175, shoulder=None, chest=None, waist=None, hip=None)),
]
for label, kw in cases:
    out = gradio_app.size_advice(charts[0], **kw)
    print(f'\n--- {label} ---\n{out[:400]}', flush=True)

print('\n=== 4. 서버 기동 및 응답 확인 ===', flush=True)
_app, local_url, _share = gradio_app.demo.launch(
    share=False, prevent_thread_lock=True, show_api=False, quiet=True)
print('local_url:', local_url, flush=True)

with urllib.request.urlopen(local_url, timeout=30) as r:
    body = r.read().decode('utf-8', 'ignore')
    print('HTTP', r.status, f'({len(body)} bytes)', flush=True)
    assert r.status == 200

# 페이지에 우리 UI 텍스트가 실려 있는지 (gradio가 뼈대만 주는 경우도 있어 참고용)
for token in ('gradio', 'config'):
    print(f'  page contains {token!r}:', token in body.lower(), flush=True)

gradio_app.demo.close()
print('\n모든 검증 통과', flush=True)
