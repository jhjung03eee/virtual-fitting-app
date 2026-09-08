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

print('\n=== 2. 옷 종류별 치수표 필터링 ===', flush=True)
print('  전체   :', gradio_app.list_charts(), flush=True)
upper_charts = gradio_app.list_charts('upper')
lower_charts = gradio_app.list_charts('lower')
print('  상의용 :', upper_charts, flush=True)
print('  하의용 :', lower_charts, flush=True)
for label in ('상의', '하의', '아우터'):
    upd = gradio_app.charts_for_cloth(label)
    picked = upd.get('choices') if isinstance(upd, dict) else getattr(upd, 'choices', None)
    print(f'  라디오 {label} -> {picked}', flush=True)

assert upper_charts, '상의 치수표가 없습니다'
assert lower_charts, '하의 치수표가 없습니다'
assert not (set(upper_charts) & set(lower_charts)), '상의/하의 치수표가 섞였습니다'

upper_chart, lower_chart = upper_charts[0], lower_charts[0]

print('\n=== 3. 사이즈 추천 (UI 함수 경로) ===', flush=True)
cases = [
    ('상의 - 전부 입력', upper_chart,
     dict(height=175, shoulder=45, chest=96, waist=80, hip=94)),
    ('상의 - 가슴만 입력', upper_chart,
     dict(height=175, shoulder=None, chest=86, waist=None, hip=None)),
    ('하의 - 허리/엉덩이', lower_chart,
     dict(height=175, shoulder=None, chest=None, waist=80, hip=94)),
    ('불일치 - 상의 치수에 하의 치수표', lower_chart,
     dict(height=175, shoulder=None, chest=96, waist=None, hip=None)),
    ('단위 오타', upper_chart,
     dict(height=175, shoulder=None, chest=9.6, waist=None, hip=None)),
    ('치수 미입력', upper_chart,
     dict(height=175, shoulder=None, chest=None, waist=None, hip=None)),
]
for label, chart, kw in cases:
    out = gradio_app.size_advice(chart, **kw)
    print(f'\n--- {label} ---\n{out[:420]}', flush=True)

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
