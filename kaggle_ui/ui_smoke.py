"""UI가 실제로 뜨고 응답하는지 검증한다 (GPU·모델 로딩 없이).

배치 커널에서는 사람이 브라우저로 접속할 수 없으므로,
서버를 띄운 뒤 로컬에서 HTTP 요청을 보내 확인하는 방식으로 검증한다.

확인 항목
1. gradio_app 모듈이 import 되는가 (문법·의존성)
2. 옷 종류에 맞는 치수표만 걸러지는가
3. 사이즈 추천 로직이 UI 함수 경로로 동작하는가
4. **run() 을 실제로 호출해 try_on 에 넘기는 인자가 맞는가**
5. 서버가 실제로 뜨고 200을 돌려주는가
"""
import ast
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

# gradio_app 은 tryon_core 에서 기본값 상수들도 가져온다(DEFAULT_STEPS 등).
# 손으로 나열하면 상수를 추가할 때마다 여기서 ImportError가 난다(실제로 났다).
# **진짜 소스를 AST로 읽어** 모듈 수준 상수를 그대로 옮겨온다. torch를 임포트하지
# 않으므로 GPU 없이도 안전하다.
_core_source = os.path.join(REPO_DIR, 'app', 'tryon_core.py')
with open(_core_source, encoding='utf-8') as f:
    _core_tree = ast.parse(f.read())
_copied = []
for _node in _core_tree.body:
    if not isinstance(_node, ast.Assign):
        continue
    for _target in _node.targets:
        if not isinstance(_target, ast.Name) or _target.id.startswith('_'):
            continue
        try:
            setattr(fake, _target.id, ast.literal_eval(_node.value))
            _copied.append(_target.id)
        except ValueError:
            pass  # 리터럴이 아닌 값(호출 결과 등)은 UI가 쓰지 않는다
print('가짜 tryon_core 에 옮긴 상수:', _copied, flush=True)


def _fake_try_on(**kwargs):
    from PIL import Image
    return Image.new('RGB', (76, 102), 'gray'), Image.new('RGB', (76, 102), 'black')


fake.try_on = _fake_try_on
fake.default_guidance = lambda cloth_type: getattr(
    fake, 'DEFAULT_GUIDANCE_BY_TYPE', {}).get(cloth_type, fake.DEFAULT_GUIDANCE)
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

print('\n=== 4. run() 전체 경로 ===', flush=True)
# 배선 테스트(tests/test_gradio_wiring.py)는 AST로 개수만 본다. 여기서는 실제로
# 호출해서 try_on 에 넘기는 키워드 인자까지 확인한다.
# "앱은 멀쩡히 뜨는데 버튼을 누르면 터지는" 부류의 버그를 잡는 자리다.
captured = {}


def _recording_try_on(**kwargs):
    from PIL import Image
    captured.update(kwargs)
    return Image.new('RGB', (76, 102), 'gray'), Image.new('RGB', (76, 102), 'black')


# gradio_app 은 `from tryon_core import try_on` 으로 **이미 이름을 바인딩**했다.
# 여기서 fake.try_on 을 바꿔도 gradio_app.try_on 은 옛 함수를 그대로 가리킨다.
# 바꿔치기는 gradio_app 의 이름 공간에 해야 한다.
gradio_app.try_on = _recording_try_on

from PIL import Image  # noqa: E402

dummy = Image.new('RGB', (76, 102), 'white')
result, advice, mask_vis = gradio_app.run(
    dummy, dummy, '상의', 'dpm', 8, 2.5, 42,
    True,                       # 배경 정규화 체크박스
    upper_chart, 175, 45, 96, 80, 94,
)
print('  반환:', type(result).__name__, type(mask_vis).__name__, flush=True)
print('  try_on 이 받은 인자:', sorted(captured), flush=True)
print('  사이즈 추천 첫 줄:', advice.splitlines()[0][:80], flush=True)

for required in ('person', 'garment', 'cloth_type', 'steps',
                 'guidance_scale', 'seed', 'scheduler', 'normalize_background'):
    assert required in captured, f'try_on 에 {required} 가 전달되지 않았습니다'
assert captured['cloth_type'] == 'upper', captured['cloth_type']
assert captured['normalize_background'] is True, '배경 정규화 체크박스가 전달되지 않았습니다'
assert advice.strip(), '사이즈 추천이 비어 있습니다'

# 하의를 고르면 cloth_type 이 lower 로 바뀌는지
gradio_app.run(dummy, dummy, '하의', 'dpm', 8, 2.5, 42, False,
               lower_chart, 175, None, None, 80, 94)
assert captured['cloth_type'] == 'lower', captured['cloth_type']
assert captured['normalize_background'] is False
print('  옷 종류 전달 확인: 상의 -> upper, 하의 -> lower', flush=True)

print('\n=== 5. 서버 기동 및 응답 확인 ===', flush=True)
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
