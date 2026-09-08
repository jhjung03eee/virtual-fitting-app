"""Colab에서 Gradio 웹 UI를 띄운다.

전제: scripts/setup_py39_and_run.py 로 /content/venv39 환경이 이미 구성돼 있을 것.
app/ 은 이 스크립트 위치를 기준으로 찾는다 (저장소를 어디에 clone하든 동작).
"""
import os
import subprocess
import sys

VENV_PY = '/content/venv39/bin/python'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(REPO_ROOT, 'app')
APP_PY = os.path.join(APP_DIR, 'gradio_app.py')

if not os.path.exists(APP_PY):
    sys.exit(f'{APP_PY} 를 찾을 수 없습니다. 저장소가 온전히 clone됐는지 확인하세요.')

if not os.path.exists(VENV_PY):
    sys.exit(
        'Python 3.9 환경(/content/venv39)이 없습니다.\n'
        '런타임이 재시작되면 /content 아래가 전부 사라지므로 환경 구성부터 다시 해야 합니다.\n\n'
        '아래 셀을 먼저 실행하세요 (최초 1회, 약 5분):\n'
        '    !python /content/vfa/scripts/setup_py39_and_run.py\n'
    )

# gradio는 setup 단계 의존성 목록에 없으므로 여기서 설치.
# CatVTON repo는 4.39.0을 고정하지만 그 버전은 최신 pydantic과 조합 시
#   TypeError: argument of type 'bool' is not iterable  (gradio_client/utils.py get_type)
# 로 /info 라우트가 터진다. 4.44.1에서 수정됨. 모델 코드와는 무관한 UI 라이브러리라 올려도 안전.
print('gradio 설치 중...', flush=True)
subprocess.run([VENV_PY, '-m', 'pip', 'install', '-q', 'gradio==4.44.1'], check=True)

# gradio_client 버그를 설치된 파일에 직접 패치한다.
# pydantic 스키마의 `additionalProperties: true` 는 dict가 아니라 bool인데
# gradio_client가 dict로 가정하고 `"const" in schema` 를 해서 터진다:
#   TypeError: argument of type 'bool' is not iterable
# gradio 메인 라우트가 show_api 설정과 무관하게 api_info()를 호출하므로 페이지가 안 열린다.
PATCH = r'''
import re
import gradio_client.utils as u

path = u.__file__
src = open(path, encoding='utf-8').read()
guard = "    if not isinstance(schema, dict):\n        return \"Any\"\n"

changed = False
for pattern in (r"(def get_type\([^)]*\)[^:]*:\n)", r"(def _json_schema_to_python_type\([^)]*\)[^:]*:\n)"):
    m = re.search(pattern, src)
    if m and not src[m.end():m.end() + len(guard) + 40].lstrip().startswith("if not isinstance(schema, dict)"):
        src = src[:m.end()] + guard + src[m.end():]
        changed = True

if changed:
    open(path, 'w', encoding='utf-8').write(src)
    print('gradio_client 패치 적용:', path)
else:
    print('gradio_client 이미 패치됨')
'''
subprocess.run([VENV_PY, '-c', PATCH], check=True)

# Colab이 걸어둔 inline 백엔드가 상속되면 venv쪽 matplotlib이 죽는다 (docs/ENVIRONMENT.md #7)
env = dict(os.environ, MPLBACKEND='Agg')

print('Gradio 앱 실행 — 아래 *.gradio.live 링크로 접속하세요.', flush=True)
proc = subprocess.Popen(
    [VENV_PY, APP_PY],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for line in proc.stdout:
    print(line, end='', flush=True)
