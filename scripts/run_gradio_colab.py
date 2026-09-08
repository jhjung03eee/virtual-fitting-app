"""Colab에서 Gradio 웹 UI를 띄운다.

전제: scripts/setup_py39_and_run.py 로 /content/venv39 환경이 이미 구성돼 있을 것.
app/tryon_core.py, app/gradio_app.py 가 /content/app/ 에 있어야 한다.
"""
import os
import subprocess
import sys

VENV_PY = '/content/venv39/bin/python'
APP_DIR = '/content/app'

if not os.path.exists(VENV_PY):
    sys.exit(
        'Python 3.9 환경(/content/venv39)이 없습니다.\n'
        '런타임이 재시작되면 /content 아래가 전부 사라지므로 환경 구성부터 다시 해야 합니다.\n\n'
        '아래 셀을 먼저 실행하세요 (최초 1회, 약 5분):\n'
        '    !python /content/vfa/scripts/setup_py39_and_run.py\n'
    )

# gradio는 setup 단계 의존성 목록에 없으므로 여기서 설치 (repo가 고정한 버전)
print('gradio 설치 중...', flush=True)
subprocess.run([VENV_PY, '-m', 'pip', 'install', '-q', 'gradio==4.39.0'], check=True)

# Colab이 걸어둔 inline 백엔드가 상속되면 venv쪽 matplotlib이 죽는다 (docs/ENVIRONMENT.md #7)
env = dict(os.environ, MPLBACKEND='Agg')

print('Gradio 앱 실행 — 아래 *.gradio.live 링크로 접속하세요.', flush=True)
proc = subprocess.Popen(
    [VENV_PY, os.path.join(APP_DIR, 'gradio_app.py')],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for line in proc.stdout:
    print(line, end='', flush=True)
