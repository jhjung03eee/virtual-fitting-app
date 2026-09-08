"""(구버전 호환) scripts/setup_env.py 로 대체됨.

기존 노트북 셀이 이 파일을 부르고 있어서 그대로 넘겨준다.
"""
import os
import subprocess
import sys

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'setup_env.py')
print('setup_env.py 로 대체되었습니다. 그대로 실행합니다.\n', flush=True)
raise SystemExit(subprocess.call([sys.executable, '-u', target, '--smoke'] + sys.argv[1:]))
