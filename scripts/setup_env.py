"""Python 3.9 환경 구성 — Colab / Kaggle 공용.

    python scripts/setup_env.py [--smoke]

CatVTON repo가 요구하는 Python 3.9 + torch 2.1.2 조합을 격리된 venv로 만든다.
왜 이렇게 해야 하는지는 docs/ENVIRONMENT.md 참고 (repo에 cp39 전용으로 컴파일된
detectron2 .so가 들어있어 최신 Python에서는 import 자체가 불가능하다).

--smoke 를 주면 마지막에 추론 1회를 돌려 환경이 실제로 동작하는지 확인한다.
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'app'))
from paths import SCRATCH, VENV, VENV_PY, CATVTON_REPO, child_env, describe  # noqa: E402

CATVTON_GIT = 'https://github.com/Zheng-Chong/CatVTON.git'

# repo requirements.txt에 빠져 있는 것들:
#   fvcore~cloudpickle : detectron2 런타임 의존성
#   av                 : densepose가 import 체인에서 요구 (비디오는 안 쓰는데도)
PIP_PACKAGES = (
    'accelerate==0.31.0 diffusers==0.29.2 huggingface_hub==0.23.4 '
    'transformers==4.27.3 numpy==1.26.4 opencv-python==4.10.0.84 pillow==10.3.0 '
    'PyYAML==6.0.1 scipy==1.13.1 scikit-image==0.24.0 tqdm==4.66.4 matplotlib==3.9.1 av '
    'fvcore iopath pycocotools omegaconf hydra-core termcolor yacs tabulate cloudpickle'
)

t0 = time.time()


def sh(cmd, check=False):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tail = (r.stdout or '')[-1200:] + (r.stderr or '')[-1200:]
    if tail.strip():
        print(tail, flush=True)
    print(f'   -> exit {r.returncode}  ({time.time() - t0:.0f}s)', flush=True)
    if check and r.returncode != 0:
        sys.exit(f'실패: {cmd}')
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='환경 구성 후 추론 1회로 검증')
    a = ap.parse_args()

    print(describe(), flush=True)
    os.makedirs(SCRATCH, exist_ok=True)

    if not os.path.exists(CATVTON_REPO):
        sh(f'git clone -q {CATVTON_GIT} {CATVTON_REPO}', check=True)
    print('CatVTON:', CATVTON_REPO, flush=True)

    if not os.path.exists('/usr/bin/python3.9'):
        sh('add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1')
        sh('apt-get install -y -qq python3.9 python3.9-venv python3.9-dev > /dev/null 2>&1')
    if not os.path.exists('/usr/bin/python3.9'):
        sys.exit('python3.9 설치 실패. apt 로그를 확인하세요.')

    if not os.path.exists(VENV_PY):
        sh(f'python3.9 -m venv {VENV}', check=True)
        sh(f'{VENV_PY} -m pip install -q --upgrade pip')

    # cu121 휠은 sm_50~sm_80 지원 -> Kaggle P100(sm_60)에서도 동작한다
    sh(f'{VENV_PY} -m pip install -q torch==2.1.2 torchvision==0.16.2 '
       '--index-url https://download.pytorch.org/whl/cu121', check=True)
    sh(f'{VENV_PY} -m pip install -q {PIP_PACKAGES}', check=True)

    check = (
        'import sys, torch; '
        "print('python:', sys.version.split()[0]); "
        "print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available()); "
        "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); "
        "print('arch:', torch.cuda.get_arch_list()[:6] if torch.cuda.is_available() else '-')"
    )
    subprocess.run([VENV_PY, '-c', check], env=child_env())

    if a.smoke:
        print('\n=== 스모크 테스트 (추론 1회) ===', flush=True)
        worker = os.path.join(REPO_ROOT, 'scripts', 'worker_argparse.py')
        rc = subprocess.call([VENV_PY, '-u', worker], env=child_env())
        if rc != 0:
            sys.exit(f'스모크 테스트 실패 (exit {rc})')

    print(f'\n환경 준비 완료. 총 {time.time() - t0:.0f}초', flush=True)


if __name__ == '__main__':
    main()
