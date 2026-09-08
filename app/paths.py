"""실행 환경(Colab / Kaggle / 로컬)에 따라 경로를 결정한다.

각 값은 환경변수로 덮어쓸 수 있다.

| 이름 | 환경변수 | Colab | Kaggle |
|---|---|---|---|
| SCRATCH | VFA_SCRATCH | /content/scratch | /kaggle/tmp/scratch |
| VENV | VFA_VENV | /content/venv39 | /kaggle/tmp/venv39 |
| OUT_ROOT | VFA_OUT | <repo>/outputs | /kaggle/working |
| CATVTON_REPO | CATVTON_REPO | <SCRATCH>/CatVTON | <SCRATCH>/CatVTON |

Kaggle에서 clone/venv를 `/kaggle/working` 밖에 두는 이유: 그 아래는 전부
커널 output으로 잡혀서 결과를 받아올 때 수천 개 파일을 내려받게 된다
(docs/ENVIRONMENT.md 참고).
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IS_KAGGLE = os.path.isdir('/kaggle/working')
IS_COLAB = os.path.isdir('/content') and not IS_KAGGLE

if IS_KAGGLE:
    _default_scratch = '/kaggle/tmp/scratch'
    _default_venv = '/kaggle/tmp/venv39'
    _default_out = '/kaggle/working'
elif IS_COLAB:
    _default_scratch = '/content/scratch'
    _default_venv = '/content/venv39'
    _default_out = os.path.join(REPO_ROOT, 'outputs')
else:
    _default_scratch = os.path.join(REPO_ROOT, '.scratch')
    _default_venv = os.path.join(REPO_ROOT, '.venv39')
    _default_out = os.path.join(REPO_ROOT, 'outputs')

SCRATCH = os.environ.get('VFA_SCRATCH', _default_scratch)
VENV = os.environ.get('VFA_VENV', _default_venv)
VENV_PY = os.path.join(VENV, 'bin', 'python')
OUT_ROOT = os.environ.get('VFA_OUT', _default_out)
CATVTON_REPO = os.environ.get('CATVTON_REPO', os.path.join(SCRATCH, 'CatVTON'))

PLATFORM = 'kaggle' if IS_KAGGLE else ('colab' if IS_COLAB else 'local')


def child_env(**extra):
    """venv 자식 프로세스에 넘길 환경변수.

    MPLBACKEND: Colab이 걸어둔 inline 백엔드가 상속되면 venv의 matplotlib이 죽는다.
    PYTHONUNBUFFERED: 파이프로 실행할 때 자식 출력이 버퍼에 갇히는 걸 막는다.
    """
    env = dict(
        os.environ,
        MPLBACKEND='Agg',
        PYTHONUNBUFFERED='1',
        CATVTON_REPO=CATVTON_REPO,
        VFA_SCRATCH=SCRATCH,
        VFA_VENV=VENV,
        VFA_OUT=OUT_ROOT,
    )
    env.update({k: str(v) for k, v in extra.items()})
    return env


def describe():
    return (f'platform={PLATFORM} scratch={SCRATCH} venv={VENV} '
            f'out={OUT_ROOT} catvton={CATVTON_REPO}')
