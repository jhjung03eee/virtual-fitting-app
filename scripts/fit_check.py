"""사진 + 키 + 치수표 -> 몸 치수 추정과 사이즈 추천을 출력한다.

    python scripts/fit_check.py --person <사진> --height 175 \
        --chart data/size_charts/sample_tshirt.json

사진을 생략하면 CatVTON 저장소의 데모 인물 이미지를 쓴다.
mediapipe가 필요하므로 venv39에서 실행된다(자동 재실행).
"""
import argparse
import glob
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'app'))
from paths import VENV_PY, CATVTON_REPO, child_env  # noqa: E402


def _reexec_in_venv():
    if os.environ.get('VFA_NO_REEXEC') == '1':
        return
    if os.path.abspath(sys.executable) == os.path.abspath(VENV_PY):
        return
    if not os.path.exists(VENV_PY):
        sys.exit(f'Python 3.9 환경({VENV_PY})이 없습니다. scripts/setup_env.py 를 먼저 실행하세요.')
    raise SystemExit(subprocess.call(
        [VENV_PY, '-u', os.path.abspath(__file__)] + sys.argv[1:], env=child_env()))


_reexec_in_venv()

from PIL import Image  # noqa: E402

import body_measure  # noqa: E402
from size_fit import SizeChart, recommend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--person', default=None, help='인물 사진 (생략 시 데모 이미지)')
    ap.add_argument('--height', type=float, required=True, help='키 (cm)')
    ap.add_argument('--chart', default=os.path.join(REPO_ROOT, 'data/size_charts/sample_tshirt.json'))
    a = ap.parse_args()

    person = a.person
    if person is None:
        cands = sorted(glob.glob(os.path.join(CATVTON_REPO, 'resource/demo/example/person/men/*')))
        if not cands:
            sys.exit('데모 인물 이미지를 찾지 못했습니다. --person 으로 직접 지정하세요.')
        person = cands[0]

    print(f'사진: {person}')
    print(f'키  : {a.height:.0f}cm\n')

    measurement = body_measure.measure(Image.open(person), a.height)
    print(measurement.describe())

    if not measurement.measurements_cm:
        sys.exit(1)

    with open(a.chart, encoding='utf-8') as f:
        chart = SizeChart.from_dict(json.load(f))

    print(f'\n치수표: {chart.name} ({chart.category})')
    rec = recommend(chart, measurement.measurements_cm)
    print(f'\n>>> {rec.message}\n')
    for fit in rec.ranked:
        mark = '  ' if fit.wearable else '✗ '
        print(f'{mark}{fit.summary}   [score {fit.score}]')
    for note in rec.notes:
        print(f'\n[참고] {note}')

    print('\n※ 단일 정면 사진 기반 추정입니다. 절대 치수보다 사이즈 간 상대 비교로 보세요.')


if __name__ == '__main__':
    main()
