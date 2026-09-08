"""배치 추론 워커 — 경로를 인자로 받아 결과 이미지를 저장한다.

Python 3.9 venv에서 실행:
    MPLBACKEND=Agg /content/venv39/bin/python scripts/worker_argparse.py \
        --cloth-type lower --out /content/out_lower
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from tryon_core import try_on, load_models, REPO_DIR, CLOTH_TYPES

ap = argparse.ArgumentParser()
ap.add_argument('--person', default=None, help='인물 사진 경로 (생략 시 저장소 데모 이미지)')
ap.add_argument('--garment', default=None, help='옷 사진 경로 (생략 시 저장소 데모 이미지)')
ap.add_argument('--cloth-type', default='upper', choices=CLOTH_TYPES)
ap.add_argument('--steps', type=int, default=30)
ap.add_argument('--out', default='/content/outputs')
a = ap.parse_args()

load_models()
print('pipeline + automasker ready', flush=True)

person_path = a.person or sorted(glob.glob(os.path.join(REPO_DIR, 'resource/demo/example/person/men/*')))[0]
garment_path = a.garment or sorted(glob.glob(os.path.join(REPO_DIR, 'resource/demo/example/condition/upper/*')))[0]
print('person:', person_path, '| garment:', garment_path, '| type:', a.cloth_type, flush=True)

result, mask_vis = try_on(person_path, garment_path, a.cloth_type, steps=a.steps)

os.makedirs(a.out, exist_ok=True)
result.save(os.path.join(a.out, 'result.png'))
mask_vis.save(os.path.join(a.out, 'mask.png'))
print('DONE ->', a.out, flush=True)
