"""app/size_warp.py — 옷 영역을 사이즈 비율만큼 늘리고 줄이는 기하 변환 (F-35).

그림 대신 단순한 사각형 '옷'으로 확인한다: 어깨선 아래 기장이 scale_y 배가 되는지,
폭이 몸 가운데 기준 scale_x 배가 되는지, 줄어서 드러난 자리에 옷 색이 남지 않는지.
"""
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from size_warp import scales_for_size, warp_garment  # noqa: E402

BLUE, WHITE = (20, 30, 120), (240, 240, 240)
SHOULDER, HEM, LEFT, RIGHT = 60, 120, 50, 110      # 옷: 어깨선 60 ~ 밑단 120, 폭 60 (가운데 80)


def tee():
    rgb = np.full((240, 160, 3), WHITE, np.uint8)
    rgb[SHOULDER:HEM, LEFT:RIGHT] = BLUE
    mask = np.zeros(rgb.shape[:2], bool)
    mask[SHOULDER:HEM, LEFT:RIGHT] = True
    return Image.fromarray(rgb), mask


def extent(mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask[SHOULDER + 20])[0]
    return rows[-1] + 1, cols[0], cols[-1] + 1


class TestSizeWarp(unittest.TestCase):
    def test_longer_and_wider(self):
        image, mask = tee()
        out, new_mask = warp_garment(image, mask, SHOULDER, 80, scale_x=1.2, scale_y=1.25)
        hem, left, right = extent(new_mask)
        self.assertAlmostEqual(hem - SHOULDER, (HEM - SHOULDER) * 1.25, delta=2)
        self.assertAlmostEqual(right - left, (RIGHT - LEFT) * 1.2, delta=2)
        # 늘어난 밑단 자리는 옷 색으로 덮인다
        self.assertLess(np.asarray(out)[hem - 3, 80, 0], 60)

    def test_narrower_fills_revealed_sides(self):
        image, mask = tee()
        out, new_mask = warp_garment(image, mask, SHOULDER, 80, scale_x=0.8, scale_y=1.0)
        _hem, left, right = extent(new_mask)
        self.assertAlmostEqual(right - left, (RIGHT - LEFT) * 0.8, delta=2)
        # 줄어서 드러난 옆자리(원래 옷이었던 곳)는 배경색으로 메워진다
        side = np.asarray(out)[SHOULDER + 20, LEFT + 2]
        self.assertGreater(int(side[0]), 150)

    def test_above_shoulder_untouched(self):
        image, mask = tee()
        rgb = np.asarray(image).copy()
        rgb[10:30, 70:90] = (200, 120, 90)          # 얼굴 자리
        out, _ = warp_garment(Image.fromarray(rgb), mask, SHOULDER, 80, scale_x=1.3, scale_y=1.3)
        np.testing.assert_array_equal(np.asarray(out)[10:30, 70:90], rgb[10:30, 70:90])

    def test_scales_for_size(self):
        sx, sy = scales_for_size({'chest': 0.7, 'length': 1.0}, {'chest': 0.63, 'length': 1.1})
        self.assertAlmostEqual(sx, 0.9)
        self.assertAlmostEqual(sy, 1.1)


if __name__ == '__main__':
    unittest.main()
