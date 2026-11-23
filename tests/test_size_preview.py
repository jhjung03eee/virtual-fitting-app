"""app/size_preview.py — 사이즈별 예상 착용 모습(총장만) (F-39).

mediapipe 없이 확인하려고 부위 구분 결과(classes)와 자세(frame)를 직접 만든다:
남색 윗옷(어깨선 60 ~ 밑단 140) 아래에 짙은 갈색 바지가 붙어 있고, 둘 다 '옷'(4) 분류다.
"""
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

try:
    import cv2  # noqa: F401  size_warp 가 쓴다 (mediapipe 와 같이 설치됨)
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from size_preview import PreviewError, garment_rows, length_previews, length_scale, top_mask  # noqa: E402

H, W = 300, 200
SHOULDER, HEM, LEFT, RIGHT = 60, 140, 60, 140
NAVY, BROWN, WALL = (20, 30, 52), (20, 15, 12), (235, 235, 230)
FRAME = (SHOULDER / H, 100 / H, 0.5)      # 어깨선 y, 몸통 길이(100px), 몸 가운데 x
SIZES = {'M': {'chest': 51.5, 'length': 68.0}, 'L': {'chest': 54.5, 'length': 71.0},
         'XL': {'chest': 58.5, 'length': 74.0}}


def person(pants=BROWN):
    rgb = np.full((H, W, 3), WALL, np.uint8)
    classes = np.zeros((H, W), np.uint8)
    rgb[SHOULDER:HEM, LEFT:RIGHT] = NAVY
    rgb[HEM:280, 70:130] = pants
    classes[SHOULDER:HEM, LEFT:RIGHT] = 4
    classes[HEM:280, 70:130] = 4
    return rgb, classes


def hem_of(image):
    """남색 픽셀이 있는 마지막 줄 + 1."""
    rgb = np.asarray(image).astype(int)
    navy = (rgb[..., 2] - rgb[..., 0]) > 20
    return np.where(navy[:, 100])[0][-1] + 1


class TestSizePreview(unittest.TestCase):
    def test_top_mask_excludes_pants(self):
        rgb, classes = person()
        mask = top_mask(rgb, classes, FRAME)
        self.assertTrue(mask[100, 100])
        self.assertFalse(mask[200, 100])          # 바지는 '옷' 분류여도 빠진다

    def test_rows(self):
        rgb, classes = person()
        armpit, hem, sy = garment_rows(top_mask(rgb, classes, FRAME), FRAME)
        self.assertEqual(sy, SHOULDER)
        self.assertAlmostEqual(hem, HEM - 1, delta=2)

    def test_same_color_pants_refused(self):
        rgb, classes = person(pants=NAVY)
        with self.assertRaises(PreviewError):
            garment_rows(top_mask(rgb, classes, FRAME), FRAME)

    def test_length_scale(self):
        # 어깨선 0 · 겨드랑이 20 · 밑단 100, 총장 68 → 74: 전체 100 → 108.8, 늘어난 8.8 을 아래 80 이 맡는다
        self.assertAlmostEqual(length_scale(68, 74, 0, 20, 100), (100 * 74 / 68 - 20) / 80)
        self.assertEqual(length_scale(68, 68, 0, 20, 100), 1.0)

    @unittest.skipUnless(HAS_CV2, 'cv2 없음')
    def test_previews_follow_chart_length(self):
        rgb, classes = person()
        out = length_previews(Image.fromarray(rgb), SIZES, 'M', FRAME, classes)
        self.assertEqual([c.split(' ')[0] for _, c in out], ['M', 'L', 'XL'])
        np.testing.assert_array_equal(np.asarray(out[0][0]), rgb)        # 추천 사이즈는 합성 그대로
        lengths = [hem_of(pic) - SHOULDER for pic, _ in out]
        self.assertAlmostEqual(lengths[1] / lengths[0], 71 / 68, delta=0.03)
        self.assertAlmostEqual(lengths[2] / lengths[0], 74 / 68, delta=0.03)
        self.assertIn('3cm 길어요', out[1][1])

    @unittest.skipUnless(HAS_CV2, 'cv2 없음')
    def test_smaller_size_is_shorter(self):
        rgb, classes = person()
        out = length_previews(Image.fromarray(rgb), SIZES, 'L', FRAME, classes)
        self.assertLess(hem_of(out[0][0]), hem_of(out[1][0]))
        self.assertIn('3cm 짧아요', out[0][1])

    def test_missing_length(self):
        rgb, classes = person()
        with self.assertRaises(PreviewError):
            length_previews(Image.fromarray(rgb), {'M': {'chest': 50}}, 'M', FRAME, classes)


if __name__ == '__main__':
    unittest.main()
