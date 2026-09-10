"""옷 색 보정 테스트 — 특히 **목표 색을 제대로 잡는지**.

여기서 실제로 버그가 났다. 고정 임계값(RGB 합 < 720)으로 배경을 걸렀는데
쇼핑몰 상품 사진의 배경이 밝은 회색이라 배경 69%가 '옷'으로 섞여 들어왔고,
목표 색이 진한 네이비에서 흐린 파랑으로 희석돼 보정이 헛돌았다.
Kaggle에서 20분 돌리고 나서야 알았다.

torch를 쓰지 않으므로 로컬에서 돈다.
"""
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from color_match import garment_pixels, match_garment_color  # noqa: E402


def make_garment(background, garment_color, box=(20, 15, 80, 85), size=(100, 100)):
    """배경 위에 사각형 옷이 놓인 상품 사진을 흉내낸다.

    기본 상자는 화면의 42% — 실제 상품 사진(바지 36.7%, 맨투맨 60.2%)과
    같은 범위로, min_fraction 폴백에 걸리지 않는다.
    """
    image = Image.new('RGB', size, background)
    array = np.asarray(image).copy()
    left, top, right, bottom = box
    array[top:bottom, left:right] = garment_color
    return Image.fromarray(array)


class TestGarmentPixels(unittest.TestCase):
    def test_white_background_excluded(self):
        garment = make_garment('white', (50, 58, 82))
        median = np.median(garment_pixels(garment), axis=0)
        np.testing.assert_allclose(median, [50, 58, 82], atol=2)

    def test_light_gray_background_excluded(self):
        """실제로 터졌던 경우. 배경이 순백이 아니라 밝은 회색이다."""
        garment = make_garment((238, 238, 235), (50, 58, 82))
        median = np.median(garment_pixels(garment), axis=0)
        np.testing.assert_allclose(median, [50, 58, 82], atol=2)

    def test_fixed_threshold_would_have_failed(self):
        """왜 모서리 기반이어야 하는지 — 고정 임계값은 이 경우를 놓친다."""
        garment = make_garment((238, 238, 235), (50, 58, 82))
        array = np.asarray(garment, dtype=float)
        naive = array[array.sum(axis=2) < 720]   # 옛 방식
        self.assertGreater(np.median(naive, axis=0)[2], 100,
                           '고정 임계값이 배경을 걸렀다면 이 테스트의 전제가 틀린 것')

    def test_garment_similar_to_background_uses_whole_image(self):
        """흰 배경의 흰 옷. 전경으로는 프린트만 잡히므로 전체를 써야 한다.

        안 그러면 프린트 색(예: 갈색 오랑우탄)이 목표가 되어
        옷 전체가 그 색으로 물든다.
        """
        image = Image.new('RGB', (100, 100), 'white')
        array = np.asarray(image).copy()
        array[45:55, 45:55] = (166, 99, 60)   # 작은 갈색 프린트
        garment = Image.fromarray(array)

        median = np.median(garment_pixels(garment), axis=0)
        self.assertGreater(median.min(), 200, f'흰 옷인데 목표가 {median} 이다')

    def test_returns_pixels_not_image(self):
        pixels = garment_pixels(make_garment('white', (10, 20, 30)))
        self.assertEqual(pixels.ndim, 2)
        self.assertEqual(pixels.shape[1], 3)


class TestMatchGarmentColor(unittest.TestCase):
    def setUp(self):
        self.result = Image.new('RGB', (40, 40), (200, 200, 200))
        self.mask = Image.new('L', (40, 40), 0)
        mask_array = np.asarray(self.mask).copy()
        mask_array[10:30, 10:30] = 255
        self.mask = Image.fromarray(mask_array, mode='L')
        self.garment = make_garment('white', (50, 58, 82))

    def test_strength_zero_is_identity(self):
        out = match_garment_color(self.result, self.mask, self.garment, strength=0.0)
        self.assertEqual(list(np.asarray(out).ravel()[:3]), [200, 200, 200])

    def test_empty_mask_is_identity(self):
        empty = Image.new('L', (40, 40), 0)
        out = match_garment_color(self.result, empty, self.garment, strength=1.0)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(self.result))

    def test_returns_same_size(self):
        out = match_garment_color(self.result, self.mask, self.garment, strength=1.0)
        self.assertEqual(out.size, self.result.size)

    def test_only_mask_area_changes(self):
        """마스크 밖은 절대 건드리면 안 된다 — 피부와 배경이 거기 있다."""
        out = np.asarray(match_garment_color(
            self.result, self.mask, self.garment, strength=1.0))
        try:
            import skimage  # noqa: F401
        except ImportError:
            self.skipTest('skimage 없음 — 보정이 건너뛰어진다')
        outside = np.asarray(self.mask) < 128
        np.testing.assert_array_equal(
            out[outside], np.asarray(self.result)[outside])


if __name__ == '__main__':
    unittest.main()
