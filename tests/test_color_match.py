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


class TestNoHaloInsideLooseMask(unittest.TestCase):
    """AutoMasker 마스크는 옷 모양이 아니라 몸 주위를 넉넉히 덮는 영역이다.

    그 안에서 모델은 옷만 그리는 게 아니라 **옷 주변 배경도 다시 그린다.**
    마스크 전체를 옷으로 보고 색을 옮기면 그 배경이 옷 색으로 물들어 후광이 생긴다.
    실제로 셔츠 주위에 파란 후광, 바지 주위에 초록 후광이 생겼다.

    원본 인물 사진과 거의 같은 픽셀은 새로 그려진 옷이 아니므로 건드리지 않는다.
    """

    def setUp(self):
        try:
            import skimage  # noqa: F401
        except ImportError:
            self.skipTest('skimage 없음')
        size = 120
        self.person = Image.new('RGB', (size, size), (200, 200, 200))   # 원본: 회색 배경
        array = np.full((size, size, 3), 201, dtype=np.uint8)           # 생성 결과의 배경(원본과 거의 같음)
        array[30:90, 40:80] = (120, 170, 120)                            # 새로 그려진 옷(연한 초록)
        self.result = Image.fromarray(array)
        loose = np.zeros((size, size), dtype=np.uint8)
        loose[15:105, 25:95] = 255                                       # 옷보다 넉넉한 마스크
        self.mask = Image.fromarray(loose, mode='L')
        self.garment = make_garment('white', (40, 70, 40))              # 진한 초록 옷
        self.painted = np.zeros((size, size), dtype=bool)
        self.painted[30:90, 40:80] = True
        self.loose = loose > 0

    def test_background_inside_mask_is_untouched(self):
        background_in_mask = self.loose & ~self.painted
        out = np.asarray(match_garment_color(
            self.result, self.mask, self.garment, strength=1.0, person=self.person)).astype(int)
        before = np.asarray(self.result).astype(int)
        drift = np.abs(out[background_in_mask] - before[background_in_mask]).max()
        self.assertLessEqual(drift, 3, f'마스크 안 배경이 {drift}만큼 물들었다 (후광)')

    def test_painted_garment_still_changes(self):
        """후광을 막느라 보정 자체가 꺼지면 안 된다."""
        out = np.asarray(match_garment_color(
            self.result, self.mask, self.garment, strength=1.0, person=self.person)).astype(int)
        before = np.asarray(self.result).astype(int)
        core = np.zeros_like(self.painted)
        core[45:75, 50:70] = True
        self.assertGreater(np.abs(out[core] - before[core]).mean(), 10)

    def test_without_person_whole_mask_is_used(self):
        """person을 주지 않으면 옛 동작(마스크 전체)이다 — 후광이 생기는 걸 확인해 둔다."""
        background_in_mask = self.loose & ~self.painted
        out = np.asarray(match_garment_color(
            self.result, self.mask, self.garment, strength=1.0)).astype(int)
        before = np.asarray(self.result).astype(int)
        self.assertGreater(np.abs(out[background_in_mask] - before[background_in_mask]).max(), 3)


if __name__ == '__main__':
    unittest.main()
