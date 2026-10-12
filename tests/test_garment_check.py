"""옷 사진 적합성 검사 테스트. 합성 이미지로 규칙을 고정한다.

실제 샘플 6벌(set2 3벌 + 초기 3벌)로 상수를 정했다. 흰 옷·사진 프린트 경고는
FASHN 전환 후 뺐다(docs/PLAN.md F-23).
흰 배경의 흰 옷에서 프린트(갈색 오랑우탄)가 '피부'로 잡혀 착용컷으로 오판할 뻔했다.
"""
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

try:
    import skimage  # noqa: F401
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from garment_check import check_garment  # noqa: E402


def product_shot(color, background=(245, 245, 245), size=(600, 700), box=(120, 100, 480, 620)):
    array = np.full((size[1], size[0], 3), background, dtype=np.uint8)
    left, top, right, bottom = box
    array[top:bottom, left:right] = color
    return array


@unittest.skipUnless(HAS_SKIMAGE, 'skimage 없음')
class TestGarmentCheck(unittest.TestCase):
    def test_plain_khaki_product_shot_is_suitable(self):
        rng = np.random.default_rng(0)
        array = product_shot((110, 108, 82)).astype(float)
        array += rng.normal(0, 2, array.shape)
        check = check_garment(Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)))
        self.assertTrue(check.suitable, check.message())

    def test_white_tee_with_photo_print_is_not_warned(self):
        """흰 옷·사진 프린트는 CatVTON에서 실패했지만 FASHN에서 재현됐다(PLAN F-14·F-19).

        경고를 그대로 두면 잘 되는 옷에 "안 될 수 있다"고 말하게 된다.
        흰 배경의 갈색 프린트를 '피부'(착용컷)로 오판하지 않는 것도 같이 본다.
        """
        array = np.full((700, 600, 3), 250, dtype=np.uint8)
        rng = np.random.default_rng(1)
        patch = rng.integers(0, 2, (160, 120, 1)) * np.array([[[166, 99, 60]]]) + \
            (1 - rng.integers(0, 2, (160, 120, 1))) * np.array([[[60, 30, 20]]])
        array[250:410, 240:360] = patch.astype(np.uint8)       # 사진 같은 갈색 프린트
        check = check_garment(Image.fromarray(array))
        self.assertTrue(check.suitable, check.message())

    def test_worn_photo_suggests_model_type(self):
        array = product_shot((40, 60, 120))
        array[100:220, 120:480] = (205, 150, 120)             # 얼굴·목 피부
        array[250:620, 40:120] = (205, 150, 120)              # 팔
        message = check_garment(Image.fromarray(array)).message()
        self.assertIn('입은 사진', message)
        self.assertIn('착용 사진', message)

    def test_worn_photo_in_model_mode_is_not_warned(self):
        """착용 사진 모드에서는 몸·배경이 있는 게 정상이다."""
        rng = np.random.default_rng(4)
        array = rng.integers(0, 255, (700, 600, 3), dtype=np.uint8)   # 복잡한 배경
        array[100:220, 120:480] = (205, 150, 120)
        array[250:620, 40:120] = (205, 150, 120)
        check = check_garment(Image.fromarray(array), photo_type='model')
        self.assertTrue(check.suitable, check.message())

    def test_busy_print_is_not_warned(self):
        rng = np.random.default_rng(2)
        array = product_shot((40, 60, 120))
        noise = rng.integers(0, 255, (520, 360, 3), dtype=np.uint8)
        array[100:620, 120:480] = noise
        self.assertNotIn('프린트', check_garment(Image.fromarray(array)).message())

    def test_cluttered_background_warns(self):
        rng = np.random.default_rng(3)
        array = rng.integers(0, 255, (700, 600, 3), dtype=np.uint8)
        array[100:620, 120:480] = (40, 60, 120)
        self.assertIn('단색 배경', check_garment(Image.fromarray(array)).message())

    def test_low_resolution_warns(self):
        small = Image.fromarray(product_shot((40, 60, 120))).resize((200, 240))
        self.assertIn('해상도', check_garment(small).message())


if __name__ == '__main__':
    unittest.main()
