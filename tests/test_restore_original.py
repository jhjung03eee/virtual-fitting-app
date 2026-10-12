"""restore_original 단위 테스트 — 구분 결과(클래스 배열)를 직접 넣어 mediapipe 없이 돈다."""
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from restore_original import garment_mask  # noqa: E402

W, H = 576, 768
BG, HAIR, SKIN, FACE, CLOTHES = 0, 1, 2, 3, 4


def scene(top_color, pants_color):
    """얼굴(위) · 상의(가운데) · 바지(아래)가 있는 단순한 사람 그림과 클래스 배열."""
    img = np.full((H, W, 3), 230, np.uint8)
    cls = np.zeros((H, W), np.uint8)
    img[60:160, 238:338] = (200, 160, 140)
    cls[60:160, 238:338] = FACE
    img[180:420, 188:388] = top_color
    cls[180:420, 188:388] = CLOTHES
    img[420:700, 208:368] = pants_color
    cls[420:700, 208:368] = CLOTHES
    return Image.fromarray(img), cls


class TestGarmentMask(unittest.TestCase):
    def setUp(self):
        self.person, self.cls = scene((120, 60, 40), (150, 150, 150))

    def mask_of(self, result, result_cls, category):
        m = garment_mask(self.person, result, category, self.cls, result_cls)
        return np.asarray(m) > 127

    def test_top_change_covers_top_only(self):
        result, rcls = scene((40, 90, 160), (150, 150, 150))
        m = self.mask_of(result, rcls, 'tops')
        self.assertTrue(m[300, 288], '상의 가운데는 결과를 써야 한다')
        self.assertFalse(m[600, 288], '바꾸지 않은 바지는 원본이어야 한다')
        self.assertFalse(m[110, 288], '얼굴은 항상 원본')
        self.assertFalse(m[50, 50], '먼 배경은 원본')

    def test_similar_color_top_is_still_covered(self):
        """새 옷 색이 원래 옷과 비슷한 부분(문턱값 아래)도 상의 범위 안이면 덮는다."""
        img = np.asarray(scene((40, 90, 160), (150, 150, 150))[0]).copy()
        img[180:260, 188:388] = (125, 62, 42)   # 어깨 부분만 원래 색과 거의 같음
        _p, rcls = scene((40, 90, 160), (150, 150, 150))
        m = self.mask_of(Image.fromarray(img), rcls, 'tops')
        self.assertTrue(m[200, 288])

    def test_bottom_change_covers_pants_only(self):
        result, rcls = scene((120, 60, 40), (50, 80, 40))
        m = self.mask_of(result, rcls, 'bottoms')
        self.assertTrue(m[600, 288])
        self.assertFalse(m[300, 288], '바꾸지 않은 상의는 원본이어야 한다')

    def test_nothing_changed_keeps_everything(self):
        m = self.mask_of(self.person, self.cls, 'tops')
        self.assertFalse(m.any())


if __name__ == '__main__':
    unittest.main()
