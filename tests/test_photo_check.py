"""사진 자세 검사 테스트 — mediapipe 없이 랜드마크를 직접 넣는다.

좌표는 카톡 사진 3장(정자세 / 팔 벌림 / 엄지척)에서 실제로 잰 비율을 흉내낸다.
팔을 비스듬히 벌린 사진을 '손이 몸 앞'으로 잘못 안내한 적이 있어 그 경우를 고정한다.
"""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from body_measure import POSE_LANDMARKS  # noqa: E402
from photo_check import check_photo  # noqa: E402

W, H = 1500, 2000


def landmarks(**overrides):
    """정자세 전신 (픽셀 좌표). overrides 로 부위를 옮긴다. 값: (x, y) 또는 (x, y, visibility)."""
    points = {
        'nose': (750, 300),
        'left_shoulder': (870, 480), 'right_shoulder': (630, 480),
        'left_elbow': (900, 750), 'right_elbow': (600, 750),
        'left_hip': (820, 870), 'right_hip': (680, 870),
        'left_wrist': (920, 900), 'right_wrist': (580, 900),
        'left_knee': (810, 1250), 'right_knee': (690, 1250),
        'left_ankle': (800, 1650), 'right_ankle': (700, 1650),
        'left_heel': (800, 1680), 'right_heel': (700, 1680),
    }
    points.update(overrides)
    out = [SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
    for name, value in points.items():
        x, y = value[:2]
        vis = value[2] if len(value) > 2 else 0.99
        out[POSE_LANDMARKS[name]] = SimpleNamespace(x=x / W, y=y / H, visibility=vis)
    return out


def image(width=W, height=H):
    return Image.new('RGB', (width, height), (200, 200, 200))


class TestPhotoCheck(unittest.TestCase):
    def test_guide_pose_passes(self):
        check = check_photo(image(), landmarks())
        self.assertTrue(check.ok, check.message())
        self.assertEqual(check.warnings, [])

    def test_thumbs_up_is_hands_in_front(self):
        check = check_photo(image(), landmarks(left_wrist=(830, 620), right_wrist=(670, 620)))
        self.assertFalse(check.ok)
        self.assertIn('내려주세요', check.message())

    def test_diagonal_spread_is_reported_as_spread(self):
        """손목이 엉덩이보다 올라가도 옆으로 벌어졌으면 '벌림'으로 안내한다."""
        check = check_photo(image(), landmarks(left_wrist=(1100, 740), right_wrist=(400, 740)))
        self.assertFalse(check.ok)
        self.assertIn('벌렸어요', check.message())
        self.assertNotIn('내려주세요', check.message())

    def test_cut_off_feet(self):
        check = check_photo(image(), landmarks(left_ankle=(800, 1650, 0.1)))
        self.assertIn('발끝', check.message())

    def test_sideways(self):
        check = check_photo(image(), landmarks(left_shoulder=(780, 480), right_shoulder=(720, 480)))
        self.assertIn('정면', check.message())

    def test_landscape_photo(self):
        check = check_photo(image(H, W), landmarks())
        self.assertIn('세로', check.message())

    def test_no_person(self):
        check = check_photo(image(), [])
        self.assertFalse(check.ok)

    def test_small_person_must_retake(self):
        """멀리서 찍은 전신(카톡 원본 0.54)은 바지가 안 입혀졌다. 합성 전에 다시 찍게 한다."""
        small = landmarks(nose=(750, 580), left_ankle=(800, 1660), right_ankle=(700, 1660))  # 0.54
        check = check_photo(image(), small)
        self.assertFalse(check.ok)
        self.assertIn('가까이', check.message())

    def test_person_filling_frame_passes(self):
        """카톡 크롭본(0.78) 수준이면 통과."""
        close = landmarks(nose=(750, 200), left_ankle=(800, 1760), right_ankle=(700, 1760))  # 0.78
        self.assertTrue(check_photo(image(), close).ok)

    def test_arm_touching_body_warns(self):
        mask = np.zeros((H, W), dtype=np.float32)
        mask[250:1700, 560:940] = 1.0          # 팔까지 한 덩어리
        check = check_photo(image(), landmarks(), seg_mask=mask)
        self.assertTrue(check.ok)
        self.assertIn('붙어', check.message())

    def test_arm_with_gap_does_not_warn(self):
        mask = np.zeros((H, W), dtype=np.float32)
        mask[250:1700, 640:860] = 1.0          # 몸통
        mask[450:950, 890:950] = 1.0           # 왼팔
        mask[450:950, 550:610] = 1.0           # 오른팔
        check = check_photo(image(), landmarks(), seg_mask=mask)
        self.assertNotIn('붙어', check.message())


if __name__ == '__main__':
    unittest.main()
