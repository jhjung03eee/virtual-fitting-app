"""photo_guide 단위 테스트 — 가짜 랜드마크로 그림만 확인한다 (mediapipe 없이 돈다)."""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from body_measure import POSE_LANDMARKS  # noqa: E402
from photo_check import PhotoCheck, check_photo  # noqa: E402
from photo_guide import BAD_COLOR, DISPLAY_HEIGHT, OK_COLOR, TARGET_COLOR, draw_guide  # noqa: E402

W, H = 900, 1200


def landmarks(nose_y=0.08, ankle_y=0.92, wrist_dx=0.06, wrist_y=0.52):
    """정면으로 선 사람 (0~1 좌표). 기본값은 가이드를 통과하는 자세."""
    pts = {
        'nose': (0.5, nose_y),
        'left_shoulder': (0.40, 0.25), 'right_shoulder': (0.60, 0.25),
        'left_elbow': (0.37, 0.38), 'right_elbow': (0.63, 0.38),
        'left_hip': (0.44, 0.52), 'right_hip': (0.56, 0.52),
        'left_wrist': (0.44 - wrist_dx, wrist_y), 'right_wrist': (0.56 + wrist_dx, wrist_y),
        'left_knee': (0.45, 0.72), 'right_knee': (0.55, 0.72),
        'left_ankle': (0.45, ankle_y), 'right_ankle': (0.55, ankle_y),
        'left_heel': (0.45, ankle_y), 'right_heel': (0.55, ankle_y),
    }
    out = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    for name, (x, y) in pts.items():
        out[POSE_LANDMARKS[name]] = SimpleNamespace(x=x, y=y, visibility=0.99)
    return out


def has_color(image, color, tol=12):
    a = np.asarray(image).astype(int)
    return bool((np.abs(a - np.array(color)).max(axis=2) <= tol).any())


class TestDrawGuide(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (W, H), (200, 200, 200))

    def draw(self, lm):
        return draw_guide(self.image, lm, check_photo(self.image, lm))

    def test_resized_to_display_height(self):
        out = self.draw(landmarks())
        self.assertEqual(out.height, DISPLAY_HEIGHT)
        self.assertEqual(out.width, round(W * DISPLAY_HEIGHT / H))

    def test_good_pose_is_green(self):
        lm = landmarks()
        self.assertTrue(check_photo(self.image, lm).ok)
        out = self.draw(lm)
        self.assertTrue(has_color(out, OK_COLOR))
        self.assertFalse(has_color(out, BAD_COLOR))

    def test_far_away_is_red_and_shows_needed_length(self):
        """멀리서 찍으면 목표선이 화면 밖 — 그래도 노란 막대로 필요한 길이를 보여준다."""
        lm = landmarks(nose_y=0.40, ankle_y=0.92)   # 화면의 52%
        out = self.draw(lm)
        self.assertTrue(has_color(out, BAD_COLOR))
        self.assertTrue(has_color(out, TARGET_COLOR))

    def test_spread_arms_are_red(self):
        out = self.draw(landmarks(wrist_dx=0.20))
        self.assertTrue(has_color(out, BAD_COLOR))

    def test_no_landmarks_returns_photo_only(self):
        out = draw_guide(self.image, None, PhotoCheck())
        self.assertEqual(out.height, DISPLAY_HEIGHT)

    def test_without_check_draws_gray(self):
        """검사 결과(숫자)가 없으면 판정 색 대신 회색 — 그림이 판정을 지어내지 않는다."""
        out = draw_guide(self.image, landmarks(), None)
        self.assertFalse(has_color(out, OK_COLOR))
        self.assertFalse(has_color(out, BAD_COLOR))


if __name__ == '__main__':
    unittest.main()
