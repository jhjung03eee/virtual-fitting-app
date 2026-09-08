"""body_measure 테스트 — mediapipe 없이 랜드마크를 주입해 계산부만 검증한다.

    python -m unittest discover -s tests

mediapipe 자체(사람 검출)는 여기서 못 재고, 실제 사진 검증은 별도 스크립트에서 한다.
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

import numpy as np  # noqa: E402

import body_measure as bm  # noqa: E402


class FakeLandmark:
    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def make_landmarks(img_w=800, img_h=1200, shoulder_px=180.0, hip_px=140.0,
                   visibility=1.0):
    """정면 정자세 사람을 흉내낸 랜드마크. 좌표는 0~1 정규화."""
    cx = 0.5
    top_y, bottom_y = 0.10, 0.95          # 머리끝/발끝에 해당하는 대략 위치
    nose_y = top_y + (bottom_y - top_y) * 0.12
    shoulder_y = top_y + (bottom_y - top_y) * 0.22
    hip_y = top_y + (bottom_y - top_y) * 0.52
    ankle_y = bottom_y - (bottom_y - top_y) * 0.04

    half_sh = shoulder_px / 2.0 / img_w
    half_hip = hip_px / 2.0 / img_w

    lms = [FakeLandmark(cx, nose_y, visibility) for _ in range(33)]
    lms[bm.POSE_LANDMARKS['nose']] = FakeLandmark(cx, nose_y, visibility)
    lms[bm.POSE_LANDMARKS['left_shoulder']] = FakeLandmark(cx - half_sh, shoulder_y, visibility)
    lms[bm.POSE_LANDMARKS['right_shoulder']] = FakeLandmark(cx + half_sh, shoulder_y, visibility)
    lms[bm.POSE_LANDMARKS['left_hip']] = FakeLandmark(cx - half_hip, hip_y, visibility)
    lms[bm.POSE_LANDMARKS['right_hip']] = FakeLandmark(cx + half_hip, hip_y, visibility)
    lms[bm.POSE_LANDMARKS['left_ankle']] = FakeLandmark(cx - 0.03, ankle_y, visibility)
    lms[bm.POSE_LANDMARKS['right_ankle']] = FakeLandmark(cx + 0.03, ankle_y, visibility)
    return lms


def blank_image(w=800, h=1200):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestEllipseMath(unittest.TestCase):
    def test_circle_perimeter(self):
        """폭==깊이면 원이므로 둘레는 pi*d 여야 한다."""
        self.assertAlmostEqual(bm._ellipse_perimeter(10.0, 10.0), math.pi * 10.0, places=3)

    def test_half_circumference_exceeds_front_width(self):
        """단면(둘레/2)은 정면 폭보다 항상 커야 한다. 아니면 환산이 잘못된 것."""
        for part in ('chest', 'waist', 'hip'):
            width = 34.0
            half = bm._circumference_half_from_width(width, part)
            self.assertGreater(half, width, f'{part}: {half} <= {width}')

    def test_monotonic_in_width(self):
        a = bm._circumference_half_from_width(30.0, 'chest')
        b = bm._circumference_half_from_width(36.0, 'chest')
        self.assertGreater(b, a)


class TestMeasure(unittest.TestCase):
    def test_scale_uses_input_height(self):
        """같은 사진이라도 입력한 키가 크면 픽셀당 cm가 줄어야 한다."""
        img, lms = blank_image(), make_landmarks()
        small = bm.measure(img, 160.0, pose_landmarks=lms)
        large = bm.measure(img, 190.0, pose_landmarks=lms)
        self.assertGreater(small.px_per_cm, large.px_per_cm)
        # 키가 크면 같은 픽셀 폭이 더 큰 cm로 환산된다
        self.assertGreater(large.measurements_cm['shoulder'],
                           small.measurements_cm['shoulder'])

    def test_plausible_range_for_average_adult(self):
        """175cm 성인 남성 정도의 입력이면 어깨너비가 상식적인 범위에 있어야 한다."""
        m = bm.measure(blank_image(), 175.0, pose_landmarks=make_landmarks())
        shoulder = m.measurements_cm['shoulder']
        self.assertTrue(35.0 < shoulder < 60.0, f'어깨너비 {shoulder}cm 는 비현실적')
        chest = m.measurements_cm['chest']
        self.assertTrue(35.0 < chest < 65.0, f'가슴단면 {chest}cm 는 비현실적')

    def test_wider_shoulders_give_larger_measurement(self):
        narrow = bm.measure(blank_image(), 175.0,
                            pose_landmarks=make_landmarks(shoulder_px=150.0))
        wide = bm.measure(blank_image(), 175.0,
                          pose_landmarks=make_landmarks(shoulder_px=210.0))
        self.assertGreater(wide.measurements_cm['shoulder'],
                           narrow.measurements_cm['shoulder'])

    def test_low_visibility_warns(self):
        m = bm.measure(blank_image(), 175.0,
                       pose_landmarks=make_landmarks(visibility=0.2))
        self.assertTrue(any('가려졌거나' in w for w in m.warnings))
        self.assertFalse(m.reliable)

    def test_fallback_scale_is_flagged(self):
        """세그멘테이션 없이 랜드마크로 스케일을 잡으면 오차가 크므로 경고해야 한다."""
        m = bm.measure(blank_image(), 175.0, pose_landmarks=make_landmarks())
        self.assertTrue(any('근사' in w for w in m.warnings))

    def test_no_person_returns_warning_not_crash(self):
        """사람이 안 보이는 사진이면 예외가 아니라 경고를 담아 반환해야 한다."""
        original = bm._run_pose
        bm._run_pose = lambda image: (None, None)
        try:
            m = bm.measure(blank_image(), 175.0)
        finally:
            bm._run_pose = original
        self.assertFalse(m.reliable)
        self.assertEqual(m.measurements_cm, {})
        self.assertTrue(any('사람을 찾지 못했' in w for w in m.warnings))

    def test_describe_is_readable(self):
        m = bm.measure(blank_image(), 175.0, pose_landmarks=make_landmarks())
        text = m.describe()
        self.assertIn('shoulder', text)
        self.assertIn('175', text)


class TestPixelHeight(unittest.TestCase):
    """세그멘테이션 마스크로 키 픽셀을 잡는 경로. Tasks API는 (H,W,1)로 준다."""

    def _mask(self, shape, top=100, bottom=900):
        m = np.zeros(shape, dtype=np.float32)
        m[top:bottom, 300:500] = 1.0
        return m

    def test_2d_mask(self):
        px, note = bm._pixel_height(blank_image(), make_landmarks(),
                                    self._mask((1200, 800)), 800, 1200)
        self.assertAlmostEqual(px, 799.0, delta=1.0)
        self.assertEqual(note, '')

    def test_3d_mask_from_tasks_api(self):
        """(H, W, 1) 마스크에서도 터지지 않고 같은 값이 나와야 한다."""
        px, note = bm._pixel_height(blank_image(), make_landmarks(),
                                    self._mask((1200, 800, 1)), 800, 1200)
        self.assertAlmostEqual(px, 799.0, delta=1.0)
        self.assertEqual(note, '')

    def test_empty_mask_falls_back(self):
        empty = np.zeros((1200, 800), dtype=np.float32)
        px, note = bm._pixel_height(blank_image(), make_landmarks(), empty, 800, 1200)
        self.assertGreater(px, 0)
        self.assertIn('근사', note)


class TestPlausibility(unittest.TestCase):
    """스케일이 틀어지면 조용히 이상한 값을 내지 말고 이유를 남겨야 한다."""

    def test_reasonable_values_pass(self):
        notes = bm._implausible_notes(
            {'shoulder': 44.0, 'chest': 50.0, 'hip': 48.0, 'torso_length': 52.0}, 175.0)
        self.assertEqual(notes, [])

    def test_absurd_shoulder_is_caught(self):
        notes = bm._implausible_notes({'shoulder': 60.0}, 175.0)  # 비율 0.34
        self.assertEqual(len(notes), 1)
        self.assertIn('shoulder', notes[0])
        self.assertIn('스케일', notes[0])

    def test_too_small_is_caught(self):
        notes = bm._implausible_notes({'chest': 30.0}, 175.0)  # 비율 0.17
        self.assertTrue(notes)

    def test_cropped_photo_is_flagged(self):
        """몸이 이미지 위/아래 끝에 닿으면 잘린 사진으로 보고 경고해야 한다."""
        mask = np.zeros((1200, 800), dtype=np.float32)
        mask[0:1200, 300:500] = 1.0   # 위아래 꽉 참
        _px, note = bm._pixel_height(blank_image(), make_landmarks(), mask, 800, 1200)
        self.assertIn('잘린', note)
        self.assertIn('머리', note)
        self.assertIn('발', note)

    def test_uncropped_photo_has_no_note(self):
        mask = np.zeros((1200, 800), dtype=np.float32)
        mask[100:900, 300:500] = 1.0
        _px, note = bm._pixel_height(blank_image(), make_landmarks(), mask, 800, 1200)
        self.assertEqual(note, '')


class TestIntegrationWithSizeFit(unittest.TestCase):
    """추정 치수를 그대로 사이즈 추천에 넣었을 때 말이 되는지."""

    def test_end_to_end(self):
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))
        from size_fit import SizeChart, recommend

        m = bm.measure(blank_image(), 175.0, pose_landmarks=make_landmarks())
        chart = SizeChart(
            name='테스트 티셔츠', category='upper',
            sizes={
                'S': {'chest': m.measurements_cm['chest'] - 2},
                'M': {'chest': m.measurements_cm['chest'] + 5},
                'L': {'chest': m.measurements_cm['chest'] + 12},
            },
        )
        rec = recommend(chart, m.measurements_cm)
        self.assertEqual(rec.best.size, 'M')  # +5cm 가 레귤러 목표치


if __name__ == '__main__':
    unittest.main()
