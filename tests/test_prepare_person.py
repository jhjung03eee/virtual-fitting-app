"""인물 사진 전처리 테스트 — EXIF 회전과 크롭 계산.

GPU도 mediapipe도 쓰지 않는다. 포즈 검출은 건너뛰고, 검출 결과가 주어졌을 때
어떤 영역을 자르는지(순수 계산)와 EXIF 처리만 본다.
"""
import os
import sys
import tempfile
import unittest

from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from prepare_person import TARGET_RATIO, crop_box, prepare  # noqa: E402


class TestExifOrientation(unittest.TestCase):
    """폰 사진은 가로로 저장되고 EXIF 태그로 세로임을 표시한다.

    이걸 적용하지 않으면 인물이 옆으로 누운 채 합성된다.
    """

    def _write_rotated(self, directory):
        path = os.path.join(directory, 'phone.jpg')
        image = Image.new('RGB', (400, 300), 'white')
        exif = image.getexif()
        exif[274] = 6  # orientation: 시계방향 90도 회전해서 보여라
        image.save(path, exif=exif)
        return path

    def test_pillow_does_not_rotate_on_its_own(self):
        """전제 확인 — 그냥 열면 가로 그대로다. 이래서 명시적 처리가 필요하다."""
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rotated(directory)
            with Image.open(path) as raw:
                self.assertEqual(raw.size, (400, 300))
                self.assertEqual(ImageOps.exif_transpose(raw).size, (300, 400))

    def test_prepare_outputs_portrait(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rotated(directory)
            out_path, _coverage = prepare(path, os.path.join(directory, 'out'))
            with Image.open(out_path) as result:
                self.assertEqual(result.size, (768, 1024))


class TestCropBox(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (3024, 4032), 'white')

    def _ratio(self, box):
        return (box[2] - box[0]) / (box[3] - box[1])

    def test_ratio_is_three_by_four(self):
        box = crop_box(self.image, (1200, 700, 1800, 3600))
        self.assertAlmostEqual(self._ratio(box), TARGET_RATIO, places=2)

    def test_stays_inside_image(self):
        box = crop_box(self.image, (1200, 700, 1800, 3600))
        self.assertGreaterEqual(box[0], 0)
        self.assertGreaterEqual(box[1], 0)
        self.assertLessEqual(box[2], self.image.width)
        self.assertLessEqual(box[3], self.image.height)

    def test_person_is_inside_the_crop(self):
        person = (1200, 700, 1800, 3600)
        box = crop_box(self.image, person)
        self.assertLessEqual(box[0], person[0])
        self.assertLessEqual(box[1], person[1])
        self.assertGreaterEqual(box[2], person[2])
        self.assertGreaterEqual(box[3], person[3])

    def test_person_fills_most_of_the_frame(self):
        """전신 사진에서 천장·바닥이 잔뜩 들어가면 옷 픽셀이 남지 않는다."""
        person = (1300, 900, 1700, 3400)
        box = crop_box(self.image, person)
        coverage = (person[3] - person[1]) / (box[3] - box[1])
        self.assertGreater(coverage, 0.7)

    def test_wide_pose_is_not_cut_off(self):
        """팔을 벌린 자세는 가로가 넓다. 세로 기준으로만 자르면 손이 잘린다."""
        person = (300, 900, 2700, 3400)  # 팔을 크게 벌린 경우
        box = crop_box(self.image, person)
        self.assertLessEqual(box[0], person[0])
        self.assertGreaterEqual(box[2], person[2])
        self.assertAlmostEqual(self._ratio(box), TARGET_RATIO, places=2)

    def test_person_larger_than_frame_still_fits_box(self):
        """검출 상자가 이미지를 꽉 채워도 크롭이 이미지 밖으로 나가면 안 된다."""
        box = crop_box(self.image, (0, 0, 3024, 4032))
        self.assertGreaterEqual(box[0], 0)
        self.assertGreaterEqual(box[1], 0)
        self.assertLessEqual(box[2], self.image.width)
        self.assertLessEqual(box[3], self.image.height)
        self.assertAlmostEqual(self._ratio(box), TARGET_RATIO, places=2)


if __name__ == '__main__':
    unittest.main()
