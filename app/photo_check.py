"""인물 사진이 합성에 맞는 자세·조건인지 GPU 없이 미리 검사한다.

앱은 사용자에게 **정면, 팔 내리고 몸에서 주먹 하나 간격, 빈손, 머리~발끝 전신** 사진을
요구한다(docs/PLAN.md "촬영 가이드"). 이유:
  - 학습 데이터(VITON-HD/DressCode)가 정면 선 자세라 벗어날수록 성공률이 떨어진다.
  - 팔을 크게 벌리거나(소매 길이 오류) 손을 몸 앞에 두면(엄지척·카드) 결과가 틀렸다.

규칙에 안 맞는 사진으로 30초짜리 합성을 돌리고 실패하느니, 올리는 순간 재촬영을
안내하는 게 낫다. 포즈 실행은 body_measure._run_pose 를 재사용한다.

판정 상수는 **초기값**이다. 가이드대로 찍은 A/B/C(팔 붙임/주먹 하나/45도) 사진으로
보정해야 한다. torch를 임포트하지 않는다.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from body_measure import POSE_LANDMARKS

MIN_VISIBILITY = 0.5
MIN_SHORT_SIDE = 720          # 이보다 작으면 줌 합성의 이득이 없다
MIN_BODY_PX = 600             # 머리~발목 픽셀 높이. 합성 모델 입력(세로 864~1024px)보다 한참 작으면 뭉개진다
# 화면 세로 중 인물(코~발목) 비율. 미달이면 다시 찍게 한다(docs/PLAN.md F-12).
# FASHN이 사진을 줄여 넣으면서 멀리 찍힌 사람은 다리가 너무 가늘어져 바지를 못 입혔다.
# 실측: 카톡 원본 0.54 바지 실패 / 카톡 크롭본 0.78·데모 여성 0.88 바지 성공. 0.65는 초기값.
MIN_BODY_FRACTION = 0.65
MIN_FRONTAL_RATIO = 0.45      # 어깨 폭 / 몸통 길이. 옆으로 돌면 줄어든다
ARMS_DOWN_MARGIN = 0.15       # 손목이 엉덩이보다 몸통 길이의 이만큼 위까지는 '내림'
MAX_ARM_SPREAD = 0.5          # 손목-엉덩이 가로 거리 / 몸통 길이. 정자세 0.25, 팔 벌림 0.67~0.70
# 얼굴 평균 밝기 / 배경 평균 밝기. 인물 전체로 재면 옷 색에 끌려간다
# (검은 옷 + 흰 벽 사진 3장이 전부 0.25~0.29로 역광 판정됐다). 그래서 얼굴만 본다.
BACKLIGHT_RATIO = 0.45
POSE_MAX_SIDE = 1280


@dataclass
class PhotoCheck:
    errors: List[str] = field(default_factory=list)     # 합성을 막는 문제
    warnings: List[str] = field(default_factory=list)   # 합성은 하되 알려줄 문제
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def message(self) -> str:
        if self.ok and not self.warnings:
            return '사진이 가이드에 맞아요.'
        lines = [f'다시 찍어주세요: {e}' for e in self.errors]
        lines += [f'참고: {w}' for w in self.warnings]
        return '\n'.join(lines)


def check_photo(image, landmarks=None, seg_mask=None) -> PhotoCheck:
    """image: PIL 이미지(EXIF 회전 적용된 것). landmarks/seg_mask 를 주면 포즈를 다시 돌리지 않는다."""
    check = PhotoCheck()
    width, height = image.size
    check.metrics['short_side'] = min(width, height)
    if min(width, height) < MIN_SHORT_SIDE:
        check.warnings.append('사진 해상도가 낮아요. 옷 질감이 흐리게 나올 수 있어요.')
    if width > height:
        check.errors.append('세로 사진으로 찍어주세요.')

    if landmarks is None:
        try:
            from body_measure import _run_pose
            # 폰 원본(3024x4032)을 그대로 넣으면 mediapipe가 segfault로 죽었다.
            # 랜드마크는 0~1 정규화 좌표라 줄인 사진으로 돌려도 원본에 그대로 쓸 수 있다.
            small = image.convert('RGB')
            small.thumbnail((POSE_MAX_SIDE, POSE_MAX_SIDE))
            landmarks, seg_mask = _run_pose(np.array(small))
        except Exception as e:  # mediapipe 미설치·모델 다운로드 실패
            check.warnings.append(f'자세 검사를 건너뛰었어요 ({type(e).__name__}).')
            return check
    if not landmarks:
        check.errors.append('사람을 찾지 못했어요. 한 명만 화면에 나오게 찍어주세요.')
        return check

    def point(name):
        lm = landmarks[POSE_LANDMARKS[name]]
        return np.array([lm.x * width, lm.y * height]), float(getattr(lm, 'visibility', 1.0))

    def seen(name):
        (x, y), vis = point(name)
        return vis >= MIN_VISIBILITY and 0 <= x <= width and 0 <= y <= height

    # 전신: 머리와 양 발목이 화면 안에 보여야 한다
    missing = [n for n in ('nose', 'left_ankle', 'right_ankle') if not seen(n)]
    if missing:
        check.errors.append('머리부터 발끝까지 전부 나오게 찍어주세요.')

    l_sh, r_sh = point('left_shoulder')[0], point('right_shoulder')[0]
    l_hip, r_hip = point('left_hip')[0], point('right_hip')[0]
    torso = np.linalg.norm((l_sh + r_sh) / 2 - (l_hip + r_hip) / 2)
    if torso < 1:
        check.errors.append('자세를 인식하지 못했어요. 정면으로 서서 찍어주세요.')
        return check

    frontal = np.linalg.norm(l_sh - r_sh) / torso
    check.metrics['frontal_ratio'] = round(float(frontal), 3)
    if frontal < MIN_FRONTAL_RATIO:
        check.errors.append('몸을 카메라 정면으로 향해 주세요.')

    if not missing:
        nose_y = point('nose')[0][1]
        ankle_y = max(point('left_ankle')[0][1], point('right_ankle')[0][1])
        body_px = ankle_y - nose_y
        check.metrics['body_px'] = round(float(body_px), 1)
        check.metrics['body_fraction'] = round(float(body_px / height), 3)
        if body_px / height < MIN_BODY_FRACTION:
            check.errors.append('더 가까이서 찍어주세요. 머리부터 발끝까지 화면을 꽉 채워야 옷이 정확하게 입혀져요.')
        elif body_px < MIN_BODY_PX:
            check.warnings.append('사진 해상도가 낮아 옷 질감이 흐리게 나올 수 있어요.')

    # 팔: 손목이 엉덩이 높이 근처까지 내려와 있고, 옆으로 너무 벌어지지 않아야 한다
    for side, label in (('left', '왼'), ('right', '오른')):
        wrist, vis = point(f'{side}_wrist')
        hip = l_hip if side == 'left' else r_hip
        if vis < MIN_VISIBILITY:
            check.warnings.append(f'{label}손이 잘 안 보여요. 손이 몸에 가려지지 않게 해주세요.')
            continue
        raised = (hip[1] - wrist[1]) / torso
        spread = abs(wrist[0] - hip[0]) / torso
        check.metrics[f'{side}_wrist_raise'] = round(float(raised), 3)
        check.metrics[f'{side}_arm_spread'] = round(float(spread), 3)
        # 벌림을 먼저 본다. 팔을 비스듬히 벌리면 손목이 엉덩이보다 올라가서
        # '손이 몸 앞' 으로 잘못 안내했다(카톡 팔 벌림 사진: 올라감 0.36, 벌림 0.67).
        if spread > MAX_ARM_SPREAD:
            check.errors.append('팔을 너무 벌렸어요. 몸에서 주먹 하나 정도만 떼 주세요.')
        elif raised > ARMS_DOWN_MARGIN:
            check.errors.append('팔을 자연스럽게 내려주세요. 손이 몸 앞에 있으면 옷이 가려져요.')
        elif seg_mask is not None and _arm_touches_body(seg_mask, point, side, width, height):
            check.warnings.append('팔이 몸에 붙어 있어요. 주먹 하나 정도 떼면 소매가 더 정확해요.')

    if seg_mask is not None and seen('nose'):
        face = point('nose')[0]
        ratio = _backlight_ratio(image, seg_mask, face, radius=0.12 * torso)
        if ratio is not None:
            check.metrics['person_bg_brightness'] = round(ratio, 3)
            if ratio < BACKLIGHT_RATIO:
                check.warnings.append('역광이에요. 빛이 앞에서 오게 찍어주세요.')

    # 같은 문구가 양쪽 팔에서 두 번 나오지 않게
    check.errors = list(dict.fromkeys(check.errors))
    check.warnings = list(dict.fromkeys(check.warnings))
    return check


def _mask_array(seg_mask, width, height):
    mask = np.asarray(seg_mask, dtype=np.float32).squeeze()
    if mask.shape != (height, width):
        from PIL import Image
        mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((width, height)),
                          dtype=np.float32) / 255.0
    return mask > 0.5


def _arm_touches_body(seg_mask, point, side, width, height):
    """팔꿈치~손목 사이 높이에서 팔과 몸통 사이에 배경 틈이 있는지 본다.

    엉덩이 관절에서 손목 쪽으로 한 줄을 훑어 사람 픽셀만 이어지면 붙은 것이다.
    """
    mask = _mask_array(seg_mask, width, height)
    elbow, wrist = point(f'{side}_elbow')[0], point(f'{side}_wrist')[0]
    hip = point(f'{side}_hip')[0]
    y = int(np.clip((elbow[1] + wrist[1]) / 2, 0, height - 1))
    x0, x1 = sorted((int(np.clip(hip[0], 0, width - 1)), int(np.clip(wrist[0], 0, width - 1))))
    if x1 - x0 < 3:
        return True
    return bool(mask[y, x0:x1 + 1].all())


def _backlight_ratio(image, seg_mask, face, radius) -> Optional[float]:
    """코 주변 얼굴 밝기 / 배경 밝기."""
    width, height = image.size
    mask = _mask_array(seg_mask, width, height)
    if mask.mean() < 0.05 or mask.mean() > 0.95:
        return None
    gray = np.asarray(image.convert('L'), dtype=np.float32)
    x, y, r = int(face[0]), int(face[1]), max(int(radius), 2)
    patch = gray[max(y - r, 0):y + r, max(x - r, 0):x + r]
    patch_mask = mask[max(y - r, 0):y + r, max(x - r, 0):x + r]
    background = gray[~mask].mean()
    if background < 1 or not patch_mask.any():
        return None
    return float(patch[patch_mask].mean() / background)
