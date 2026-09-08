"""사진 한 장 + 키(cm)로 몸 치수를 추정한다 (MediaPipe Pose).

**정확도에 대한 정직한 전제**

단일 정면 사진에서 실측 치수를 얻는 것은 불가능하다. 이 모듈이 하는 일은
`추정`이고, 아래 가정 위에서만 성립한다.

1. 정면·전신·정자세, 카메라가 대략 몸 높이에서 수평으로 촬영
2. 입력한 키(cm)로 픽셀→cm 스케일을 잡는다 (머리끝~발끝 픽셀 = 키)
3. MediaPipe 랜드마크는 관절 중심이라 옷 치수의 기준점(어깨 솔기 등)과 다르다.
   보정계수(SHOULDER_LANDMARK_TO_BREADTH 등)로 메우며, 이 값들은 경험적 상수다.
4. 정면 사진에서는 두께(깊이)를 볼 수 없다. 가슴/허리 둘레는 타원 가정으로 환산한다.

따라서 이 값들은 **절대 치수보다 상대 비교(어떤 사이즈가 더 맞는가)** 에 쓰는 것이 맞다.
계획서의 "정밀 치수보다 상대적 사이즈 추천으로 스코프를 잡는다"와 같은 맥락이다.

출력은 전부 cm이고, 옷 치수표와 맞추기 위해 **단면(폭) 기준**으로 낸다.
(가슴단면 = 가슴둘레 / 2)
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --- 경험적 보정 상수 -------------------------------------------------------
# MediaPipe 어깨 랜드마크는 어깨관절 중심이라 실제 어깨너비(견봉~견봉)보다 좁다.
SHOULDER_LANDMARK_TO_BREADTH = 1.18
# 가슴/허리/엉덩이 단면은 정면 폭의 절반이 아니라, 타원 둘레의 절반이다.
# 몸통 단면을 타원으로 보고 깊이/폭 비를 아래 값으로 가정한다.
DEPTH_TO_WIDTH = {'chest': 0.72, 'waist': 0.78, 'hip': 0.80}
# 랜드마크 신뢰도가 이보다 낮으면 경고를 붙인다.
MIN_VISIBILITY = 0.6

POSE_LANDMARKS = {
    'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
}


@dataclass
class BodyMeasurement:
    height_cm: float
    px_per_cm: float
    measurements_cm: Dict[str, float] = field(default_factory=dict)  # 단면 기준
    raw_widths_cm: Dict[str, float] = field(default_factory=dict)    # 정면 폭 (참고용)
    warnings: List[str] = field(default_factory=list)
    landmark_visibility: Dict[str, float] = field(default_factory=dict)

    @property
    def reliable(self) -> bool:
        return not self.warnings

    def describe(self) -> str:
        lines = [f'키 {self.height_cm:.0f}cm 기준 추정 (단면):']
        for k, v in sorted(self.measurements_cm.items()):
            lines.append(f'  {k:9s} {v:5.1f} cm')
        for w in self.warnings:
            lines.append(f'  [주의] {w}')
        return '\n'.join(lines)


def _ellipse_perimeter(width: float, depth: float) -> float:
    """라마누잔 근사. 타원 둘레."""
    a, b = width / 2.0, depth / 2.0
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def _circumference_half_from_width(front_width_cm: float, part: str) -> float:
    """정면 폭 -> 둘레의 절반(=단면). 깊이는 가정값으로 채운다."""
    depth = front_width_cm * DEPTH_TO_WIDTH[part]
    return _ellipse_perimeter(front_width_cm, depth) / 2.0


def _dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def measure(image, height_cm: float, pose_landmarks=None) -> BodyMeasurement:
    """이미지에서 몸 치수를 추정한다.

    image: PIL.Image 또는 numpy 배열(RGB)
    height_cm: 사용자가 입력한 키
    pose_landmarks: 미리 뽑아둔 랜드마크가 있으면 재사용 (테스트/재현용)
    """
    import numpy as np

    if hasattr(image, 'convert'):  # PIL
        image = np.asarray(image.convert('RGB'))
    h, w = image.shape[:2]

    if pose_landmarks is None:
        pose_landmarks, seg_mask = _run_pose(image)
    else:
        seg_mask = None

    if pose_landmarks is None:
        return BodyMeasurement(height_cm=height_cm, px_per_cm=0.0,
                               warnings=['사람을 찾지 못했습니다. 전신이 나온 정면 사진인지 확인하세요.'])

    def pt(name):
        lm = pose_landmarks[POSE_LANDMARKS[name]]
        return (lm.x * w, lm.y * h)

    visibility = {name: float(pose_landmarks[idx].visibility)
                  for name, idx in POSE_LANDMARKS.items()}
    warnings: List[str] = []

    low_vis = [n for n in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')
               if visibility[n] < MIN_VISIBILITY]
    if low_vis:
        warnings.append(f"가려졌거나 흐릿한 부위가 있습니다: {', '.join(low_vis)}. "
                        '정면 전신 사진이면 정확도가 올라갑니다.')

    px_height, scale_note = _pixel_height(image, pose_landmarks, seg_mask, w, h)
    if px_height <= 0:
        return BodyMeasurement(height_cm=height_cm, px_per_cm=0.0,
                               warnings=['몸 전체 높이를 잡지 못했습니다. 발끝까지 나오게 찍어주세요.'])
    if scale_note:
        warnings.append(scale_note)

    px_per_cm = px_height / height_cm

    shoulder_px = _dist(pt('left_shoulder'), pt('right_shoulder'))
    hip_px = _dist(pt('left_hip'), pt('right_hip'))
    shoulder_mid = ((pt('left_shoulder')[0] + pt('right_shoulder')[0]) / 2,
                    (pt('left_shoulder')[1] + pt('right_shoulder')[1]) / 2)
    hip_mid = ((pt('left_hip')[0] + pt('right_hip')[0]) / 2,
               (pt('left_hip')[1] + pt('right_hip')[1]) / 2)
    torso_px = _dist(shoulder_mid, hip_mid)

    shoulder_cm = shoulder_px / px_per_cm * SHOULDER_LANDMARK_TO_BREADTH
    hip_width_cm = hip_px / px_per_cm
    torso_cm = torso_px / px_per_cm

    # 가슴 폭은 어깨~엉덩이 사이 가슴 높이에서의 실루엣 폭이 이상적이지만,
    # 팔이 몸통에 붙어 있으면 실루엣에 팔이 섞인다. 랜드마크만으로 추정한다.
    chest_width_cm = shoulder_cm / SHOULDER_LANDMARK_TO_BREADTH * 0.92

    measurements = {
        'shoulder': round(shoulder_cm, 1),
        'chest': round(_circumference_half_from_width(chest_width_cm, 'chest'), 1),
        'hip': round(_circumference_half_from_width(hip_width_cm, 'hip'), 1),
        'torso_length': round(torso_cm, 1),
    }
    raw_widths = {
        'chest_front_width': round(chest_width_cm, 1),
        'hip_front_width': round(hip_width_cm, 1),
    }

    return BodyMeasurement(
        height_cm=height_cm,
        px_per_cm=round(px_per_cm, 3),
        measurements_cm=measurements,
        raw_widths_cm=raw_widths,
        warnings=warnings,
        landmark_visibility={k: round(v, 2) for k, v in visibility.items()},
    )


def _pixel_height(image, landmarks, seg_mask, w: int, h: int):
    """머리끝~발끝 픽셀 높이. 세그멘테이션이 있으면 그걸 쓰고, 없으면 랜드마크로 추정."""
    import numpy as np

    if seg_mask is not None:
        ys, _xs = np.where(seg_mask > 0.5)
        if ys.size > 0:
            return float(ys.max() - ys.min()), ''

    # 폴백: 코~발목 거리에 머리 윗부분과 발 높이를 보정해서 더한다.
    nose_y = landmarks[POSE_LANDMARKS['nose']].y * h
    ankles = [landmarks[POSE_LANDMARKS[n]].y * h for n in ('left_ankle', 'right_ankle')]
    ankle_y = max(ankles)
    if ankle_y <= nose_y:
        return 0.0, ''
    # 코는 정수리에서 약 12% 아래, 발목은 바닥에서 약 4% 위 (인체 비율 통계 근사)
    span = (ankle_y - nose_y) / (1.0 - 0.12 - 0.04)
    return float(span), '세그멘테이션을 못 써서 키 스케일을 인체 비율로 근사했습니다(오차 큼).'


def _run_pose(image):
    """MediaPipe Pose 실행. (landmarks, segmentation_mask) 반환."""
    try:
        import mediapipe as mp
    except ImportError as e:
        raise ImportError(
            'mediapipe가 필요합니다. venv에 설치하세요:\n'
            '    <venv>/bin/pip install mediapipe'
        ) from e

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
    ) as pose:
        result = pose.process(image)

    if not result.pose_landmarks:
        return None, None
    return result.pose_landmarks.landmark, result.segmentation_mask
