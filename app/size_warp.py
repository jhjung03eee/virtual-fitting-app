"""합성 결과의 옷을 사이즈 비율만큼 늘리거나 줄인다 — 실험 단계, 앱에는 아직 안 붙임 (docs/PLAN.md F-35).

합성 모델은 사이즈를 넣을 입구가 없어 사이즈와 무관하게 한 장을 만든다(F-28, F-31).
그 한 장에서 **옷 영역만** 어깨선을 기준으로 세로(기장)·가로(몸판 폭)로 늘려 사이즈 차이를 근사한다.
- 세로: 어깨선 아래만 scale_y 배. 어깨선 위(목·얼굴)는 그대로.
- 가로: 몸 가운데 기준 scale_x 배. 목 부분이 벌어지지 않게 어깨선 위 ramp 구간에서 1 → scale_x 로 서서히.
- 옷이 줄어 드러난 자리는 주변 픽셀로 메운다(cv2 inpaint). 늘어난 옷은 바지·배경 위에 덮는다.
학습은 없다. 주름·그림자는 늘어난 비율대로 같이 늘어난다(근사의 한계).
cv2 는 mediapipe 가 의존해 같이 설치된다.
"""
import numpy as np
from PIL import Image


def warp_garment(image, mask, shoulder_y, center_x, scale_x, scale_y, ramp=None, feather=None):
    """image: PIL, mask: bool HxW(옷 영역), shoulder_y·center_x: 픽셀 좌표.

    ramp: 가로 배율이 1에서 scale_x 까지 바뀌는 어깨선 위 구간(픽셀). 기본은 높이의 3%.
    feather: 옷 경계를 섞는 폭(픽셀). 기본은 높이의 0.4%.
    반환: (PIL 이미지, 새 옷 영역 bool HxW)
    """
    import cv2

    rgb = np.asarray(image.convert('RGB'))
    h, w = mask.shape
    ramp = ramp or max(4.0, 0.03 * h)
    feather = feather or max(1.0, 0.004 * h)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    # 역방향 지도: 새 그림의 (x, y) 가 원래 그림의 어디에서 오는가
    src_y = np.where(ys > shoulder_y, shoulder_y + (ys - shoulder_y) / scale_y, ys).astype(np.float32)
    t = np.clip((ys - (shoulder_y - ramp)) / ramp, 0.0, 1.0)
    sx = 1.0 + (scale_x - 1.0) * t
    src_x = (center_x + (xs - center_x) / sx).astype(np.float32)

    warped = cv2.remap(rgb, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    new_mask = cv2.remap(mask.astype(np.uint8) * 255, src_x, src_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127

    # 옷이 줄어 드러난 자리를 먼저 메운 바탕 위에 새 옷을 얹는다.
    # 드러난 띠만 메우면 바로 옆 원래 옷 색이 띠 안으로 번진다(소매 바깥 줄무늬).
    # 그래서 원래 옷 자리 전체를 바깥(배경·팔)에서 메운다. 안쪽은 어차피 새 옷이 덮는다
    # 넓은 자리를 원래 크기로 메우면 수십 분 걸린다 → 1/4 크기에서 메우고 키운다(드러나는 곳은 배경이라 흐려도 된다)
    base = rgb.copy()
    if (mask & ~new_mask).any():
        k = int(2 * feather) | 1
        grown = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((k, k), np.uint8))
        small = (max(1, w // 4), max(1, h // 4))
        filled = cv2.inpaint(cv2.resize(rgb, small, interpolation=cv2.INTER_AREA),
                             (cv2.resize(grown, small, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255,
                             3, cv2.INPAINT_TELEA)
        filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
        # 바꾸는 건 새 옷 밖의 드러난 자리만. 새 옷 경계(목둘레 등)의 바탕까지 바꾸면 메운 색이 테두리로 비친다
        reveal = (grown > 0) & ~new_mask
        base[reveal] = filled[reveal]
    alpha = cv2.GaussianBlur(new_mask.astype(np.float32), (0, 0), feather)[..., None]
    out = (warped * alpha + base * (1.0 - alpha)).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(out), new_mask


def scales_for_size(current, target):
    """current·target: {'chest': 몸판 폭, 'length': 기장} (같은 단위). → (scale_x, scale_y)"""
    return target['chest'] / current['chest'], target['length'] / current['length']
