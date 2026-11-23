"""합성 결과의 옷을 사이즈 비율만큼 늘리거나 줄인다 — 실험 단계, 앱에는 아직 안 붙임 (docs/PLAN.md F-35).

합성 모델은 사이즈를 넣을 입구가 없어 사이즈와 무관하게 한 장을 만든다(F-28, F-31).
그 한 장에서 **겨드랑이 아래 몸판만** 세로(기장)·가로(몸판 폭)로 늘려 사이즈 차이를 근사한다.
- 겨드랑이 위(목둘레·어깨·소매)는 그대로 둔다. 처음(v1)엔 어깨선 아래를 통째로 늘렸더니 소매가 팔꿈치까지 길어지고,
  가로로 줄일 때 목둘레가 움직여 살색 테두리가 생겼다.
- 가로는 몸 가운데 기준, 겨드랑이에서 ramp 구간 동안 1 → scale_x 로 서서히(옆선이 꺾이지 않게).
- 옷이 줄어 드러난 자리는 바깥(배경)에서 메운다. 팔 같은 keep 픽셀은 맨 위에 원래대로 다시 얹는다
  (몸판이 넓어져도 팔 앞을 덮지 않게).
학습은 없다. 주름·그림자는 늘어난 비율대로 같이 늘어난다(근사의 한계).
cv2 는 mediapipe 가 의존해 같이 설치된다.
"""
import numpy as np
from PIL import Image


def _below_hem_alpha(new_mask, top_y, band):
    """새 밑단 바로 아래 band 줄의 섞는 비율: 밑단에서 1, band 아래에서 0 (열마다 밑단 높이가 다르다)."""
    h, w = new_mask.shape
    below = new_mask.copy()
    below[:int(top_y) + 1] = False
    has = below.any(axis=0)
    hem = np.where(has, h - 1 - np.argmax(below[::-1], axis=0), -1)
    dy = np.arange(h)[:, None] - hem[None, :]
    alpha = np.clip(1.0 - dy / float(band), 0.0, 1.0) * (dy > 0) * has[None, :]
    return alpha.astype(np.float32)


def warp_garment(image, mask, top_y, center_x, scale_x, scale_y, keep=None, ramp=None, feather=None, hem_band=0):
    """image: PIL, mask: bool HxW(옷 영역), top_y: 이 줄(겨드랑이) 아래만 바꾼다, center_x: 몸 가운데 x(픽셀).

    scale_y: top_y 아래 몸판 길이 배율. scale_x: 몸판 폭 배율.
    keep: bool HxW — 결과 맨 위에 원래 픽셀을 다시 얹을 자리(팔·손). 없으면 생략.
    ramp: 가로 배율이 1에서 scale_x 까지 바뀌는 top_y 아래 구간(픽셀). 기본은 높이의 2%.
    feather: 옷 경계를 섞는 폭(픽셀). 기본은 높이의 0.3%.
    hem_band: 0보다 크면 새 밑단 아래 이 줄 수만큼 **변환한 그림을 이어 쓰고** 아래로 갈수록 원래 그림으로 섞는다.
      밑단 경계에서 늘린 옷과 원래 바지를 바로 섞으면 뿌연 띠가 생기고 원래 밑단 아래 그림자가 사라져
      칼로 자른 듯 보였다(F-39). 변환한 그림의 밑단 아래는 원래 밑단 아래(그림자 포함)를 옮겨 온 것이다.
    반환: (PIL 이미지, 새 옷 영역 bool HxW)
    """
    import cv2

    rgb = np.asarray(image.convert('RGB'))
    h, w = mask.shape
    ramp = ramp or max(4.0, 0.02 * h)
    feather = feather or max(1.0, 0.003 * h)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    # 역방향 지도: 새 그림의 (x, y) 가 원래 그림의 어디에서 오는가. top_y 위는 제자리
    src_y = np.where(ys > top_y, top_y + (ys - top_y) / scale_y, ys).astype(np.float32)
    t = np.clip((ys - top_y) / ramp, 0.0, 1.0)
    sx = 1.0 + (scale_x - 1.0) * t
    src_x = (center_x + (xs - center_x) / sx).astype(np.float32)

    warped = cv2.remap(rgb, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    new_mask = cv2.remap(mask.astype(np.uint8) * 255, src_x, src_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127

    # 옷이 줄어 드러난 자리를 먼저 메운 바탕 위에 새 옷을 얹는다.
    # 드러난 띠만 메우면 바로 옆 원래 옷 색이 띠 안으로 번진다 → 원래 옷 자리 전체를 바깥에서 메운 값을 쓴다.
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
        # 바꾸는 건 새 옷 밖의 드러난 자리만. 새 옷 경계의 바탕까지 바꾸면 메운 색이 테두리로 비친다.
        # 옷 테두리 전체(grown)를 바꾸면 변하지 않은 어깨·소매 테두리까지 흐린 메움 색으로 덮여 흰 줄·점이 생겼다(F-39)
        # → 실제로 옷이 비켜난 자리(mask & ~new_mask) 근처, top_y 아래만 바꾼다
        moved = cv2.dilate((mask & ~new_mask).astype(np.uint8) * 255, np.ones((k, k), np.uint8)) > 0
        reveal = (grown > 0) & ~new_mask & moved & (ys > top_y)
        base[reveal] = filled[reveal]
    alpha = cv2.GaussianBlur(new_mask.astype(np.float32), (0, 0), feather)
    if hem_band > 0:
        alpha = np.maximum(alpha, _below_hem_alpha(new_mask, top_y, hem_band))
        # 옷 옆 몇 픽셀도 변환한 그림을 쓴다. 옷 자리만 바꾸면 늘어난 옷 옆에 원래 바지 허리 모서리가 그대로 남아
        # 옆선이 원래 밑단 높이에서 꺾이고 바지와 엉겨 보였다(F-39, 사용자 "경계 양쪽 끝이 이상하다").
        # 변환한 그림에서는 옷 옆선·벽·바지 모서리가 원래 관계 그대로 함께 내려온다.
        side = max(2, int(hem_band // 2))
        wide = cv2.dilate(alpha, np.ones((1, 2 * side + 1), np.uint8))
        wide = cv2.GaussianBlur(wide, (0, 0), sigmaX=side / 2.0, sigmaY=0.5)
        if keep is not None:
            # 팔은 맨 위에 원래대로 다시 얹지만, 변환한 그림 속 팔이 원래 팔 밖으로 비치면 팔이 둘로 보인다 → 그 자리는 뺀다
            moved_arm = cv2.remap(keep.astype(np.uint8) * 255, src_x, src_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            moved_arm = cv2.dilate(moved_arm, np.ones((5, 5), np.uint8)) > 0
            wide[moved_arm] = alpha[moved_arm]
        alpha = np.where(ys > top_y, np.maximum(alpha, wide), alpha)
    alpha = alpha[..., None]
    out = warped * alpha + base * (1.0 - alpha)
    if keep is not None:
        a = cv2.GaussianBlur((keep & (ys > top_y)).astype(np.float32), (0, 0), feather)[..., None]
        out = rgb * a + out * (1.0 - a)
    return Image.fromarray(out.round().clip(0, 255).astype(np.uint8)), new_mask


def scales_for_size(current, target, top=0.0):
    """current·target: {'chest': 몸판 폭, 'length': 어깨선~밑단 기장} (같은 단위).

    top: 어깨선~겨드랑이(top_y) 길이. 그 위는 그대로 두므로 기장 차이를 아래 몸판이 모두 맡는다.
    → (scale_x, scale_y)
    """
    return target['chest'] / current['chest'], (target['length'] - top) / (current['length'] - top)
