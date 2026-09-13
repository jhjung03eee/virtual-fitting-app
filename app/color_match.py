"""생성된 옷 영역의 색조를 원본 옷 사진에 맞춘다.

**torch를 임포트하지 않는다.** numpy와 PIL만 쓰므로 GPU 없이 테스트할 수 있다
(`size_fit.py`·`chart_extract.py`와 같은 이유로 분리했다).

왜 필요한가: 확산 설정으로는 하의 색을 못 맞췄다. guidance 2.5는 크림색,
5.0은 밝은 워싱 데님, 7.5는 얼룩. 스텝 8/30/50도 차이가 없었다. 원본은
다크 네이비 코듀로이인데 어느 설정도 그 색조를 내지 못한다.
그런데 **정답 색은 입력 옷 사진에 있다.** 모델이 못 맞추면 후처리로 맞춘다.
"""
import numpy as np
from PIL import Image


def garment_pixels(garment, tolerance=45.0, min_fraction=0.15):
    """옷 사진에서 옷에 해당하는 픽셀만 (N,3) 배열로.

    배경색을 **네 모서리에서 추정**해 그 색과 가까운 픽셀을 뺀다.
    고정 임계값(예: RGB 합 < 720)을 쓰면 안 되는 이유가 있다 — 쇼핑몰 상품
    사진의 배경은 순백이 아니라 밝은 회색인 경우가 많다. 실제로 코듀로이 바지
    사진의 배경이 회색이라 고정 임계값으로는 배경 69%가 '옷'으로 섞여 들어왔고,
    목표 색이 [51,58,82](진한 네이비)에서 [76,86,115]로 희석돼 보정이 헛돌았다.

    전경이 min_fraction 미만이면 이미지 전체를 쓴다. 흰 배경의 흰 티셔츠가
    그런 경우인데, 이때 전경으로 잡히는 건 **프린트뿐이다**(실제로 오랑우탄만
    7.8% 잡혔다). 그 색을 목표로 삼으면 티셔츠 전체가 갈색이 된다.
    전체를 쓰면 목표가 옷 색에 가까워져 보정이 거의 안 걸린다 — 안전한 쪽이다.

    min_fraction 은 실측으로 정했다. 문제 케이스(흰 티셔츠)가 7.8%,
    정상 케이스(코듀로이 바지) 36.7% · (맨투맨) 60.2% 였다. 0.15면 둘을
    여유 있게 가르면서, 프레임에서 작게 찍힌 옷을 잘못 폴백시키지 않는다.
    """
    array = np.asarray(garment.convert('RGB'), dtype=np.float64)
    height, width = array.shape[:2]
    box_h, box_w = max(1, height // 20), max(1, width // 20)
    corners = np.concatenate([
        array[:box_h, :box_w].reshape(-1, 3),
        array[:box_h, -box_w:].reshape(-1, 3),
        array[-box_h:, :box_w].reshape(-1, 3),
        array[-box_h:, -box_w:].reshape(-1, 3),
    ])
    background = np.median(corners, axis=0)

    foreground = np.linalg.norm(array - background, axis=2) > tolerance
    if foreground.mean() < min_fraction:
        foreground = np.ones(array.shape[:2], dtype=bool)
    return array[foreground]


def _robust_stats(values):
    """중앙값과 MAD. 평균/표준편차보다 이상치에 덜 끌린다."""
    median = np.median(values, axis=0)
    spread = np.median(np.abs(values - median), axis=0) + 1e-6
    return median, spread


def painted_weight(result, person, mask, threshold=18.0, width=20.0):
    """마스크 안에서 **실제로 새로 그려진 픽셀**의 가중치(0~1)를 돌려준다.

    AutoMasker 마스크는 옷 모양이 아니라 몸 주위를 넉넉히 덮는 영역이라,
    그 안에서 모델은 옷 주변 배경도 다시 그린다. 마스크 전체를 옷으로 보고
    색을 옮기면 그 배경이 옷 색으로 물들어 **후광**이 생긴다
    (실제로 셔츠 주위에 파란 후광, 바지 주위에 초록 후광이 생겼다).

    원본 인물 사진과 거의 같은 픽셀은 옷이 아니다(배경이나 남은 부분).
    RGB 차이가 threshold 이하면 0, threshold+width 이상이면 1, 사이는 부드럽게 잇는다.
    """
    inside = np.asarray(mask.convert('L'), dtype=np.float64) / 255.0
    if person is None:
        return inside
    a = np.asarray(result.convert('RGB'), dtype=np.float64)
    b = np.asarray(person.convert('RGB').resize(result.size), dtype=np.float64)
    difference = np.abs(a - b).mean(axis=2)
    changed = np.clip((difference - threshold) / width, 0.0, 1.0)
    return inside * changed


def skin_pixels(rgb, target_lab=None, lab=None, min_distance=20.0):
    """피부로 보이는 픽셀(bool 배열). 반팔 결과에서 새로 그려진 팔이 여기 걸린다.

    YCbCr 피부 범위를 쓴다. 다만 카키·베이지 옷은 이 범위에 걸칠 수 있어서,
    목표 옷 색(target_lab)과 LAB 거리가 min_distance 이하인 픽셀은 피부로 보지 않는다.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    skin = (cr >= 138) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    if target_lab is not None and lab is not None:
        skin &= np.linalg.norm(lab - target_lab, axis=-1) > min_distance
    return skin


# 한 번에 옮길 수 있는 최대 이동량 (LAB). 모델이 옷을 완전히 다른 색으로 그렸거나
# 목표 추정이 틀렸을 때 보정이 옷을 망가뜨리지 않게 막는다.
MAX_SHIFT = {'L': 25.0, 'ab': 20.0}

# 생성된 옷 색과의 LAB 거리가 COLOR_NEAR 이하면 완전히, COLOR_FAR 이상이면 보정하지 않는다.
COLOR_NEAR, COLOR_FAR = 12.0, 30.0


def match_garment_color(result, mask, garment, strength=1.0, person=None):
    """result의 mask 안쪽 색조를 garment에 맞춘다. PIL 이미지를 돌려준다.

    person 을 주면 마스크 안에서도 원본과 달라진 픽셀만 보정한다(후광 방지).
    파이프라인에서는 항상 주는 게 맞다. 없으면 마스크 전체를 옷으로 본다.

    LAB 색공간에서 **중앙값만 옮긴다**(L·a·b 모두, 산포는 그대로).
    예전에는 a·b 산포(MAD)까지 목표에 맞췄는데, 카키 같은 저채도 옷은
    생성 결과의 a·b 산포가 거의 0이라 배율이 커져 **작은 노이즈가 얼룩으로
    증폭**됐다. 산포에는 주름·음영·직물 결이 들어있으니 모델 것을 살린다.

    피부(반팔의 팔)는 통계와 보정에서 뺀다. 안 빼면 팔이 옷 색으로 물든다
    (카키 반팔에서 팔이 주황색이 됐다).
    이동량은 MAX_SHIFT 로 제한한다.

    strength: 0이면 그대로, 1이면 완전히 맞춘다.
    skimage가 없으면 원본을 그대로 돌려준다(조용히 건너뜀).
    """
    if strength <= 0:
        return result
    try:
        from skimage.color import rgb2lab, lab2rgb
    except ImportError:
        return result

    weight = painted_weight(result, person, mask)
    if not (weight > 0.9).any():
        return result

    rgb = np.asarray(result.convert('RGB'), dtype=np.float64)
    lab = rgb2lab(rgb / 255.0)
    target = rgb2lab(garment_pixels(garment).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    target_median, _ = _robust_stats(target)

    weight = weight * ~skin_pixels(rgb, target_median, lab)
    # 통계는 확실히 옷인 픽셀로만 낸다. 경계의 반쯤 섞인 픽셀이 들어가면 목표가 흐려진다.
    confident = weight > 0.9
    if not confident.any():
        return result
    source_median, _ = _robust_stats(lab[confident])

    # 모델은 옷 주변 배경도 원본과 조금 다르게 다시 그린다. painted_weight 는 그걸
    # '새로 그린 옷'으로 보므로, 파란 셔츠 옆 회색 벽이 파랗게 물드는 후광이 생겼다.
    # 생성된 옷 색(source_median)과 색조가 먼 픽셀은 가중치를 줄인다.
    # 밝기는 주름·음영으로 크게 흔들리므로 절반만 반영한다.
    distance = np.sqrt(((lab[..., 1:] - source_median[1:]) ** 2).sum(axis=-1)
                       + (0.5 * (lab[..., 0] - source_median[0])) ** 2)
    weight = weight * np.clip((COLOR_FAR - distance) / (COLOR_FAR - COLOR_NEAR), 0.0, 1.0)

    shift = target_median - source_median
    shift[0] = np.clip(shift[0], -MAX_SHIFT['L'], MAX_SHIFT['L'])
    shift[1:] = np.clip(shift[1:], -MAX_SHIFT['ab'], MAX_SHIFT['ab'])

    # 픽셀마다 '새로 그려진 정도'만큼만 옮긴다. 배경·원래 남은 부분·피부는 0이라 그대로다.
    corrected = lab + shift * (weight * strength)[..., None]
    return Image.fromarray(np.clip(lab2rgb(corrected) * 255.0, 0, 255).astype('uint8'))
