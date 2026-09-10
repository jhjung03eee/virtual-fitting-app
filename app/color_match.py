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


def match_garment_color(result, mask, garment, strength=1.0):
    """result의 mask 안쪽 색조를 garment에 맞춘다. PIL 이미지를 돌려준다.

    LAB 색공간에서 중앙값과 MAD를 맞춘다. 로버스트 통계를 쓰는 이유는
    마스크 안에 피부(반팔의 팔)나 그림자가 섞여도 덜 끌려가기 때문이다.

    **밝기(L)는 중앙값만 옮기고 산포는 유지한다.** 주름과 음영이 거기
    들어있어서 산포까지 맞추면 옷이 평평해진다. 색조(a·b)만 완전히 옮긴다.

    strength: 0이면 그대로, 1이면 완전히 맞춘다.
    skimage가 없으면 원본을 그대로 돌려준다(조용히 건너뜀).
    """
    if strength <= 0:
        return result
    try:
        from skimage.color import rgb2lab, lab2rgb
    except ImportError:
        return result

    selected = np.asarray(mask.convert('L')) > 127
    if not selected.any():
        return result

    lab = rgb2lab(np.asarray(result.convert('RGB'), dtype=np.float64) / 255.0)
    target = rgb2lab(garment_pixels(garment).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)

    source_median, source_spread = _robust_stats(lab[selected])
    target_median, target_spread = _robust_stats(target)

    adjusted = lab[selected].copy()
    adjusted[:, 0] += (target_median[0] - source_median[0]) * strength
    for channel in (1, 2):
        scaled = ((adjusted[:, channel] - source_median[channel])
                  * (target_spread[channel] / source_spread[channel])
                  + target_median[channel])
        adjusted[:, channel] += (scaled - adjusted[:, channel]) * strength

    lab[selected] = adjusted
    return Image.fromarray(np.clip(lab2rgb(lab) * 255.0, 0, 255).astype('uint8'))
