"""옷 사진이 합성에 맞는지 미리 검사하고, 안 되는 옷은 경고한다 (A안: 지원 범위 명시).

실험으로 확인한 범위:
  - 잘 됨: 단색·중간 톤 기본 의류, 큰 글자 프린트, 단색 배경의 제품 단독 컷
  - 안 됨: 사진 같은 프린트(오랑우탄 티셔츠가 얼룩으로 뭉개짐), 흰 옷(색·프린트 불안정),
          모델이 입은 착용컷(몸·배경까지 옷으로 들어감)

**경고만 하고 합성은 막지 않는다.** 규칙이 틀릴 수 있고, 사용자가 결과를 보고 판단하면 된다.
판정 상수는 샘플 6벌(set2 3벌 + 초기 3벌)로 정한 초기값이다.

알려진 한계: 네이비 코듀로이 바지(실패)는 이 지표로는 성공한 올리브 바지와 구별되지 않는다.
특수 질감은 축소한 사진에서 잘 안 드러난다. torch를 임포트하지 않는다.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from PIL import Image, ImageFilter

from color_match import skin_pixels

ANALYSIS_SIDE = 512
BACKGROUND_TOLERANCE = 45.0   # color_match.garment_pixels 와 같은 값
MIN_FOREGROUND = 0.15         # 이보다 작으면 배경과 같은 색 옷(흰 배경의 흰 옷)
MAX_SKIN = 0.05               # 전경 중 피부 비율. 넘으면 착용컷
MAX_OFF_COLOR = 0.15          # 옷 중앙값 색에서 먼 픽셀 비율. 샘플: 단색 0.02~0.06, 사진 프린트 0.65
MAX_EDGE = 0.12               # 에지 밀도. 샘플: 단색·글자 0.03~0.07, 사진 프린트 0.21
MAX_CORNER_STD = 25.0         # 모서리 색 편차. 크면 단색 배경이 아니다
WHITE_L, WHITE_CHROMA = 85.0, 10.0
MIN_SHORT_SIDE = 512


@dataclass
class GarmentCheck:
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def suitable(self) -> bool:
        return not self.warnings

    def message(self) -> str:
        if self.suitable:
            return '합성에 잘 맞는 옷 사진이에요.'
        return '\n'.join(f'주의: {w}' for w in self.warnings)


def check_garment(garment) -> GarmentCheck:
    try:
        from skimage.color import rgb2lab
    except ImportError:
        return GarmentCheck(warnings=['옷 사진 검사를 건너뛰었어요 (scikit-image 없음).'])

    check = GarmentCheck()
    image = garment.convert('RGB')
    check.metrics['short_side'] = min(image.size)
    if min(image.size) < MIN_SHORT_SIDE:
        check.warnings.append('옷 사진 해상도가 낮아요. 질감이 흐리게 나올 수 있어요.')
    image = image.copy()
    image.thumbnail((ANALYSIS_SIDE, ANALYSIS_SIDE))

    rgb = np.asarray(image, dtype=np.float64)
    h, w = rgb.shape[:2]
    bh, bw = max(1, h // 20), max(1, w // 20)
    corners = np.concatenate([rgb[:bh, :bw].reshape(-1, 3), rgb[:bh, -bw:].reshape(-1, 3),
                              rgb[-bh:, :bw].reshape(-1, 3), rgb[-bh:, -bw:].reshape(-1, 3)])
    corner_std = float(corners.std(axis=0).mean())
    background = np.median(corners, axis=0)
    foreground = np.linalg.norm(rgb - background, axis=2) > BACKGROUND_TOLERANCE
    fraction = float(foreground.mean())
    check.metrics.update(corner_std=round(corner_std, 1), foreground=round(fraction, 3))

    if corner_std > MAX_CORNER_STD:
        check.warnings.append('단색 배경에 옷만 놓인 제품 사진이 가장 잘 돼요.')

    lab = rgb2lab(rgb / 255.0)
    if fraction < MIN_FOREGROUND:
        # 배경과 같은 색의 옷이다. 이때 전경으로 잡히는 건 프린트뿐이라
        # 피부·색 지표를 옷 전체에 대해 낼 수 없다(오랑우탄 프린트가 '피부 56%'로 잡혔다).
        whole = np.median(lab.reshape(-1, 3), axis=0)
        if whole[0] > WHITE_L - 10:
            check.warnings.append('흰색·아주 밝은 옷은 색과 프린트가 정확하지 않을 수 있어요.')
        if fraction > 0.01 and _edge_density(image, foreground) > MAX_EDGE:
            check.warnings.append('사진 같은 프린트는 뭉개질 수 있어요. 무지나 큰 글자 프린트가 잘 돼요.')
        return check

    median = np.median(lab[foreground], axis=0)
    chroma = float(np.hypot(median[1], median[2]))
    skin = float(skin_pixels(rgb, median, lab)[foreground].mean())
    off_color = float((np.linalg.norm(lab[foreground] - median, axis=1) > 25).mean())
    edge = _edge_density(image, foreground)
    check.metrics.update(lightness=round(float(median[0]), 1), chroma=round(chroma, 1),
                         skin=round(skin, 3), off_color=round(off_color, 3), edge=round(edge, 3))

    if skin > MAX_SKIN:
        check.warnings.append('사람이 입은 사진 같아요. 옷만 찍힌 제품 사진을 올려주세요.')
    if median[0] > WHITE_L and chroma < WHITE_CHROMA:
        check.warnings.append('흰색·아주 밝은 옷은 색과 프린트가 정확하지 않을 수 있어요.')
    if off_color > MAX_OFF_COLOR or edge > MAX_EDGE:
        check.warnings.append('사진 같은 프린트나 복잡한 무늬는 뭉개질 수 있어요. 무지나 큰 글자 프린트가 잘 돼요.')
    return check


def _edge_density(image, foreground):
    edges = np.asarray(image.convert('L').filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    return float((edges[foreground] > 40).mean()) if foreground.any() else 0.0
