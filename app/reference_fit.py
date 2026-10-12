"""기준 옷 비교 사이즈 추천 — "잘 맞는 내 옷 한 벌"과 새 옷 치수표를 비교한다.

**왜 몸 치수 대신 옷 치수인가**

`size_fit.py` 는 몸 치수와 옷 치수의 차이(여유분)로 핏을 판정한다. 두 가지 약점이 있다.
  1. 몸 치수를 재기 어렵다. 사진 추정(`body_measure.py`)은 보정 데이터가 없어 믿을 수 없고,
     줄자로 재도 가슴·엉덩이 둘레는 혼자 재기 어렵다.
  2. 여유분 기준(가슴 단면 +3~7cm = 레귤러)은 쇼핑몰 가이드의 일반값이다.
     넉넉하게 입는 사람과 붙게 입는 사람의 차이를 반영하지 못한다.

반면 **잘 맞는 옷의 치수에는 그 사람의 몸과 취향이 같이 들어 있다.** 옷을 눕혀 놓고
쇼핑몰과 같은 방법(단면)으로 재면 되므로 몸을 재는 것보다 쉽고, 단위·기준점이 치수표와 같다.
그래서 이 모듈은 여유분을 다시 판정하지 않고 **기준 옷과의 차이만** 본다.

    차이 = 새 옷 사이즈의 치수 - 기준 옷의 치수   (단면 cm, + 면 기준 옷보다 큼)

**한계.** 소재(신축성)와 핏 설계가 다른 옷끼리는 같은 치수라도 착용감이 다르다.
같은 종류(반팔 티끼리, 청바지끼리)를 기준으로 고를수록 정확하다.

`size_fit.py` 와 같은 이유로 순수 계산이다(GPU·mediapipe 없이 테스트 가능).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from size_fit import SizeChart, ko

# 항목별 "거의 같음"으로 볼 차이(cm). 옷을 손으로 재면 1cm 안팎은 측정 오차다.
# 기장 항목은 재는 위치가 조금만 달라도 차이가 커서 더 넓게 잡는다.
SAME_TOLERANCE = {
    'chest': 1.0, 'shoulder': 1.0, 'waist': 1.0, 'hip': 1.0, 'thigh': 1.0,
    'hem': 1.0, 'cuff': 1.0, 'armhole': 1.0,
    'length': 2.0, 'sleeve': 2.0, 'rise': 1.5,
}
DEFAULT_SAME_TOLERANCE = 1.0
# 이 배수를 넘으면 '조금'이 아니라 확실히 크거나 작다.
CLEAR_MULTIPLIER = 3.0

# 추천 점수에서 항목별 중요도. size_fit.DIMENSION_WEIGHT 와 같은 생각이다:
# 가슴·허리가 맞아야 입을 수 있고, 기장·소매는 취향 차이가 크다.
WEIGHT = {
    'chest': 1.0, 'waist': 1.0, 'hip': 0.8, 'shoulder': 0.6, 'thigh': 0.5,
    'length': 0.3, 'sleeve': 0.3, 'rise': 0.3, 'hem': 0.2, 'cuff': 0.1, 'armhole': 0.3,
}
DEFAULT_WEIGHT = 0.2

# 기준 옷보다 작은 쪽은 큰 쪽보다 나쁘다(작으면 못 입고, 크면 입을 수는 있다).
# 둘레 항목에서 기준보다 작을 때 차이에 이 배수를 곱한다.
TIGHT_DIMENSIONS = ('chest', 'waist', 'hip', 'thigh')
TIGHT_PENALTY = 1.5

# 종합 판정(작음/맞음/큼)에 쓰는 항목. 검증 스크립트(scripts/eval_size_fit.py)가
# 사람이 느낀 착용감과 비교하는 값이다.
VERDICT_DIMENSIONS = {
    'upper': ('chest', 'shoulder', 'length'),
    'lower': ('waist', 'hip', 'thigh', 'length'),
}
# 가중 평균 차이가 이 범위 밖이면 작음/큼. 큰 쪽을 더 너그럽게 둔다(TIGHT_PENALTY와 같은 이유).
# 초기값이다 — 팀원 착용 기록으로 보정한다(docs/PLAN.md).
VERDICT_SMALL_BELOW = -1.5
VERDICT_LARGE_ABOVE = 2.5

VERDICTS = ('작음', '맞음', '큼')


@dataclass
class ReferenceGarment:
    """잘 맞는 내 옷 한 벌의 치수. 모든 값은 cm, 단면 기준(치수표와 같은 방식)."""
    category: str  # 'upper' | 'lower'
    dims: Dict[str, float]
    name: str = '기준 옷'

    @classmethod
    def from_chart(cls, chart: SizeChart, size: str) -> 'ReferenceGarment':
        """치수표가 있는 옷을 이미 갖고 있으면 그 사이즈 행을 그대로 기준으로 쓴다."""
        return cls(category=chart.category, dims=dict(chart.sizes[size]),
                   name=f'{chart.name} {size}')

    @classmethod
    def from_inputs(cls, category: str, name: str = '기준 옷', **values) -> 'ReferenceGarment':
        """UI에서 온 값(빈 칸·0 포함)을 정리한다. 빈 칸은 모르는 항목이다."""
        dims = {}
        for key, value in values.items():
            if value in (None, '', 0):
                continue
            dims[key] = float(value)
        return cls(category=category, dims=dims, name=name or '기준 옷')


@dataclass
class DimensionDiff:
    dimension: str
    garment_cm: float
    reference_cm: float
    diff_cm: float
    label: str


@dataclass
class SizeComparison:
    size: str
    diffs: List[DimensionDiff]
    score: float  # 낮을수록 기준 옷에 가깝다
    verdict: str  # '작음' | '맞음' | '큼'


@dataclass
class ReferenceRecommendation:
    best: Optional[SizeComparison]
    ranked: List[SizeComparison] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        if self.best is None:
            return '기준 옷과 비교할 수 있는 항목이 없습니다.'
        close = [d for d in self.best.diffs if d.label == '거의 같음']
        if len(close) == len(self.best.diffs):
            return f'{self.best.size} 추천 — 기준 옷과 거의 같은 크기입니다.'
        return f'{self.best.size} 추천 — 기준 옷과 가장 비슷합니다 ({self.best.verdict}).'


def label_diff(dimension: str, diff_cm: float) -> str:
    """차이(cm)를 사람이 읽는 말로 바꾼다."""
    same = SAME_TOLERANCE.get(dimension, DEFAULT_SAME_TOLERANCE)
    if abs(diff_cm) <= same:
        return '거의 같음'
    direction = '큼' if diff_cm > 0 else '작음'
    if abs(diff_cm) <= same * CLEAR_MULTIPLIER:
        return f'조금 {direction}'
    return direction


def verdict(category: str, diffs: Dict[str, float]) -> str:
    """기준 옷 대비 종합 판정. 판정 항목이 하나도 없으면 공통 항목 전체로 본다."""
    keys = [d for d in VERDICT_DIMENSIONS.get(category, ()) if d in diffs] or list(diffs)
    if not keys:
        return '맞음'
    total = sum(WEIGHT.get(d, DEFAULT_WEIGHT) for d in keys)
    mean = sum(WEIGHT.get(d, DEFAULT_WEIGHT) * diffs[d] for d in keys) / total
    if mean < VERDICT_SMALL_BELOW:
        return '작음'
    if mean > VERDICT_LARGE_ABOVE:
        return '큼'
    return '맞음'


def compare_size(chart: SizeChart, size: str, ref: ReferenceGarment) -> SizeComparison:
    """한 사이즈를 기준 옷과 항목별로 비교한다."""
    diffs: List[DimensionDiff] = []
    score = 0.0
    for dimension, garment_cm in sorted(chart.sizes[size].items()):
        if dimension not in ref.dims:
            continue
        diff = garment_cm - ref.dims[dimension]
        diffs.append(DimensionDiff(
            dimension=dimension, garment_cm=garment_cm, reference_cm=ref.dims[dimension],
            diff_cm=round(diff, 1), label=label_diff(dimension, diff),
        ))
        same = SAME_TOLERANCE.get(dimension, DEFAULT_SAME_TOLERANCE)
        # 측정 오차 범위 안의 차이는 벌점을 주지 않는다
        penalty = max(0.0, abs(diff) - same)
        if dimension in TIGHT_DIMENSIONS and diff < 0:
            penalty *= TIGHT_PENALTY
        score += WEIGHT.get(dimension, DEFAULT_WEIGHT) * penalty
    return SizeComparison(
        size=size, diffs=diffs, score=round(score, 2),
        verdict=verdict(chart.category, {d.dimension: d.diff_cm for d in diffs}),
    )


def recommend_by_reference(chart: SizeChart, ref: ReferenceGarment) -> ReferenceRecommendation:
    """치수표의 모든 사이즈를 기준 옷과 비교해 가장 가까운 사이즈를 고른다."""
    notes: List[str] = []
    if ref.category != chart.category:
        return ReferenceRecommendation(best=None, notes=[
            '기준 옷과 새 옷의 종류(상의/하의)가 다릅니다. 같은 종류의 옷을 기준으로 넣어주세요.'])

    common = [d for d in chart.dimensions() if d in ref.dims]
    if not common:
        return ReferenceRecommendation(best=None, notes=[
            '기준 옷 치수와 치수표에 공통 항목이 없습니다. '
            f"치수표 항목: {', '.join(ko(d) for d in chart.dimensions())}"])

    missing = [d for d in chart.dimensions() if d not in ref.dims]
    if missing:
        notes.append(f"기준 옷에서 재지 않아 비교하지 못한 항목: {', '.join(ko(d) for d in missing)}")
    unused = [d for d in sorted(ref.dims) if d not in chart.dimensions()]
    if unused:
        notes.append(f"이 치수표에 없는 항목이라 사용하지 않았습니다: {', '.join(ko(d) for d in unused)}")
    main = VERDICT_DIMENSIONS.get(chart.category, ())
    if main and not any(d in common for d in main[:1]):
        notes.append(f'{ko(main[0])}를 재서 넣으면 추천이 더 정확해집니다.')

    ranked = sorted((compare_size(chart, size, ref) for size in chart.sizes),
                    key=lambda c: c.score)
    if all(c.verdict == '작음' for c in ranked):
        notes.append('모든 사이즈가 기준 옷보다 작습니다.')
    elif all(c.verdict == '큼' for c in ranked):
        notes.append('모든 사이즈가 기준 옷보다 큽니다.')
    return ReferenceRecommendation(best=ranked[0], ranked=ranked, notes=notes)
