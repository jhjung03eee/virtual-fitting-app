"""여유분 계산과 사이즈 추천 — 이 프로젝트의 핵심 기능.

옷 치수표(단면 기준)와 추정된 몸 치수를 비교해 사이즈별 여유분을 구하고,
어떤 사이즈가 어떤 핏이 되는지 판정한다.

**용어**: 한국 쇼핑몰 치수표는 대부분 '단면'(flat, 눕혀서 잰 폭)으로 표기한다.
가슴단면 50cm 는 가슴둘레 100cm 에 대응한다. 이 모듈은 전부 단면 기준으로 계산한다.

측정 정확도의 한계는 body_measure 쪽에 있고, 이 모듈은 순수 계산이라
mediapipe 없이도 동작한다(테스트 가능).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 핏 판정 구간 (단위 cm, 단면 기준 여유분 = 옷 단면 - 몸 단면)
# 근거: 상의 가슴단면은 몸 단면보다 3~7cm 크면 일반적인 레귤러 핏으로 입힌다.
#      값 자체는 국내 쇼핑몰 사이즈 가이드에서 흔히 쓰이는 범위를 참고한 것이고,
#      브랜드마다 다르므로 튜닝 가능한 상수로 둔다.
FIT_BANDS = {
    'upper': {
        'chest': [
            (-99.0, 0.0, '작음'),
            (0.0, 3.0, '타이트'),
            (3.0, 7.0, '레귤러'),
            (7.0, 12.0, '루즈'),
            (12.0, 99.0, '오버핏'),
        ],
        'shoulder': [
            (-99.0, -1.5, '작음'),
            (-1.5, 1.5, '레귤러'),
            (1.5, 4.0, '루즈'),
            (4.0, 99.0, '오버핏'),
        ],
    },
    'lower': {
        'waist': [
            (-99.0, 0.0, '작음'),
            (0.0, 2.0, '타이트'),
            (2.0, 5.0, '레귤러'),
            (5.0, 99.0, '루즈'),
        ],
        'hip': [
            (-99.0, 0.0, '작음'),
            (0.0, 3.0, '타이트'),
            (3.0, 8.0, '레귤러'),
            (8.0, 99.0, '루즈'),
        ],
    },
}

# 추천 점수를 매길 때 어떤 항목을 얼마나 중요하게 볼지.
# 상의는 가슴이 맞아야 입을 수 있고 어깨는 어느 정도 여유가 허용된다.
DIMENSION_WEIGHT = {
    'chest': 1.0,
    'shoulder': 0.6,
    'waist': 1.0,
    'hip': 0.8,
    'length': 0.3,
}

# '가장 좋은 핏'으로 볼 여유분 목표치 (단면 cm). 점수는 이 값과의 거리로 계산한다.
IDEAL_EASE = {
    'upper': {'chest': 5.0, 'shoulder': 0.5, 'length': 0.0},
    'lower': {'waist': 3.0, 'hip': 5.0, 'length': 0.0},
}

# 이 항목들은 "작으면 못 입는다"에 해당해 음수 여유분에 큰 벌점을 준다.
HARD_LIMIT_DIMENSIONS = ('chest', 'waist', 'hip')
UNDERSIZE_PENALTY = 3.0


@dataclass
class SizeChart:
    """옷 한 벌의 사이즈별 치수표. 모든 값은 cm, 단면 기준."""
    name: str
    category: str  # 'upper' | 'lower'
    # {'M': {'chest': 52.0, 'shoulder': 45.0, 'length': 70.0}, ...}
    sizes: Dict[str, Dict[str, float]]
    source: str = ''  # 어디서 가져온 치수표인지 (출처 표기용)

    def dimensions(self) -> List[str]:
        keys = set()
        for row in self.sizes.values():
            keys.update(row)
        return sorted(keys)

    @classmethod
    def from_dict(cls, d: dict) -> 'SizeChart':
        return cls(name=d['name'], category=d['category'], sizes=d['sizes'],
                   source=d.get('source', ''))


@dataclass
class DimensionFit:
    dimension: str
    garment_cm: float
    body_cm: float
    ease_cm: float
    label: str


@dataclass
class SizeFit:
    size: str
    dimensions: List[DimensionFit]
    score: float  # 낮을수록 잘 맞음
    wearable: bool

    @property
    def summary(self) -> str:
        parts = [f'{d.dimension} {d.ease_cm:+.1f}cm({d.label})' for d in self.dimensions]
        return f"{self.size}: {' / '.join(parts)}"


@dataclass
class Recommendation:
    best: Optional[SizeFit]
    ranked: List[SizeFit] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        if self.best is None:
            return '맞는 사이즈를 찾지 못했습니다.'
        if not self.best.wearable:
            return (f'{self.best.size}이(가) 그나마 가깝지만 몸 치수보다 작습니다. '
                    f'더 큰 사이즈를 권합니다.')
        labels = {d.dimension: d.label for d in self.best.dimensions}
        main = labels.get('chest') or labels.get('waist') or '레귤러'
        return f'{self.best.size} 추천 — {main} 핏으로 입힙니다.'


def classify_ease(category: str, dimension: str, ease_cm: float) -> str:
    """여유분(cm)을 핏 라벨로 바꾼다."""
    bands = FIT_BANDS.get(category, {}).get(dimension)
    if not bands:
        return '기준없음'
    for low, high, label in bands:
        if low <= ease_cm < high:
            return label
    return '기준없음'


def evaluate_size(chart: SizeChart, size: str, body_cm: Dict[str, float]) -> SizeFit:
    """한 사이즈에 대해 항목별 여유분과 점수를 계산한다."""
    row = chart.sizes[size]
    ideal = IDEAL_EASE.get(chart.category, {})

    dims: List[DimensionFit] = []
    score = 0.0
    wearable = True

    for dimension, garment_cm in sorted(row.items()):
        if dimension not in body_cm:
            continue  # 몸 치수를 모르는 항목은 판정에서 뺀다
        ease = garment_cm - body_cm[dimension]
        dims.append(DimensionFit(
            dimension=dimension,
            garment_cm=garment_cm,
            body_cm=body_cm[dimension],
            ease_cm=round(ease, 1),
            label=classify_ease(chart.category, dimension, ease),
        ))

        weight = DIMENSION_WEIGHT.get(dimension, 0.5)
        target = ideal.get(dimension, 0.0)
        penalty = abs(ease - target)
        if dimension in HARD_LIMIT_DIMENSIONS and ease < 0:
            penalty += abs(ease) * UNDERSIZE_PENALTY
            wearable = False
        score += weight * penalty

    return SizeFit(size=size, dimensions=dims, score=round(score, 2), wearable=wearable)


def recommend(chart: SizeChart, body_cm: Dict[str, float]) -> Recommendation:
    """치수표 전체를 평가해 가장 잘 맞는 사이즈를 고른다."""
    notes: List[str] = []

    usable = [d for d in chart.dimensions() if d in body_cm]
    if not usable:
        return Recommendation(best=None, ranked=[], notes=[
            '치수표와 몸 치수에 공통 항목이 없습니다. '
            f'치수표 항목={chart.dimensions()}, 몸 치수 항목={sorted(body_cm)}'
        ])

    missing = [d for d in chart.dimensions() if d not in body_cm]
    if missing:
        notes.append(f"몸 치수가 없어 판정에서 제외한 항목: {', '.join(missing)}")

    fits = [evaluate_size(chart, size, body_cm) for size in chart.sizes]
    # 입을 수 있는 것을 우선하고, 그 안에서 점수가 낮은 순
    ranked = sorted(fits, key=lambda f: (not f.wearable, f.score))

    if not any(f.wearable for f in ranked):
        notes.append('모든 사이즈가 몸 치수보다 작습니다.')

    return Recommendation(best=ranked[0], ranked=ranked, notes=notes)
