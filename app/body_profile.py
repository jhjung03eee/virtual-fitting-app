"""사용자가 직접 입력하는 신체 치수 프로필.

**둘레 vs 단면**

사용자는 자기 몸을 '둘레'로 안다 (가슴둘레 95cm).
쇼핑몰 치수표는 '단면'으로 적혀 있다 (가슴단면 52cm = 옷을 눕혀서 잰 폭).

    가슴단면 = 가슴둘레 / 2

이 모듈은 **입력은 둘레로 받고, 비교용으로는 단면으로 변환**해서 내보낸다.
어깨너비만은 둘레가 아니라 폭이라 변환하지 않는다.
사용자에게 보여줄 때는 둘레로, size_fit에 넘길 때는 단면으로 쓴다.
"""
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

# 입력 항목 정의: (필드명, 화면 표시명, 둘레인가?, 상식적인 범위)
FIELDS = (
    ('height_cm', '키', False, (120.0, 220.0)),
    ('shoulder_cm', '어깨너비', False, (30.0, 60.0)),
    ('chest_circumference_cm', '가슴둘레', True, (60.0, 150.0)),
    ('waist_circumference_cm', '허리둘레', True, (50.0, 150.0)),
    ('hip_circumference_cm', '엉덩이둘레', True, (60.0, 160.0)),
)

# 프로필 필드 -> 치수표 항목 이름
TO_CHART_DIMENSION = {
    'shoulder_cm': 'shoulder',
    'chest_circumference_cm': 'chest',
    'waist_circumference_cm': 'waist',
    'hip_circumference_cm': 'hip',
}


@dataclass
class BodyProfile:
    """키는 필수, 나머지는 아는 것만 채우면 된다.

    모르는 항목은 None으로 두면 사이즈 추천에서 그 항목만 빠진다
    (size_fit.recommend가 공통 항목만 보고 판단한다).
    """
    height_cm: float
    shoulder_cm: Optional[float] = None
    chest_circumference_cm: Optional[float] = None
    waist_circumference_cm: Optional[float] = None
    hip_circumference_cm: Optional[float] = None
    name: str = ''

    def validate(self) -> List[str]:
        """입력값이 상식 범위인지 본다. 오타(95를 9.5로 입력 등)를 잡는 게 목적."""
        problems = []
        for field, label, _is_circ, (low, high) in FIELDS:
            value = getattr(self, field)
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                problems.append(f'{label}: 숫자가 아닙니다 ({value!r})')
            elif not (low <= value <= high):
                problems.append(f'{label} {value}cm 는 입력 범위({low:.0f}~{high:.0f}cm)를 벗어납니다. '
                                '단위가 cm인지 확인해주세요.')
        if self.height_cm is None:
            problems.append('키는 반드시 입력해야 합니다.')
        return problems

    def to_chart_dimensions(self) -> Dict[str, float]:
        """치수표와 비교할 수 있게 단면(cm)으로 변환한다."""
        out = {}
        for field, dimension in TO_CHART_DIMENSION.items():
            value = getattr(self, field)
            if value is None:
                continue
            is_circumference = next(c for f, _l, c, _r in FIELDS if f == field)
            out[dimension] = round(value / 2.0, 1) if is_circumference else round(value, 1)
        return out

    def missing_fields(self) -> List[str]:
        return [label for field, label, _c, _r in FIELDS if getattr(self, field) is None]

    def describe(self) -> str:
        lines = [f'{self.name or "내"} 신체 치수:']
        for field, label, is_circ, _r in FIELDS:
            value = getattr(self, field)
            unit = ' (둘레)' if is_circ else ''
            lines.append(f'  {label}{unit}: ' + (f'{value:.1f} cm' if value is not None else '미입력'))
        return '\n'.join(lines)

    # --- 저장/불러오기 (매번 다시 입력하지 않도록) ---------------------------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> 'BodyProfile':
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_inputs(cls, **kwargs) -> 'BodyProfile':
        """UI에서 온 값(빈 문자열/0 포함)을 정리해서 만든다."""
        clean = {}
        for key, value in kwargs.items():
            if value in (None, '', 0):
                continue
            clean[key] = float(value) if key != 'name' else value
        if 'height_cm' not in clean:
            raise ValueError('키(height_cm)는 필수입니다.')
        return cls(**clean)
