"""치수를 모를 때의 사이즈 추천 — 키·몸무게 → 몸 치수 예측 → 사이즈별 '맞을 확률' → 가장 큰 확률의 사이즈.

기존 추천(`size_fit.recommend`)은 줄자로 잰 치수가 있어야 한다. 대부분의 사람은 가슴둘레를 모른다.
이 모듈은 그 앞단만 채운다. **사이즈를 고르는 규칙은 `size_fit.recommend` 그대로 쓴다** (여기서 바꾸지 않음).

1) 예측 (Bartol 외, "Linear regression vs. deep learning: a simple yet effective baseline for human body
   measurement", Sensors 2022 의 방식): 성별마다, 부위마다 최소제곱 선형 회귀
       치수 = a + b·키(cm) + c·몸무게(kg)
   계수는 사이즈코리아 8차(2020) 직접측정 원자료로 학습했다 (`scripts/fit_body_model.py`,
   결과 `data/body_models/sizekorea8_2020.json`). 부위 정의는 앱 안내와 같다:
   어깨 = No.088 어깨사이너비(양쪽 어깨점 사이 수평거리), 가슴 = No.042 젖가슴둘레, 허리 = No.045 허리둘레.

2) 확률: 예측값은 '평균'일 뿐이다. 같은 키·몸무게라도 사람마다 회귀 잔차(표준편차 resid_sd_cm,
   부위끼리 상관 resid_corr)만큼 다르다. 그 분포에서 몸을 n개 뽑아 각각 `size_fit.recommend` 를 돌리고,
   1등으로 뽑힌 비율 = 그 사이즈가 맞을 확률로 본다 (몬테카를로). seed 는 입력에서 만들어
   같은 사람에겐 늘 같은 답을 준다.

3) 추천: 확률이 가장 큰 사이즈. 확률이 60% 미만이면 경계에 있다는 뜻 — 줄자로 주 부위(상의 가슴,
   하의 허리) 하나만 재 넣으라고 안내한다. 줄자로 잰 항목이 있으면 그 항목은 예측하지 않고 그대로 쓴다.

평가(사이즈코리아 평가용 20%, 1019명, 치수표 9개): '치수를 알 때'의 추천과 같은 사이즈를 고른 비율은
성별 평균값만 쓸 때보다 높고, 확률 60% 이상으로 답한 경우 대부분 일치했다 — 수치는 docs/PLAN.md F-37.
"""
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from size_fit import Recommendation, SizeChart, recommend

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'body_models', 'sizekorea8_2020.json')

# BodyProfile 필드 → 회귀식 부위. 상의는 어깨·가슴, 하의는 허리 (size_fit.SIZE_DIMENSIONS 와 같은 범위)
PREDICT = {
    'upper': {'shoulder_cm': 'shoulder', 'chest_circumference_cm': 'chest'},
    'lower': {'waist_circumference_cm': 'waist'},
}
# 둘레는 치수표 단면과 비교할 때 ÷2 (body_profile.BodyProfile.to_chart_dimensions 와 같은 규칙)
TO_CHART = {'shoulder_cm': ('shoulder', False), 'chest_circumference_cm': ('chest', True),
            'waist_circumference_cm': ('waist', True)}
SEX = {'남': 'm', '남자': 'm', 'm': 'm', 'male': 'm', '여': 'f', '여자': 'f', 'f': 'f', 'female': 'f'}
CONFIDENT = 0.6
DEFAULT_SAMPLES = 2000

_MODEL = None


def load_model(path: str = MODEL_PATH) -> dict:
    global _MODEL
    if path != MODEL_PATH:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    if _MODEL is None:
        with open(path, encoding='utf-8') as f:
            _MODEL = json.load(f)
    return _MODEL


@dataclass
class Prediction:
    profile: Dict[str, float]            # BodyProfile 필드 이름 → cm (키 포함)
    sd: Dict[str, float]                 # 예측한 필드만 → 잔차 표준편차 cm (줄자로 잰 필드는 없음)
    corr: Dict[str, float]               # '필드a|필드b' → 상관
    sources: List[str] = field(default_factory=list)   # 화면 안내용 ('가슴둘레 96.1cm 예측(평균 오차 2.4cm)')
    seed: int = 1


@dataclass
class UnknownRecommendation:
    recommendation: Recommendation       # best 를 확률 최대 사이즈로 바꾼 size_fit.Recommendation
    probabilities: Dict[str, float]      # 사이즈 → 확률 (치수표 순서)
    point_best: Optional[str]            # 예측값 한 점으로만 고른 사이즈 (비교용)
    prediction: Prediction
    samples: int

    @property
    def confident(self) -> bool:
        best = self.recommendation.best
        return best is not None and self.probabilities.get(best.size, 0.0) >= CONFIDENT


LABEL = {'shoulder_cm': '어깨너비', 'chest_circumference_cm': '가슴둘레', 'waist_circumference_cm': '허리둘레'}


def predict(category: str, sex: str, height_cm: float, weight_kg: float,
            measured: Optional[Dict[str, float]] = None, model: Optional[dict] = None) -> Prediction:
    """키·몸무게로 몸 치수를 예측한다. measured 에 줄자로 잰 값이 있으면 그 항목은 예측하지 않는다."""
    model = model or load_model()
    sx = SEX.get(str(sex).strip().lower())
    if sx is None:
        raise ValueError('성별을 골라 주세요 (회귀식이 남녀 따로입니다).')
    if not (120 <= float(height_cm) <= 220):
        raise ValueError('키를 cm로 입력해 주세요 (120~220).')
    if not (30 <= float(weight_kg) <= 200):
        raise ValueError('몸무게를 kg로 입력해 주세요 (30~200).')
    if category not in PREDICT:
        raise ValueError(f'키·몸무게 추천은 상의·하의만 됩니다 ({category}).')
    m = model['sex'][sx]
    h, w = float(height_cm), float(weight_kg)
    measured = {k: v for k, v in (measured or {}).items() if v not in (None, '', 0)}
    profile, sd, part_of, sources = {'height_cm': h}, {}, {}, []
    for fld, part in PREDICT[category].items():
        if fld in measured:
            profile[fld] = float(measured[fld])
            sources.append(f'{LABEL[fld]} 줄자 {profile[fld]:g}cm')
            continue
        c = m['parts'][part]
        profile[fld] = round(c['a'] + c['b_height'] * h + c['c_weight'] * w, 1)
        sd[fld] = c['resid_sd_cm']
        part_of[part] = fld
        sources.append(f'{LABEL[fld]} {profile[fld]:g}cm 예측(평균 오차 {c["test_mae_cm"]:g}cm)')
    corr = {}
    for key, r in m.get('resid_corr', {}).items():
        a, b = key.split('|')
        if a in part_of and b in part_of:
            corr[f'{part_of[a]}|{part_of[b]}'] = r
    seed = round(h * 10) * 100003 + round(w * 10) * 31 + (7 if sx == 'm' else 13)
    return Prediction(profile, sd, corr, sources, seed)


def to_chart_body(profile: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for fld, (dim, is_circ) in TO_CHART.items():
        if fld in profile:
            out[dim] = round(profile[fld] / 2.0, 1) if is_circ else round(profile[fld], 1)
    return out


def _cholesky(cov: List[List[float]]) -> List[List[float]]:
    n = len(cov)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = cov[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:   # 양의 정부호가 아니면 상관을 버리고 각자 흔든다
                    return [[math.sqrt(cov[a][a]) if a == b else 0.0 for b in range(n)] for a in range(n)]
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def size_probabilities(chart: SizeChart, pred: Prediction, samples: int = DEFAULT_SAMPLES) -> Dict[str, float]:
    """예측 분포에서 몸을 samples 개 뽑아 size_fit.recommend 1등의 비율을 센다."""
    fields = [f for f in pred.sd if pred.sd[f] > 0]
    cov = [[pred.sd[a] * pred.sd[b] * (1.0 if i == j else pred.corr.get(f'{a}|{b}', pred.corr.get(f'{b}|{a}', 0.0)))
            for j, b in enumerate(fields)] for i, a in enumerate(fields)]
    L = _cholesky(cov) if fields else []
    rng = random.Random(pred.seed)
    counts = {s: 0 for s in chart.sizes}
    used = 0
    for _ in range(samples if fields else 1):
        z = [rng.gauss(0.0, 1.0) for _ in fields]
        p = dict(pred.profile)
        for i, f in enumerate(fields):
            p[f] = pred.profile[f] + sum(L[i][j] * z[j] for j in range(i + 1))
        rec = recommend(chart, to_chart_body(p))
        if rec.best is not None:
            counts[rec.best.size] += 1
            used += 1
    return {s: (c / used if used else 0.0) for s, c in counts.items()}


def size_input_mode(category: str, sex, height_cm, weight_kg, measured: Dict[str, float]) -> str:
    """입력으로 어떤 추천을 할지 고른다: 'measured'(입력 치수) · 'estimate'(키·몸무게 추정) · 'missing'(입력 부족).

    옷에 필요한 치수를 모두 줄자로 쟀으면 추정하지 않는다. 일부만 있으면 빈 칸을 추정으로 채운다.
    추정에 필요한 값이 없으면 지금까지처럼 입력한 치수만으로 추천한다.
    """
    needed = PREDICT.get(category)
    filled = {k for k, v in measured.items() if v not in (None, '', 0)}
    if needed is None or set(needed) <= filled:
        return 'measured'
    if sex and height_cm not in (None, '', 0) and weight_kg not in (None, '', 0):
        return 'estimate'
    return 'measured' if filled else 'missing'


def recommend_unknown(chart: SizeChart, sex: str, height_cm: float, weight_kg: float,
                      measured: Optional[Dict[str, float]] = None, samples: int = DEFAULT_SAMPLES,
                      model: Optional[dict] = None) -> UnknownRecommendation:
    """치수를 모를 때의 추천. 결과의 recommendation 은 size_fit.Recommendation 이라 기존 화면 코드로 그릴 수 있다."""
    pred = predict(chart.category, sex, height_cm, weight_kg, measured, model)
    point = recommend(chart, to_chart_body(pred.profile))
    probs = size_probabilities(chart, pred, samples)
    point_best = point.best.size if point.best else None
    if point.best is not None and any(probs.values()):
        order = list(chart.sizes)                    # 동률이면 치수표 순서상 앞(작은) 사이즈
        top = max(order, key=lambda s: (probs[s], -order.index(s)))
        point.best = next((f for f in point.ranked if f.size == top), point.best)
    return UnknownRecommendation(point, probs, point_best, pred, samples)
