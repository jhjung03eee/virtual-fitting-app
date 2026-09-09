"""치수표 이미지 → SizeChart 변환의 **검증·정규화층**.

사용자가 쇼핑몰(무신사 등)에서 옷 치수표를 스크린샷으로 올리면
Vision 모델이 표를 읽어 JSON으로 돌려주고, 이 모듈이 그 JSON을 검사해
`size_fit.SizeChart`로 바꾼다.

**이 모듈은 네트워크를 쓰지 않는다.** 추출기(VLM/OCR/수기입력)가 무엇이든
결과 검사는 똑같이 필요하므로 순수 계산으로 분리했다. 덕분에 GPU도 API 키도
없이 테스트할 수 있다 (size_fit.py와 같은 이유).

검사하는 것:
  1. 라벨 정규화 — '가슴단면', '가슴 단면', '가슴단먼'(오독) → 'chest'
  2. 사이즈명 정리 — '화이트S' 처럼 색상이 붙은 행을 색상과 사이즈로 분리
  3. 값 범위 — 단면 기준으로 말이 되는 값인지. 치수표가 둘레로 적혀 있으면 감지
  4. 사이즈 순서 — S < M < L 이어야 한다. 어긋나면 그 칸을 잘못 읽은 것이다

4번이 핵심이다. Vision 모델은 숫자를 지어낼 수 있는데, 잘못 읽으면 대개
이 단조성이 깨지므로 **어느 칸이 의심스러운지 특정**할 수 있다.
치수가 틀리면 곧바로 잘못된 사이즈 추천이 되므로 그냥 믿으면 안 된다.

2번이 4번을 지키기 위해 필요하다. 무신사는 색상별로 행을 나눠서
'화이트S / 블랙S' 처럼 적는데, 이걸 그대로 두면 사이즈 순서를 알 수 없어
4번 검사가 통째로 무력해진다 (실제 샘플에서 발견).
"""
import difflib
import re
from typing import Dict, List, Optional, Tuple

from size_fit import SizeChart

# --- 라벨 사전 -------------------------------------------------------------
# 국내 쇼핑몰 치수표에 쓰이는 표현 → 내부 항목명.
# 표기 흔들림(띄어쓰기/조사)은 정규화 후 퍼지 매칭으로 흡수하므로
# 여기에는 대표형만 적는다.
#
# 소매 관련 항목이 셋이라 헷갈리기 쉽다. 서로 다른 치수이므로 반드시 구분한다:
#   소매길이(sleeve) 어깨끝~소매끝 / 소매부리단면(cuff) 소매 끝의 폭 / 암홀(armhole) 겨드랑이 둘레
LABEL_ALIASES = {
    'length': ('총장', '기장', '옷길이', '전체길이', '총기장'),
    'shoulder': ('어깨너비', '어깨', '견폭', '어깨단면', '숄더'),
    'chest': ('가슴단면', '가슴둘레', '가슴', '흉위', '가슴너비'),
    'waist': ('허리단면', '허리둘레', '허리', '웨이스트'),
    'hip': ('엉덩이단면', '엉덩이둘레', '엉덩이', '힙'),
    'sleeve': ('소매길이', '소매', '팔길이'),
    'cuff': ('소매부리단면', '소매부리', '소매통', '소매단면'),
    'armhole': ('암홀', '암홀단면', '진동둘레'),
    'hem': ('밑단단면', '밑단', '밑단너비'),
    'thigh': ('허벅지단면', '허벅지', '허벅지너비'),
    'rise': ('밑위',),
}

# size_fit이 실제로 핏 판정에 쓰는 항목. 나머지는 참고용으로 보관만 한다
# (evaluate_size가 몸 치수에 없는 항목은 건너뛴다).
FIT_DIMENSIONS = ('chest', 'shoulder', 'waist', 'hip', 'length')

# 단면(눕혀서 잰 폭) 기준으로 상식적인 범위 (cm).
# 범위를 벗어나면 오독이거나 치수표가 둘레로 적힌 것이다.
# length는 반바지(약 30)부터 롱코트(약 120)까지 넓게 잡는다.
PLAUSIBLE_RANGES = {
    'length': (25.0, 130.0),
    'shoulder': (28.0, 68.0),
    'chest': (28.0, 78.0),
    'waist': (22.0, 65.0),
    'hip': (28.0, 75.0),
    'sleeve': (10.0, 80.0),
    'cuff': (5.0, 30.0),
    'armhole': (12.0, 35.0),
    'hem': (10.0, 45.0),
    'thigh': (18.0, 48.0),
    'rise': (14.0, 48.0),
}

# 표 안에 섞여 들어오는 안내/헤더 행. 값이 없으므로 조용히 버린다
# (경고를 내면 정상 동작인데도 사용자에게 문제처럼 보인다).
IGNORED_ROW_LABELS = (
    '내사이즈', '사이즈를직접입력해주세요', 'cm', '구분', 'size', '사이즈',
    '단위', '측정부위',
)

# 국가별 사이즈 환산표(한국 90 = 미국 M = 유럽 46)의 열 이름.
# 이 표에는 실측 치수가 없어서 사이즈 추천에 쓸 수 없다.
# 무신사는 '실측 사이즈'와 '기준표 사이즈' 탭이 나란히 있어서 자주 헷갈린다.
COUNTRY_LABELS = ('한국', '미국', '영국', '일본', '프랑스', '유럽', '이탈리아',
                  '독일', '중국', '국제')

# 알파벳 사이즈의 표준 순서. 숫자 사이즈(90·95·100 / 28·30)는 숫자로 정렬한다.
ALPHA_SIZE_ORDER = ('XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL', '4XL')

# 같은 뜻인데 다르게 적히는 사이즈 표기
SIZE_ALIASES = {
    '2XS': 'XXS', '2XL': 'XXL', '3XL': 'XXXL',
    'SMALL': 'S', 'MEDIUM': 'M', 'LARGE': 'L',
    'FREE': 'FREE', 'F': 'FREE', 'ONESIZE': 'FREE', 'ONE SIZE': 'FREE',
}

# 사이즈 토큰. 숫자가 붙은 표기(2XL)를 먼저 봐야 한다 —
# 'XL'을 먼저 매칭하면 '2XL'이 변형 '2' + 사이즈 'XL'로 잘못 쪼개진다.
_SIZE_TOKEN = re.compile(
    r'(4XL|3XL|2XL|2XS|XXXL|XXL|XL|XXS|XS|FREE|S|M|L|F|\d{2,3})$', re.IGNORECASE
)

# Vision 모델에 보낼 지시문. 표 구조 파악은 모델에 맡기고,
# 라벨은 원문 그대로 받아 이쪽에서 정규화한다 (모델이 항목명을 임의로
# 바꾸면 원본 대조가 불가능해지기 때문).
EXTRACTION_PROMPT = """이 이미지는 온라인 쇼핑몰의 옷 실측 치수표입니다.
표를 읽어 아래 JSON 형식 그대로 출력하세요. 설명 문장 없이 JSON만 출력합니다.

{
  "name": "상품명 (보이지 않으면 빈 문자열)",
  "category": "upper 또는 lower 중 하나",
  "unit": "표에 적힌 단위 (보통 cm)",
  "sizes": {
    "S": {"가슴단면": 49, "어깨너비": 43, "총장": 66},
    "M": {"가슴단면": 52, "어깨너비": 45, "총장": 68}
  }
}

규칙:
- 항목명(가슴단면, 총장 등)은 표에 적힌 한글 표현을 **그대로** 쓰세요. 번역하거나 바꾸지 마세요.
- 표에 있는 사이즈와 항목을 빠짐없이 넣으세요.
- 행 이름에 색상이 붙어 있으면('화이트S', '블랙 M') 그대로 두세요. 임의로 합치지 마세요.
- 숫자는 표에 적힌 값 그대로 넣으세요. 계산하거나 반올림하지 마세요.
- 값이 '-'이거나 비어 있는 칸은 그 항목을 아예 넣지 마세요. 0으로 채우지 마세요.
- '내 사이즈'처럼 입력을 받는 안내 행은 넣지 마세요.
- 상의(티셔츠/셔츠/아우터)면 category는 "upper", 하의(바지/스커트)면 "lower"입니다.
- 이 표가 실측 치수가 아니라 국가별 사이즈 환산표(한국/미국/유럽 등을 비교하는 표)라면
  {"error": "국가별 환산표"} 를 출력하세요.
- 표를 읽을 수 없으면 {"error": "이유"} 를 출력하세요."""


def _clean(text: str) -> str:
    """비교용으로 라벨을 납작하게 만든다: 공백·괄호·단위 제거."""
    text = re.sub(r'\([^)]*\)', '', str(text))
    text = re.sub(r'[\s_\-/]+', '', text)
    return re.sub(r'(cm|CM|밀리|미리)$', '', text).strip()


def normalize_label(raw: str, cutoff: float = 0.6) -> Optional[str]:
    """치수표 항목명을 내부 항목명으로 바꾼다. 못 알아보면 None.

    OCR/VLM이 '가슴단먼'처럼 한 글자를 틀려도 복구하려고 퍼지 매칭을 쓴다.
    치수표 항목은 어휘가 고정돼 있어서 이 방식이 잘 통한다.
    """
    key = _clean(raw)
    if not key:
        return None

    # 1순위: 정확히 일치
    for dimension, aliases in LABEL_ALIASES.items():
        if key in (_clean(a) for a in aliases):
            return dimension

    # 2순위: 별칭을 포함. 긴 별칭부터 봐야 '소매부리단면'이 '소매'(sleeve)로
    # 새지 않고 cuff로 간다.
    candidates = sorted(
        ((dimension, _clean(a)) for dimension, aliases in LABEL_ALIASES.items()
         for a in aliases),
        key=lambda pair: len(pair[1]), reverse=True,
    )
    for dimension, alias in candidates:
        if len(alias) >= 2 and alias in key:
            return dimension

    # 3순위: 오독 복구
    best, best_score = None, 0.0
    for dimension, alias in candidates:
        score = difflib.SequenceMatcher(None, key, alias).ratio()
        if score > best_score:
            best, best_score = dimension, score
    return best if best_score >= cutoff else None


def is_ignored_row(raw: str) -> bool:
    """치수 데이터가 아닌 안내/헤더 행인지."""
    return _clean(raw).lower() in {_clean(x).lower() for x in IGNORED_ROW_LABELS}


def split_variant_size(raw: str) -> Tuple[str, str]:
    """'화이트S' → ('화이트', 'S'). 색상이 없으면 ('', 'S').

    무신사는 색상별로 행을 나눠 적는다. 이걸 분리하지 않으면 사이즈 순서를
    알 수 없어 단조성 검사가 통째로 무력해진다.
    """
    key = re.sub(r'[\s_\-]+', '', str(raw)).upper()
    if key in SIZE_ALIASES:
        return '', SIZE_ALIASES[key]

    match = _SIZE_TOKEN.search(key)
    if not match:
        return '', key or str(raw)

    variant = key[:match.start()]
    size = SIZE_ALIASES.get(match.group().upper(), match.group().upper())
    return variant, size


def normalize_size_name(raw: str) -> str:
    """사이즈 표기를 통일한다. 'small' → 'S', '화이트S' → 'S'."""
    return split_variant_size(raw)[1]


def size_sort_key(size: str) -> Tuple[int, float, str]:
    """사이즈를 작은 것부터 정렬한다. 알파벳/숫자 혼용을 처리한다."""
    name = normalize_size_name(size)
    if name in ALPHA_SIZE_ORDER:
        return (0, float(ALPHA_SIZE_ORDER.index(name)), name)
    number = re.fullmatch(r'(\d+(?:\.\d+)?)', name)
    if number:
        return (1, float(number.group(1)), name)
    return (2, 0.0, name)  # FREE 등 순서를 알 수 없는 것은 맨 뒤


def _to_float(value) -> Optional[float]:
    """'52', '52cm', '52.5', '52~53' → float. 못 읽으면 None."""
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r'\d+(?:\.\d+)?', str(value))
    return float(match.group()) if match else None


def check_ranges(sizes: Dict[str, Dict[str, float]]) -> List[str]:
    """값이 단면 기준으로 말이 되는지 본다.

    치수표가 '둘레'로 적혀 있으면 값이 대략 2배가 되는데, 이 경우
    반으로 나누면 정상 범위에 들어오므로 그렇게 안내한다.
    """
    problems = []
    for size in sorted(sizes, key=size_sort_key):
        for dimension, value in sorted(sizes[size].items()):
            bounds = PLAUSIBLE_RANGES.get(dimension)
            if bounds is None:
                continue
            low, high = bounds
            if low <= value <= high:
                continue
            if low <= value / 2 <= high:
                problems.append(
                    f'{size} {dimension} {value:g}cm — 단면 치고 큽니다. '
                    f'치수표가 둘레로 적혀 있다면 {value / 2:g}cm가 맞습니다'
                )
            else:
                problems.append(
                    f'{size} {dimension} {value:g}cm — 예상 범위'
                    f'({low:g}~{high:g}cm)를 벗어납니다. 잘못 읽었을 수 있습니다'
                )
    return problems


def check_monotonic(sizes: Dict[str, Dict[str, float]]) -> List[str]:
    """사이즈가 커지면 치수도 커져야 한다. 뒤집힌 칸을 찾아낸다.

    Vision 모델의 오독을 잡는 가장 확실한 신호다. 사람이 만든 치수표에서
    L이 M보다 작은 경우는 사실상 없으므로, 어긋나면 읽기 오류로 본다.

    색상별 행이 남아 있으면(화이트S/블랙S) 같은 색끼리만 비교한다.
    """
    groups: Dict[str, List[str]] = {}
    for size in sizes:
        if size_sort_key(size)[0] == 2:
            continue  # FREE 등 순서를 알 수 없는 사이즈는 비교 불가
        groups.setdefault(split_variant_size(size)[0], []).append(size)

    problems = []
    for variant in sorted(groups):
        ordered = sorted(groups[variant], key=size_sort_key)
        for prev, cur in zip(ordered, ordered[1:]):
            for dimension in sorted(set(sizes[prev]) & set(sizes[cur])):
                before, after = sizes[prev][dimension], sizes[cur][dimension]
                if after < before:
                    problems.append(
                        f'{cur} {dimension} {after:g}cm < {prev} {dimension} {before:g}cm — '
                        f'큰 사이즈가 더 작습니다. 이 두 칸을 확인해주세요'
                    )
    return problems


def normalize_sizes(raw_sizes: dict) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """추출된 sizes 딕셔너리의 라벨·값·사이즈명을 정규화한다.

    색상별로 나뉜 행은 치수가 같으면 하나로 합치고, 다르면 색상을 붙여 남긴다.
    """
    parsed: List[Tuple[str, str, Dict[str, float]]] = []  # (variant, size, row)
    problems: List[str] = []

    for raw_size, raw_row in raw_sizes.items():
        if is_ignored_row(raw_size):
            continue
        if not isinstance(raw_row, dict):
            problems.append(f'{raw_size} 행을 읽을 수 없습니다')
            continue

        variant, size = split_variant_size(raw_size)
        row: Dict[str, float] = {}
        for raw_label, raw_value in raw_row.items():
            dimension = normalize_label(raw_label)
            if dimension is None:
                problems.append(f"'{raw_label}' 항목이 무엇인지 알 수 없어 제외했습니다")
                continue
            value = _to_float(raw_value)
            if value is None or value <= 0:
                continue  # '-'로 비워둔 칸. 정상이므로 경고하지 않는다
            if dimension in row and row[dimension] != value:
                problems.append(
                    f'{raw_size}에 {dimension} 값이 두 개({row[dimension]:g}, {value:g}) 있습니다'
                )
            row[dimension] = value

        if row:
            parsed.append((variant, size, row))
        else:
            problems.append(f'{raw_size}에서 읽어낸 항목이 없습니다')

    # 색상별 행 합치기
    grouped: Dict[str, List[Tuple[str, Dict[str, float]]]] = {}
    for variant, size, row in parsed:
        grouped.setdefault(size, []).append((variant, row))

    sizes: Dict[str, Dict[str, float]] = {}
    merged_variants, split_sizes = set(), set()
    for size, entries in grouped.items():
        if len(entries) == 1:
            sizes[size] = entries[0][1]
            continue
        first = entries[0][1]
        if all(row == first for _, row in entries[1:]):
            sizes[size] = first
            merged_variants.update(v for v, _ in entries if v)
        else:
            split_sizes.add(size)
            for variant, row in entries:
                sizes[f'{variant}{size}' if variant else size] = row

    if merged_variants:
        problems.append(
            f"{'/'.join(sorted(merged_variants))} 색상별 행의 치수가 같아 하나로 합쳤습니다"
        )
    if split_sizes:
        problems.append(
            f"{'/'.join(sorted(split_sizes))}는 색상마다 치수가 달라 따로 두었습니다"
        )

    return sizes, problems


def looks_like_country_chart(raw_sizes: dict) -> bool:
    """국가별 사이즈 환산표인지. 실측 치수가 없어 추천에 쓸 수 없다."""
    labels = {_clean(label) for row in raw_sizes.values()
              if isinstance(row, dict) for label in row}
    if not labels:
        return False
    hits = labels & {_clean(c) for c in COUNTRY_LABELS}
    return len(hits) >= 2


def build_chart(extracted: dict) -> Tuple[Optional[SizeChart], List[str]]:
    """추출 결과 JSON을 검사해 SizeChart로 만든다.

    반환: (chart, problems). chart가 None이면 쓸 수 없는 결과다.
    problems가 비어 있지 않아도 chart는 나올 수 있다 — 사용자에게 보여주고
    확인받으라는 뜻이지 실패가 아니다.
    """
    if not isinstance(extracted, dict):
        return None, ['추출 결과가 JSON 객체가 아닙니다']

    country_note = (
        '국가별 사이즈 환산표로 보입니다. 이 표에는 실측 치수(가슴단면·총장 등)가 없어 '
        '사이즈를 추천할 수 없습니다. 상품 페이지에서 **실측 사이즈** 탭을 캡처해주세요'
    )
    if extracted.get('error'):
        message = str(extracted['error'])
        if '환산' in message or '기준표' in message:
            return None, [country_note]
        return None, [f'치수표를 읽지 못했습니다: {message}']

    raw_sizes = extracted.get('sizes')
    if not isinstance(raw_sizes, dict) or not raw_sizes:
        return None, ['치수표에서 사이즈를 찾지 못했습니다']
    if looks_like_country_chart(raw_sizes):
        return None, [country_note]

    sizes, problems = normalize_sizes(raw_sizes)
    if not sizes:
        return None, problems or ['치수표에서 읽어낸 값이 없습니다']

    category = str(extracted.get('category', '')).strip().lower()
    if category not in ('upper', 'lower'):
        # 하의에만 있는 항목이 보이면 하의로 본다
        lower_only = {'rise', 'thigh', 'hip'}
        found = {d for row in sizes.values() for d in row}
        category = 'lower' if found & lower_only and 'chest' not in found else 'upper'
        problems.append(f'옷 종류를 표에서 알 수 없어 {category}로 추정했습니다. 확인해주세요')

    problems += check_ranges(sizes)
    problems += check_monotonic(sizes)

    usable = {d for row in sizes.values() for d in row} & set(FIT_DIMENSIONS)
    if not usable:
        problems.append(
            '핏 판정에 쓸 수 있는 항목(가슴/어깨/허리/엉덩이/총장)이 없어 '
            '사이즈 추천이 안 됩니다'
        )

    chart = SizeChart(
        name=str(extracted.get('name') or '이름 없는 치수표').strip(),
        category=category,
        sizes=sizes,
        source=str(extracted.get('source') or '사용자가 올린 치수표 이미지에서 추출'),
    )
    return chart, problems
