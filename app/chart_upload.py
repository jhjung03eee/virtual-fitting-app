"""치수표 JSON 저장·수정과 선택 기능인 Claude API 읽기.

앱의 기본 읽기는 비용이 없는 `chart_ocr.py` 다. 이 파일의 API 읽기는 CHART_READER=claude 일 때만 쓰며 호출마다 요금이 든다.
흐름: 사진 → Claude API(Vision) → JSON 문자열 → `chart_extract.build_chart` 로 검사 → `data/size_charts/uploaded_*.json`.

어떤 방식이든 숫자를 잘못 읽을 수 있으므로 저장 결과와 경고를 화면에 보여 주고 사용자가 고칠 수 있게 한다.
API 키는 코드에 넣지 않는다. 실행 환경의 ANTHROPIC_API_KEY 를 SDK가 읽는다.
"""
import base64
import io
import json
import os
import re
import time
from typing import List, Optional, Tuple

from chart_extract import EXTRACTION_PROMPT, LABEL_ALIASES, build_chart
from size_fit import SizeChart

MODEL = os.environ.get('CHART_VISION_MODEL', 'claude-opus-5-5')
MAX_TOKENS = 16000
FALLBACK_BETA = 'server-side-fallback-2026-07-01'
MAX_IMAGE_BYTES = 4_500_000
MAX_IMAGE_SIDE = 2576
UPLOADED_PREFIX = 'uploaded_'


class ChartReadError(Exception):
    """화면에 그대로 보여 줄 수 있는 사진 읽기 실패."""


def encode_image(image) -> Tuple[str, str]:
    """PIL 이미지를 API 형식(미디어 타입, base64)으로 바꾼다. 글자가 뭉개지지 않게 PNG를 먼저 쓴다."""
    image = image.convert('RGB')
    if max(image.size) > MAX_IMAGE_SIDE:
        image = image.copy()
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    media_type = 'image/png'
    if buffer.tell() > MAX_IMAGE_BYTES:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        media_type = 'image/jpeg'
    return media_type, base64.standard_b64encode(buffer.getvalue()).decode('ascii')


def parse_model_json(text: str) -> dict:
    """모델 답변에서 JSON 객체를 꺼낸다. 코드 블록이나 앞뒤 문장이 붙어도 읽는다."""
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ChartReadError('모델 답변에서 치수표 JSON을 찾지 못했습니다. 사진을 다시 올려 주세요.')
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        raise ChartReadError(f'모델 답변의 JSON 형식이 잘못됐습니다: {error}') from error
    if not isinstance(parsed, dict):
        raise ChartReadError('모델 답변이 치수표 형식이 아닙니다.')
    return parsed


def _default_client():
    try:
        import anthropic
    except ImportError as error:
        raise ChartReadError('anthropic 패키지가 없습니다. pip install anthropic 후 다시 실행해 주세요.') from error
    return anthropic.Anthropic()


def _request(client, media_type: str, data: str):
    return client.beta.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={'effort': 'high'},  # 숫자 한 칸의 오독이 곧 잘못된 추천이다
        betas=[FALLBACK_BETA],
        fallbacks='default',
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': data}},
                {'type': 'text', 'text': EXTRACTION_PROMPT},
            ],
        }],
    )


def extract_from_image(image, client=None) -> dict:
    """치수표 사진을 모델에 보내 추출 JSON을 돌려준다. 실패하면 ChartReadError."""
    media_type, data = encode_image(image)
    try:
        import anthropic
    except ImportError:
        anthropic = None
    client = client or _default_client()
    try:
        response = _request(client, media_type, data)
    except TypeError as error:
        # 인증 수단이 없으면 SDK가 요청을 만들 때 TypeError를 낸다
        if 'authentication' in str(error).lower() or 'api_key' in str(error).lower():
            raise ChartReadError('ANTHROPIC_API_KEY가 설정되지 않아 사진을 읽을 수 없습니다. '
                                 '키를 설정하거나 아래 칸에 치수표 JSON을 직접 넣어 주세요.') from error
        raise
    except Exception as error:
        if anthropic is None or not isinstance(error, anthropic.AnthropicError):
            raise
        if isinstance(error, anthropic.AuthenticationError):
            raise ChartReadError('API 키가 올바르지 않습니다. ANTHROPIC_API_KEY를 확인해 주세요.') from error
        if isinstance(error, anthropic.RateLimitError):
            raise ChartReadError('요청이 많아 잠시 거절됐습니다. 조금 뒤 다시 올려 주세요.') from error
        if isinstance(error, anthropic.APIStatusError):
            raise ChartReadError(f'치수표 읽기 서버 오류입니다 (HTTP {error.status_code}). 다시 시도해 주세요.') from error
        if isinstance(error, anthropic.APIConnectionError):
            raise ChartReadError('치수표 읽기 서버에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요.') from error
        raise ChartReadError(f'치수표를 읽지 못했습니다: {error}') from error

    if response.stop_reason == 'refusal':
        raise ChartReadError('모델이 이 사진을 처리하지 않았습니다. 치수표 부분만 잘라 다시 올려 주세요.')
    if response.stop_reason == 'max_tokens':
        raise ChartReadError('치수표가 너무 길어 끝까지 읽지 못했습니다. 표 부분만 잘라 다시 올려 주세요.')
    text = ''.join(block.text for block in response.content if block.type == 'text')
    return parse_model_json(text)


def read_chart_image(image, client=None) -> Tuple[Optional[SizeChart], List[str]]:
    """사진 → 검사된 SizeChart. chart가 None이면 쓸 수 없는 결과이고 problems에 이유가 있다."""
    return build_chart(extract_from_image(image, client))


def chart_to_dict(chart: SizeChart) -> dict:
    return {
        'name': chart.name,
        'category': chart.category,
        'source': chart.source,
        'unit': 'cm (단면)',
        'sizes': chart.sizes,
    }


def chart_to_json(chart: SizeChart) -> str:
    return json.dumps(chart_to_dict(chart), ensure_ascii=False, indent=2)


def save_chart(chart: SizeChart, directory: str) -> str:
    """치수표를 JSON으로 저장하고 파일 이름을 돌려준다. 같은 이름의 기존 파일을 덮어쓰지 않는다."""
    os.makedirs(directory, exist_ok=True)
    slug = re.sub(r'[^0-9A-Za-z가-힣]+', '_', chart.name).strip('_')[:40] or 'chart'
    stamp = time.strftime('%Y%m%d_%H%M%S')
    for attempt in range(1000):
        suffix = f'_{attempt}' if attempt else ''
        filename = f'{UPLOADED_PREFIX}{stamp}_{slug}{suffix}.json'
        try:
            with open(os.path.join(directory, filename), 'x', encoding='utf-8') as file:
                file.write(chart_to_json(chart))
            return filename
        except FileExistsError:
            continue
    raise OSError('치수표 파일 이름을 만들지 못했습니다.')


def chart_from_text(text: str) -> Tuple[Optional[SizeChart], List[str]]:
    """사용자가 고치거나 직접 넣은 JSON을 같은 검사에 통과시킨다.

    화면의 JSON은 저장 형식(chest 등)이고 검사층은 표의 표현(가슴단면 등)을 받으므로 먼저 바꾼다.
    """
    try:
        data = json.loads(text or '')
    except json.JSONDecodeError as error:
        return None, [f'JSON 형식이 잘못됐습니다: {error}']
    if not isinstance(data, dict):
        return None, ['JSON 최상위는 치수표 객체여야 합니다.']
    sizes = data.get('sizes')
    if isinstance(sizes, dict):
        data = dict(data, sizes={
            size: ({LABEL_ALIASES[label][0] if label in LABEL_ALIASES else label: value
                    for label, value in row.items()} if isinstance(row, dict) else row)
            for size, row in sizes.items()
        })
    data.setdefault('source', '사용자가 직접 입력하거나 고친 치수표')
    return build_chart(data)
