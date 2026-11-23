"""치수표 사진 → 추출 JSON. EasyOCR을 실행 환경 안에서 돌리므로 API 비용이 없고 사진이 외부로 나가지 않는다.

OCR은 글자와 위치만 준다. 이 모듈이 위치로 표를 복원하고, 결과 검사는 `chart_extract.build_chart` 에 맡긴다.
표 복원(`table_from_boxes`)은 OCR 없이 테스트할 수 있게 순수 계산으로 분리했다.

지원하는 표 모양:
  - 사이즈가 행:  헤더 줄에 항목(가슴단면·총장), 왼쪽 열에 사이즈 (무신사 등)
  - 사이즈가 열:  헤더 줄에 사이즈(M·L·XL), 왼쪽 열에 항목
"""
import difflib
import os
import re
import statistics
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from chart_extract import (COUNTRY_LABELS, LABEL_ALIASES, build_chart, is_ignored_row, normalize_label,
                           normalize_size_name, size_sort_key)
from chart_upload import ChartReadError
from size_fit import SizeChart

LOW_CONFIDENCE = 0.5     # 이보다 낮은 숫자는 확인을 요청한다. 실제 캡처로 평가해 조정할 값
UPSCALE_BELOW = 1200     # 긴 변이 이보다 작으면 2배로 키운다. 작은 숫자의 소수점을 놓치지 않기 위해
ROW_TOLERANCE = 0.6      # 글자 높이의 이 비율 안이면 같은 줄
COLUMN_TOLERANCE = 0.6   # 열 간격의 이 비율 안이면 같은 열
LOWER_ONLY = {'waist', 'hip', 'thigh', 'rise'}


HEADER_LABEL_CUTOFF = 0.5
SINGLE_COLUMN_GAP = 200.0  # 열이 하나뿐일 때 쓰는 값
DIGITS = '0123456789.,'
SIZE_LETTERS = 'XSMLFREOxsmlfreo0123456789'
CROP_PADDING = 6
Recover = Callable[[float, float, float, float], Optional[str]]


@dataclass(order=True)
class Cell:
    x: float
    y: float
    height: float
    text: str
    confidence: float


def _number(text: str) -> Optional[float]:
    """'51.5', '51,5', '52cm', '52~53' → 첫 숫자. 숫자 칸이 아니면 None."""
    value = text.strip().replace(',', '.')
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:cm|CM)?(?:\s*[~\-]\s*\d+(?:\.\d+)?\s*(?:cm|CM)?)?', value)
    return float(match.group(1)) if match else None


def _is_size(text: str) -> bool:
    """사이즈 이름처럼 보이는지. 표 아래 안내 문장을 사이즈 행으로 잡지 않기 위해 쓴다."""
    return size_sort_key(text)[0] < 2 or normalize_size_name(text) == 'FREE'


def _label(text: str) -> Optional[str]:
    if _number(text) is not None or _is_size(text) or is_ignored_row(text):
        return None
    return normalize_label(text, cutoff=0.7)


def _cells(boxes: Sequence) -> List[Cell]:
    """OCR 상자를 칸으로 바꾼다. 'M 51.5 43'처럼 붙어 나온 상자는 글자 수 비율로 위치를 나눈다."""
    cells: List[Cell] = []
    for corners, text, confidence in boxes:
        xs, ys = [float(p[0]) for p in corners], [float(p[1]) for p in corners]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        y, height = (top + bottom) / 2, bottom - top
        tokens = str(text).split()
        if len(tokens) < 2 or not any(_number(token) is not None for token in tokens):
            cells.append(Cell((left + right) / 2, y, height, ' '.join(tokens), float(confidence)))
            continue
        total, used = sum(len(token) for token in tokens), 0
        for token in tokens:
            middle = left + (right - left) * (used + len(token) / 2) / total
            cells.append(Cell(middle, y, height, token, float(confidence)))
            used += len(token)
    return cells


def _rows(cells: List[Cell]) -> List[List[Cell]]:
    if not cells:
        return []
    tolerance = statistics.median(cell.height for cell in cells) * ROW_TOLERANCE
    rows: List[List[Cell]] = []
    for cell in sorted(cells, key=lambda c: c.y):
        if rows and abs(cell.y - statistics.mean(c.y for c in rows[-1])) <= tolerance:
            rows[-1].append(cell)
        else:
            rows.append([cell])
    return [sorted(row, key=lambda c: c.x) for row in rows]


def _nearest(x: float, columns: List[Cell], gap: float) -> Optional[Cell]:
    column = min(columns, key=lambda c: abs(c.x - x))
    return column if abs(column.x - x) <= gap * COLUMN_TOLERANCE else None


def _gap(columns: List[Cell]) -> float:
    return _gap_of([column.x for column in columns])


def _gap_of(positions: List[float]) -> float:
    if len(positions) < 2:
        return SINGLE_COLUMN_GAP
    return statistics.median(b - a for a, b in zip(positions, positions[1:]))


def _read_value(cell: Cell, name: str, notes: List[str]) -> Optional[float]:
    value = _number(cell.text)
    if value is None:
        if re.search(r'\d', cell.text):
            notes.append(f"{name}의 '{cell.text}'을 숫자로 읽지 못했어요. 사진을 보고 넣어 주세요.")
        return None
    if cell.confidence < LOW_CONFIDENCE:
        notes.append(f'{name} {value:g}는 흐리게 읽혔어요. 사진과 같은지 확인해 주세요.')
    return value


def _header_label(text: str) -> Optional[str]:
    """표의 머리글 자리에서는 항목일 가능성이 높으므로 더 느슨하게 맞춘다.

    단, 다른 항목과 점수가 같으면 고르지 않는다. '문위'는 밑위와 흉위(가슴)에 똑같이 가까운데,
    가슴으로 잘못 넣으면 경고 없이 추천이 틀어진다. 모르면 경고하고 사람에게 맡긴다.
    """
    if _number(text) is not None or _is_size(text) or is_ignored_row(text):
        return None
    strict = _label(text)
    if strict:
        return strict
    key = re.sub(r'[\s_\-/]+', '', re.sub(r'\([^)]*\)', '', text))
    scores: Dict[str, float] = {}
    for dimension, aliases in LABEL_ALIASES.items():
        scores[dimension] = max(difflib.SequenceMatcher(None, key, re.sub(r'\s+', '', alias)).ratio() for alias in aliases)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if ranked[0][1] >= HEADER_LABEL_CUTOFF and ranked[0][1] > ranked[1][1]:
        return ranked[0][0]
    return None


def _recover_size(recover: Optional[Recover], x0: float, y0: float, x1: float, y1: float) -> Optional[str]:
    """OCR이 놓친 사이즈 칸을 위치로 다시 읽는다. 사이즈로 보이지 않으면 버린다."""
    if recover is None:
        return None
    text = recover(x0, y0, x1, y1)
    return text if text and _is_size(text) else None


def _sizes_in_rows(rows: List[List[Cell]], notes: List[str], recover: Optional[Recover]) -> Dict[str, Dict[str, float]]:
    """헤더에 항목, 왼쪽에 사이즈가 있는 표."""
    for index, row in enumerate(rows):
        if sum(1 for cell in row if _label(cell.text)) >= 2:
            break
    else:
        return {}
    # 느슨하게 맞춘 항목은 대표 이름으로 바꿔 둔다. 그래야 뒤의 build_chart 검사에서 다시 버려지지 않는다
    columns = [Cell(cell.x, cell.y, cell.height, LABEL_ALIASES[_header_label(cell.text)][0], cell.confidence)
               for cell in row if _header_label(cell.text)]
    first = columns[0].x
    for cell in row:
        if cell.x > first and not _header_label(cell.text) and not is_ignored_row(cell.text):
            notes.append(f"표 머리글의 '{cell.text}'을 알아보지 못해 그 열을 뺐어요. JSON에 직접 넣어 주세요.")
    # 못 읽은 머리글이 있어도 간격이 벌어지지 않게 머리글 줄의 모든 칸으로 잰다
    gap = _gap_of([cell.x for cell in row])
    boundary = first - gap / 2
    sizes: Dict[str, Dict[str, float]] = {}
    for body in rows[index + 1:]:
        name = ''.join(cell.text for cell in body if cell.x < boundary)
        numbers = [cell for cell in body if cell.x >= boundary and _number(cell.text) is not None]
        if not name and numbers:
            y = statistics.mean(cell.y for cell in numbers)
            height = max(cell.height for cell in numbers)
            name = _recover_size(recover, min(cell.x for cell in row) - gap / 2, y - height, boundary, y + height) or ''
            if not name:
                notes.append('사이즈 이름을 읽지 못한 행을 뺐어요. JSON에 직접 넣어 주세요.')
                continue
        if not _is_size(name):
            continue
        values: Dict[str, float] = {}
        for cell in body:
            if cell.x < boundary:
                continue
            column = _nearest(cell.x, columns, gap)
            if column is None:
                continue
            value = _read_value(cell, f'{name} {column.text}', notes)
            if value is not None:
                values[column.text] = value
        if values:
            sizes[name] = values
    return sizes


def _number_columns(rows: List[List[Cell]]) -> List[float]:
    """숫자 칸들의 x 위치를 묶어 열 중심을 구한다. 머리글을 놓쳐도 숫자로 열을 알 수 있다."""
    numbers = sorted(cell for row in rows for cell in row[1:] if _number(cell.text) is not None)
    if not numbers:
        return []
    tolerance = statistics.median(cell.height for cell in numbers) * 2
    groups: List[List[float]] = [[numbers[0].x]]
    for cell in numbers[1:]:
        if cell.x - statistics.mean(groups[-1]) <= tolerance:
            groups[-1].append(cell.x)
        else:
            groups.append([cell.x])
    return [statistics.mean(group) for group in groups]


def _sizes_in_columns(rows: List[List[Cell]], notes: List[str], recover: Optional[Recover]) -> Dict[str, Dict[str, float]]:
    """헤더에 사이즈, 왼쪽에 항목이 있는 표."""
    led = [i for i, row in enumerate(rows) if _label(row[0].text)
           and any(_number(cell.text) is not None for cell in row[1:])]
    if not led:
        return {}
    start = led[0]
    body_indexes = [i for i in range(start, len(rows)) if len(rows[i]) > 1
                    and any(_number(cell.text) is not None for cell in rows[i][1:])]
    centers = _number_columns([rows[i] for i in body_indexes])
    if not centers:
        return {}
    gap = _gap_of(centers)
    header = rows[start - 1] if start > 0 else []
    body_y = [statistics.mean(cell.y for cell in rows[i]) for i in body_indexes]
    step = statistics.median(b - a for a, b in zip(body_y, body_y[1:])) if len(body_y) > 1 else rows[start][0].height * 2
    header_y = statistics.mean(cell.y for cell in header) if header else body_y[0] - step
    height = statistics.median(cell.height for i in body_indexes for cell in rows[i])
    columns: List[Cell] = []
    for center in centers:
        found = [cell for cell in header if abs(cell.x - center) <= gap * COLUMN_TOLERANCE and _is_size(cell.text)]
        name = found[0].text if found else _recover_size(
            recover, center - gap / 2, header_y - height, center + gap / 2, header_y + height)
        if name:
            columns.append(Cell(center, header_y, height, name, 1.0))
        else:
            notes.append('사이즈 이름을 읽지 못한 열을 뺐어요. JSON에 직접 넣어 주세요.')
    if not columns:
        return {}
    sizes: Dict[str, Dict[str, float]] = {cell.text: {} for cell in columns}
    for index in body_indexes:
        label = rows[index][0].text
        dimension = _header_label(label)
        if not dimension:
            notes.append(f"표의 항목 '{label}'을 알아보지 못해 그 행을 뺐어요. JSON에 직접 넣어 주세요.")
            continue
        label = LABEL_ALIASES[dimension][0]
        for cell in rows[index][1:]:
            column = _nearest(cell.x, columns, gap)
            if column is None:
                continue
            value = _read_value(cell, f'{column.text} {label}', notes)
            if value is not None:
                sizes[column.text][label] = value
    return {size: row for size, row in sizes.items() if row}


def _is_country_chart(rows: List[List[Cell]]) -> bool:
    return any(sum(cell.text.strip() in COUNTRY_LABELS for cell in row) >= 2 for row in rows)


def table_from_boxes(boxes: Sequence, recover: Optional[Recover] = None) -> Tuple[dict, List[str]]:
    """OCR 상자 [(네 꼭짓점, 글자, 신뢰도)] → (build_chart가 받는 추출 JSON, 확인 요청 문구).

    recover(x0, y0, x1, y1)를 주면 OCR이 놓친 사이즈 칸을 그 위치에서 다시 읽는다.
    """
    rows = _rows(_cells(boxes))
    if _is_country_chart(rows):
        return {'error': '국가별 환산표'}, []
    notes: List[str] = []
    sizes = _sizes_in_rows(rows, notes, recover)
    if not sizes:
        notes = []
        sizes = _sizes_in_columns(rows, notes, recover)
    if not sizes:
        return {'error': '사진에서 표를 찾지 못했어요. 실측 사이즈 표 부분만 잘라 다시 올려 주세요'}, []
    found = {normalize_label(label) for row in sizes.values() for label in row}
    category = 'lower' if found & LOWER_ONLY and 'chest' not in found else 'upper'
    return {
        'name': '사진에서 읽은 치수표',
        'category': category,
        'source': '사용자가 올린 치수표 사진을 OCR로 읽음',
        'sizes': sizes,
    }, notes


_readers: Dict[str, object] = {}
_reader_lock = threading.Lock()


def _load_reader(languages: List[str]):
    """OCR 모델을 처음 쓸 때 한 번만 올린다. 합성 모델의 GPU 메모리를 뺏지 않게 기본은 CPU로 돌린다."""
    key = '+'.join(languages)
    with _reader_lock:
        if key not in _readers:
            try:
                import easyocr
            except ImportError as error:
                raise ChartReadError('easyocr 패키지가 없습니다. pip install easyocr 후 다시 실행해 주세요.') from error
            try:
                _readers[key] = easyocr.Reader(languages, gpu=os.environ.get('CHART_OCR_GPU') == '1', verbose=False)
            except Exception as error:
                raise ChartReadError(f'OCR 모델을 준비하지 못했습니다. 인터넷 연결을 확인해 주세요: {error}') from error
        return _readers[key]


def prepare_readers() -> Optional[str]:
    """앱 시작 때 OCR 모델을 미리 올린다. 첫 사진에서 약 50초 기다리지 않게 한다. 실패해도 앱은 뜬다."""
    try:
        _load_reader(['ko', 'en'])
        _load_reader(['en'])
    except ChartReadError as error:
        return str(error)
    return None


def _recognize(reader, array, x0, y0, x1, y1, allowlist) -> Tuple[Optional[str], float]:
    """이미지의 한 영역을 허용 글자만으로 다시 읽는다."""
    height, width = array.shape[:2]
    crop = array[max(int(y0) - CROP_PADDING, 0):min(int(y1) + CROP_PADDING, height),
                 max(int(x0) - CROP_PADDING, 0):min(int(x1) + CROP_PADDING, width)]
    if crop.size == 0:
        return None, 0.0
    results = reader.recognize(crop, allowlist=allowlist, detail=1, paragraph=False)
    if not results:
        return None, 0.0
    return results[0][1].strip(), float(results[0][2])


def _reread_numbers(array, boxes: Sequence, digit_reader) -> List:
    """숫자 칸을 숫자 전용으로 다시 읽는다. 한글 모델은 1을 7로 자주 읽는다(51.5 → 57.5, 106 → 706)."""
    refined = []
    for corners, text, confidence in boxes:
        tokens = str(text).split()
        if len(tokens) == 1 and re.search(r'\d', text) and not re.search(r'[가-힣]', text):
            xs, ys = [p[0] for p in corners], [p[1] for p in corners]
            again, score = _recognize(digit_reader, array, min(xs), min(ys), max(xs), max(ys), DIGITS)
            if again and _number(again) is not None:
                text, confidence = again, score
        refined.append((corners, text, confidence))
    return refined


def _prepare(image):
    import numpy as np
    image = image.convert('RGB')
    if max(image.size) < UPSCALE_BELOW:
        image = image.resize((image.width * 2, image.height * 2))
    return np.array(image)


def read_chart_image(image, reader=None, digit_reader=None) -> Tuple[Optional[SizeChart], List[str]]:
    """사진 → 검사된 SizeChart와 확인 문구. chart가 None이면 problems에 이유가 있다.

    글자 위치와 한글은 한글 모델로, 숫자와 놓친 사이즈 칸은 영어 모델로 다시 읽는다.
    """
    if reader is None:
        reader = _load_reader(['ko', 'en'])
        digit_reader = digit_reader or _load_reader(['en'])
    array = _prepare(image)
    recover = None
    try:
        boxes = reader.readtext(array, detail=1, paragraph=False)
        if digit_reader is not None:
            boxes = _reread_numbers(array, boxes, digit_reader)

            def recover(x0, y0, x1, y1):
                return _recognize(digit_reader, array, x0, y0, x1, y1, SIZE_LETTERS)[0]
        extracted, notes = table_from_boxes(boxes, recover)
    except ChartReadError:
        raise
    except Exception as error:
        raise ChartReadError(f'사진의 글자를 읽지 못했습니다: {error}') from error
    chart, problems = build_chart(extracted)
    return chart, notes + problems
