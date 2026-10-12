"""FASHN 결과에서 옷 부분만 남기고 나머지(얼굴·머리·손·배경)는 원본 사진으로 되돌린다.

**왜 필요한가.** FASHN은 마스크 없이 사진 전체를 576x768로 다시 그린다. 옷은 잘 입히지만
얼굴·배경까지 살짝 매끈해지고 해상도도 낮아진다. FitDiT가 "더 자연스럽다"고 느낀 이유가
마스크 밖을 원본 그대로 두기 때문이었다(docs/PLAN.md F-15). 생성이 끝난 뒤 옷 자리만
오려 붙이면 FASHN의 핏·글자와 원본의 얼굴·배경을 같이 가져갈 수 있다.

**옷 자리 마스크 = (옷 종류에 맞는 세로 범위의 옷) ∪ (그 범위에서 달라진 피부).**
  - 옷/피부/얼굴/머리 구분은 MediaPipe selfie_multiclass(CPU) — 원본과 결과 양쪽에서 구해 합친다.
    양쪽을 합치는 이유: 긴팔 → 반팔이면 원본에서 옷이던 팔이 결과에서 피부가 되고,
    반팔 → 긴팔이면 그 반대다. 어느 쪽이든 바뀐 자리는 결과를 써야 한다.
  - 이 모델은 상의와 바지를 구분하지 못한다. 상의만 바꿨는데 '옷 전체'를 결과로 덮으면 바지까지
    FASHN이 다시 그린 것이 된다. 그래서 **실제로 크게 달라진 곳**과 교집합을 낸다.
    FASHN은 바뀌지 않은 부분도 미세하게 다시 그리므로(차이 평균 몇/255) 문턱값으로 거른다.
  - 얼굴·머리·배경은 처음부터 후보에서 뺀다 → 항상 원본.

원본과 결과는 같은 비율이어야 한다(FASHN은 3:4로 줄이기만 하므로 맞는다. 배경 차이 평균 0.5/255로 확인).
numpy·PIL만 쓰고, 구분 모델을 못 쓰면 ImportError 대신 None 을 돌려 호출부가 원래 결과를 쓰게 한다.
"""
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SEGMENTER_URL = ('https://storage.googleapis.com/mediapipe-models/image_segmenter/'
                 'selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite')
# selfie_multiclass 클래스: 0 배경, 1 머리카락, 2 몸 피부, 3 얼굴 피부, 4 옷, 5 기타(소품)
CLOTHES_CLASSES = (4,)       # 5(소품)는 신발·가방이라 원본으로 둔다

# 아래 크기 값은 가로 576px(FASHN 출력) 기준이고, 다른 크기에서는 비례해서 쓴다.
CHANGE_THRESHOLD = 24.0   # 0~255 RGB 평균 차이. 바뀌지 않은 바지의 재생성 잡음보다 크고 새 옷보다 작게
NOISE_OPEN_PX = 5         # 이보다 작은 점(주름·질감 잡음)은 지운다
BAND_MARGIN_PX = 6        # 바뀐 옷의 세로 끝에서 이만큼 더 (밑단 그림자)
CLOSE_PX = 15             # 새 옷 안에서 원래 옷과 색이 비슷해 구멍 난 곳을 메운다
EDGE_THRESHOLD = 8.0      # 옷 근처에서는 이 정도 차이도 덮는다. 배경은 원본과 결과 차이가 1 안팎이라 안전
EDGE_REACH_PX = 12        # 옷·바뀐 피부에서 이 거리 안의 크게 달라진 픽셀도 덮는다 (원래 팔 윤곽)
GROW_PX = 4               # 경계 바깥으로 조금 넓혀 원래 옷 테두리가 남지 않게
FEATHER_PX = 3.0          # 경계를 흐리게 섞는다 (2px에서는 배경 밝기 차이가 선으로 보였다)


def _segmenter_model_path() -> str:
    import urllib.request

    cache_dir = os.environ.get('VFA_SCRATCH') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.scratch')
    cache_dir = os.path.join(cache_dir, 'mediapipe')
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, 'selfie_multiclass_256x256.tflite')
    if not os.path.exists(path):
        urllib.request.urlretrieve(SEGMENTER_URL, path)
    return path


def segment(image: Image.Image) -> Optional[np.ndarray]:
    """픽셀별 클래스(0~5) 배열. mediapipe 가 없으면 None."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError:
        return None
    options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_segmenter_model_path()),
        output_category_mask=True,
    )
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=np.asarray(image.convert('RGB'))))
        # numpy_view()는 segmenter 내부 버퍼다. 닫히기 전에 복사한다(body_measure 와 같은 이유)
        return np.squeeze(result.category_mask.numpy_view()).copy()


def _scaled(px: float, width: int) -> int:
    return max(1, int(round(px * width / 576)))


def _morph(mask: Image.Image, size: int, grow: bool) -> Image.Image:
    size = size + 1 if size % 2 == 0 else size   # PIL 필터 크기는 홀수
    return mask.filter(ImageFilter.MaxFilter(size) if grow else ImageFilter.MinFilter(size))


def _fill_holes(mask: Image.Image) -> Image.Image:
    """바깥에서 닿지 않는 빈 곳(옷 안의 구멍)을 채운다."""
    padded = Image.new('L', (mask.width + 2, mask.height + 2), 0)
    padded.paste(mask, (1, 1))
    ImageDraw.floodfill(padded, (0, 0), 128)
    outside = np.asarray(padded)[1:-1, 1:-1] == 128
    return Image.fromarray(np.where(outside, 0, 255).astype(np.uint8), 'L')


def garment_mask(person_small: Image.Image, result: Image.Image, category: str = 'tops',
                 person_classes: Optional[np.ndarray] = None,
                 result_classes: Optional[np.ndarray] = None) -> Optional[Image.Image]:
    """결과를 쓸 자리(흰색) 마스크. person_small 은 result 와 같은 크기로 줄인 원본.

    category: FASHN 옷 종류 'tops' | 'bottoms' | 'one-pieces'. 상의면 바뀐 옷의 아래 끝보다 위,
    하의면 위 끝보다 아래만 결과로 덮는다(selfie_multiclass 가 상의·바지를 구분하지 못해서).
    구분 결과(클래스 배열)를 주면 그걸 쓰고(테스트용), 없으면 segment() 로 구한다.
    구분 모델을 못 쓰면 None.
    """
    if person_classes is None:
        person_classes = segment(person_small)
    if result_classes is None:
        result_classes = segment(result)
    if person_classes is None or result_classes is None:
        return None

    width = result.width
    a = np.asarray(person_small.convert('RGB'), dtype=np.float32)
    b = np.asarray(result.convert('RGB'), dtype=np.float32)
    diff = Image.fromarray(np.clip(np.abs(a - b).mean(axis=2), 0, 255).astype(np.uint8), 'L')
    diff = np.asarray(diff.filter(ImageFilter.GaussianBlur(_scaled(2, width))))
    clothes = np.isin(person_classes, CLOTHES_CLASSES) | np.isin(result_classes, CLOTHES_CLASSES)
    skin = (person_classes == 2) | (result_classes == 2)

    changed = Image.fromarray(((diff > CHANGE_THRESHOLD) & (clothes | skin)).astype(np.uint8) * 255, 'L')
    changed = np.asarray(_morph(_morph(changed, _scaled(NOISE_OPEN_PX, width), False),
                                _scaled(NOISE_OPEN_PX, width), True)) > 0

    # 색 차이만으로는 새 옷과 원래 옷 색이 비슷한 곳(카키 티 ↔ 갈색 소매)을 놓친다.
    # 그래서 '달라진 곳'은 세로 범위를 정하는 데만 쓰고, 그 범위 안의 옷은 전부 결과로 덮는다.
    rows = np.nonzero((changed & clothes).any(axis=1))[0]
    if rows.size == 0:
        return Image.new('L', result.size, 0)   # 바뀐 옷이 없다 → 전부 원본
    margin = _scaled(BAND_MARGIN_PX, width)
    ys = np.arange(result.height)[:, None]
    if category == 'tops':
        band = ys <= np.percentile(rows, 99) + margin
    elif category == 'bottoms':
        band = ys >= np.percentile(rows, 1) - margin
    else:
        band = np.ones_like(clothes)
    # 피부는 실제로 달라진 곳만(반팔 ↔ 긴팔로 팔이 바뀐 자리). 손·목은 원본으로 남긴다
    base = (clothes | (changed & skin)) & band
    # 옷 근처에서 크게 달라졌는데 구분 모델이 '배경'이라고 한 픽셀도 덮는다. 넓은 셔츠 소매가 원래 팔을
    # 덮을 때 팔 윤곽 몇 픽셀이 배경으로 분류돼 원본이 남으면서 소매 옆에 흰 선이 보였다(첫 시도).
    near_base = np.asarray(_morph(Image.fromarray(base.astype(np.uint8) * 255, 'L'),
                                  _scaled(EDGE_REACH_PX, width), True)) > 0
    face_or_hair = np.isin(person_classes, (1, 3)) & np.isin(result_classes, (1, 3))
    covered = base | ((diff > EDGE_THRESHOLD) & near_base & band & ~face_or_hair)

    mask = Image.fromarray(covered.astype(np.uint8) * 255, 'L')
    mask = _morph(_morph(mask, _scaled(CLOSE_PX, width), True), _scaled(CLOSE_PX, width), False)
    mask = _fill_holes(mask)
    mask = _morph(mask, _scaled(GROW_PX, width), True)
    # 메우고 넓히는 동안 옷에서 먼 배경까지 결과로 덮지 않게 덮을 자리 근처로 다시 자르고,
    # 얼굴·머리로 넘어간 부분도 뺀다
    near = np.asarray(_morph(Image.fromarray(covered.astype(np.uint8) * 255, 'L'),
                             _scaled(GROW_PX, width), True)) > 0
    mask = Image.fromarray(np.where(face_or_hair | ~near, 0, np.asarray(mask)).astype(np.uint8), 'L')
    return mask.filter(ImageFilter.GaussianBlur(FEATHER_PX * width / 576))


def restore_original(person: Image.Image, result: Image.Image, category: str = 'tops',
                     scale: float = 2.0, upscaled: Optional[Image.Image] = None,
                     ) -> Tuple[Image.Image, Optional[Image.Image]]:
    """옷 자리만 결과, 나머지는 원본인 이미지를 result 의 scale 배 크기로 돌려준다.

    scale 을 키우면 얼굴·배경은 원본 해상도로 또렷해지고, 옷은 결과를 늘린 것이라 그만큼 흐리다.
    upscaled: 결과를 업스케일러(Real-ESRGAN 등)로 키운 이미지. 주면 옷 자리에 단순 확대 대신 이걸 쓴다.
    마스크는 언제나 원래 결과(576px) 기준으로 구한다.
    반환: (합친 이미지, 마스크). 구분 모델을 못 쓰면 (결과를 늘린 것, None).
    """
    person = person.convert('RGB')
    result = result.convert('RGB')
    size = (int(round(result.width * scale)), int(round(result.height * scale)))
    mask = garment_mask(person.resize(result.size, Image.LANCZOS), result, category)
    source = result if upscaled is None else upscaled.convert('RGB')
    big_result = source.resize(size, Image.LANCZOS)
    if mask is None:
        return big_result, None
    big_mask = mask.resize(size, Image.BILINEAR)
    return Image.composite(big_result, person.resize(size, Image.LANCZOS), big_mask), big_mask
