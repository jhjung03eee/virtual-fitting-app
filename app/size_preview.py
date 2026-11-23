"""사이즈별 예상 착용 모습 — 합성 결과 한 장의 **총장만** 치수표 비율대로 바꾼다 (docs/PLAN.md F-39).

합성 모델은 사이즈를 모른다(F-28). F-35에서 결과 사진의 몸판을 늘려 보니 기장은 실제 착용과 2% 안으로 맞았고,
몸판 폭은 3~7%에 가로로 줄일 때 옆선 얼룩이 남았다. 그래서 여기서는 **세로(총장)만** 바꾼다.
- 합성 결과 = 추천 사이즈(기준)로 본다. 다른 사이즈는 치수표 총장 비율만큼 겨드랑이 아래 몸판만 늘리거나 줄인다.
- 절대 크기(사진 속 1cm가 몇 픽셀인지)는 쓰지 않는다. 키로 그 비율을 잡는 방법은 아직 검증하지 않았다(F-35).
  그래서 "기준 사이즈보다 몇 cm 길다/짧다"는 상대 비교만 보여 준다.
- 상의만. 옷 영역은 '옷' 분류 중 가슴 높이 옷 색과 비슷한 픽셀이다(바지도 '옷' 분류라서).
  윗옷과 바지 색이 비슷하면 밑단을 못 찾으므로 그때는 만들지 않고 알린다.
어깨선·소매·핏(붙음·뜸)은 바꾸지 못한다.
"""
import numpy as np
from PIL import Image, ImageFilter

from size_warp import warp_garment

CLOTHES, BODY_SKIN = 4, 2      # restore_original.segment 클래스: 2 몸 피부(팔·손) · 4 옷
CHEST_LEVEL = 0.55             # 어깨선 아래 몸통 길이의 55% = 가슴 높이 (measure_size_effect 와 같음)
SLEEVE_BAND = 0.40             # 소매 폭을 찾는 어깨선 아래 구간
SEARCH_DEPTH = 1.6             # 밑단은 어깨선 아래 몸통 길이의 1.6배 안에서 찾는다
COLOR_DISTANCE = 12.0          # 가슴 높이 옷 색과의 Lab 거리 기준(밝기 절반 가중)
HEM_BAND = 0.12                # 밑단 아래 원래 그림(그림자 포함)을 옮겨 와 섞는 폭 = 몸통 길이의 12%


class PreviewError(ValueError):
    """미리보기를 만들 수 없는 사진·치수표."""


def _lab(rgb):
    """RGB uint8 → CIE Lab (L 0~100, a·b 대략 −128~127). cv2 없이 numpy로."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]],
                       np.float32)
    xyz /= np.array([0.9505, 1.0, 1.089], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def body_frame(landmarks):
    """(어깨선 y, 몸통 길이, 몸 가운데 x) — 0~1 좌표. scripts/measure_size_effect.body_frame 과 같다."""
    from body_measure import POSE_LANDMARKS
    p = {n: np.array([landmarks[POSE_LANDMARKS[n]].x, landmarks[POSE_LANDMARKS[n]].y])
         for n in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')}
    shoulder = (p['left_shoulder'] + p['right_shoulder']) / 2
    hip = (p['left_hip'] + p['right_hip']) / 2
    return float(shoulder[1]), float(np.linalg.norm(shoulder - hip)), float(shoulder[0])


def top_mask(rgb, classes, frame):
    """윗옷 영역 bool HxW: '옷' 분류 ∩ 가슴 높이 옷 색과 가까운 픽셀 ∩ 어깨선~탐색 깊이."""
    h, w = classes.shape
    sy, t, cx = frame[0] * h, frame[1] * h, frame[2] * w
    clothes = classes == CLOTHES
    band = np.zeros_like(clothes)
    y0, y1 = int(sy + 0.35 * t), int(sy + CHEST_LEVEL * t) + 1
    band[y0:y1, max(0, int(cx - 0.25 * t)):int(cx + 0.25 * t) + 1] = True
    lab = _lab(rgb)
    sample = lab[clothes & band]
    if len(sample) < 20:
        raise PreviewError('가슴 높이에서 옷을 찾지 못했어요. 상의 합성 결과에서만 만들 수 있어요.')
    color = np.median(sample, axis=0)
    # 밝기는 주름·그림자로 크게 흔들리므로 절반만 본다. 남색 티(20,30,52)와 짙은 갈색 바지(20,15,12)는
    # RGB 거리가 43뿐이라 RGB로는 못 갈랐다 — 색상(a·b) 차이는 크다
    diff = lab - color
    near = np.sqrt((0.5 * diff[..., 0]) ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2) < COLOR_DISTANCE
    ys = np.arange(h)[:, None]
    mask = clothes & near & (ys >= sy - 0.1 * t) & (ys < sy + SEARCH_DEPTH * t)
    img = Image.fromarray(mask.astype(np.uint8) * 255, 'L')
    k = max(3, w // 120) | 1
    # 주름·프린트 사이 구멍을 메우고 작은 점을 지운다
    img = img.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
    img = img.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))
    return np.asarray(img) > 0


def garment_rows(mask, frame):
    """(겨드랑이 y, 밑단 y, 어깨선 y) 픽셀. 밑단을 못 찾으면 PreviewError."""
    h = mask.shape[0]
    sy, t = frame[0] * h, frame[1] * h
    counts = mask.sum(axis=1)
    chest_row = int(sy + CHEST_LEVEL * t)
    chest_w = counts[chest_row]
    if chest_w == 0:
        raise PreviewError('가슴 높이에서 옷을 찾지 못했어요.')
    limit = min(h - 1, int(sy + SEARCH_DEPTH * t) - 1)
    hem = chest_row
    while hem < limit and counts[hem + 1] >= 0.5 * chest_w:
        hem += 1
    if hem >= limit:
        raise PreviewError('윗옷 밑단을 찾지 못했어요. 윗옷과 바지 색이 비슷하면 그럴 수 있어요.')
    band = counts[int(sy):int(sy + SLEEVE_BAND * t)]
    sleeve_row = int(sy) + int(band.argmax()) if len(band) else chest_row
    cut = (counts[sleeve_row] + chest_w) / 2
    armpit = sleeve_row
    while armpit < chest_row and counts[armpit] > cut:
        armpit += 1
    return armpit, hem, int(sy)


def length_scale(base_length, target_length, shoulder_y, armpit_y, hem_y):
    """겨드랑이 아래 몸판 세로 배율. 어깨선~밑단 길이가 치수표 총장 비율만큼 바뀌게, 바뀌는 몫은 겨드랑이 아래가 모두 맡는다."""
    full, top = hem_y - shoulder_y, armpit_y - shoulder_y
    return (full * target_length / base_length - top) / (full - top)


def length_previews(image, sizes, base, frame, classes):
    """image: 합성 결과(PIL), sizes: 치수표 {사이즈: {'length': cm, ...}} (순서대로), base: 합성 결과로 볼 사이즈.

    frame: body_frame() 값, classes: restore_original.segment() 값.
    → [(PIL, 캡션)] 치수표 순서. 총장이 없는 사이즈는 뺀다.
    """
    if base not in sizes or sizes[base].get('length') is None:
        raise PreviewError(f'{base} 사이즈의 총장이 치수표에 없어요.')
    rgb = np.asarray(image.convert('RGB'))
    if classes is None or classes.shape != rgb.shape[:2]:
        raise PreviewError('신체 부위 구분 결과가 없어 만들 수 없어요.')
    mask = top_mask(rgb, classes, frame)
    armpit, hem, sy = garment_rows(mask, frame)
    arms = classes == BODY_SKIN
    base_len = float(sizes[base]['length'])
    out = []
    for size, row in sizes.items():
        length = row.get('length')
        if length is None:
            continue
        diff = float(length) - base_len
        if size == base:
            picture, caption = image.convert('RGB'), f'{size} · 총장 {length:g}cm (추천, 합성 그대로)'
        else:
            scale = length_scale(base_len, float(length), sy, armpit, hem)
            band = max(3, int(HEM_BAND * frame[1] * rgb.shape[0]))
            picture, _ = warp_garment(image, mask, armpit, frame[2] * rgb.shape[1], 1.0, scale,
                                      keep=arms, hem_band=band)
            caption = f'{size} · 총장 {length:g}cm ({base}보다 {abs(diff):g}cm {"길어요" if diff > 0 else "짧아요"})'
        out.append((picture, caption))
    return out


def make_previews(image, sizes, base):
    """앱용: 포즈·부위 구분을 직접 돌린다 (mediapipe 필요)."""
    from photo_check import run_pose
    from restore_original import segment
    landmarks, _ = run_pose(image)
    if not landmarks:
        raise PreviewError('합성 결과에서 사람 자세를 찾지 못했어요.')
    return length_previews(image, sizes, base, body_frame(landmarks), segment(image.convert('RGB')))
