"""사진 검사 결과(`app/photo_check.py`)를 사진 위에 그림으로 겹쳐 보여준다.

지금까지는 "더 가까이서 찍어주세요" 같은 **글로만** 알려줬다. 무엇이 얼마나 모자란지는
사진 위에 선을 그어 보여주는 쪽이 빨리 고친다. 촬영 유도선의 **1단계**다
(2단계는 촬영 중 실시간 안내, `app/static/capture_guide.html`).

**글자는 영문·숫자만 쓴다.** PIL 기본 폰트는 한글을 못 그리고 Colab에는 한글 폰트가 없어서
한글을 넣으면 네모로 나온다. 한국어 설명은 앱 화면의 `check.message()` 가 맡고,
이 그림은 **기하와 숫자**만 담당한다.

**숫자는 다시 계산하지 않고 `check.metrics` 에서 가져온다.** 여기서 비율을 다시 구하면
`photo_check` 의 판정식이 바뀔 때 그림과 판정이 조용히 어긋난다. 이 모듈이 직접 쓰는 건
랜드마크의 **위치**(선을 어디 그을지)뿐이다.

torch·mediapipe 를 임포트하지 않는다(랜드마크는 인자로 받는다). 그래서 단위 테스트가 된다.
"""
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from body_measure import POSE_LANDMARKS
from photo_check import (ARMS_DOWN_MARGIN, MAX_ARM_SPREAD, MIN_BODY_FRACTION,
                         MIN_FRONTAL_RATIO)

DISPLAY_HEIGHT = 720          # 앱 화면에 맞춰 줄여 그린다. 폰 원본(4032px)을 그대로 그리면 선이 안 보인다
OK_COLOR = (60, 200, 110)
BAD_COLOR = (235, 70, 70)
TARGET_COLOR = (255, 200, 0)  # 맞춰야 할 목표선
DIM_COLOR = (170, 170, 170)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:         # Pillow 10.1 미만은 size 인자가 없다
        return ImageFont.load_default()


def _color(value: Optional[float], ok: bool) -> tuple:
    """값을 모르면 회색, 알면 통과 여부에 따라 초록/빨강."""
    if value is None:
        return DIM_COLOR
    return OK_COLOR if ok else BAD_COLOR


def _dashed_line(draw, y, width, color, dash=14, gap=10, thickness=3):
    for x in range(0, width, dash + gap):
        draw.line([(x, y), (min(x + dash, width), y)], fill=color, width=thickness)


def _text(draw, xy, text, color, font, anchor_box=True):
    """읽히도록 흰 바탕을 깔고 쓴다. 사진 배경이 밝든 어둡든 보여야 한다."""
    x, y = xy
    if anchor_box:
        box = draw.textbbox((x, y), text, font=font)
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(255, 255, 255))
    draw.text((x, y), text, fill=color, font=font)


def draw_guide(image, landmarks=None, check=None, height: int = DISPLAY_HEIGHT) -> Image.Image:
    """검사 결과를 겹쳐 그린 사진을 돌려준다.

    landmarks: `photo_check.run_pose()` 가 준 33점. None 이면 사진만 줄여서 돌려준다.
    check: `photo_check.check_photo()` 결과. `metrics` 의 숫자를 라벨로 쓴다.
    """
    canvas = image.convert('RGB')
    if canvas.height != height:
        canvas = canvas.resize((max(1, round(canvas.width * height / canvas.height)), height),
                               Image.LANCZOS)
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    font = _font(17)

    if not landmarks:
        _text(draw, (12, 12), 'pose not detected', BAD_COLOR, font)
        return canvas

    metrics = dict(check.metrics) if check is not None else {}

    def point(name):
        lm = landmarks[POSE_LANDMARKS[name]]
        return np.array([lm.x * width, lm.y * height], dtype=float)

    summary = []

    # --- 전신 비율: 발목이 목표선 아래로 내려와야 한다 (MIN_BODY_FRACTION) --------------
    nose = point('nose')
    ankle_y = max(point('left_ankle')[1], point('right_ankle')[1])
    fraction = metrics.get('body_fraction')
    fraction_ok = fraction is not None and fraction >= MIN_BODY_FRACTION
    body_color = _color(fraction, fraction_ok)

    target_y = nose[1] + MIN_BODY_FRACTION * height
    # 필요한 길이(노랑)와 실제 코~발목(초록/빨강)을 나란히 세운다. 멀리서 찍으면 목표선이
    # 화면 밖으로 나가 '얼마나 모자란지'가 안 보였다 — 막대 두 개는 언제나 화면 안에서 비교된다.
    draw.line([(30, nose[1]), (30, min(target_y, height - 1))], fill=TARGET_COLOR, width=7)
    draw.line([(16, nose[1]), (16, ankle_y)], fill=body_color, width=7)
    draw.line([(8, nose[1]), (38, nose[1])], fill=body_color, width=3)
    draw.line([(8, ankle_y), (24, ankle_y)], fill=body_color, width=3)
    if target_y < height:
        _dashed_line(draw, target_y, width, TARGET_COLOR)
        _text(draw, (46, target_y + 6), f'feet below this line (need {MIN_BODY_FRACTION:.0%})',
              (120, 90, 0), font)
    else:
        _text(draw, (46, height - 30), f'come closer: body {fraction or 0:.0%} < need {MIN_BODY_FRACTION:.0%}',
              BAD_COLOR, font)
    summary.append(('body', f'{fraction:.0%}' if fraction is not None else '-', fraction_ok))

    # --- 정면: 어깨 폭 / 몸통 길이 -------------------------------------------------
    l_sh, r_sh = point('left_shoulder'), point('right_shoulder')
    l_hip, r_hip = point('left_hip'), point('right_hip')
    frontal = metrics.get('frontal_ratio')
    frontal_ok = frontal is not None and frontal >= MIN_FRONTAL_RATIO
    draw.line([tuple(l_sh), tuple(r_sh)], fill=_color(frontal, frontal_ok), width=5)
    draw.line([tuple((l_sh + r_sh) / 2), tuple((l_hip + r_hip) / 2)], fill=DIM_COLOR, width=2)
    summary.append(('front', f'{frontal:.2f}' if frontal is not None else '-', frontal_ok))

    # --- 팔: 벌림(엉덩이~손목 가로 거리)과 올림 ------------------------------------
    for side, short in (('left', 'L'), ('right', 'R')):
        wrist, hip = point(f'{side}_wrist'), point(f'{side}_hip')
        spread = metrics.get(f'{side}_arm_spread')
        raised = metrics.get(f'{side}_wrist_raise')
        arm_ok = (spread is not None and spread <= MAX_ARM_SPREAD
                  and raised is not None and raised <= ARMS_DOWN_MARGIN)
        color = _color(spread, arm_ok)
        # 손목 높이에서 몸(엉덩이)까지의 가로 거리 = 얼마나 벌렸는가
        draw.line([(hip[0], wrist[1]), (wrist[0], wrist[1])], fill=color, width=3)
        draw.ellipse((wrist[0] - 9, wrist[1] - 9, wrist[0] + 9, wrist[1] + 9), outline=color, width=4)
        # 문제가 '올림'이면 벌림 숫자를 보여 봐야 헷갈린다(엄지척 사진: 벌림 0.16인데 NG)
        if spread is None:
            value = '-'
        elif spread <= MAX_ARM_SPREAD and raised is not None and raised > ARMS_DOWN_MARGIN:
            value = 'up'
        else:
            value = f'{spread:.2f}'
        summary.append((f'arm {short}', value, arm_ok))

    # --- 요약 (좌상단) -------------------------------------------------------------
    line_height = 24
    box_height = line_height * len(summary) + 12
    draw.rectangle((8, 8, 186, 8 + box_height), fill=(255, 255, 255))
    y = 14
    for name, value, ok in summary:
        # 숫자가 없으면(검사를 못 함) 실패로 표시하지 않는다
        verdict, color = ('-', DIM_COLOR) if value == '-' else (('OK', OK_COLOR) if ok else ('NG', BAD_COLOR))
        draw.text((16, y), f'{name:<6s} {value:>5s}', fill=(40, 40, 40), font=font)
        draw.text((146, y), verdict, fill=color, font=font)
        y += line_height
    return canvas
