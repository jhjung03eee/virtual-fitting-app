"""계획서 발표(9월, 과제제안) PPT를 생성한다.

**결과·실험 데이터를 넣지 않는다.** 제안 발표이지 중간보고가 아니므로
"왜 이 과제를 하는지 · 어떻게 구현할 계획인지 · 무엇을 기대하는지"만 담는다.

**한 장에 한 가지만 말한다.** 큰 문장 하나 + 짧은 보조선으로 짜고,
나머지는 발표자가 말로 채운다.

**기술 용어를 깊게 쓰지 않는다.** 청중은 모델 구조를 몰라도 되게 쓴다.
diffusion·GAN·SDXL·GPU 같은 말은 화면과 대본에서 "AI 합성", "장비" 같은 말로 바꾼다
(모델 이름과 논문 제목은 Appendix에만 그대로 둔다).

슬라이드 내용과 **발표 대본**은 SLIDES 리스트에 있다(대본은 각 항목의 notes).
장별 소요 시간은 대본 글자 수에서 계산해 노트 첫 줄에 붙인다.

**출력은 docs/fitcheck_proposal_v2.pptx.** docs/proposal_presentation.pptx는 사용자가
PowerPoint에서 직접 고친 파일이라 이 스크립트로 덮어쓰면 안 된다.

2번 슬라이드 사진(data/samples/deck_size_fail.jpg)은 외부 이미지라 gitignore 폴더에 둔다.
로고는 scripts/make_logo.py가 만든 docs/brand/*.png.

    python scripts/make_logo.py        # 로고가 없으면 먼저
    python scripts/build_proposal_deck.py
"""
import os
import re

from pptx import Presentation
from pptx.util import Emu, Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCKUP_DIR = os.path.join(ROOT, 'docs', 'mockups')
BRAND_DIR = os.path.join(ROOT, 'docs', 'brand')
SAMPLE_DIR = os.path.join(ROOT, 'data', 'samples')
OUT_PATH = os.path.join(ROOT, 'docs', 'fitcheck_proposal_v2.pptx')

# Noto Sans KR은 굵기별로 별도 패밀리라 이름으로 골라 쓴다. 발표 PC에 없으면
# 아래 네 값을 전부 '맑은 고딕'으로 바꾸면 된다 (굵기 대비는 죽는다).
F_THIN = 'Noto Sans KR Light'
F_LIGHT = 'Noto Sans KR DemiLight'
F_REG = 'Noto Sans KR'
F_MED = 'Noto Sans KR Medium'

BG = RGBColor(0x00, 0x00, 0x00)
SURFACE = RGBColor(0x1E, 0x1E, 0x22)  # 이보다 어두우면 프로젝터에서 배경과 구분이 안 된다
FG = RGBColor(0xFF, 0xFF, 0xFF)
FG_SUB = RGBColor(0x98, 0x98, 0x9E)
FG_DIM = RGBColor(0x5A, 0x5A, 0x5E)
LINE = RGBColor(0x2C, 0x2C, 0x2E)
GREEN = RGBColor(0x30, 0xD1, 0x58)    # 로고 체크와 같은 초록
BAR = RGBColor(0x4A, 0x4A, 0x50)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
M = Inches(0.9)             # 좌우 여백
COL_W = SLIDE_W - 2 * M

# TODO: 발표 전에 채워 넣기
TEAM_INFO = '성균관대학교 종합설계프로젝트   ·   [팀명]   ·   [조원 이름]'

# 대본 소요 시간을 글자 수에서 계산할 때 쓰는 발표 속도. 낭독 속도(330~350)보다 낮게
# 잡은 값으로, 슬라이드 전환과 호흡을 포함한 실제 발표 속도에 가깝다.
SPEAK_CPM = 290
QA_MARKER = '--- 예상 질문 ---'  # 이 아래는 낭독분이 아니라 시간 계산에서 뺀다

SLIDES = [
    {
        'type': 'cover',
        'title': '사이즈 정보를 반영한 가상 피팅 앱',
        'subtitle': '종합설계프로젝트  ·  과제 계획 발표',
        'notes': '안녕하십니까. 사이즈 정보를 반영한 가상 피팅 앱, FitCheck을 제안하는 '
                 '[팀명]입니다. 온라인에서 옷을 살 때 겪는 사이즈 문제를 가상 피팅과 사이즈 '
                 '추천으로 풀어보려고 합니다.',
    },
    {
        'type': 'story',
        'eyebrow': '개인적 경험에서 출발',
        'headline': '무신사에서 산 옷을\n또 환불했다',
        'sub': '사이즈도, 핏도 나와 맞지 않았다.',
        'quote': '“돈도 시간도 아깝다..”',
        'image': os.path.join(SAMPLE_DIR, 'deck_size_fail.jpg'),
        'notes': '먼저 제 이야기로 시작하겠습니다.\n\n'
                 '저는 옷을 주로 무신사에서 삽니다. 그런데 막상 받아 보면 사이즈나 핏이 저와 '
                 '맞지 않아서 환불한 경험이 정말 많았습니다. 그때마다 돈도 시간도 아깝다는 '
                 '생각이 들었습니다.\n\n'
                 '화면으로만 보고 고른 옷은 받아 보기 전까지 어떻게 맞을지 알 수 없습니다. '
                 '그리고 이건 저만의 문제가 아니었습니다.',
    },
    {
        'type': 'hero_stat',
        'eyebrow': '문제',
        'headline': '온라인에서는 입어볼 수 없고,\n반품 1위 원인은 사이즈다',
        'evidence': [
            ('조사에 따라 사이즈 · 핏이 전체 반품 사유의 45 – 53%',
             'Narvar 소비자 조사(색상 포함) · Coresight Research 2023'),
            ('국내 연구에서도 화면 속 사이즈와 실제 치수의 차이가 주요 반품 사유',
             '김지수 · 나영주, 감성과학 23(1), 2020'),
        ],
        'support': [
            ('약 30%', '온라인 패션 평균 반품률 — 오프라인 매장의 3배 이상', '물류신문, 2024'),
            ('연 4조 원대', '국내 이커머스 반품 처리 비용 (2023년 추정)', '머니투데이, 2024'),
        ],
        'notes': '온라인에서는 옷을 입어볼 수 없습니다. 그래서 반품이 생기는데, 그 1순위 원인이 '
                 '사이즈입니다.\n\n'
                 '조사에 따라 사이즈와 핏이 전체 반품 사유의 45에서 53퍼센트를 차지합니다. '
                 '국내 연구에서도 화면에서 본 사이즈와 실제 치수의 차이가 주요 반품 사유로 '
                 '보고됐습니다.\n\n'
                 '규모도 큽니다. 온라인 패션의 평균 반품률은 30퍼센트 수준으로 오프라인 매장의 '
                 '세 배가 넘고, 국내 이커머스에서 반품을 처리하는 데만 한 해 4조 원대의 비용이 '
                 '드는 것으로 추정됩니다.\n\n'
                 + QA_MARKER + '\n'
                 '· 45~53%의 출처 → 45%는 Narvar 소비자 조사(물류신문 2024 인용)인데 '
                 '사이즈·핏에 색상이 포함된 수치이고, 53%는 의류 판매자 대상 조사(Coresight '
                 'Research 2023)입니다. 조사 대상과 정의가 달라 범위로 제시했습니다.\n'
                 '· 더 권위 있는 수치가 필요하면 → 미국소매협회(NRF)는 2025년 온라인 판매의 '
                 '19.3%가 반품될 것으로 전망했습니다(전 품목 기준).\n'
                 '· 4조 원은 어떻게 나온 숫자인가 → 공식 통계가 아니라 언론 추정치입니다. 2023년 택배 '
                 '물량 46억여 건에 이커머스 반품률 20%를 곱해 반품 약 9억 3천만 건을 잡고 비용을 '
                 '추산했습니다. 정확한 금액보다 "수조 원대"라는 규모를 보여 드리려는 숫자입니다.',
    },
    {
        'type': 'solution',
        'eyebrow': '해결 방안',
        'headline': '입어보기 전에,\n어울리는지 그리고 맞는지까지',
        'sub': 'FitCheck — 입어보지 않고도 어울림과 사이즈를 함께 확인하는 가상 피팅 앱',
        'columns': [
            ('가상 착용 이미지', '내 사진에 옷을 합성해\n어울리는지 먼저 확인한다'),
            ('사이즈 추천', '신체 치수와 옷 치수표를 비교해\n맞는 사이즈를 알려준다'),
        ],
        'notes': '그래서 저희가 제안하는 것이 FitCheck입니다. 입어보기 전에, 어울리는지 그리고 '
                 '맞는지까지 함께 확인하게 하는 앱입니다.\n\n'
                 '기능은 두 가지입니다. 하나는 가상 착용 이미지로, 내 사진에 옷을 합성해 '
                 '어울리는지 먼저 봅니다. 다른 하나는 사이즈 추천으로, 신체 치수와 옷 치수표를 '
                 '비교해 맞는 사이즈를 알려 줍니다.\n\n'
                 '기존 가상 피팅이 입은 모습만 보여 줬다면, 저희는 사이즈까지 함께 답합니다.',
    },
    {
        'type': 'columns',
        'title': '기대 효과',
        'columns': [
            ('소비자', '사이즈 실패로 인한\n반품을 줄인다',
             '사기 전에 맞는 사이즈로 입은 모습을 본다'),
            ('판매 기업', '반품 물류 · 재고\n비용을 줄인다',
             '사이즈 문의와 교환 응대 부담도 준다'),
            ('확장성', '치수표만 있으면\n어떤 상품에도',
             '쇼핑몰이 가진 데이터로 바로 적용된다'),
        ],
        'notes': '기대 효과는 세 가지입니다.\n\n'
                 '소비자는 사기 전에 맞는 사이즈로 입은 모습을 확인해, 사이즈 실패로 인한 반품을 '
                 '줄일 수 있습니다. 판매 기업은 반품 물류와 재고 비용, 사이즈 문의 응대 부담을 '
                 '줄일 수 있습니다.\n\n'
                 '그리고 치수표와 상품 이미지만 있으면 동작하기 때문에, 어떤 상품에도 바로 적용할 '
                 '수 있습니다.',
    },
    {
        'type': 'flow',
        'title': '어떻게 만드는가',
        'cards': [
            ('INPUT', '입력', ['사용자 — 전신 사진, 신체 치수', '옷 — 상품 이미지, 치수표']),
            ('PROCESS', '처리', ['가상 피팅 — AI 이미지 합성', '사이즈 추천 — 여유분 계산']),
            ('OUTPUT', '출력', ['사이즈별 착용 모습', 'S · M · L 추천과 그 근거']),
        ],
        'caption': '별도의 실측 장비도, 추가 촬영도 없다. 쇼핑몰이 이미 갖고 있는 정보만으로 동작한다.',
        'notes': '어떻게 만드는지 전체 흐름입니다.\n\n'
                 '입력은 사용자의 전신 사진과 신체 치수, 그리고 옷의 상품 이미지와 치수표입니다. '
                 '처리 단계에서는 AI 이미지 합성으로 착용 모습을 만들고, 동시에 신체와 '
                 '옷 치수의 차이인 여유분을 계산합니다. 출력은 사이즈별 착용 모습과 S·M·L 추천, '
                 '그리고 그 근거입니다.\n\n'
                 '별도의 실측 장비나 추가 촬영 없이, 쇼핑몰이 이미 가진 정보만으로 동작합니다.',
    },
    {
        'type': 'mockup',
        'eyebrow': '구현  01  ·  가상 피팅',
        'headline': '사진 두 장으로\n착용 모습을 만든다',
        'lines': [
            '사용자 전신 사진 + 옷 이미지 입력',
            'AI가 입은 모습을 합성',
            '상의 · 하의 · 원피스 카테고리별 지원',
        ],
        'image': 'mockup_upload.png',
        'notes': '첫 번째 구현은 가상 피팅입니다. 사용자 전신 사진과 옷 이미지 두 장을 받아, '
                 '그 사람이 그 옷을 입은 모습을 합성합니다.\n\n'
                 '상의, 하의, 원피스처럼 카테고리를 고르면 해당 부위만 바꿉니다. 사용자 사진은 '
                 '조명과 배경이 제각각이라, 촬영 가이드도 함께 제공할 계획입니다.',
    },
    {
        'type': 'cards4',
        'eyebrow': '모델 후보',
        'headline': '무엇으로 합성할지,\n네 후보를 같은 조건에서 비교한다',
        'sub': '같은 인물 사진 · 같은 옷으로 비교해, 학기 여건에서 가장 균형 잡힌 모델을 주력으로 정한다',
        'cards': [
            ('CatVTON', '가벼운 최신 모델',
             ['일반 장비로도 실행', '주름 표현이 자연스러움', '프린트가 흐려질 수 있음'], '주력 후보'),
            ('IDM-VTON', '고품질 · 무거운 모델',
             ['옷 디테일을 잘 살림', '로고 · 프린트 재현 우수', '무겁고 생성이 느림'], '품질 비교 기준'),
            ('FASHN VTON v1.5', '최근 공개된 모델',
             ['옷 디테일 보존에 유리', '누구나 쓸 수 있게 공개', '검증 사례가 아직 적음'], '대안 후보'),
            ('HR-VITON', '초기 방식 · 빠른 모델',
             ['가볍고 빠름', '옷 사진을 몸에 맞춰 변형', '깔끔한 배경 사진에 강함'], '속도 비교 기준'),
        ],
        'notes': '사진을 합성할 AI 모델은 네 후보를 같은 조건에서 비교해 고릅니다.\n\n'
                 'CatVTON은 가벼워서 일반 장비로도 돌릴 수 있어 주력 후보입니다. IDM-VTON은 결과 '
                 '품질이 좋지만 무거워서, 품질 기준을 확인하는 비교용으로 씁니다.\n\n'
                 'FASHN VTON은 최근 공개된 모델이라 대안 후보로 두었고, HR-VITON은 가볍고 빠른 '
                 '초기 방식이라 속도를 비교하는 기준으로 삼습니다.',
    },
    {
        'type': 'mockup',
        'eyebrow': '구현  02  ·  사이즈 추천',
        'headline': '아는 치수만 넣어도\n추천이 나온다',
        'lines': [
            '키 · 어깨너비 · 가슴둘레 · 허리둘레',
            '둘레 ↔ 단면 자동 변환',
            '여유분 계산 → S / M / L 판정',
        ],
        'image': 'mockup_measure.png',
        'notes': '두 번째는 사이즈 추천입니다. 키, 어깨너비, 가슴둘레, 허리둘레 중 아는 것만 넣어도 '
                 '동작합니다.\n\n'
                 '사용자가 아는 둘레 값과 쇼핑몰 치수표의 단면 값은 기준이 달라서, 앱이 자동으로 '
                 '변환해 비교합니다. 그다음 여유분을 계산해 S, M, L 중 맞는 사이즈를 추천하고, '
                 '어느 부위가 왜 그런지도 함께 보여 줍니다.',
    },
    {
        'type': 'mockup',
        'eyebrow': '구현  03  ·  두 기능의 통합',
        'headline': '같은 옷,\n사이즈만 다르게',
        'lines': [
            '추천 사이즈별 착용 모습을 나란히 비교',
            '여유분을 실루엣에 반영',
            '이 연결이 과제의 핵심',
        ],
        'image': 'mockup_compare.png',
        'notes': '세 번째가 과제의 핵심인 두 기능의 연결입니다. 추천된 사이즈별로 착용 모습을 '
                 '합성해 나란히 보여 줍니다. 같은 옷을 M으로 입었을 때와 L로 입었을 때를 한눈에 '
                 '비교하는 것이 목표입니다.\n\n'
                 '가장 어려운 부분이라, 사이즈별 합성 비교를 먼저 만들고 여유분을 실루엣에 '
                 '반영하는 작업을 그 위에 얹겠습니다.',
    },
    {
        'type': 'cards4',
        'eyebrow': '실험 계획',
        'headline': '같은 입력으로 비교하고,\n정해둔 기준으로 고른다',
        'sub': '후보 4종을 동일 입력으로 비교 → 실제 휴대폰 사진으로 검증 → 판정 기준 합격률로 주력 모델 확정',
        'cards': [
            ('01  동일 조건 비교', '9 – 10월',
             ['같은 인물 · 옷 입력', '같은 설정으로 실행', '후보 4종 결과 수집'], '산출물 — 모델 비교표'),
            ('02  실사진 검증', '10월',
             ['휴대폰 사진 · 일반 배경', '자세 · 조명 바꿔 확인', '실패 유형 정리'], '산출물 — 실패 사례집'),
            ('03  품질 판정', '10월',
             ['색 · 프린트 · 얼룩 확인', '형태 · 얼굴 · 배경 보존', '항목별 합격률 산출'], '산출물 — 판정 기준표'),
            ('04  주력 모델 선정', '11월 초',
             ['품질 · 속도 · 비용 균형', '앱에 쓸 설정 확정', '탈락 이유도 기록'], '산출물 — 선정 근거'),
        ],
        'notes': '모델 선택은 감이 아니라 정해 둔 기준으로 합니다.\n\n'
                 '먼저 9월부터 10월까지 같은 인물, 같은 옷으로 후보 네 개의 결과를 모읍니다. '
                 '10월에는 실제 휴대폰 사진으로 검증해 실패 유형을 정리하고, 색과 프린트, 얼룩, '
                 '형태, 얼굴과 배경 보존 항목으로 합격률을 매깁니다.\n\n'
                 '11월 초에 품질과 속도, 실행 비용을 함께 보고 주력 모델을 확정합니다.',
    },
    {
        'type': 'gantt',
        'eyebrow': '진행 계획',
        'headline': '언제까지 무엇을 끝내는가',
        'sub': '9월 조사 · 설계  →  10월 모델 검증  →  11월 구현 · 통합  →  12월 평가 · 발표',
        'months': ['9월', '10월', '11월', '12월'],
        # (작업, 시작, 끝, 강조) — 시작·끝은 9월 초를 0, 12월 말을 4로 둔 월 단위
        'tasks': [
            ('조사 · 설계', 0.0, 1.0, False),
            ('모델 비교 실험', 0.5, 2.0, True),
            ('품질 개선', 1.5, 2.75, False),
            ('사이즈 추천 구현', 1.9, 3.0, True),
            ('통합 · 평가 · 발표', 2.5, 4.0, False),
        ],
        'milestones': [('중간발표', 2.1), ('최종발표', 3.6)],
        'notes': '전체 일정입니다.\n\n'
                 '9월에 조사와 설계를 하고, 10월까지 모델 비교 실험을 마칩니다. 10월 중순부터는 '
                 '품질 개선과 사이즈 추천 구현을 함께 진행합니다. 11월 초 중간발표에서 각 기능이 '
                 '동작하는 모습을 보여 드리고, 12월까지 두 기능을 통합해 평가한 뒤 최종발표에서 '
                 '시연하겠습니다.',
    },
    {
        'type': 'cards4',
        'eyebrow': '역할 분담',
        'headline': '네 갈래로 나눠\n동시에 진행한다',
        'sub': '담당자는 팀 구성 확정 후 기입한다 · 품질 판정과 평가는 전원이 함께 참여한다',
        'cards': [
            ('AI 합성', '담당  [          ]',
             ['모델 비교 실험', '품질 개선 실험', '앱 적용 설정 확정'], '산출물 — 모델 선정 근거'),
            ('사이즈 추천', '담당  [          ]',
             ['치수표 수집 · 정리', '여유분 계산 로직', '추천 규칙 설계'], '산출물 — 추천 규칙 문서'),
            ('앱 개발', '담당  [          ]',
             ['화면 설계 · 구현', '두 기능 통합', '결과 비교 화면'], '산출물 — 프로토타입 앱'),
            ('데이터 · 평가', '담당  [          ]',
             ['테스트 사진 · 옷 수집', '품질 판정 진행', '사용성 평가 설계'], '산출물 — 평가 결과 보고'),
        ],
        'notes': '역할은 AI 합성, 사이즈 추천, 앱 개발, 데이터와 평가 네 갈래로 나눠 동시에 '
                 '진행합니다. 담당자는 팀 구성이 확정되면 채우고, 품질 판정과 평가는 전원이 함께 '
                 '참여합니다.',
    },
    {
        'type': 'closing',
        'title': '감사합니다',
        'subtitle': 'Q  &  A',
        'notes': '이상으로 발표를 마치겠습니다. 감사합니다.\n\n'
                 + QA_MARKER + '\n'
                 '· 가상 피팅 모델을 직접 학습하나? → 아니요. 공개된 사전학습 모델을 비교해 고르고, '
                 '사이즈 반영과 두 기능의 연결에 집중합니다.\n'
                 '· 왜 CatVTON이 주력 후보인가? → 학생 팀이 쓸 수 있는 장비로도 돌아가는 가벼운 '
                 '모델이기 때문입니다. 다만 실험 계획의 판정 기준 결과로 최종 확정합니다.\n'
                 '· 신체 치수를 사진으로 자동 측정하지 않는 이유는? → 정확도를 신뢰하기 어려워 '
                 '사용자 직접 입력을 기본으로 두고, 자동 측정은 보조 수단으로 검토합니다.\n'
                 '· 치수표 형식이 쇼핑몰마다 다른데? → 항목 이름과 단위를 정규화하는 처리를 두고, '
                 '없는 항목은 판정에서 제외합니다.',
    },
    {
        'type': 'references',
        'eyebrow': 'APPENDIX  ·  참고 자료',
        'refs': [
            ('국내 이커머스 반품 약 9억 3,500만 건, 반품 처리 비용 약 4조 6,700억 원 (2023년 추정)',
             '머니투데이 (2024. 9. 11.) — 택배 물량 × 반품률 20%(eMarketer)로 산출한 언론 추정치',
             'mt.co.kr/living/2024/09/11/2024091022404176342'),
            ('패션 평균 반품률 약 30%, 오프라인의 3배 이상 · 사이즈·핏·색상 문제로 인한 반품 45%',
             '물류신문, "성장하는 패션 물류 시장, 경쟁 키워드 \'반품\'" (2024) — Narvar 소비자 조사 인용',
             'klnews.co.kr/news/articleView.html?idxno=312851'),
            ('의류 판매자의 53%가 사이즈·핏을 반품 사유 1순위로 지목',
             'Coresight Research (2023) — 업계 보고서 2차 인용',
             ''),
            ('화면 속 사이즈와 실제 치수의 차이가 주요 반품 사유',
             '김지수 · 나영주 (2020). 패션 온라인 쇼핑몰 유형에 따른 반품이유와 반품물류서비스 만족도. '
             '감성과학, 23(1), 3–16.',
             ''),
            ('모델 후보 — CatVTON',
             'Chong, Z. et al. (2024). CatVTON: Concatenation Is All You Need for Virtual Try-On '
             'with Diffusion Models.',
             'arXiv:2407.15886'),
            ('모델 후보 — IDM-VTON',
             'Choi, Y. et al. (2024). Improving Diffusion Models for Authentic Virtual Try-on in the Wild. '
             'ECCV 2024.',
             'arXiv:2403.05139'),
            ('모델 후보 — FASHN VTON v1.5',
             'FASHN AI. FASHN VTON v1.5: Efficient Maskless Virtual Try-On in Pixel Space '
             '(Apache 2.0 공개 모델).',
             'github.com/fashn-AI/fashn-vton-1.5'),
            ('모델 후보 — HR-VITON',
             'Lee, S. et al. (2022). High-Resolution Virtual Try-On with Misalignment and '
             'Occlusion-Handled Conditions. ECCV 2022.',
             'arXiv:2206.14180'),
        ],
        'note': '구현 화면은 팀이 직접 제작한 UI 목업이며, 2번 슬라이드 사진은 사이즈 실패 예시로 인용한 '
                '온라인 이미지다.',
    },
]


# ---------------------------------------------------------------- 기본 도구

def _timed_notes(notes, elapsed):
    """대본 앞에 '이 장 소요 · 누적' 머리말을 붙이고 누적 시간을 돌려준다.

    손으로 적어 두면 대본을 고칠 때마다 어긋나므로 글자 수에서 계산한다.
    """
    spoken = len(re.sub(r'\s', '', notes.split(QA_MARKER)[0]))
    seconds = spoken / SPEAK_CPM * 60
    total = elapsed + seconds
    head = f'[약 {round(seconds / 5) * 5}초  ·  누적 {int(total // 60)}:{int(total % 60):02d}]'
    return f'{head}\n\n{notes}', total


def _emu(value):
    """좌표를 정수 EMU로 맞춘다.

    나눗셈으로 만든 소수점 좌표를 커넥터에 그대로 넘기면 PowerPoint가 파일 자체를
    열지 못한다("PowerPoint could not open the file"). 도형·텍스트 상자는 조용히
    넘어가므로 증상이 특정 슬라이드에서만 나타나 원인을 찾기 어렵다.
    """
    return Emu(int(round(value)))


def _set_font(run, size, font=F_REG, color=FG, tracking=None):
    f = run.font
    f.name = font
    f.size = Pt(size)
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    # font.name은 라틴 자소만 지정한다. 한글은 a:ea를 따로 안 주면 테마 폰트로 떨어진다
    latin = rPr.find(qn('a:latin'))
    ea = rPr.find(qn('a:ea'))
    if ea is None:
        ea = rPr.makeelement(qn('a:ea'), {})
        latin.addnext(ea)
    ea.set('typeface', font)
    if tracking is not None:
        rPr.set('spc', str(int(round(tracking * 100))))  # 1/100 pt


def _textbox(slide, left, top, width, height, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(_emu(left), _emu(top), _emu(width), _emu(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.paragraphs[0].alignment = align
    return tf


def _lines(tf, texts, size, font=F_REG, color=FG, spacing=1.0, gap=0, align=PP_ALIGN.LEFT,
           tracking=None):
    for i, text in enumerate(texts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        if gap:
            p.space_after = Pt(gap)
        run = p.add_run()
        run.text = text
        _set_font(run, size, font=font, color=color, tracking=tracking)


def _rule(slide, left, top, width, color=LINE, weight=Pt(0.75)):
    line = slide.shapes.add_connector(1, _emu(left), _emu(top), _emu(left + width), _emu(top))
    line.line.color.rgb = color
    line.line.width = weight


def _vrule(slide, left, top, height, color=LINE, weight=Pt(0.75)):
    line = slide.shapes.add_connector(1, _emu(left), _emu(top), _emu(left), _emu(top + height))
    line.line.color.rgb = color
    line.line.width = weight


def _rect(slide, left, top, width, height, color, rounded=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                   _emu(left), _emu(top), _emu(width), _emu(height))
    if rounded:
        shape.adjustments[0] = rounded
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def _eyebrow(slide, text, top=Inches(0.78)):
    tf = _textbox(slide, M, top, COL_W, Inches(0.3))
    _lines(tf, [text], 11.5, font=F_MED, color=FG_SUB, tracking=0.9)


def _picture(slide, path, left, top, height):
    return slide.shapes.add_picture(path, _emu(left), _emu(top), height=_emu(height))


def _brand(path_name):
    return os.path.join(BRAND_DIR, path_name)


def _footer(slide, number):
    """모든 내용 슬라이드 하단: 흐린 워드마크 + 쪽번호.

    옷걸이 마크는 이 크기에서 눈금이 뭉개지므로 워드마크만 쓴다.
    """
    pic = _picture(slide, _brand('fitcheck_wordmark_dim.png'), M, SLIDE_H - Inches(0.67),
                   Inches(0.1))
    tf = _textbox(slide, SLIDE_W - M - Inches(1.0), SLIDE_H - Inches(0.75), Inches(1.0),
                  Inches(0.3), align=PP_ALIGN.RIGHT)
    _lines(tf, [f'{number:02d}'], 10.5, font=F_REG, color=FG_DIM, align=PP_ALIGN.RIGHT,
           tracking=0.8)
    return pic


def _headline_block(slide, spec, size=34):
    """eyebrow + 헤드라인(1~2줄) + 한 줄 설명. 설명 아래 여백의 시작 높이를 돌려준다."""
    _eyebrow(slide, spec['eyebrow'])
    lines = spec['headline'].split('\n')
    tf = _textbox(slide, M, Inches(1.45), COL_W, Inches(1.4))
    _lines(tf, lines, size, font=F_THIN, spacing=1.25)
    # Noto Sans KR은 글꼴 자체 줄높이가 em의 약 1.45배라, 줄간격 1.25를 곱하면 줄당 약 1.8em
    sub_top = Inches(1.45) + Pt(size * 1.8 * len(lines)) + Inches(0.12)
    tf = _textbox(slide, M, sub_top, COL_W, Inches(0.35))
    _lines(tf, [spec['sub']], 14.5, font=F_LIGHT, color=FG_SUB)
    return sub_top + Inches(0.55)


# ---------------------------------------------------------------- 슬라이드

def build_cover(slide, spec, number):
    lockup = _picture(slide, _brand('fitcheck_lockup_v_white.png'), Inches(0), Inches(1.35),
                      Inches(2.2))
    lockup.left = int((SLIDE_W - lockup.width) / 2)

    tf = _textbox(slide, M, Inches(4.05), COL_W, Inches(0.7), align=PP_ALIGN.CENTER)
    _lines(tf, [spec['title']], 29, font=F_THIN, align=PP_ALIGN.CENTER)

    tf = _textbox(slide, M, Inches(4.78), COL_W, Inches(0.4), align=PP_ALIGN.CENTER)
    _lines(tf, [spec['subtitle']], 14, font=F_LIGHT, color=FG_SUB, align=PP_ALIGN.CENTER,
           tracking=0.8)

    tf = _textbox(slide, M, SLIDE_H - Inches(1.15), COL_W, Inches(0.4), align=PP_ALIGN.CENTER)
    _lines(tf, [TEAM_INFO], 11.5, font=F_REG, color=FG_DIM, align=PP_ALIGN.CENTER)


def build_story(slide, spec, number):
    """큰 문장 + 보조선 + 인용, 오른쪽에 사진."""
    img_h = Inches(5.55)
    pic = _picture(slide, spec['image'], Inches(0), Inches(0.95), img_h)
    pic.left = _emu(SLIDE_W - M - pic.width)
    text_w = pic.left - M - Inches(0.6)

    _eyebrow(slide, spec['eyebrow'])

    tf = _textbox(slide, M, Inches(2.35), text_w, Inches(2.2))
    _lines(tf, spec['headline'].split('\n'), 42, font=F_THIN, spacing=1.28)

    tf = _textbox(slide, M, Inches(4.75), text_w, Inches(0.4))
    _lines(tf, [spec['sub']], 16, font=F_LIGHT, color=FG_SUB)

    tf = _textbox(slide, M, Inches(5.25), text_w, Inches(0.45))
    _lines(tf, [spec['quote']], 18, font=F_REG, color=FG)

    _footer(slide, number)


def build_solution(slide, spec, number):
    _eyebrow(slide, spec['eyebrow'])

    tf = _textbox(slide, M, Inches(1.65), COL_W, Inches(1.8))
    _lines(tf, spec['headline'].split('\n'), 40, font=F_THIN, spacing=1.28)

    tf = _textbox(slide, M, Inches(3.55), COL_W, Inches(0.4))
    _lines(tf, [spec['sub']], 16, font=F_LIGHT, color=FG_SUB)

    _rule(slide, M, Inches(4.35), COL_W)

    half = _emu(COL_W / 2)
    for i, (label, body) in enumerate(spec['columns']):
        left = _emu(M + i * half)
        tf = _textbox(slide, left, Inches(4.65), _emu(half - Inches(0.4)), Inches(0.35))
        _lines(tf, [label], 14, font=F_MED, color=FG_SUB, tracking=0.6)
        tf = _textbox(slide, left, Inches(5.1), _emu(half - Inches(0.4)), Inches(1.1))
        _lines(tf, body.split('\n'), 21, font=F_THIN, spacing=1.3)

    _footer(slide, number)


def build_cards4(slide, spec, number):
    """헤드라인 아래 4칸: 제목 / 부제 / 불릿 3개 / 하단 태그."""
    top = _headline_block(slide, spec)
    _rule(slide, M, top, COL_W)

    gap = Inches(0.3)
    col_w = _emu((COL_W - 3 * gap) / 4)
    y = top + Inches(0.28)
    for i, (title, subtitle, bullets, tag) in enumerate(spec['cards']):
        left = _emu(M + i * (col_w + gap))

        tf = _textbox(slide, left, y, col_w, Inches(0.4))
        _lines(tf, [title], 17, font=F_REG, color=FG)

        tf = _textbox(slide, left, y + Inches(0.42), col_w, Inches(0.3))
        _lines(tf, [subtitle], 12, font=F_REG, color=FG_DIM)

        tf = _textbox(slide, left, y + Inches(0.85), col_w, Inches(1.1))
        _lines(tf, ['·  ' + b for b in bullets], 13, font=F_LIGHT, color=FG_SUB, spacing=1.2,
               gap=5)

        tf = _textbox(slide, left, y + Inches(2.05), col_w, Inches(0.3))
        _lines(tf, [tag], 12.5, font=F_REG, color=GREEN if i == 0 and spec['eyebrow'] == '모델 후보'
               else FG)

    _footer(slide, number)


def build_gantt(slide, spec, number):
    top = _headline_block(slide, spec)
    _rule(slide, M, top, COL_W)

    label_w = Inches(2.5)
    x0 = M + label_w
    unit = _emu((COL_W - label_w) / len(spec['months']))
    head_y = top + Inches(0.2)
    row0 = head_y + Inches(0.5)
    row_step = Inches(0.46)
    rows_bottom = row0 + row_step * len(spec['tasks'])

    for i, month in enumerate(spec['months']):
        tf = _textbox(slide, x0 + unit * i + Inches(0.12), head_y, unit, Inches(0.3))
        _lines(tf, [month], 13, font=F_REG, color=FG_SUB)
        _vrule(slide, x0 + unit * i, head_y, rows_bottom - head_y)
    _vrule(slide, x0 + unit * len(spec['months']), head_y, rows_bottom - head_y)

    bar_h = Inches(0.18)
    for r, (task, start, end, accent) in enumerate(spec['tasks']):
        y = row0 + row_step * r
        tf = _textbox(slide, M, y - Inches(0.05), label_w - Inches(0.2), Inches(0.3))
        _lines(tf, [task], 13.5, font=F_LIGHT, color=FG if accent else FG_SUB)
        _rect(slide, x0 + unit * start, y + Inches(0.03), unit * (end - start), bar_h,
              GREEN if accent else BAR)

    for name, at in spec['milestones']:
        x = x0 + unit * at
        _vrule(slide, x, head_y + Inches(0.35), rows_bottom - head_y - Inches(0.35), color=FG_SUB,
               weight=Pt(1))
        tf = _textbox(slide, x - Inches(0.6), rows_bottom + Inches(0.05), Inches(1.2), Inches(0.3),
                      align=PP_ALIGN.CENTER)
        _lines(tf, [name], 11, font=F_REG, color=FG_SUB, align=PP_ALIGN.CENTER)

    _footer(slide, number)


def build_flow(slide, spec, number):
    _eyebrow(slide, spec['title'])

    card_w = Inches(3.3)
    gap = _emu((COL_W - 3 * card_w) / 2)
    top = Inches(2.35)
    height = Inches(2.5)

    for i, (tag, title, items) in enumerate(spec['cards']):
        left = _emu(M + i * (card_w + gap))
        _rect(slide, left, top, card_w, height, SURFACE, rounded=0.075)

        pad = Inches(0.34)
        tf = _textbox(slide, left + pad, top + Inches(0.36), card_w - 2 * pad, Inches(0.28))
        _lines(tf, [tag], 10.5, font=F_MED, color=FG_DIM, tracking=1.8)

        tf = _textbox(slide, left + pad, top + Inches(0.72), card_w - 2 * pad, Inches(0.4))
        _lines(tf, [title], 19, font=F_REG, color=FG)

        tf = _textbox(slide, left + pad, top + Inches(1.32), card_w - 2 * pad, Inches(1.0))
        _lines(tf, items, 13, font=F_LIGHT, color=FG_SUB, spacing=1.35, gap=7)

        if i < 2:
            tf = _textbox(slide, left + card_w, top + Inches(1.05), gap, Inches(0.5),
                          align=PP_ALIGN.CENTER)
            _lines(tf, ['→'], 20, font=F_LIGHT, color=FG_DIM, align=PP_ALIGN.CENTER)

    tf = _textbox(slide, M, Inches(5.45), COL_W, Inches(0.5))
    _lines(tf, [spec['caption']], 15, font=F_LIGHT, color=FG_SUB)

    _footer(slide, number)


def build_mockup(slide, spec, number):
    # 목업이 세로로 길어서 그냥 두면 오른쪽이 텅 빈다. 패널을 깔아 무게를 준다
    panel_left = Inches(8.1)
    panel_top = Inches(0.95)
    panel_w = SLIDE_W - M - panel_left
    panel_h = Inches(5.55)
    _rect(slide, panel_left, panel_top, panel_w, panel_h, SURFACE, rounded=0.05)

    img_h = Inches(4.85)
    pic = _picture(slide, os.path.join(MOCKUP_DIR, spec['image']), Inches(0),
                   _emu(panel_top + (panel_h - img_h) / 2), img_h)
    pic.left = _emu(panel_left + (panel_w - pic.width) / 2)

    text_w = Inches(6.4)

    _eyebrow(slide, spec['eyebrow'])

    tf = _textbox(slide, M, Inches(1.9), text_w, Inches(1.7))
    _lines(tf, spec['headline'].split('\n'), 33, font=F_THIN, spacing=1.28)

    _rule(slide, M, Inches(4.05), Inches(5.1))

    tf = _textbox(slide, M, Inches(4.35), text_w, Inches(1.7))
    _lines(tf, spec['lines'], 15, font=F_LIGHT, color=FG_SUB, spacing=1.3, gap=13)

    _footer(slide, number)


def build_columns(slide, spec, number):
    _eyebrow(slide, spec['title'])

    col_w = Inches(3.3)
    gap = _emu((COL_W - 3 * col_w) / 2)
    top = Inches(2.45)

    for i, (tag, head, sub) in enumerate(spec['columns']):
        left = _emu(M + i * (col_w + gap))

        tf = _textbox(slide, left, top, col_w, Inches(0.3))
        _lines(tf, [tag], 11.5, font=F_MED, color=FG_DIM, tracking=1.4)

        _rule(slide, left, top + Inches(0.45), col_w)

        tf = _textbox(slide, left, top + Inches(0.75), col_w, Inches(1.5))
        _lines(tf, head.split('\n'), 23, font=F_THIN, spacing=1.3)

        tf = _textbox(slide, left, top + Inches(2.25), col_w, Inches(1.0))
        _lines(tf, [sub], 13.5, font=F_LIGHT, color=FG_SUB, spacing=1.4)

    _footer(slide, number)


def build_hero_stat(slide, spec, number):
    """주장 하나를 크게 세우고, 근거와 규모 숫자를 그 아래에 받친다."""
    _eyebrow(slide, spec['eyebrow'])

    tf = _textbox(slide, M, Inches(1.65), Inches(11.0), Inches(1.8))
    _lines(tf, spec['headline'].split('\n'), 38, font=F_THIN, spacing=1.28)

    top = Inches(3.62)
    for text, source in spec['evidence']:
        tf = _textbox(slide, M, top, Inches(11.0), Inches(0.32))
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = text
        _set_font(run, 14.5, font=F_LIGHT, color=FG_SUB)
        cite = p.add_run()
        cite.text = '   ' + source
        _set_font(cite, 11, font=F_REG, color=FG_DIM)
        top += Inches(0.42)

    _rule(slide, M, Inches(4.95), COL_W)

    half = _emu(COL_W / 2)
    for i, (value, label, source) in enumerate(spec['support']):
        left = _emu(M + i * half)

        tf = _textbox(slide, left, Inches(5.2), _emu(half - Inches(0.4)), Inches(0.6))
        _lines(tf, [value], 30, font=F_THIN)

        tf = _textbox(slide, left, Inches(5.82), _emu(half - Inches(0.4)), Inches(0.4))
        _lines(tf, [label], 13, font=F_LIGHT, color=FG_SUB)

        tf = _textbox(slide, left, Inches(6.18), _emu(half - Inches(0.4)), Inches(0.35))
        _lines(tf, [source], 11, font=F_REG, color=FG_DIM)

    _footer(slide, number)


def build_closing(slide, spec, number):
    tf = _textbox(slide, M, Inches(2.85), COL_W, Inches(0.9), align=PP_ALIGN.CENTER)
    _lines(tf, [spec['title']], 42, font=F_THIN, align=PP_ALIGN.CENTER)

    tf = _textbox(slide, M, Inches(3.95), COL_W, Inches(0.5), align=PP_ALIGN.CENTER)
    _lines(tf, [spec['subtitle']], 15, font=F_LIGHT, color=FG_SUB, align=PP_ALIGN.CENTER,
           tracking=1.2)

    lockup = _picture(slide, _brand('fitcheck_lockup_v_dim.png'), Inches(0), Inches(5.05),
                      Inches(1.0))
    lockup.left = int((SLIDE_W - lockup.width) / 2)


def build_references(slide, spec, number):
    _eyebrow(slide, spec['eyebrow'])

    top = Inches(1.35)
    step = Inches(0.6)
    num_w = Inches(0.55)
    text_w = COL_W - num_w
    for i, (claim, citation, url) in enumerate(spec['refs'], start=1):
        tf = _textbox(slide, M, top, num_w, Inches(0.26))
        _lines(tf, [f'{i:02d}'], 10.5, font=F_MED, color=FG_DIM)

        tf = _textbox(slide, M + num_w, top, text_w, Inches(0.26))
        _lines(tf, [claim], 12, font=F_REG, color=FG)

        tf = _textbox(slide, M + num_w, top + Inches(0.25), text_w, Inches(0.3))
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = citation
        _set_font(run, 10, font=F_LIGHT, color=FG_SUB)
        if url:
            link = p.add_run()
            link.text = '   ' + url
            _set_font(link, 9.5, font=F_REG, color=FG_DIM)
        top += step

    tf = _textbox(slide, M + num_w, top + Inches(0.02), text_w, Inches(0.3))
    _lines(tf, [spec['note']], 10, font=F_LIGHT, color=FG_DIM)

    _footer(slide, number)


BUILDERS = {
    'cover': build_cover,
    'story': build_story,
    'hero_stat': build_hero_stat,
    'solution': build_solution,
    'columns': build_columns,
    'flow': build_flow,
    'mockup': build_mockup,
    'cards4': build_cards4,
    'gantt': build_gantt,
    'closing': build_closing,
    'references': build_references,
}


def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    elapsed = 0.0
    for i, spec in enumerate(SLIDES, start=1):
        slide = prs.slides.add_slide(blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = BG
        BUILDERS[spec['type']](slide, spec, i)
        if spec.get('notes'):
            notes, elapsed = _timed_notes(spec['notes'], elapsed)
            slide.notes_slide.notes_text_frame.text = notes

    prs.save(OUT_PATH)
    print(f'{len(SLIDES)}장 생성 완료 (대본 약 {int(elapsed // 60)}분 {int(elapsed % 60)}초) '
          f'-> {OUT_PATH}')


if __name__ == '__main__':
    main()
