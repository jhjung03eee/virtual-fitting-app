# FitCheck — 사이즈 정보를 반영한 가상 피팅 앱

종합설계프로젝트(캡스톤). 사용자 사진과 옷 사진을 합성해 입은 모습을 보여주고,
옷 치수표와 입력한 신체 치수를 비교해 S/M/L을 추천한다.

모델은 **직접 구동(self-hosted)** 한다. 외부 합성 API는 쓰지 않는다.

> - 전체 진행 현황·실험 결과·판단 근거: **[docs/PLAN.md](docs/PLAN.md)**
> - 환경 함정 모음: **[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)**

## 팀원용 빠른 시작 — 데모 보기

[![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhjung03eee/virtual-fitting-app/blob/master/notebooks/run_app.ipynb)

1. 위 배지를 눌러 Colab에서 `notebooks/run_app.ipynb` 를 연다
2. **런타임 → 런타임 유형 변경 → GPU (T4)**
3. 셀을 위에서부터 실행 → 마지막 셀 출력의 **공개 링크(`*.gradio.live`)** 를 연다

최초 1회 설치·가중치 내려받기에 약 5분. 합성은 T4 기준 한 장 약 2분(품질 우선 설정).
사진은 **정면 · 팔은 몸에서 주먹 하나 간격 · 머리부터 발끝까지 화면 가득** 이어야 한다
(앱이 올리는 순간 검사해서 안내한다).

## 진행 상황

- [x] 합성 모델 구동·비교: CatVTON → HR-VITON → 하이브리드 → **FASHN VTON v1.5 채택**, FitDiT 비교 후 보류
- [x] 환경 구성 문제 전부 규명·문서화 → `docs/ENVIRONMENT.md`
- [x] 속도/품질 실험: 정밀도·스텝·guidance 실측, Kaggle T4 2장 병렬 실험 커널
- [x] 옷 치수표 대비 여유분 계산 → S/M/L 추천 (사용자 입력 방식, 단위 테스트)
- [x] 사진 자세·거리 검사(`app/photo_check.py`)를 앱에 연결
- [x] **앱에 FASHN 연결 + 실제 GPU에서 합성·사이즈 추천·서버 기동 검증** (docs/PLAN.md F-17)
- [ ] 촬영 가이드대로 찍은 새 전신 사진으로 **하의(바지)** 확인 (F-12)
- [ ] 고성능 GPU(A100 등)에서 속도 측정, 시연 환경 확정
- [ ] 모바일 앱 UI + 추론 서버 분리

## 모델 구성

| 역할 | 사용 |
|---|---|
| 가상 피팅 합성 (**현재**) | [FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5) (Apache-2.0, 마스크 없이 픽셀 공간 생성) |
| 기본 설정 | fp16 · 50스텝 · guidance 2.5 (`app/fashn_core.py`, 근거는 docs/PLAN.md F-11·F-13·F-14) |
| 가상 피팅 합성 (이전) | [CatVTON](https://github.com/Zheng-Chong/CatVTON) — `app/tryon_core.py` 에 남아 있음(벤치마크·과거 실험 재현용) |
| 사진 자세·거리 검사 | MediaPipe Pose (`app/photo_check.py`) |
| 사이즈 추천 | 치수표 대비 여유분 계산 (`app/size_fit.py`, `app/body_profile.py`) |

## 구성

```
app/fashn_core.py          합성 엔진 (FASHN). 앱·실험이 공유하는 기본값
app/gradio_app.py          웹 데모 화면 (합성 + 사진 검사 + 사이즈 추천)
app/photo_check.py         인물 사진 자세·거리·역광 검사 (GPU 불필요)
app/size_fit.py            치수표 대비 여유분 계산 → S/M/L 추천
app/tryon_core.py          CatVTON 추론 (과거 실험 재현용)
notebooks/run_app.ipynb    Colab에서 데모 띄우기
kaggle_*/                  Kaggle 커널로 돌린 실험 기록 (모델 비교·속도·품질)
docs/PLAN.md               진행 현황과 모든 실험 결과·판단 근거
data/samples/results/      실험 결과 이미지 (커밋 제외)
```

## CatVTON(이전 엔진)을 돌려야 할 때

과거 실험을 재현할 때만 필요하다. CatVTON 저장소에 **cp39 전용으로 컴파일된 detectron2 바이너리**가
들어 있어 **Python 3.9 venv** 에서만 돌아간다(`scripts/setup_env.py` 가 만든다). 자세한 이유는
`docs/ENVIRONMENT.md`. 현재 앱(FASHN)은 반대로 Python 3.10 이상이 필요하므로 두 엔진은 같이 못 쓴다.

## 주의

- GPU는 T4(16GB) 권장. FASHN 가중치는 약 2GB(최초 1회 자동 다운로드), 추론에 fp16 기준 3.2GB 사용.
- 사람 얼굴 사진(`data/person/`)과 쇼핑몰 상품 사진(`data/samples/`)은 커밋하지 않는다(.gitignore).
