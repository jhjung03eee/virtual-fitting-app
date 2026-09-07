# 사이즈 정보를 반영한 가상 피팅 앱

종합설계프로젝트 — 사용자 사진과 옷 이미지를 diffusion으로 합성하고, 옷 치수와 체형을 비교해
"이 사이즈를 사면 이렇게 핏이 나온다"를 보여주는 앱.

모델은 **직접 구동(self-hosted)** 한다. 외부 API는 쓰지 않는다.

## 진행 상황

- [x] CatVTON 파이프라인 구동 검증 (Colab T4, 30스텝 63초, 옷 합성 성공)
- [x] 환경 구성 문제 전부 규명 및 문서화 → `docs/ENVIRONMENT.md`
- [ ] AutoMasker(DensePose+SCHP) 포함 원본 그대로 구동 (Python 3.9 venv)
- [ ] 팀원 실제 사진 + 쇼핑몰 옷 이미지로 테스트, 실패 케이스 수집
- [ ] MediaPipe Pose로 체형(어깨너비 등) 추정 + 키 기반 스케일 보정
- [ ] 옷 치수표 대비 여유분 계산 → S/M/L 추천
- [ ] 모바일 앱 UI + 추론 서버 연동

## 모델 구성

| 역할 | 사용 |
|---|---|
| 가상 피팅 합성 | [CatVTON](https://github.com/Zheng-Chong/CatVTON) (HF: `zhengchong/CatVTON`) |
| 베이스 diffusion | `runwayml/stable-diffusion-inpainting` |
| 옷 영역 마스크 자동 생성 | CatVTON 내장 `AutoMasker` (DensePose + SCHP) |
| 체형 추정 (예정) | MediaPipe Pose |

## 구성

```
notebooks/catvton_tryon.ipynb        Colab/Kaggle용 추론 노트북
scripts/setup_py39_and_run.py        Python 3.9 venv 구성 + AutoMasker 포함 추론 (권장 경로)
scripts/colab_minimal_rectmask.py    최신 환경에서 detectron2 없이 돌리는 최소 버전 (검증됨)
kaggle/                              Kaggle CLI로 커널 푸시해 돌린 기록
docs/ENVIRONMENT.md                  환경 구성 시행착오 전부 (필독)
data/person, data/garment            테스트 이미지
outputs/                             결과
```

## 빠른 시작 (Colab)

1. Colab에서 새 노트북 → **런타임 유형 변경 → T4 GPU** (P100은 최신 torch에서 지원 종료됨)
2. `scripts/setup_py39_and_run.py` 내용을 셀에 붙여넣고 실행

Python 3.9 환경을 따로 만드는 이유는 `docs/ENVIRONMENT.md` 참고 —
**repo에 cp39 전용으로 컴파일된 detectron2 바이너리가 들어있어서 Python 3.12에서는 import가 안 된다.**

## 주의

- GPU는 T4(16GB) 이상. 1024×768 추론에 약 8GB 사용.
- 가중치는 HuggingFace에서 자동 다운로드 (최초 1회 약 4GB).
