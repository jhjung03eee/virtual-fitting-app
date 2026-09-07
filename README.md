# 사이즈 정보를 반영한 가상 피팅 앱

종합설계프로젝트 — 사용자 사진과 옷 이미지를 diffusion 기반으로 합성하고, 옷 치수와 체형을 비교해 사이즈별 핏을 보여주는 앱.

## 현재 단계

1차 목표: **사람 사진 + 옷 사진 → 합성 이미지** 프로토타입 (CatVTON)

사이즈표 매칭 / 여유분 계산 / MediaPipe Pose / 모바일 앱 UI는 다음 단계.

## 구성

```
notebooks/catvton_tryon.ipynb   Colab·Kaggle에서 실행하는 CatVTON 추론 노트북
data/person/                    테스트용 인물 사진 (정자세 전신)
data/garment/                   테스트용 옷 이미지
outputs/                        합성 결과 저장
```

## 실행 방법

`notebooks/catvton_tryon.ipynb`를 Colab 또는 Kaggle에 업로드해서 위에서부터 실행.

- **Colab**: Gradio 셀까지 실행하면 공개 링크가 나와서 웹 UI로 이미지 업로드 테스트 가능
- **Kaggle**: 노트북 설정에서 Internet을 켜야 함. Gradio 공개 링크는 막히므로 배치 추론 셀 사용

GPU는 16GB(T4, P100) 이상이면 1024×768 해상도로 동작.

## 참고

- CatVTON: https://github.com/Zheng-Chong/CatVTON
- 가중치: HuggingFace `zhengchong/CatVTON` (자동 다운로드)
- 베이스 모델: `runwayml/stable-diffusion-inpainting`
