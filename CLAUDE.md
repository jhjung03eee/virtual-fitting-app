# 사이즈 정보를 반영한 가상 피팅 앱

**성균관대 종합설계프로젝트(캡스톤).** 학부 졸업 과제이지 상용 제품이 아니다.
목표는 **발표에서 방어할 수 있는 결과물**이다 — 대단한 novelty가 아니라,
"무엇을 시도했고 무엇이 왜 안 됐는지"를 근거와 함께 보여주는 것.
그래서 **실패한 실험도 지우지 말고 기록으로 남긴다.**

사용자 사진 + 옷 사진 → diffusion 합성 + 옷 치수표 대비 사이즈 추천.

## 먼저 읽을 것

| 문서 | 내용 |
|---|---|
| `docs/PLAN.md` | 진행 현황, 남은 작업, **판정 기준**, 실험 결과 전부 |
| `docs/ENVIRONMENT.md` | 환경 함정 13건. **구동이 안 되면 여기부터** |
| `docs/TEST_RESULTS.md` | 벤치마크 45건 자동 생성 리포트 |

## 절대 어기면 안 되는 것

**1. Python 3.9 venv에서만 돌아간다.**
CatVTON 저장소에 **cp39 전용으로 컴파일된 detectron2 `.so`** 가 들어있다.
3.10+ 에서는 import 자체가 안 된다. `scripts/setup_env.py` 가 venv를 만든다.
"최신 환경으로 어떻게든 돌려보자"는 이미 여러 번 실패했다. 시도하지 말 것.

**2. 커밋 전에 추적 중인 `.py` 전체를 syntax 검사한다.**
테스트가 임포트하지 않는 스크립트는 깨져도 테스트가 통과한다. 실제로 그렇게
깨진 파일을 커밋한 적 있다.
```sh
python scripts/check_syntax.py          # 추적 중인 .py 전체
python -m unittest discover -s tests    # 117건
```

**3. 저작권·개인정보 파일은 커밋하지 않는다.**
`data/person/*` (사용자 얼굴), `data/samples/*` (쇼핑몰 상품 사진)은 `.gitignore` 처리돼 있다.
치수표에서 뽑은 **숫자**는 저작물이 아니므로 `data/size_charts/*.json` 으로 커밋해도 된다.

**4. heredoc으로 파이썬 파일을 패치하지 말 것.**
f-string 안의 `\n` 이 반복해서 망가졌다. Write/Edit 도구를 쓴다.

## 현재 기본값과 근거

`app/tryon_core.py` 상단 주석에 근거가 다 적혀 있다. 요약:

| 설정 | 값 | 이유 |
|---|---|---|
| 샘플러 | DPM++ | DDIM은 스텝을 줄이면 무너진다 |
| 스텝 | 8 | 50스텝과 눈으로 구별 안 됨. 5.8배 빠름 |
| guidance | 상의 2.5 / 하의 5.0 | 하의는 2.5에서 색이 아예 틀림 |
| `composite` | **켬** | 마스크 밖을 원본으로 복원. 아래 참고 |
| `normalize_background` | 끔 | 옷에 따라 갈림. 검증 중 |

**`composite`는 이 프로젝트에서 찾은 가장 중요한 수정이다.** CatVTON 파이프라인은
latent 전체를 디코딩해 돌려주므로 **마스크 밖도 다시 생성한다.** 얼굴·손·배경이
매번 뭉개지고 guidance를 올리면 급격히 나빠진다. 인페인팅 표준대로 마스크 밖을
원본 픽셀로 되돌리면 해결된다.

## 이미 폐기된 가설 — 다시 하지 말 것

| 시도 | 결과 |
|---|---|
| CFG 끄기 (2배 가속) | ❌ 사진 프린트 옷이 통째로 붕괴 |
| 스텝 줄이기 6·5·4 | ❌ 6스텝부터 프린트 소멸 |
| 스텝 **올리기** 30·50 | ❌ 8스텝과 차이 없음. 스텝은 변동 요인이 아니다 |
| `torch.compile` / `channels_last` | ❌ 효과 없음 (P100은 Triton 미지원, T4는 libcuda 문제) |
| 모델 교체 (IDM-VTON) | ❌ **기각.** 더 낫긴 하나 29GB·SDXL이라 T4에 못 들어갈 수 있음. `docs/PLAN.md` F절 |

**결과를 좌우하는 건 시드다.** 같은 설정에서 시드만 바꾸면 프린트가 나왔다 말았다 한다.
그래서 "설정 A에서 안 나온다"는 판단은 **시드 여러 개로 확인하기 전까지 하면 안 된다.**
seed=42 하나로 여섯 가지 설정을 비교하고 "모델 한계"라고 결론냈다가 뒤집힌 적 있다.

## 알려진 한계 (발표에서 그대로 보고할 것)

**사진 프린트는 재현되지 않는다.** 무지 옷과 큰 글자 프린트는 잘 되지만,
사진 같은 프린트는 작은 얼룩으로 나온다. IDM-VTON을 평가해 더 낫다는 것을
확인했으나 비용 때문에 교체하지 않기로 했다(`docs/PLAN.md` F절).
숨기지 말고 근거 이미지와 함께 한계로 보고한다.

## 판정 기준

"뭔가 나오면 성공"으로 보면 개선 여부를 알 수 없다. **전부** 만족해야 합격:
색조 / 프린트 형태 / 없던 얼룩 없음 / 실루엣 / 얼굴·배경 보존.
`docs/PLAN.md` 의 "판정 기준" 절 참고. 현재 성적은 후하지 않다.

## 실행 방법

GPU 작업은 전부 Kaggle 커널로 돌린다 (`kaggle_*/` 디렉터리).

```sh
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 kaggle kernels push -p kaggle_quality
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 kaggle kernels status jefferyjung/vfa-quality-ceiling
```

- **`PYTHONUTF8=1` 없으면 한글 때문에 cp949 에러가 난다** (Windows)
- 커널의 **GPU 종류는 API로 못 바꾼다.** 커널 페이지 → Session options → Accelerator
- `kernel-metadata.json` 의 `title` 슬러그와 `id` 가 다르면 커널은 title 쪽으로 만들어지고,
  `status` 가 엉뚱한 슬러그를 찾아 권한 오류처럼 보인다
- clone은 `/kaggle/tmp`, 결과만 `/kaggle/working` (안 그러면 output이 수천 개 파일)

GPU 없이 로컬에서 되는 것:
```sh
python scripts/compare_grid.py data/samples/results/<폴더> --garment-dir data/samples
python scripts/prepare_person.py data/person -o data/person/prepared
```
`compare_grid.py` 는 torch를 안 쓴다. Kaggle 결과로 비교 이미지를 다시 만들 때 쓴다.

## 구조

```
app/tryon_core.py     추론 단일 소스 (Gradio·벤치마크가 공유)
app/size_fit.py       여유분 계산 → S/M/L 추천 (순수 계산, GPU 불필요)
app/body_profile.py   사용자 입력 신체 치수. 둘레↔단면 변환
app/chart_extract.py  치수표 이미지 추출 결과 검증층 (네트워크 안 씀)
app/gradio_app.py     웹 UI
scripts/              벤치마크·비교·전처리
kaggle_*/             Kaggle 커널 진입점
```

사이즈 파트는 **사용자가 직접 치수를 입력**한다. 사진으로 몸을 재는
`app/body_measure.py` 는 보조 기능으로만 남아 있다(정확도를 신뢰할 수 없음).
