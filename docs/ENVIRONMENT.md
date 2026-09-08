# 환경 구성 기록 — CatVTON을 실제로 돌리기까지

2026-09-08 기준. Kaggle/Colab에서 CatVTON을 돌리며 실제로 부딪힌 문제와 해결책 전부.
**결론부터: repo가 요구하는 Python 3.9 환경을 그대로 만드는 게 정답이다.** 최신 환경에 억지로
끼워맞추면 아래 문제들을 순서대로 다 밟게 된다.

## 핵심 원인

CatVTON repo는 2024년 환경(Python 3.9.0 + torch 2.1.2)을 전제로 만들어졌고,
**detectron2를 소스 빌드하지 않고 cp39용으로 미리 컴파일된 `.so` 바이너리를 repo에 포함**시켰다.

```
CatVTON/detectron2/_C.cpython-39-x86_64-linux-gnu.so   <- Python 3.9 전용
```

컴파일된 확장 모듈은 파이썬 버전이 다르면 ABI가 안 맞아 import 자체가 불가능하다.
2026년 현재 Colab/Kaggle 기본은 **Python 3.12 + torch 2.10**이라 그대로는 절대 안 된다.

## 밟은 지뢰들 (시간순)

### 1. `ModuleNotFoundError: No module named 'fvcore'`
`AutoMasker` → `model/DensePose` → `densepose` → `detectron2` 체인에서 발생.
repo의 `requirements.txt`에 **detectron2 런타임 의존성이 아예 빠져 있다.**

추가로 필요: `fvcore iopath pycocotools omegaconf hydra-core termcolor yacs tabulate cloudpickle`

### 2. Kaggle P100 GPU가 최신 torch에서 지원 종료
```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
```
torch 2.10은 Pascal(sm_60) 지원을 뺐다. Kaggle이 P100을 배정하면 CPU로 떨어진다.
→ **T4(sm_75) 이상을 쓰거나, repo가 고정한 torch 2.1.2를 쓰면 P100도 동작한다.**

### 3. `transformers==4.27.3` 설치 실패 (Python 3.12에서)
```
Building wheel for tokenizers (pyproject.toml) did not run successfully
```
구버전 tokenizers는 cp312용 prebuilt wheel이 없어 소스 빌드를 시도하다 실패.
**Python 3.9에서는 cp39 wheel이 있어 정상 설치된다.**

### 4. `huggingface_hub==0.23.4`로 다운그레이드 → 최신 transformers 파손
```
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```
베이스 이미지의 최신 transformers/accelerate/datasets가 최신 huggingface_hub API에 의존하는데,
구버전으로 강제 다운그레이드하면 그 위에 얹힌 것들이 다 깨진다.
→ **최신 환경에서는 버전을 낮추지 말고, 격리된 venv 안에서만 맞출 것.**

### 5. `StableDiffusionSafetyChecker` API 불일치
```
AttributeError: 'StableDiffusionSafetyChecker' object has no attribute 'all_tied_weights_keys'
```
diffusers 0.29.2(2024)의 safety checker를 transformers 5.x(2026)가 로드하면서 발생.
→ `CatVTONPipeline(..., skip_safety_check=True)` 로 회피 가능 (repo에 이미 옵션 있음).

### 6. `ModuleNotFoundError: No module named 'av'`
densepose의 `data/video/video_keyframe_dataset.py`가 PyAV를 import한다.
가상 피팅에 비디오는 안 쓰지만 **import 체인에 걸려 있어서 반드시 설치해야 한다.**
이것도 repo `requirements.txt`에 없다. → `pip install av`

### 7. `ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value`
Colab이 `MPLBACKEND` 환경변수를 inline 백엔드로 설정해두는데, 이게 subprocess로 상속되면
venv 안의 matplotlib이 그 백엔드를 못 찾아 죽는다.
→ 워커 실행 시 `MPLBACKEND=Agg`로 덮어쓸 것.

### 8. `TypeError: argument of type 'bool' is not iterable` (gradio)
Gradio 앱을 띄우면 페이지가 안 열리고 이 에러가 반복된다.
pydantic이 만드는 JSON 스키마에는 `additionalProperties: true` 처럼 값이 dict가 아니라
bool인 항목이 있는데, `gradio_client/utils.py`의 `get_type()`이 dict로 가정하고
`"const" in schema` 를 실행해서 터진다.

- gradio 4.44.1로 올려도 발생한다.
- `launch(show_api=False)` 로도 못 막는다 — 메인 라우트(`routes.py`)가 옵션과 무관하게
  `api_info()`를 호출한다.
- **해결**: venv에 설치된 `gradio_client/utils.py`의 `get_type` / `_json_schema_to_python_type`
  앞에 `if not isinstance(schema, dict): return "Any"` 가드를 삽입한다.
  `scripts/run_gradio_colab.py`가 실행 시 자동으로 패치한다(멱등).

### 9. Gradio 공개 링크가 출력되지 않음
`로딩 완료` 까지 찍히고 `Running on public URL: ...` 이 안 보인다. 앱은 정상 실행 중인데
**출력 버퍼링** 때문에 안 보이는 것.
자식 프로세스의 stdout이 파이프면 블록 버퍼링(4KB)이라, `flush=True` 없는 gradio 내부
print가 버퍼에 갇힌다.
→ 자식을 `python -u` 로 실행하고 `PYTHONUNBUFFERED=1` 을 준다. 추가로 `launch(prevent_thread_lock=True)`
가 반환하는 URL을 직접 `flush=True`로 출력한 뒤 `demo.block_thread()` 로 대기한다.

## 검증된 실행 방법

### A. Colab + Python 3.9 venv (권장, AutoMasker 포함 전부 동작)

`scripts/setup_py39_and_run.py` 참고. 요지:

```bash
add-apt-repository -y ppa:deadsnakes/ppa
apt-get install -y python3.9 python3.9-venv python3.9-dev
python3.9 -m venv /content/venv39
/content/venv39/bin/pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
/content/venv39/bin/pip install accelerate==0.31.0 diffusers==0.29.2 huggingface_hub==0.23.4 \
    transformers==4.27.3 numpy==1.26.4 opencv-python==4.10.0.84 pillow==10.3.0 PyYAML==6.0.1 \
    scipy==1.13.1 scikit-image==0.24.0 tqdm==4.66.4 matplotlib==3.9.1 \
    fvcore iopath pycocotools omegaconf hydra-core termcolor yacs tabulate cloudpickle
```
추가로 `av`도 설치해야 한다. 그 다음 추론 스크립트를
`MPLBACKEND=Agg /content/venv39/bin/python worker.py` 로 실행 (노트북 커널이 아니라 subprocess).

**검증 완료 (2026-09-08, Colab T4)**:
- 환경 구성 219초 (Python 3.9 설치 21초, torch 2.1.2 171초, 나머지 48초)
- `pipeline + automasker ready` → 추론 30스텝 71초 (2.40s/it)
- **AutoMasker가 사람 몸 형태에 맞는 마스크를 정상 생성**, 배경 아티팩트 없는 결과 확인
- venv의 torch 2.1.2는 `sm_50~sm_80` 지원 → **이 환경이면 Kaggle P100도 쓸 수 있다**

### B. 최신 환경 + 사각형 마스크 (검증 완료, 품질 제한)

`scripts/colab_minimal_rectmask.py`. detectron2/AutoMasker를 전혀 안 쓰고
`skip_safety_check=True` + 직접 만든 사각형 마스크로 동작시킨 버전.

**결과**: T4에서 30스텝 63초. 옷 합성 자체는 성공(질감·단추·프린트까지 재현)하지만,
사각형 마스크가 몸 바깥 배경까지 덮어서 인물 뒤에 갈색 사각형 아티팩트가 남는다.

**같은 코드를 CPU + 4스텝으로 돌리면** 형체 없는 갈색 덩어리만 나온다 (Kaggle 1차 시도).
diffusion은 스텝 수가 부족하면 수렴 자체를 못 한다.

## 성능 참고 (T4, 768x1024, 30 steps)

| 항목 | 시간 |
|---|---|
| 가중치 다운로드 (최초 1회) | ~90초 (SD inpainting 3.44GB + CatVTON) |
| 추론 30스텝 | 63초 (2.10s/it) |

## Kaggle CLI 자동화 메모

로컬에서 커널을 밀어넣고 결과를 받는 방법 (`kaggle/` 참고):
```bash
kaggle kernels push -p kaggle/          # kernel-metadata.json 필요
kaggle kernels status <owner>/<slug>
```
- Windows에서는 `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` 없으면 cp949 인코딩 에러가 난다.
- 로그만 빠르게 받으려면 `api.kernels_logs(...)` (파일 다운로드 없이 로그만).
- `/kaggle/working`에 clone하면 repo 전체가 kernel output으로 잡혀 다운로드가 지옥이 된다.
  **clone은 `/kaggle/tmp`에, 결과만 `/kaggle/working`에 저장할 것.**
