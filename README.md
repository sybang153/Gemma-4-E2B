# Gemma 4 E2B 활성화 뷰어

Gemma 4 E2B(QAT Q4_0 GGUF)와 대화하면서 모델의 전체 구조와 뉴런 활성화를 실시간으로 보는 도구입니다.
Google과 무관한 비공식 프로젝트입니다.

**바로 쓰기: https://sybang153.github.io/Gemma-4-E2B** — 설치 없이 브라우저의 WebGPU로 모델을 직접 실행합니다.

## 기능

- **구조도**: 35개 층의 어텐션·MLP·PLE(층별 임베딩)·잔차를 위에서 아래로 그리고, 각 모듈 칸을 그 순간의 활성값 히트맵으로 채웁니다.
- **확대 보기**: 모듈을 누르면 모든 값을 한 칸씩 크게 보여 줍니다. 파란 테두리 모듈은 칸을 눌러 그 뉴런을 해석할 수 있습니다.
- **뉴런 해석**: MLP 뉴런(GELU(gate)×up)과 PLE 뉴런(GELU(inp_gate)×층별 임베딩)을 골라 다음을 봅니다.
  - 위키백과 한·영 말뭉치에서 가장 강하게 발화한 예시 (± 양극)
  - 출력 가중치가 직접 밀어 올리고 내리는 토큰
  - 뉴런을 끄거나 조향했을 때 다음 토큰 확률·생성 결과의 변화

## 웹 버전 (GitHub Pages)

- 데스크톱 Chrome·Edge 최신 버전(WebGPU)과 넉넉한 GPU 메모리(2 GB 이상)가 필요합니다.
- 처음 한 번 Hugging Face `google/gemma-4-E2B-it-qat-q4_0-gguf`에서 모델(3.35 GB)을 받아 브라우저 저장소에 보관합니다. 이미 받아 둔 GGUF 파일을 골라 쓸 수도 있습니다. 저장 공간이 모자라면 저장하지 않고 필요한 부분(약 1.4 GB)만 매번 받아 실행합니다.
- 추론은 `engine.js`(GGUF 읽기, 토크나이저, WebGPU 커널), 서버 역할은 `web.js`가 맡습니다. 토큰화는 llama.cpp와 같고, 활성화는 f32로 계산해 llama.cpp CPU 결과와 1~5% 정도 다릅니다(그리디 생성은 같은 결과를 확인했습니다). 컨텍스트는 4096 토큰입니다.
- 뉴런 해석의 말뭉치 색인은 로컬에서 미리 만든 `data/`를 씁니다.

## 로컬 서버 버전

llama.cpp(llama-cpp-python, CPU)로 실행하며 GPU가 필요 없습니다. Windows와 Python 3.13에서 확인했습니다. llama.cpp의 `ggml-base.dll`을 직접 읽으므로 현재 Windows 전용입니다.

```bash
pip install -r requirements.txt
python server.py
```

브라우저에서 http://127.0.0.1:8800 을 엽니다.

첫 실행 때 하는 일:

- Hugging Face `google/gemma-4-E2B-it-qat-q4_0-gguf`에서 모델(약 3.4 GB)을 받습니다. Gemma 라이선스 동의와 `huggingface-cli login`이 필요할 수 있습니다.
- `wikimedia/wikipedia`(2023-11-01 한국어·영어)에서 문서 160개의 앞부분을 받아, 모든 MLP·PLE 뉴런의 발화를 한 번 색인합니다. 노트북 CPU에서 약 7~8분 걸리며, 그동안 대화는 기다립니다. 결과는 `cache/`에 저장되어 다음 실행부터 바로 불러옵니다.

웹 버전의 `data/`는 이 색인으로 `python build_web_data.py`를 실행해 만듭니다.

## 라이선스

코드는 [MIT 라이선스](LICENSE)입니다. 모델 가중치는 이 저장소에 포함되지 않으며 Gemma 이용 약관을 따릅니다. 위키백과 문서와 그 토큰·색인(`data/`)은 CC BY-SA 4.0입니다.
