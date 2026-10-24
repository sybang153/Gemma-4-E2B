# Gemma 4 E2B 활성화 뷰어

Gemma 4 E2B(QAT Q4_0 GGUF)와 대화하면서 모델의 전체 구조와 뉴런 활성화를 실시간으로 보는 로컬 도구입니다.
llama.cpp(llama-cpp-python, CPU)로 실행하며 GPU가 필요 없습니다. Google과 무관한 비공식 프로젝트입니다.

## 기능

- **구조도**: 35개 층의 어텐션·MLP·PLE(층별 임베딩)·잔차를 위에서 아래로 그리고, 각 모듈 칸을 그 순간의 활성값 히트맵으로 채웁니다.
- **확대 보기**: 모듈을 누르면 모든 값을 한 칸씩 크게 보여 줍니다.
- **뉴런 해석**: MLP 뉴런(GELU(gate)×up)과 PLE 뉴런(GELU(inp_gate)×층별 임베딩)을 골라 다음을 봅니다.
  - 위키백과 한·영 말뭉치에서 가장 강하게 발화한 예시 (± 양극)
  - 출력 가중치가 직접 밀어 올리고 내리는 토큰
  - 뉴런을 끄거나 조향했을 때 다음 토큰 확률·생성 결과의 변화

## 실행

Windows와 Python 3.13에서 확인했습니다. llama.cpp의 `ggml-base.dll`을 직접 읽으므로 현재 Windows 전용입니다.

```bash
pip install -r requirements.txt
python server.py
```

브라우저에서 http://127.0.0.1:8800 을 엽니다.

첫 실행 때 하는 일:

- Hugging Face `google/gemma-4-E2B-it-qat-q4_0-gguf`에서 모델(약 3.4 GB)을 받습니다. Gemma 라이선스 동의와 `huggingface-cli login`이 필요할 수 있습니다.
- `wikimedia/wikipedia`(2023-11-01 한국어·영어)에서 문서 160개의 앞부분을 받아, 모든 MLP·PLE 뉴런의 발화를 한 번 색인합니다. 노트북 CPU에서 약 7~8분 걸리며, 그동안 대화는 기다립니다. 결과는 `cache/`에 저장되어 다음 실행부터 바로 불러옵니다.
