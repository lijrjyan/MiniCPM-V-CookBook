# MiniCPM-o 4.5 — Overview

The omni-modal flagship of the MiniCPM-o series — vision, speech, and full-duplex live streaming, all in a single model.

## Highlights

- **End-to-end omnimodal** — vision encoder + speech encoder + LLM + TTS densely connected via hidden states
- **Full-duplex live streaming** — hear, see, and speak simultaneously
- **Voice cloning** and configurable assistant voice
- HuggingFace: [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5) · ModelScope: [`OpenBMB/MiniCPM-o-4_5`](https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5)

For real-time, full-duplex demos see the [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) repository — it ships a turn-key web demo for the o-series with PyTorch + CUDA inference and a frontend.

## Where to start

- **Audio recipes:** [Speech-to-Text](inference/speech2text.html), [Text-to-Speech](inference/text2speech.html), [Voice Cloning](inference/voice-clone.html)
- **Deploy:** [vLLM](deployment/vllm.html), [SGLang (sglang-omni, full-duplex)](deployment/sglang.html), [llama.cpp](deployment/llamacpp.html), [Ollama](deployment/ollama.html)
- **Quantize:** [GGUF](quantization/gguf.html)
