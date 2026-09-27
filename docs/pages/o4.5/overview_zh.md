# MiniCPM-o 4.5 — 概览

MiniCPM-o 系列的全模态旗舰版本 —— 视觉、语音、全双工实时流媒体一体化。

## 主要特性

- **端到端全模态** —— 视觉编码器 + 语音编码器 + LLM + TTS 通过隐状态紧耦合
- **全双工实时流媒体** —— 同时听、看、说
- **语音克隆** 与可配置的助手声线
- HuggingFace：[`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5) · 魔搭：[`OpenBMB/MiniCPM-o-4_5`](https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5)

实时全双工 web demo 请参考 [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) 仓库 —— 一套基于 PyTorch + CUDA 的开箱即用 demo，含前后端代码。

## 入口

- **音频示例：** [语音转文本](inference/speech2text.html)、[文本转语音](inference/text2speech.html)、[语音克隆](inference/voice-clone.html)
- **部署：** [vLLM](deployment/vllm.html)、[SGLang（sglang-omni 全双工）](deployment/sglang.html)、[llama.cpp](deployment/llamacpp.html)、[Ollama](deployment/ollama.html)
- **量化：** [GGUF](quantization/gguf.html)
