# Bitsandbytes for Qwen3-8B model
This repo provides a script that applies **4-bit** or **8-bit** quantization on the Qwen3-8B (or any Hugging Face LLM) with the **bitsandbytes** library.

It also evaluates and compares the latency, TPS, and peak memory for both original model and quantised model and automatically pushes the quantized model to your Hugging Face account.

Before running this code, please download your preferenced LLM to your local computer from huggingface.

## Reference
The code is adapted from the official HF blog post:
https://huggingface.co/blog/4bit-transformers-bitsandbytes