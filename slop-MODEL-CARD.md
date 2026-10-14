---
license: mit
language: en
pipeline_tag: text-classification
tags:
  - ai-text-detection
  - deberta-v3
  - onnx
  - webgpu
  - raid
library_name: onnx
---

# SLOP — AI-text detector (DeBERTa-v3 + FeatAttn)

Sub-200M-parameter detector for machine-generated text. DeBERTa-v3-base encoder
fused with a Feature-Attention head over 17 surface stylometric features
(readability, type-token ratio, sentence-length CV, …). **No reference LM** at
inference — runs fully self-contained, including in the browser on WebGPU.

- **Params:** 184.2M (embeddings 98.8M / encoder 85.1M / heads 0.36M)
- **Training:** HC3 + ai-text-detection-pile + RAID train (in-distribution),
  focal loss (γ=2.0, α=0.85)
- **Inputs:** `input_ids [B,S]`, `attention_mask [B,S]`, `features [B,17]`
- **Output:** `prob [B]` = P(AI) (sigmoid already applied)

## Files
| file | notes |
| --- | --- |
| `model.safetensors` | fp32 weights |
| `onnx/model_fp32.onnx` | full precision |
| `onnx/model_fp16.onnx` | half precision |
| `onnx/model_int8.onnx` | 8-bit dynamic |
| `onnx/model_q4.onnx` | **4-bit (MatMulNBits) + int8 embeddings — for WebGPU/onnxruntime-web** |

## Browser usage (onnxruntime-web, WebGPU)
The `q4` graph is `MatMulNBits` 4-bit matmuls + an int8-quantised embedding table
(~170 MB). Feed `input_ids`/`attention_mask` from the DeBERTa tokenizer and the
17-d feature vector; the graph returns `prob`.

Built with [slopdetector](https://github.com/anudit/slopdetector).
