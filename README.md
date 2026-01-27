# AIMET Playground

A benchmarking project to evaluate the effectiveness of **SmoothQuant** and **MobileQuant** pre-processing techniques on LLM quantization using [AIMET](https://github.com/quic/aimet) (AI Model Efficiency Toolkit).

## Objective

Investigate whether SmoothQuant and MobileQuant pre-processing can improve quantization quality compared to direct quantization of the original model. This is measured through perplexity (PPL) and MMLU benchmark scores.

## Target Model

- **Llama 3.2 1B Instruct** (`meta-llama/Llama-3.2-1B-Instruct`)

## Experiment Design

### Model Variants

| # | Model Variant | Description |
|---|---------------|-------------|
| 1 | Original | Base Llama 3.2 1B Instruct model |
| 2 | SmoothQuant | Pre-processed with [llmcompressor](https://github.com/vllm-project/llm-compressor) SmoothQuant |
| 3 | MobileQuant | Pre-processed with [MobileQuant](https://github.com/saic-fi/MobileQuant) technique |

### Quantization Schemes

Each model variant is quantized using AIMET [Recipe 2](https://quic.github.io/aimet-pages/releases/latest/tutorials/quantization_recipe.html) with two configurations:

| Scheme | Weights | Activations |
|--------|---------|-------------|
| W4A16  | 4-bit   | 16-bit (FP16) |
| W8A8   | 8-bit   | 8-bit |

### Total Experiment Matrix

| Model Variant | W4A16 | W8A8 |
|---------------|-------|------|
| Original      | [✓](https://huggingface.co/soiji/Llama-3.2-1B-Instruct_SmoothQuant)     | ✓    |
| SmoothQuant   | ✓     | ✓    |
| MobileQuant   | ✓     | ✓    |

**Total: 6 quantized models**

## Results

The experiment aims to produce a comparison table like:

| Model | Quantization | PPL ↓ | MMLU ↑ |
|-------|--------------|-------|--------|
| Original | W4A16 | - | - |
| Original | W8A8 | - | - |
| SmoothQuant | W4A16 | - | - |
| SmoothQuant | W8A8 | - | - |
| MobileQuant | W4A16 | - | - |
| MobileQuant | W8A8 | - | - |

## References

- [AIMET Documentation](https://quic.github.io/aimet-pages/)
- [AIMET Quantization Recipes](https://quic.github.io/aimet-pages/releases/latest/tutorials/quantization_recipe.html)
- [SmoothQuant Paper](https://arxiv.org/abs/2211.10438)
- [MobileQuant Paper](https://arxiv.org/abs/2312.11514)
- [llmcompressor](https://github.com/vllm-project/llm-compressor)

## License

See LICENSE file for details.