# AIMET Playground

A benchmarking project to evaluate the effectiveness of **SmoothQuant** and **MobileQuant** pre-processing techniques on LLM quantization using [AIMET](https://github.com/quic/aimet) (AI Model Efficiency Toolkit).

This repo is temporal and will be removed and documented in GIST after all experiment is finished

## Objective

Investigate whether SmoothQuant and MobileQuant pre-processing can improve quantization quality compared to direct quantization of the original model. This is measured through perplexity (PPL) and MMLU benchmark scores.

## Target Model

- **Llama 3.2 1B Instruct** (`meta-llama/Llama-3.2-1B-Instruct`)

## Experiment Design

### Model Variants

| # | Model Variant | Description |
|---|---------------|-------------|
| 1 | [Original](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Base Llama 3.2 1B Instruct model |
| 2 | [SmoothQuant](https://huggingface.co/soiji/Llama-3.2-1B-Instruct_SmoothQuant) | Pre-processed with [llmcompressor](https://github.com/vllm-project/llm-compressor) SmoothQuant |
| 3 | [MobileQuant](https://huggingface.co/soiji/Llama-3.2-1B-Instruct_Mobilequant) | Pre-processed with [MobileQuant](https://github.com/saic-fi/MobileQuant) technique |


### Quantization Schemes

Each model variant is quantized using AIMET [Recipe](https://quic.github.io/aimet-pages/releases/latest/tutorials/quantization_recipe.html) with two configurations:

## Results

### Test to verify whether the same workflow can be applied simply by replacing the current model with a SmoothQuant version.
| Model | recipe |  PPL(aimet.Examples.torch.evaluate) ↓ | MMLU(aimet.Examples.onnx.evaluate) ↑ |
|-------|--------|-------|--------|
| Original | pcq_spinquant_adascale | 13.658965110778809 | - |
| Original | lpbq_seqmse | 13.811701774597168 | - |
| SmoothQuant | pcq_spinquant_adascale | 13.576032638549805 | 41.72482552342971 |
| SmoothQuant | lpbq_seqmse | 13.760761260986328 | - |
| MobileQuant | pcq_spinquant_adascale | - | - |
| MobileQuant | lpbq_seqmse | - | - |

### Test the impact of technique
| Model | quantization |  PPL(aimet.Examples.torch.evaluate) ↓ | MMLU(aimet.Examples.onnx.evaluate) ↑ |
|-------|--------|-------|--------|
| Original | w4a16 | - | - |
| Original | w8a8 | - | - |
| SmoothQuant | w4a16 | - | - |
| SmoothQuant | w8a8 | - | - |
| MobileQuant | w4a16 | - | - |
| MobileQuant | w8a8 | - | - |
| OmniQuant | w4a16 | - | - |
| OmniQuant | w8a8 | - | - |
| SpinQuant | w4a16 | - | - |
| SpinQuant | w8a8 | - | - |


You can check models in [huggingface collection](https://huggingface.co/collections/soiji/aimet-quant-test)

