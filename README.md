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

| Model | recipe |  PPL ↓ | MMLU ↑ |
|-------|--------|-------|--------|
| Original | pcq_spinquant_adascale | 35.384281158447266 | - |
| Original | lpbq_seqmse | 13.811701774597168 | - |
| SmoothQuant | pcq_spinquant_adascale | 26.687795639038086 | - |
| SmoothQuant | lpbq_seqmse | 13.760761260986328 | - |
| MobileQuant | pcq_spinquant_adascale | - | - |
| MobileQuant | lpbq_seqmse | - | - |

You can check models in [huggingface collection](https://huggingface.co/collections/soiji/aimet-quant-test)
