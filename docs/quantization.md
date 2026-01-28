## Model preperation

### Smoothquant
```bash TODO: @sooh0601 fill command or python script that used for Smoothquant

```

### Mobilequant
```bash TODO: @sooh0601 fill command or python script that used for Mobilequant

```

## Quantization
LPBQ example(original model):
```bash
cd aimet && \
python -m Examples.torch.quantize  \
--model-id "meta-llama/Llama-3.2-1B-Instruct"  \
--recipe "lpbq_seqmse"  \
--export-path "./Llama-3.2-1B-Instruct_w8a8_LPBQ" \
--seqmse-num-batches 20
```

PCQ example(Smoothquanted model):
```bash
cd aimet && \
python -m Examples.torch.quantize  \
--model-id "soiji/Llama-3.2-1B-Instruct_Smoothquant"  \
--recipe "pcq_spinquant_adascale"  \
--export-path "./Llama-3.2-1B-Instruct_Smoothquant_PCQ" \
--adascale-num-batches 128 \
--adascale-num-iterations 2048
```

You can check the progress in [huggingface](https://huggingface.co/collections/soiji/aimet-quant-test)
