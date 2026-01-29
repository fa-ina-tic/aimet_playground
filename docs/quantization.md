## Model preperation

### Smoothquant
```bash TODO: @sooh0601 fill command or python script that used for Smoothquant

```

### Mobilequant
Based on [Mobilequant repo](https://github.com/saic-fi/MobileQuant)

Convert model into hf format
```
CUDA_VISIBLE_DEVICES=0 python scripts/convert_ckpt.py --checkpoint_dir /models/meta-llama/Llama-3.2-1B-Instruct --output_dir  /models/meta-llama/Llama-3.2-1B-Instruct_Converted
```

Get activation scales(Use [Smoothquant](https://github.com/mit-han-lab/smoothquant/issues))
```
python examples/generate_act_scales.py --model-name /models/meta-llama/Llama-3.2-1B-Instruct_Converted --output-path /models/meta-llama/Llama-3.2-1B-Instruct_Converted/act_scales.pth --dataset-path /workplace/wikitext/wikitext-2-raw-v1
```

Smoothquant model first:
```
CUDA_VISIBLE_DEVICES=0 python ptq/smoothquant.py --hf_path /models/meta-llama/Llama-3.2-1B-Instruct_Converted --alpha 0.5 --calib_data wikitext
```

```
CUDA_VISIBLE_DEVICES=0 python ptq/generate_act_range.py --hf_path /models/meta-llama/Llama-3.2-1B-Instruct_Converted --calib_data wikitext2
```

```
CUDA_VISIBLE_DEVICES=0 python ptq/generate_qcfg.py --hf_path /models/meta-llama/Llama-3.2-1B-Instruct_Converted --use_16bit_softmax_input --use_16bit_softmax_output
```

Run MobileQuant
run.sh:
```
MODEL=Llama-3.2-1B-Instruct_Converted
CKPT=/models/meta-llama/${MODEL}
BATCH_SIZE=1

LET_LR=1e-3
LET_MIN_LR=1e-4
LWC_LR=1e-2
LWC_MIN_LR=1e-3
LRL_LR=1e-6
LRL_MIN_LR=1e-7

EPOCHS=60
NSAMPLES=1024
OUTPUT_DIR=/models/meta-llama/${MODEL}-mobilequant-s${NSAMPLES}-e${EPOCHS}

python ptq/mobilequant.py --lwc --let --lrl \
    --act_bitwidth 16 --epochs ${EPOCHS} --nsamples ${NSAMPLES} \
    --deactive_amp --hf_path ${CKPT} --output_dir ${OUTPUT_DIR} \
    --let_lr ${LET_LR} --let_min_lr ${LET_MIN_LR} \
    --lwc_lr ${LWC_LR} --lwc_min_lr ${LWC_MIN_LR} \
    --lrl_lr ${LRL_LR} --lrl_min_lr ${LRL_MIN_LR} \
    --batch_size ${BATCH_SIZE} --mode e2e --weight_bitwidth 16
```
command:
```
CUDA_VISIBLE_DEVICES=0 ./run.sh
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
