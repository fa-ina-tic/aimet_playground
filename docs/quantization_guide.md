Quant 어떻게 했는 지를 적어두겠습닏.


일단 smoothquant는 llmcompressor를 사용하였습니다.
```bash TODO: @sooh0601 fill command or python script that used for Smoothquant

```

Mobilequant는 MobileQuant technique을 그대로 사용합니다.
여기서 smoothquant 한번 더 돌려야 하기는 합니다 ㅎㅎ
```bash TODO: @sooh0601 fill command or python script that used for Mobilequant
```

그 다음 모델 quantization은 aimet사용해서 합니다.
LPBQ 
```bash

```

PCQ
```bash
```

해서 타겟 모델이 

Progress
- [ ] Original Model
    - [ ] PCQ
        - [x] w4a16
        - [ ] w8a8
    - [x] LPBQ
        - [x] w4a16
        - [x] w8a8
- [ ] SmoothQuant Model
    - [x] [SmoothQuant](https://huggingface.co/soiji/Llama-3.2-1B-Instruct_SmoothQuant)
    - [ ] PCQ
        - [ ] w4a16
        - [ ] w8a8
    - [x] LPBQ
        - [x] [w4a16](https://huggingface.co/soiji/Llama-3.2-1B-Instruct_Smoothquant_w4a16_LPBQ)
        - [x] [w8a8](https://huggingface.co/soiji/Llama-3.2-1B-Instruct_Smoothquant_w8a8_LPBQ)
- [ ] MobileQuant Model
    - [ ] MobileQuant
    - [ ] PCQ
        - [ ] w4a16
        - [ ] w8a8
    - [ ] LPBQ
        - [ ] w4a16
        - [ ] w8a8
