evaluation은 aimet에 있는 방식 그대로 사용합니다.
다만 window device에서 진행하고 있는 터라, torch.evaluate 사용합니다.

example:
```bash
python -m Examples.torch.evaluate  \
--model-id "meta-llama/Llama-3.2-1B-Instruct"  \
--checkpoint "../models/Llama-3.2-1B-Instruct_Smoothquant_LPBQ"  \
--eval-ppl
```

evaluation output
| Recipe | bit | Original | SmoothQuant | MobileQuant |
| - | - | - | - | - |
| pcq_spinquant_adascale | w4a16 | 35.384281158447266 | 26.687795639038086 | - |
| lpbq_seqmse | w4a16 | 13.811701774597168 | 13.760761260986328 | - |
