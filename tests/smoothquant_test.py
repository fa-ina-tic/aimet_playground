from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from aimet_torch.experimental.smoothquant import apply_smoothquant
from aimet_torch.utils import place_model
import torch

import sys
sys.path.append("../aimet")

from GenAITests.shared.helpers.datasets import Wikitext

if __name__ == "__main__":
    model_path = "/models/meta-llama/Llama-3.2-1B-Instruct"
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_dataset = Wikitext.load_encoded_dataset(tokenizer, 1024, "test")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with place_model(hf_model, device):
        apply_smoothquant(
            hf_model,
            test_dataset,
            num_iterations=50,
        )
    print(f"smoothquant successfully loaded")
