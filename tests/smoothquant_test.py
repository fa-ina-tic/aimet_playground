from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from aimet_torch.experimental.smoothquant import apply_smoothquant
from aimet_torch.utils import place_model
from aimet_torch.v2.nn.transformers.models.llama.modeling_llama import (
    QuantizedLlamaRMSNorm,
)
import torch
from aimet_torch import QuantizationSimModel
from aimet_torch.common.defs import QuantScheme

import sys
sys.path.append("/home/soomin.lee/aimet-smoothquant")
import itertools
import os
from tqdm import tqdm

from GenAITests.shared.models.base import LLM
from GenAITests.shared.helpers.datasets import Wikitext
from GenAITests.shared.models.generator import Generator
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache
from GenAITests.torch.helpers.quant_recipes import _prefill_inputs

SEQUENCE_LENGTH = 2048
CONTEXT_LENGTH = 4096
CALIB_NUM_BATCHES = 20
NUM_BATCHES = 128

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calib_num_batches",
        help="",
        default=CALIB_NUM_BATCHES,
        type=int
    )
    parser.add_argument(
        "--num_batches",
        help="",
        default=NUM_BATCHES,
        type=int
    )
    parser.add_argument(
        "--export_path",
        help="",
        default="./smooth_quant"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
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
    
    traceable_model = ONNXExportableModuleWithCache(hf_model)
    dummy_input_ids = torch.zeros((1, SEQUENCE_LENGTH), dtype=torch.int)
    dummy_attention_mask = torch.ones((1, SEQUENCE_LENGTH), dtype=torch.int)
    assembled_dummy_inputs = Generator.prepare_inputs(
        model=traceable_model,
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        past_key_values=[],
        context_length=CONTEXT_LENGTH,
        sequence_length=SEQUENCE_LENGTH,
    )

    # Create QuantizationSimModel with 4-bit integer weights and 16-bit integer activations
    quantsim = QuantizationSimModel(
        model=traceable_model,
        quant_scheme=QuantScheme.post_training_tf,
        dummy_input=assembled_dummy_inputs,
        default_output_bw=16,
        default_param_bw=4,
        in_place=True,
        config_file=LLM.get_quantsim_config(),
    )

    # Apply mixed precision to model
    quantsim.model.model.lm_head.param_quantizers["weight"].bitwidth = 8
    for _, module in quantsim.model.named_modules():
        if isinstance(module, QuantizedLlamaRMSNorm):
            module.param_quantizers["weight"].bitwidth = 16

    # Create a generator object to accurately simulate inference with static graph constraints while maintaining the
    # same interface. Use the generator object to do all forward passes through the model, including calibration, eval
    generator = Generator(quantsim.model, tokenizer, SEQUENCE_LENGTH, CONTEXT_LENGTH)

    # Load WikiText dataset from Huggingface
    train_dataset = Wikitext.load_encoded_dataset(tokenizer, CONTEXT_LENGTH, "train")

    # Use CUDA if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with place_model(quantsim.model, device):
        prefilled_inputs = _prefill_inputs(
            generator, train_dataset, args.num_batches, torch.device("cpu")
        )

        def calibration_callback(model: torch.nn.Module):
            sliced_dataloader = itertools.islice(train_dataset, args.calib_num_batches)
            for batch in tqdm(
                sliced_dataloader, total=args.calib_num_batches, desc="Calibrating"
            ):
                # Use generator for forward passes
                generator(input_ids=batch["input_ids"].to(device=model.device))

        # Compute activation encodings
        quantsim.compute_encodings(calibration_callback)

    os.makedirs(args.export_path, exist_ok=True)
    # Save model weights/buffers and quantizer states
    torch.save(
        quantsim.model.model.state_dict(),
        os.path.join(args.export_path, f"model_cls{CONTEXT_LENGTH}.pt"),
    )
    # Save only quantizer states
    torch.save(
        quantsim.quantizer_state_dict(),
        os.path.join(args.export_path, f"model_cls{CONTEXT_LENGTH}_quantizer.pt"),
    )
    print(
        f"PyTorch model weights and quantizer states are saved in {args.export_path} successfully."
    )
    quantsim.model.config.save_pretrained(args.export_path)
    tokenizer.save_pretrained(args.export_path)

    print(f"smoothquant successfully loaded")
