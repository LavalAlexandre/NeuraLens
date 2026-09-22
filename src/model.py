import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "google/medgemma-4b-it"


def load_base_model(model_id=MODEL_ID):
    """Load MedGemma in 4-bit NF4 with bfloat16 compute."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("A CUDA GPU with bfloat16 support (Ampere or newer) is required.")

    return AutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        ),
    )


def load_processor(model_id=MODEL_ID):
    processor = AutoProcessor.from_pretrained(model_id)
    # Right padding for training; the validation loader left-pads on its own
    processor.tokenizer.padding_side = "right"
    return processor


def load_peft_model_from_checkpoint(checkpoint_path, model_id=MODEL_ID):
    return PeftModel.from_pretrained(load_base_model(model_id), checkpoint_path)
