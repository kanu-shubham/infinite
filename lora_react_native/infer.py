"""Generate React Native code from the LoRA-tuned model.

Usage:
    python infer.py "Build a React Native settings screen with a dark mode toggle persisted to AsyncStorage."
"""
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import CFG
from prepare_data import SYSTEM_PROMPT


def load() -> tuple:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=CFG.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, CFG.bnb_4bit_compute_dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(CFG.output_dir)
    base = AutoModelForCausalLM.from_pretrained(
        CFG.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, CFG.output_dir)
    model.eval()
    return model, tokenizer


def generate(prompt: str) -> str:
    model, tokenizer = load()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or (
        "Create a React Native screen showing a list of todos with add and delete."
    )
    print(generate(prompt))
