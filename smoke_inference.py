from dataclasses import dataclass
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class InferenceConfig:
    model_id: str
    adapter_path: str
    prompt: str
    max_new_tokens: int


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--prompt", default="Checkpoint migration test")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    return InferenceConfig(args.model_id, args.adapter_path, args.prompt, args.max_new_tokens)


def load_model(cfg: InferenceConfig) -> PeftModel:
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.float16)
    return PeftModel.from_pretrained(base, cfg.adapter_path)


def run_inference(cfg: InferenceConfig) -> str:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = load_model(cfg).to("cuda" if torch.cuda.is_available() else "cpu")
    tokens = tokenizer(cfg.prompt, return_tensors="pt").to(model.device)
    output = model.generate(**tokens, max_new_tokens=cfg.max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main() -> None:
    cfg = parse_args()
    generated = run_inference(cfg)
    print(generated)


if __name__ == "__main__":
    main()
