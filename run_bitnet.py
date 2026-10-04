"""Smoke-test microsoft/bitnet-b1.58-2B-4T-bf16 under PyTorch/transformers.

The repo ships bf16 master weights; transformers' native `bitnet` support applies
online ternary quantization (per config.json quantization_config), so generations
match the 1.58-bit model without bitnet.cpp's speed/memory wins.
"""
import argparse, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"

PROMPTS = [
    "Explain in two sentences why ternary weights save memory versus fp16.",
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "What is the capital of Australia? Answer in one word.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt", help="override the built-in prompts with a single prompt")
    args = ap.parse_args()

    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | device {args.device}")
    if args.device.startswith("cuda"):
        print(f"gpu   {torch.cuda.get_device_name(0)} | cap {torch.cuda.get_device_capability(0)}")

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    params = sum(p.numel() for p in model.parameters())
    print(f"params {params / 1e9:.2f}B")
    if args.device.startswith("cuda"):
        print(f"vram   {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")

    prompts = [args.prompt] if args.prompt else PROMPTS
    for i, prompt in enumerate(prompts, 1):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        dt = time.perf_counter() - t0

        new = out[0][inputs["input_ids"].shape[-1]:]
        n = new.shape[-1]
        print(f"\n--- [{i}/{len(prompts)}] {prompt}")
        print(tok.decode(new, skip_special_tokens=True).strip())
        print(f"    [{n} tok in {dt:.2f}s = {n / dt:.1f} tok/s]")

    print("\nOK")


if __name__ == "__main__":
    main()
