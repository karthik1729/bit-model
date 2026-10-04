# BitNet b1.58 2B4T — local test rig

Inference harness for [`microsoft/bitnet-b1.58-2B-4T-bf16`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16):
a 2.41B-parameter model whose weights are ternary (`{-1, 0, +1}`, i.e. log2(3) ≈ 1.58 bits).

| Script | Purpose |
| --- | --- |
| `chat.py` | Interactive terminal REPL with token streaming, slash commands, per-reply metrics |
| `serve.py` | Local web chat UI (stdlib HTTP + SSE, no gradio) on `127.0.0.1:7860` |
| `run_bitnet.py` | Non-interactive smoke test: loads, generates three fixed prompts, prints tok/s |

## Setup on a new machine

Requires an NVIDIA GPU and ~6 GiB of VRAM. Verified on an RTX PRO 6000 Blackwell
(sm_120) with Python 3.14.

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install transformers accelerate
```

Pin notes:

- **`cu128` or newer is mandatory on Blackwell.** sm_120 has no kernels in cu126 and
  earlier; you get `no kernel image is available for execution`. Verify with
  `.venv/bin/python -c "import torch; print(torch.cuda.get_device_capability(0))"`
  → `(12, 0)`.
- **`transformers` must be >= 5.x**, which has native `bitnet` support
  (`src/transformers/models/bitnet/`). Older advice to install the
  `shumingma/transformers` fork is obsolete.
- The repo's `config.json` has an `auto_map` pointing at `modeling_bitnet.py`, but
  **those files are not in the repo** — `trust_remote_code` cannot work. Native
  support is the only path.

First run downloads ~5 GB to `~/.cache/huggingface`.

## Usage

```bash
.venv/bin/python chat.py              # sampling, temp 0.7
.venv/bin/python chat.py --greedy     # deterministic — use this for A/B tests
.venv/bin/python serve.py             # web UI at http://127.0.0.1:7860
.venv/bin/python run_bitnet.py        # one-shot smoke test
```

In `chat.py`: Enter sends, `\` continues a line, **Ctrl-C interrupts a generation
without exiting**, Ctrl-D quits. `/help` lists commands (`/greedy`, `/sample`,
`/temp`, `/topp`, `/max`, `/system`, `/regen`, `/undo`, `/save`, `/info`).

Over SSH with no browser, prefer `chat.py`. To use `serve.py` remotely, tunnel it
rather than binding publicly — the UI has **no authentication**:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@host
```

## Measured behaviour

On an RTX PRO 6000 Blackwell, bf16, batch 1:

- loads in ~7 s warm (~56 s on first run, including download)
- **5.3 GiB** total GPU footprint (4.49 GiB weights + ~0.8 GiB CUDA context /
  cuBLAS workspace); KV cache adds ~75 KiB/token thanks to GQA (5 KV heads)
- **~8 tok/s** on sustained generation; short replies read lower (~3 tok/s)
  because fixed per-request overhead dominates
- **time-to-first-token is ~6 s on the very first generation, then ~0.3–0.6 s.**
  That is one-time `@torch.compile` JIT on the `WeightQuant`/`ActQuant` kernels.
  `chat.py` does a throwaway 2-token generation at startup to absorb it.

### Why it is not fast

`config.json` requests `quantization_mode: online`, so `AutoBitLinear` re-quantizes
**all 2.41B weights to ternary on every forward pass** — once per token. You pay a
full quantization sweep per token and gain nothing at inference time.

The advertised BitNet numbers (~0.4 GB memory, 5–7× CPU speedup, ~29 ms/token) come
from [bitnet.cpp](https://github.com/microsoft/BitNet)'s lookup-table kernels, which
are CPU-only. Don't read the tok/s here as BitNet's performance story.

Three distinct configurations:

| Path | Speed | Trainable | Faithful ternary |
| --- | --- | --- | --- |
| `-bf16` + online quant (this repo) | ~8 tok/s | **yes** (STE) | yes |
| packed `bitnet-b1.58-2B-4T` repo | faster | no | yes |
| bitnet.cpp on CPU | fastest | no | yes |

## Notes toward finetuning

Finetuning this model is **QAT**, not ordinary SFT. `transformers`'
`integrations/bitnet.py` defines:

```python
class WeightQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):        # ternary round/clamp to -1, 0, 1
    @staticmethod
    def backward(ctx, grad_output):  # straight-through estimator
        return grad_output.clone()
```

With `online_quant=True` the quantizer sits **inside** the autograd graph, so
gradients reach the bf16 master weights. That means QAT finetuning works out of the
box — but three constraints follow:

1. **Finetune the `-bf16` repo, not the packed one.** The packed repo loads with
   `online_quant=False`: a frozen `weight_scale` buffer and no quantizer in the
   graph. That path is inference-only.
2. **LoRA is not viable if the result must stay ternary.** `AutoBitLinear` subclasses
   `nn.Linear`, so PEFT attaches happily, but the adapter computes
   `W_ternary·x + BA·x` — a full-precision side path that bypasses quantization.
   Merging means folding `BA` into master weights and re-quantizing, which is lossy
   and non-equivalent.
3. **Watch the learning rate.** That STE backward is a *pure* identity with no
   clipping, and the ternary scale is global per-tensor (`1/weight.abs().mean()`).
   Too low an LR and no weight crosses a threshold: loss moves while the deployed
   ternary model is unchanged. Always verify post-training that re-packed ternary
   weights actually differ.

Memory for full-parameter QAT of 2.41B: ~4.8 GiB weights + ~4.8 GiB grads +
~19 GiB fp32 AdamW states ≈ 30–40 GiB with gradient checkpointing. Under ~17 GiB of
free VRAM, switch to 8-bit optimizer states (~4.8 GiB) to land near 14–15 GiB, which
keeps full-parameter QAT intact.
