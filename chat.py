"""Interactive terminal chat for microsoft/bitnet-b1.58-2B-4T-bf16.

    .venv/bin/python chat.py                 # sampling, temp 0.7
    .venv/bin/python chat.py --greedy        # reproducible decoding

Ctrl-C interrupts a generation without leaving the REPL; Ctrl-D (or /exit) quits.
Type /help for commands.
"""
import argparse, json, os, signal, sys, threading, time

import torch
from transformers.utils import logging as hf_logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"
DEFAULT_SYSTEM = "You are a helpful AI assistant."
HISTFILE = os.path.expanduser("~/.bitnet_chat_history")

TTY = sys.stdout.isatty()


def c(code, text):
    return f"\033[{code}m{text}\033[0m" if TTY else text


DIM, BOLD, CYAN, GREEN, YELLOW, RED = "2", "1", "36", "32", "33", "31"

CANCEL = threading.Event()


class CancelOnEvent(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return CANCEL.is_set()


class Chat:
    def __init__(self, args):
        self.system = args.system
        self.do_sample = not args.greedy
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens
        self.history = []

        hf_logging.set_verbosity_error()   # keep warnings out of the token stream
        hf_logging.disable_progress_bar()

        print(c(DIM, f"loading {MODEL_ID} …"), flush=True)
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map=args.device
        )
        self.model.eval()
        load_s = time.perf_counter() - t0

        params = sum(p.numel() for p in self.model.parameters())
        dev = str(self.model.device)
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        vram = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0

        print(c(BOLD, "\nBitNet b1.58 2B4T") + c(DIM, "  (ternary weights, online QAT)"))
        print(c(DIM, f"  {gpu} · {dev} · {params/1e9:.2f}B params · {vram:.2f} GiB · loaded {load_s:.1f}s"))
        print(c(DIM, "  /help for commands · Ctrl-C interrupts · Ctrl-D quits\n"))

        if not args.no_warmup:
            self._warmup()

    def _warmup(self):
        """Pay the torch.compile JIT for the quantization kernels up front."""
        t0 = time.perf_counter()
        print(c(DIM, "warming up quantization kernels …"), end="", flush=True)
        ids = self.tok("hi", return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            self.model.generate(**ids, max_new_tokens=2, do_sample=False,
                                pad_token_id=self.tok.eos_token_id)
        print(c(DIM, f" {time.perf_counter() - t0:.1f}s\n"), flush=True)

    # ---------------- generation ----------------

    def generate(self, prompt):
        self.history.append({"role": "user", "content": prompt})
        return self._run()

    def _run(self):
        messages = [{"role": "system", "content": self.system}] + self.history
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        streamer = TextIteratorStreamer(
            self.tok, skip_prompt=True, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,   # destructive for BPE
        )
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            pad_token_id=self.tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([CancelOnEvent()]),
        )
        if self.do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p)

        err = {}

        def run():
            try:
                with torch.inference_mode():
                    self.model.generate(**kwargs)
            except Exception as exc:
                err["msg"] = f"{type(exc).__name__}: {exc}"
                streamer.end()

        CANCEL.clear()
        prev = signal.signal(signal.SIGINT, lambda *_: CANCEL.set())

        print(c(CYAN, "bitnet") + c(DIM, " › "), end="", flush=True)
        t0 = time.perf_counter()
        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        acc, n, ttft = "", 0, None
        try:
            for chunk in streamer:
                if not chunk:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
                acc += chunk
                sys.stdout.write(chunk)
                sys.stdout.flush()
        finally:
            worker.join()
            signal.signal(signal.SIGINT, prev)

        dt = time.perf_counter() - t0
        print()

        if err:
            print(c(RED, f"  error: {err['msg']}"))
            self.history.pop()
            return

        if acc.strip():
            self.history.append({"role": "assistant", "content": acc})
        else:
            self.history.pop()

        tail = " · interrupted" if CANCEL.is_set() else ""
        print(c(DIM, f"  [{n} tok · {dt:.2f}s · {n/dt:.1f} tok/s · ttft "
                     f"{ttft:.2f}s · prompt {prompt_tokens} tok{tail}]\n") if ttft
              else c(DIM, "  [no output]\n"))

    # ---------------- commands ----------------

    def mode(self):
        return (f"greedy" if not self.do_sample
                else f"sample temp={self.temperature} top_p={self.top_p}")

    def command(self, line):
        """Return True if the REPL should exit."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            return True

        elif cmd == "/help":
            print(c(DIM, """
  /help              show this
  /clear             wipe conversation history
  /system <text>     set system prompt (also clears history)
  /greedy            deterministic decoding (for A/B comparisons)
  /sample            re-enable sampling
  /temp <float>      sampling temperature
  /topp <float>      nucleus top-p
  /max <int>         max new tokens
  /regen             rerun the last prompt
  /undo              drop the last exchange
  /history           print the conversation
  /save <path>       write transcript as JSON
  /info              model + settings
  /exit              quit (or Ctrl-D)
"""))

        elif cmd == "/clear":
            self.history = []
            print(c(DIM, "  history cleared\n"))

        elif cmd == "/system":
            if arg:
                self.system = arg
                self.history = []
                print(c(DIM, f"  system set, history cleared\n"))
            else:
                print(c(DIM, f"  system: {self.system}\n"))

        elif cmd == "/greedy":
            self.do_sample = False
            print(c(DIM, "  greedy decoding — identical input now gives identical output\n"))

        elif cmd == "/sample":
            self.do_sample = True
            print(c(DIM, f"  sampling on ({self.mode()})\n"))

        elif cmd in ("/temp", "/topp", "/max"):
            if not arg:
                print(c(RED, f"  usage: {cmd} <value>\n"))
                return False
            try:
                if cmd == "/temp":
                    self.temperature = float(arg)
                elif cmd == "/topp":
                    self.top_p = float(arg)
                else:
                    self.max_new_tokens = int(arg)
            except ValueError:
                print(c(RED, f"  not a number: {arg}\n"))
                return False
            if cmd != "/max" and not self.do_sample:
                self.do_sample = True
                print(c(DIM, "  (sampling re-enabled)"))
            print(c(DIM, f"  {self.mode()} · max_new_tokens={self.max_new_tokens}\n"))

        elif cmd == "/regen":
            while self.history and self.history[-1]["role"] == "assistant":
                self.history.pop()
            if not self.history:
                print(c(RED, "  nothing to regenerate\n"))
            else:
                print(c(DIM, f"  regenerating: {self.history[-1]['content'][:60]}"))
                self._run()

        elif cmd == "/undo":
            for role in ("assistant", "user"):
                if self.history and self.history[-1]["role"] == role:
                    self.history.pop()
            print(c(DIM, f"  {len(self.history)} messages left\n"))

        elif cmd == "/history":
            if not self.history:
                print(c(DIM, "  (empty)\n"))
            for m in self.history:
                tag = c(GREEN, "you") if m["role"] == "user" else c(CYAN, "bitnet")
                print(f"  {tag} › {m['content']}")
            print()

        elif cmd == "/save":
            path = arg or "transcript.json"
            with open(path, "w") as fh:
                json.dump({"system": self.system, "settings": self.mode(),
                           "messages": self.history}, fh, indent=2)
            print(c(DIM, f"  wrote {path} ({len(self.history)} messages)\n"))

        elif cmd == "/info":
            print(c(DIM, f"  {MODEL_ID}\n  {self.mode()} · max_new_tokens="
                         f"{self.max_new_tokens} · {len(self.history)} messages"))
            if torch.cuda.is_available():
                print(c(DIM, f"  vram {torch.cuda.memory_allocated()/2**30:.2f} GiB allocated\n"))

        else:
            print(c(RED, f"  unknown command: {cmd} (try /help)\n"))

        return False

    # ---------------- repl ----------------

    def loop(self):
        try:
            import readline  # arrow-key editing + persistent history
            if os.path.exists(HISTFILE):
                readline.read_history_file(HISTFILE)
            readline.set_history_length(1000)
        except Exception:
            readline = None

        while True:
            try:
                line = input(c(GREEN, "you") + c(DIM, " › ")).strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print(c(DIM, "  (Ctrl-D or /exit to quit)"))
                continue

            if not line:
                continue

            # trailing backslash continues onto the next line
            while line.endswith("\\"):
                try:
                    line = line[:-1] + "\n" + input(c(DIM, "  … "))
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

            if line.startswith("/"):
                if self.command(line):
                    break
                continue

            self.generate(line)

        if readline:
            try:
                readline.write_history_file(HISTFILE)
            except Exception:
                pass
        print(c(DIM, "bye"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--greedy", action="store_true", help="deterministic decoding")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-warmup", action="store_true")
    Chat(ap.parse_args()).loop()


if __name__ == "__main__":
    main()
