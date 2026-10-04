"""Local chat UI for microsoft/bitnet-b1.58-2B-4T-bf16.

Loads the model once, then serves a single-page chat interface with token
streaming over SSE. Stdlib-only on the server side (no gradio/fastapi), so it
runs on the same venv that run_bitnet.py uses.

    .venv/bin/python serve.py          # -> http://127.0.0.1:7860
"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"
DEFAULT_SYSTEM = "You are a helpful AI assistant."

TOK = MODEL = None
GPU_LOCK = threading.Lock()   # one generation at a time: single GPU
CANCEL = threading.Event()


class CancelOnEvent(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return CANCEL.is_set()


def load(device):
    global TOK, MODEL
    t0 = time.perf_counter()
    TOK = AutoTokenizer.from_pretrained(MODEL_ID)
    MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device
    )
    MODEL.eval()
    dt = time.perf_counter() - t0
    params = sum(p.numel() for p in MODEL.parameters())
    info = {
        "model": MODEL_ID,
        "device": str(MODEL.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "params_b": round(params / 1e9, 2),
        "load_s": round(dt, 1),
        "vram_gib": round(torch.cuda.memory_allocated() / 2**30, 2)
        if torch.cuda.is_available()
        else 0.0,
        "torch": torch.__version__,
    }
    print(f"[ready] {json.dumps(info)}")
    return info


INFO = {}


def stream_reply(req, write):
    """Run generation, pushing SSE events through `write`."""
    messages = [{"role": "system", "content": req.get("system") or DEFAULT_SYSTEM}]
    messages += [
        {"role": m["role"], "content": m["content"]} for m in req.get("messages", [])
    ]

    text = TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = TOK(text, return_tensors="pt").to(MODEL.device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    streamer = TextIteratorStreamer(TOK, skip_prompt=True, skip_special_tokens=True)
    do_sample = bool(req.get("do_sample", True))
    kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(req.get("max_new_tokens", 256)),
        do_sample=do_sample,
        pad_token_id=TOK.eos_token_id,
        stopping_criteria=StoppingCriteriaList([CancelOnEvent()]),
    )
    if do_sample:
        kwargs.update(
            temperature=float(req.get("temperature", 0.7)),
            top_p=float(req.get("top_p", 0.95)),
        )

    err = {}

    def run():
        try:
            with torch.inference_mode():
                MODEL.generate(**kwargs)
        except Exception as exc:  # surfaced to the browser rather than swallowed
            err["msg"] = f"{type(exc).__name__}: {exc}"
            streamer.end()

    CANCEL.clear()
    t0 = time.perf_counter()
    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    n_chunks = 0
    first_tok_s = None
    for chunk in streamer:
        if not chunk:
            continue
        if first_tok_s is None:
            first_tok_s = time.perf_counter() - t0
        n_chunks += 1
        write({"type": "token", "text": chunk})

    worker.join()
    dt = time.perf_counter() - t0

    if err:
        write({"type": "error", "message": err["msg"]})
        return

    write(
        {
            "type": "done",
            "stats": {
                "tokens": n_chunks,
                "seconds": round(dt, 2),
                "tok_s": round(n_chunks / dt, 1) if dt > 0 else 0,
                "ttft_s": round(first_tok_s, 2) if first_tok_s else None,
                "prompt_tokens": prompt_tokens,
                "cancelled": CANCEL.is_set(),
            },
        }
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console readable
        pass

    def _send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/info":
            self._send(200, json.dumps(INFO).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path == "/stop":
            CANCEL.set()
            self._send(200, b'{"ok":true}', "application/json")
            return
        if self.path != "/chat":
            self._send(404, b"not found", "text/plain")
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"bad json"}', "application/json")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def write(ev):
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()

        if not GPU_LOCK.acquire(blocking=False):
            write({"type": "error", "message": "A generation is already running."})
            return
        try:
            stream_reply(req, write)
        except (BrokenPipeError, ConnectionResetError):
            CANCEL.set()          # browser went away mid-stream
        except Exception as exc:
            try:
                write({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass
        finally:
            GPU_LOCK.release()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BitNet b1.58 2B4T</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --panel-2:#1e222b; --line:#2a2f3a;
    --fg:#e6e8ee; --muted:#9aa3b2; --accent:#7aa2f7; --user:#243049;
    --ok:#9ece6a; --warn:#e0af68; --err:#f7768e;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme: light){
    :root:not([data-theme="dark"]){
      --bg:#f6f7f9; --panel:#fff; --panel-2:#f0f2f5; --line:#dde1e7;
      --fg:#1b1f27; --muted:#5d6675; --user:#e4ecfb;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       height:100vh;display:flex;flex-direction:column}
  header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;
         padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
  h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.2px}
  .pill{font:11px/1 var(--mono);color:var(--muted);background:var(--panel-2);
        border:1px solid var(--line);border-radius:999px;padding:5px 9px;white-space:nowrap}
  .spacer{flex:1}
  button{font:inherit;color:var(--fg);background:var(--panel-2);
         border:1px solid var(--line);border-radius:8px;padding:7px 13px;cursor:pointer}
  button:hover:not(:disabled){border-color:var(--accent)}
  button:disabled{opacity:.45;cursor:not-allowed}
  button.primary{background:var(--accent);color:#0d1117;border-color:transparent;font-weight:600}
  main{flex:1;display:flex;min-height:0}
  #log{flex:1;overflow-y:auto;padding:20px 16px;display:flex;flex-direction:column;gap:14px}
  .wrap{max-width:780px;width:100%;margin:0 auto;display:flex;flex-direction:column;gap:14px}
  .msg{padding:11px 14px;border-radius:12px;background:var(--panel);
       border:1px solid var(--line);white-space:pre-wrap;overflow-wrap:anywhere}
  .msg.user{background:var(--user);border-color:transparent;align-self:flex-end;max-width:85%}
  .msg.err{border-color:var(--err);color:var(--err)}
  .role{font:10px/1 var(--mono);letter-spacing:.9px;text-transform:uppercase;
        color:var(--muted);margin-bottom:6px}
  .stats{font:11px/1 var(--mono);color:var(--muted);margin-top:9px;
         padding-top:8px;border-top:1px dashed var(--line)}
  .cursor{display:inline-block;width:7px;height:15px;background:var(--accent);
          vertical-align:-2px;animation:b 1s steps(2) infinite}
  @keyframes b{50%{opacity:0}}
  aside{width:280px;border-left:1px solid var(--line);background:var(--panel);
        padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:16px}
  aside label{display:block;font:11px/1.4 var(--mono);color:var(--muted);
              text-transform:uppercase;letter-spacing:.7px;margin-bottom:7px}
  textarea,input[type=number]{width:100%;background:var(--panel-2);color:var(--fg);
      border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit;resize:vertical}
  input[type=range]{width:100%;accent-color:var(--accent)}
  .row{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
  .val{font:12px var(--mono);color:var(--accent)}
  .chk{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--fg)}
  .chk input{accent-color:var(--accent);width:16px;height:16px}
  footer{border-top:1px solid var(--line);background:var(--panel);padding:12px 16px}
  .composer{max-width:780px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
  #prompt{flex:1;min-height:46px;max-height:180px}
  .hint{font:11px var(--mono);color:var(--muted);text-align:center;
        max-width:780px;margin:7px auto 0}
  @media (max-width:860px){
    aside{position:fixed;inset:48px 0 0 auto;z-index:9;width:min(300px,86vw);
          transform:translateX(100%);transition:transform .2s}
    aside.open{transform:none;box-shadow:-8px 0 24px #0006}
  }
  @media (min-width:861px){#toggle{display:none}}
</style>
</head>
<body>
<header>
  <h1>BitNet b1.58 2B4T</h1>
  <span class="pill" id="meta">loading…</span>
  <span class="pill" id="quant">ternary · online QAT</span>
  <span class="spacer"></span>
  <button id="toggle">Settings</button>
  <button id="clear">Clear</button>
</header>

<main>
  <div id="log"><div class="wrap" id="wrap"></div></div>
  <aside id="side">
    <div>
      <label for="system">System prompt</label>
      <textarea id="system" rows="4">You are a helpful AI assistant.</textarea>
    </div>
    <div>
      <label class="chk"><input type="checkbox" id="do_sample" checked> Sampling (off = greedy)</label>
    </div>
    <div>
      <div class="row"><label for="temperature">Temperature</label><span class="val" id="v_t">0.70</span></div>
      <input type="range" id="temperature" min="0.01" max="2" step="0.01" value="0.7">
    </div>
    <div>
      <div class="row"><label for="top_p">Top-p</label><span class="val" id="v_p">0.95</span></div>
      <input type="range" id="top_p" min="0.05" max="1" step="0.01" value="0.95">
    </div>
    <div>
      <label for="max_new_tokens">Max new tokens</label>
      <input type="number" id="max_new_tokens" min="1" max="4096" value="256">
    </div>
    <div class="pill" style="white-space:normal;line-height:1.5">
      Greedy + fixed system prompt is the reproducible setting for comparing
      checkpoints before/after finetuning.
    </div>
  </aside>
</main>

<footer>
  <div class="composer">
    <textarea id="prompt" placeholder="Ask something…  (Enter to send, Shift+Enter for newline)"></textarea>
    <button class="primary" id="send">Send</button>
    <button id="stop" disabled>Stop</button>
  </div>
  <div class="hint" id="hint">history is sent each turn · one generation at a time</div>
</footer>

<script>
const $ = id => document.getElementById(id);
const wrap = $('wrap');
let history = [], busy = false;

fetch('/info').then(r => r.json()).then(i => {
  $('meta').textContent = `${i.gpu} · ${i.params_b}B · ${i.vram_gib} GiB · torch ${i.torch}`;
}).catch(() => { $('meta').textContent = 'info unavailable'; });

$('temperature').oninput = e => $('v_t').textContent = (+e.target.value).toFixed(2);
$('top_p').oninput = e => $('v_p').textContent = (+e.target.value).toFixed(2);
$('toggle').onclick = () => $('side').classList.toggle('open');
$('clear').onclick = () => { if (!busy) { history = []; wrap.innerHTML = ''; } };

function bubble(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  const r = document.createElement('div');
  r.className = 'role';
  r.textContent = role === 'user' ? 'you' : role === 'err' ? 'error' : 'bitnet';
  const body = document.createElement('span');
  body.textContent = text || '';
  d.append(r, body);
  wrap.append(d);
  $('log').scrollTop = $('log').scrollHeight;
  return body;
}

async function send() {
  const text = $('prompt').value.trim();
  if (!text || busy) return;
  busy = true;
  $('send').disabled = true;
  $('stop').disabled = false;
  $('prompt').value = '';

  bubble('user', text);
  history.push({ role: 'user', content: text });

  const body = bubble('assistant', '');
  const cur = document.createElement('span');
  cur.className = 'cursor';
  body.after(cur);

  let acc = '';
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: history,
        system: $('system').value,
        do_sample: $('do_sample').checked,
        temperature: +$('temperature').value,
        top_p: +$('top_p').value,
        max_new_tokens: +$('max_new_tokens').value,
      }),
    });

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        const line = part.replace(/^data: /, '').trim();
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'token') {
          acc += ev.text;
          body.textContent = acc;
          $('log').scrollTop = $('log').scrollHeight;
        } else if (ev.type === 'done') {
          const s = ev.stats;
          const el = document.createElement('div');
          el.className = 'stats';
          el.textContent =
            `${s.tokens} tok · ${s.seconds}s · ${s.tok_s} tok/s · ttft ${s.ttft_s ?? '—'}s` +
            ` · prompt ${s.prompt_tokens} tok` + (s.cancelled ? ' · stopped' : '');
          body.parentElement.append(el);
        } else if (ev.type === 'error') {
          bubble('err', ev.message);
        }
      }
    }
  } catch (e) {
    bubble('err', String(e));
  } finally {
    cur.remove();
    if (acc) history.push({ role: 'assistant', content: acc });
    else history.pop();
    busy = false;
    $('send').disabled = false;
    $('stop').disabled = true;
    $('prompt').focus();
  }
}

$('send').onclick = send;
$('stop').onclick = () => fetch('/stop', { method: 'POST' });
$('prompt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
$('prompt').focus();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    global INFO
    INFO = load(args.device)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}  (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")
        srv.shutdown()


if __name__ == "__main__":
    main()
