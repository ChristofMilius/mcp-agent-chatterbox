# mcp-agent-chatterbox

MCP server for local, GPU-accelerated text-to-speech with
[Chatterbox](https://github.com/resemble-ai/chatterbox) (Resemble AI, MIT).
Text goes in, spoken audio comes out — no cloud service, no API key, no
per-character cost.

Ships three things:

- **A resident synthesis worker** that holds the model, renders sentence by
  sentence and plays as it goes. Start it once and leave it running.
- **An MCP tool surface** (`speak`, `tts_status`, `tts_unload`, `list_voices`,
  `stop_speech`) that any MCP client can call. It forwards `speak` to the worker.
- **An opencode plugin** (`.opencode/plugins/tts.js`) that adds a `speak` tool
  and a `/speak` command to opencode itself, and installs itself globally so
  it works in every project, not just this repo.

## Start the worker first

The MCP server does not synthesize. It hands the text to a separate, long-lived
worker process and waits for the result.

```bash
mcp-agent-chatterbox worker          # start it, leave it running
mcp-agent-chatterbox worker-status   # is it up? what is it doing?
mcp-agent-chatterbox worker-stop     # finish the queue, then exit
```

### How the pieces fit

```
LLM / agent                              browser
   │  MCP tool call: speak(text=...)        │  buttons: stop / replay / speak
   ▼                                        ▼
MCP server (server.py, cli: `serve`)     panel (panel.py, cli: `panel`)
   └──────────────────┬─────────────────────┘
                      │  one JSON line per request over TCP 127.0.0.1:8765
                      ▼
worker (worker.py, cli: `worker`)
                      │  Speaker → ChatterboxEngine → model on the GPU
                      │  render-ahead buffer, then playback in order
                      ▼
sentence-batched chunks → tts_output/*.wav → speakers
```

Three consequences, each of which explains a behaviour you will otherwise trip
over:

- **The `speak` tool is still the way an agent speaks.** The worker does not
  implement MCP and never sees a JSON-RPC frame; it only understands small
  newline-delimited JSON requests (`speak`, `status`, `history`, `replay`,
  `unload`, `stop`, `shutdown`). The MCP server is the adapter that turns a tool
  call into that request, and the panel is a second adapter that turns a button
  into the same request. So an agent needs the server *and* the worker; the
  server alone writes no audio, and neither does the panel.
- **The server holds no model.** It boots in milliseconds and stays responsive
  because there is nothing in it to load. That is deliberate: `tts_status` and
  `list_voices` must answer while a model is still downloading.
- **The CLI is a second front door, not a third layer.** `mac speak "..."` runs
  the same `Speaker` directly, without the MCP server; `mac worker*` commands use
  the same TCP socket the server uses. Nothing in the CLI is a parallel
  implementation of synthesis.

**Why a separate process.** Synthesis used to happen inside the MCP server, and
inside that process it reliably stopped after two sentences and never resumed.
The same code outside a server finished the whole reply every time. It failed
identically over stdio and over HTTP, so it was not the transport; and it failed
with the client's pipe held open and closed early, so it was not the shutdown
path. The measurement, for the record, on the same six sentences:

| where the synthesis ran | result |
|---|---|
| inside the MCP server process (stdio) | 2 of 6 sentences, then nothing for 300s+ |
| inside the MCP server process (HTTP) | 2 of 6 sentences, then nothing |
| in a plain process, same code | 6 of 6, ~32s |
| **server forwarding to the worker** | **6 of 6, ~34s** |

The second problem was lifetime. The server dies when its client's turn ends, so
a reply longer than the turn was killed mid-sentence no matter how the queue was
drained. A resident worker outlives any turn, and keeps the model loaded, so
successive replies pay nothing for the load.

`speak` accepts a reply and answers immediately; rendering and playback keep
running in the worker afterwards. A blocking answer was the earlier design —
the tool result was a record of what was actually written and played, taken at
the moment the reply finished — but it held the harness's tool call open for the
entire spoken duration, and LM Studio's tool-call timeout aborts such a call
long before a long reply finishes, cutting the turn off mid-playback. Returning
on acceptance makes `speak` immune to any client timeout. The audio is played
by the worker, so it keeps playing regardless of what the harness does with the
server process, and the record of the utterance (written files, played counts,
timings, errors) is served afterwards by `tts_status` under `worker.last`.

## The four models

| Key | Params | Languages | Reference clip | Notes |
|---|---|---|---|---|
| `nano` | 110M | English | **not needed** | **Default.** Turbo's small sibling: same architecture, single-step decoder, tags and built-in voice, tightest latency/memory budget (also runs on CPU). |
| `turbo` | 350M | English | **not needed** | Nano's bigger sibling: same architecture, tags and built-in voice, more VRAM headroom. Understands paralinguistic tokens inline: `[laugh]`, `[chuckle]`, `[whispering]`, `[clear throat]`, `[shush]`, and 14 more (exact set: `TURBO_TAGS` in `registry.py`). |
| `multilingual` | 500M | 23 (incl. German) | **required** | `language="de"`, `en`, `fr`, … `t3_model="v2"` (default) or `"v3"`. |
| `original` | 500M | English | **required** | The CFG / exaggeration-tuning model, for delivery style control. |

The full tag set is per model and read straight from the pulled vocabularies
(ground truth on 2026-10-22, an earlier list contained invented tokens):
nano/turbo know exactly 19 (`TURBO_TAGS` in `registry.py`, including the two
easy-to-miss spellings `[clear throat]` with a space and `[whispering]` with
`-ing`); the 500M models know a broader 28-token set (`ORIGINAL_TAGS`, which
*does* contain `[whisper]` and `[music]`). Anything else in brackets is dropped
before rendering — an invented tag such as `[soft, contented hum]` would
otherwise be spoken literally, word for word. The count of dropped tokens is
logged, never their text.

Only `turbo` and `nano` ship a stock voice. The other two raise
`AssertionError: Please prepare_conditionals first or specify audio_prompt_path`
without a reference clip, so the tool checks this up front and returns a
`reference_clip_required` payload naming the fix instead of a stack trace.

## Requirements

- Python 3.13 (see [Why 3.13 only](#why-313-only))
- An NVIDIA GPU with a driver of 525+ for the CUDA 12.4 wheels. CPU works but
  is slow.
- ~6 GB of free VRAM for a comfortable margin (turbo needs ~2-3 GB, the 500M
  models ~3-4 GB)
- Windows for audio playback. On other platforms the chunk files are still
  written; the `play` field comes back `false` and `tts_status` records the
  playback error in `speech.last.errors`.

## Installation

```bash
uv sync --extra dev
```

`uv sync` installs the CUDA build of PyTorch automatically. `pyproject.toml`
pins torch to PyTorch's official CUDA index via `[tool.uv.sources]`, so GPU
acceleration is a property of the environment rather than an accident of the
platform.

### Verify

```bash
uv run mcp-agent-chatterbox doctor
```

Prints the resolved device, per-GPU free/total VRAM, the models, and any
reference clips found. On a multi-GPU box this is also how you find out which
card is free.

Quick CUDA check:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect `2.14.0+cu132 True`.

### Why `setuptools` is floored, not capped

`pyproject.toml` requires `setuptools>=83.0.0`. That floor exists to clear
GHSA-h35f-9h28-mq5c, which is fixed in 83.0.0. It is a floor, never a cap, so
uv picks the patched release and cannot regress below it.

An earlier revision carried a hard `setuptools<81` cap here, justified by a
story that turned out to be **wrong**. It is recorded because the symptom is
still real and the wrong fix is a trap:

The **turbo** model embeds [Perth](https://github.com/resemble-ai/perth),
Resemble AI's audio watermarking library, and calls it at load time. When Perth
fails to import, `perth/__init__.py` swallows the `ImportError` and sets
`PerthImplicitWatermarker = None` — the failure is silent by design upstream —
and `ChatterboxTurboTTS.__init__` then calls that `None`:

```
TypeError: 'NoneType' object is not callable
```

The cap was added because the visible `ImportError` was believed to be
`from pkg_resources import resource_filename`, which setuptools 81 dropped.
It was not. Two independent findings:

- **`perth` 1.1.0 never imports `pkg_resources` at all.** The resolved git
  dependency carries a real package with real metadata and uses `importlib`,
  not the loose vendored copy described in the old note.
- **The actual `ImportError` was a librosa one:**
  `cannot import name 'resample' from 'librosa'`. It surfaced through
  `perth_watermarker.py`'s `from librosa import resample`, which is a red
  herring — librosa itself was broken. librosa lazily loads `resample` from
  `librosa.core.spectrum`, which imports numba; numba imports
  `numba.misc.coverage_support`, which subclasses `coverage.types.Tracer`; and
  a **half-deleted `coverage` install** made that fail with
  `AttributeError: module 'coverage' has no attribute 'types'`. lazy_loader
  swallowed the resulting chain, leaving `librosa.resample` simply *unbound* —
  so `import librosa` still succeeded, and the breakage only appeared later,
  inside an unrelated package's `from librosa import resample`.

The `coverage` install was damaged by a `uv sync` that was killed midway
because the running MCP server held `coverage/tracer.pyd` open, leaving
`coverage/__init__.py` deleted while `types.py` survived. So the cap was
fixing nothing: it merely rode along with the resolution that removed the
corruption. Verified on setuptools 84.0.0 with `pkg_resources` gone entirely —
`perth.PerthImplicitWatermarker` imports as a real class and turbo renders.

Consequences worth knowing:

- **A clean `coverage` install is what actually matters.** The tell is
  `coverage.__file__` being `None` (namespace-package shell). If turbo ever
  throws that `TypeError` again, run `uv sync` to completion and check
  `coverage/__init__.py` exists before suspecting a dependency.
- **Only turbo is affected.** `multilingual`, `original`, and `nano` never
  touch Perth.
- Never restore a `setuptools<81` cap. It cannot fix this failure, and it
  reintroduces a known advisory.

### Why `chatterbox-tts` comes from a git snapshot

We deliberately do **not** install the PyPI release. `chatterbox-tts` has
shipped nothing since 0.1.7 (2026-03-26), but upstream master carries what the
wheel lacks:

- **real v2/v3 multilingual checkpoint selection.** The 0.1.7 wheel's
  `mtl_tts.from_pretrained()` takes no `t3_model` argument, so our `t3_model`
  knob would `TypeError` the moment it was used. Master maps `v2`/`v3` to
  `t3_mtl23ls_v2/v3.safetensors`.
- **the 500M EOS-noise trim** — the final speech token decodes to ~40 ms of
  noise before EOS and is dropped from the wav.
- **the HF Xet-download fallback** — the Xet storage backend crashes on some
  environments; master retries over the HTTP/LFS path automatically.

`pyproject.toml` pins the snapshot to an exact commit (`rev = 5de7a54a…`) under
`[tool.uv.sources]`, so the environment is reproducible. To pick up newer
upstream work, bump the `rev` deliberately and re-run the four models — then
re-examine the dtype-shim reasoning, because upstream still carries the
`norm_loudness` float64 upcast that `engine._harden_reference_dtype()` exists
to contain. The snapshot also builds `resemble-perth` from git (the watermarker
behind turbo), replacing the loose vendored copy the 0.1.7 wheel shipped.

Consequences worth knowing:

- Master keeps version `0.1.7` — upstream never bumps it, so the git `rev` is
  the package's only reliable identity. Do not "upgrade" back to a PyPI
  `chatterbox-tts>=0.1.7`; you would silently lose v3, the trim, and the Xet
  fallback.
- The `t3_model` knob is now real: `v2` (default) or `v3`. v3 is a separate
  multigigabyte checkpoint downloaded on first use.
- Master retuned the multilingual `repetition_penalty` default from 2.0 to 1.2;
  the registry and its test mirror that.
- The dtype shim is independent of this snapshot — the bug exists in both the
  wheel and master, so the shim stays either way.

## Model weights

Weights are **not** bundled. They download from Hugging Face on the first
`speak()` call and are cached in the standard location
(`%USERPROFILE%\.cache\huggingface\hub` on Windows, override with `HF_HOME`).
Subsequent calls reuse the cache.

- `turbo` → `ResembleAI/chatterbox-turbo`
- `nano` → `ResembleAI/chatterbox-nano` (a separate repo, not a subset of turbo)
- `multilingual` / `original` → `ResembleAI/chatterbox`

Nothing is downloaded by `doctor` or `tts_status`, so those stay fast and work
offline.

### Putting the cache where you want it

`HF_HOME` is the official Hugging Face switch: every HF tool (`huggingface-cli`,
`transformers`, `huggingface_hub`, `datasets`) resolves its cache root from it.
Setting it system-wide is the right thing to do, and it is what you want. This
server adds one variable of its own, `CHATTERBOX_HF_HOME`, and the two are not
alternatives to each other — **`CHATTERBOX_HF_HOME` simply sets `HF_HOME`**
(see `engine.py:_apply_hf_home`). If the two are set to the same path, nothing
is in conflict and nothing is duplicated in effect.

So why is a second spelling needed at all? Because a system-wide `HF_HOME` can
fail to reach the server, and that failure is silent. Two things are involved.

**1. Not every harness forwards your environment to MCP subprocesses.** This is
the mechanism worth being precise about, because it is easy to misdiagnose.
Environment variables are not global to the machine at runtime — each process
receives an environment block from its parent, and a parent may spawn a child
with a **subset** of its own. When a harness builds the child environment from
only the `env:` block in its own config, your user-level `HF_HOME` is simply
absent from that child. The server then has no `HF_HOME`, and `huggingface_hub`
falls back to the default under `C:\` — no error, just a download to the wrong
drive.

Worse, the `env:` block itself turned out not to be a dependable carrier. LM
Studio's `mcp.json` and Hermes both accept one, both are spec-correct, and
neither was observed forwarding it to the server subprocess in practice — every
`CHATTERBOX_*` setting quietly fell back to its compiled default. This is why
the server now reads a project `.env` file it opens itself; see
[Configuration](#configuration). That file is the source of truth, and it makes
`CHATTERBOX_HF_HOME` work regardless of what the harness chooses to pass.

This is **not** about `PATH`. `PATH` is the list of directories searched for
executables and has nothing to do with where a cache is written. The variable
that matters is `HF_HOME`, and the thing that failed was process environment
inheritance, not a path lookup.

**2. `HF_HOME` must be set before `huggingface_hub` is imported.**
`huggingface_hub` reads and *seals* `HF_HOME` into its constants module on first
import, which happens the moment a Chatterbox model class is imported. Setting
the variable later in the process has no effect at all. This server therefore
applies `CHATTERBOX_HF_HOME` to `HF_HOME` inside `engine.load()` — immediately
before `load_model_class()`, and before anything has touched
`huggingface_hub` — so the cache root is fixed at the only moment that matters.

Together these are why setting the cache path **in the project `.env`** is the
robust move:

```dotenv
CHATTERBOX_HF_HOME=/path/to/hf-cache   # sets HF_HOME before the model import
```

Setting it here and in your system environment to the same path is fine — the
server detects the match and does nothing. Forward slashes work on Windows; left
unset, the variable leaves HF's own default untouched rather than pointing it
somewhere arbitrary.

**The symptom, if you get it wrong:** the first `speak()` after a config change
takes minutes again, and a second copy of the model appears under
`C:\Users\<you>\.cache\huggingface\hub\models--ResembleAI--chatterbox-<model>`.
`HF_HOME` is sealed per *process*, so a harness restart re-reads config while a
long-running server does not — check `.env`, and the server's startup log line,
when hunting this. The ready line records which config file was read.

**Corollary for `nano`.** Because `nano` is a separate HF repo, switching
`CHATTERBOX_MODEL` between `nano` and `turbo` (either direction) is a **fresh
download**, not a reuse of an already-cached sibling. A machine with turbo cached
still pays a full nano fetch the first time — into whichever cache root is
actually in effect, so get the cache path right *before* that first nano call.

## Registering the MCP server

**Settings go in `.env`, not here.** The server reads `.env` from the repo root
at startup, so registration only needs to say how to launch it:

```jsonc
{
  "mcp": {
    "mcp-agent-chatterbox": {
      "type": "local",
      "command": [
        "uv", "run", "--directory", "<absolute path to this repo>",
        "mcp-agent-chatterbox", "serve"
      ],
      "enabled": true
    }
  }
}
```

Copy `.env.example` to `.env` and edit that — see
[Configuration](#configuration) for the full key list and the reason this
indirection exists. Nothing is loaded at startup, so an `enabled: true` server
stays idle until a tool call arrives.

### The `environment` block, and why it is not the source of truth

Most harnesses accept an `env` / `environment` block and some do honour it. It
is kept here as a **per-harness override** — useful for pointing one client at a
different GPU or model than the rest — but it must not be the only place a
setting lives:

```jsonc
      "environment": {
        // Overrides for THIS client only. The durable value belongs in .env.
        "CHATTERBOX_MODEL": "turbo",
        "CHATTERBOX_GPU_INDEX": "0"
      }
```

Precedence is **environment over `.env`**: a variable already present in the
process environment is never overwritten by the file. That is what makes the
override above work, and it is also why a one-off `CHATTERBOX_MODEL=turbo` in a
shell, or a test's monkeypatch, always wins.

The trap this arrangement exists to close: LM Studio's `mcp.json` and Hermes
both accept an `env` block, both are spec-correct (`env` is the key — there is
nothing to rename), and neither forwarded it to the server subprocess in
practice. With `.env` in play that failure is inert, because the file is read by
the server itself and no harness has to pass anything.

**YAML-based harnesses** (clients that configure MCP servers in a `.yaml` file
rather than JSON) take the same command, with `command`/`args` split:

```yaml
  chatterbox:
    command: uv
    args:
      - run
      - --directory
      - <absolute path to this repo>
      - mcp-agent-chatterbox
      - serve
    env:                     # optional overrides; .env holds the durable values
      CHATTERBOX_MODEL: turbo
      CHATTERBOX_GPU_INDEX: '1'
```

Quote the numeric-looking values (`'1'`) — YAML would otherwise read them as
integers, and a non-string env value is a harness-specific coin flip.

**LM Studio / Claude Desktop / other strict-JSON clients:** the same command,
`command` split from `args`, and the environment named `env`:

```json
{
  "mcpServers": {
    "mcp-agent-chatterbox": {
      "command": "uv",
      "args": [
        "run", "--directory", "<absolute path to this repo>",
        "mcp-agent-chatterbox", "serve"
      ],
      "env": {
        "CHATTERBOX_MODEL": "turbo",
        "CHATTERBOX_GPU_INDEX": "1"
      }
    }
  }
}
```

Restart the client. The server speaks stdio, loads no weights at startup, and
stays responsive while the model is idle. Its first log line records the config
it resolved, so a wrong setting is visible without a tool call:

```
[server] mcp-agent-chatterbox v0.1.0 ready (output=…, voices=…, model=nano, voice=seven, config=I:\…\.env)
```

## Skill: teaching an agent to answer by voice

The MCP server's `instructions` field and the `speak` tool description carry
the trigger rule ("when the user asks for a spoken reply, call `speak()`"),
so any client sees it in the tool schema. Some harnesses additionally load
**Agent Skills** — a `SKILL.md` file with workflow instructions — which gives
a small local model much more to hold on to: what counts as a voice request,
that one instruction persists for the whole exchange, that the spoken text
is the model's own answer (never a repetition of the user's message),
spoken register (no markdown/lists/paths), which model and voice to pick
(prefer `nano` for English; ask the user which voice once, then remember
it, unless the server configures a default), how to summarize a long
answer for the ear, and the failure modes to avoid retrying.

This repo ships one at [`skills/voice-output/SKILL.md`](skills/voice-output/SKILL.md).

**Install in LM Studio Bionic:** Settings → Skills → add
`skills/voice-output/SKILL.md`, or ask Bionic to install the skill from this
repo's URL. Trigger it explicitly with `@voice-output` in the composer, or
invoke it with a phrase like "reply by voice from now on".

**Install in opencode:** copy the folder into the global skills directory:

```powershell
Copy-Item -Recurse skills\voice-output $env:USERPROFILE\.config\opencode\skills\
```

Any other harness that speaks the Agent Skills format (Codex, Claude Code,
and compatible tools) can consume the same file.

## The opencode plugin

`.opencode/plugins/tts.js` installs the repo's commands — `/speak`,
`/speak-stop` and `/tts-status` — into opencode. It **self-replicates** into
`~/.config/opencode/plugins/` the first time opencode loads it from this repo,
and copies every `*.md` in `.opencode/commands/` into the global commands
directory, so the commands are available in every project rather than only
sessions started inside this folder. The repo copy is the source of truth; edit
there, never the global replica. Removing a command from the repo does not
delete the installed copy — install is not reaping.

The plugin deliberately does not auto-speak assistant replies: loading a model
costs seconds and VRAM, and unsolicited audio on every turn would be awful.
Speaking stays an explicit choice, by tool call or by `/speak`.

## The panel

A page with the controls that have no home in a chat window: **Stop**, **Replay
last**, a play button on every message still on disk, **Unload model**, and a
text box that speaks what you type. Started on demand, nothing autostarts:

```bash
uv run mcp-agent-chatterbox panel            # serves http://127.0.0.1:8766/ and opens it
uv run mcp-agent-chatterbox panel --no-open  # just serve it
uv run mcp-agent-chatterbox panel --port 9000
```

It is one HTML file and the standard library's `http.server` — no toolkit to
install, no build step. It plays no audio itself and holds no state: every
button is one op forwarded to the resident worker, which owns the queue, the
model and the output directory. That split is not tidiness, it is what makes
Stop honest — playback is `winsound`, whose sound belongs to the thread that
started it, so only the worker's own playback thread can cut a chunk short. A
panel that played the WAVs itself would be uncancellable from the worker, and
Stop would only land between chunks.

**Replay is free.** Every message that was ever spoken left its chunks in
`tts_output/` as `<message_id>_<n>.wav`, so repeating one is a file read, not a
render: no GPU time, no model load, and it works even after `tts_unload`. A
replay job goes through the same queue as live speech, so it never talks over an
utterance that is still rendering, and Stop cuts it off like anything else. The
files are the history, which also means the list survives a worker restart —
and that deleting old `tts_output/*.wav` files is how you shorten it.

Two deliberate limits: the panel is bound to `127.0.0.1` (not reachable from
another machine), and every action requires a custom header that a foreign web
page cannot set without a CORS preflight this server never approves, plus a
loopback `Origin` when one is sent. Any page you visit can otherwise POST to a
localhost port and unload your model.

The page carries no external assets, so it renders the same in a normal browser
and inside an embedded browser pane. There is no desktop app integration to
build: it is a URL.

**Stop cuts the sound, not just the queue.** A blocking play starts the WAV
asynchronously and waits in its own thread, checking a stop flag every 20 ms, so
pressing Stop silences the current chunk instead of waiting for it to end. A
stopped chunk counts as played rather than as a playback error.

Further improvements to this web-based panel are **envisioned** — per-message
deletion, a progress bar for the chunk being spoken, a compact layout for narrow
panes. None of them changes the design above: a page, a queue owner, and one op
per button.

## Tools

### `speak`

The main call. Accepts text, hands it to the resident worker, and returns as
soon as the worker has accepted it — synthesis and playback then run to
completion in the worker, regardless of what the caller does with the result.
The worker renders it chunk by chunk into
`<message_id>_<n>.wav` and plays the chunks in order. Rendering runs on its own
thread, a few chunks ahead of the player, so a boundary costs the configured
pause, not a full render's worth of silence.

If no worker is running it returns `worker_not_running` with the command to
start one. It never falls back to synthesizing in the server process.

| Parameter | Default | Meaning |
|---|---|---|
| `text` | — | What to say. Required. |
| `model` | `nano` | `nano` \| `turbo` \| `multilingual` \| `original` |
| `voice` | server setting (`CHATTERBOX_VOICE`) | Reference clip name in the voices dir (filename without extension). Required for multilingual/original |
| `reference_clip` | — | Path to a wav/mp3/flac, as an alternative to `voice` |
| `language` | — | ISO 639-1 code, multilingual only |
| `t3_model` | `v2` | `v2` \| `v3`, multilingual only |
| `exaggeration` | per model | Emotional range. 500M models only — turbo ignores it |
| `cfg_weight` | per model | Guidance strength. 500M models only, turbo `0.0` / 500M `0.5` |
| `temperature` | per model | Sampling temperature (0.05–5.0) |
| `top_p` | per model | Nucleus-sampling cutoff (0.0–1.0) |
| `top_k` | per model | Top-k sampling size (0–1000). Turbo only — the 500M models have no such parameter |
| `repetition_penalty` | per model | Penalise repeated tokens (1.0–2.0) |
| `norm_loudness` | per model | Normalize to −27 LUFS. Turbo only — the 500M models have no such parameter |
| `seed` | random | Reseed torch for a reproducible re-render within the resident session; `0` means random. Not guaranteed byte-identical across a restart |
| `play_audio` | server setting | `false` writes the chunk files only |
| `filename` | generated | Base name for the chunk files; they are written as `<filename>_1.wav`, `<filename>_2.wav`, ... A millisecond-timestamped message id is used when omitted |

Style parameters left unset keep each model's own tuned values rather than a
single global default — turbo is tuned for latency at `cfg_weight=0.0`, the
500M models for fidelity at `0.5`, and forcing one number onto all three
audibly degrades two of them. A knob a model does not support (turbo ignores
`exaggeration`/`cfg_weight`; the 500M models take no `top_k`/`norm_loudness`)
is dropped with a warning rather than crashing.

Text is split into chunks of **whole sentences** — `CHATTERBOX_SENTENCES_PER_CHUNK`
of them per chunk (3 by default), because Chatterbox's own `generate()` truncates
long inputs (turbo degrades past roughly 600 characters and comes back shorter
than a shorter prompt), so a single oversized render would be garbled. Every chunk
is a separate `generate()` call, and each boundary costs a pause plus a fresh
prosody start, which is why short sentences are batched rather than spoken one at
a time — with one sentence per chunk a quick, clipped reply sounds staccato. The
first sentence always stays alone so audio starts as soon as it is rendered, and a
batch never exceeds `CHATTERBOX_MAX_CHUNK_CHARS` characters; a single sentence
longer than that is split further on word boundaries. The chunks share the same
voice, style knobs and reference clip, and are played with
`CHATTERBOX_CHUNK_PAUSE_MS` of silence between them.

A batch is also capped by **speaking time**, not just by sentence count. Measured
on this stack, speech runs at about 17.5 characters per second and rendering costs
about 0.4× the audio it produces — so a chunk may be at most ~2.2× as long as the
one before it and still be ready before that one finishes playing. Chunks therefore
ramp up from the short opening sentence instead of jumping straight to a full
batch. Without that rule the second chunk could take longer to render than the
first takes to play, and the player would sit silent with text still waiting.

Playback waits until roughly two seconds of audio are already on disk, which covers
an opening sentence too short to hide the next render, then keeps up to three
rendered chunks banked ahead. Nothing is timed at runtime: the queue paces the
renderer, so a machine that renders faster simply stays further ahead.

On acceptance `speak` reports the message id, the chunk count, the file names
the chunks will be written to and how many utterances are queued ahead — it
does not wait for the utterance to play. Once it has finished, `tts_status`'s
`worker.last` carries the record of what was written and played: the measured
synthesis and audio durations, `first_audio_s` (submission to the first audible
chunk) and `max_gap_s` (the worst silence the player could not avoid), so
dryness is measurable instead of something you have to hear. Playback is
strictly in order while rendering runs ahead, so the first sentence is audible
within seconds and the rest follow without a render gap between them. Files
stay on disk and replay in filename order; there is no stitched final file.
`stop_speech` cuts the current sentence and cancels everything still queued.
Because the audio belongs to the worker, it keeps playing even if the MCP server
process goes away when the turn ends.

```jsonc
// a quick aside in the stock voice
{ "text": "Build finished. All tests pass.", "model": "turbo" }

// German, cloned voice
{ "text": "Der Build ist fertig.", "model": "multilingual", "language": "de" }

// expressive narration — exaggeration tunes the 500M models; turbo ignores it
{ "text": "And then [chuckle] it compiled on the first try.", "model": "original",
  "exaggeration": 0.4 }
```

### What it actually costs

Measured 2026-10-21 on the development machine (Ryzen 5 5600, RTX 3060 12 GB,
`nano`, voice `seven`, `cuda:1`), same 1494-character, 21-sentence text twice —
once on a freshly started worker, once warm. Nine chunks both times.

| Run | Audio | Synthesis | First audio | Worst gap | Wall clock |
|---|---|---|---|---|---|
| Cold worker | 84.0 s | 36.9 s | 47.1 s | 0.0 s | 134.4 s |
| Warm worker | 81.9 s | 34.7 s | 2.1 s | 0.0 s | 87.3 s |

Read the warm row as the shape of the pipeline: `first_audio_s` plus the audio
duration plus the pause budget accounts for essentially the whole wall clock
(2.1 + 81.9 + 2.0 = 86.0 s against 87.3 s measured), and `max_gap_s` is zero, so
every boundary was covered by a chunk that had finished rendering while the
previous one played. Cold and warm wall clock here is the worker pipeline
(submission to the last chunk spoken), not the tool call — since speak returns
on acceptance, the call itself lasts milliseconds. The cold row is the same
pipeline behind a model load —
the first `generate()` pays for torch, the weights and CUDA init, and there is
nothing to play until it returns. That cost belongs to the first call of a worker
session, not to the streaming design.

Chunk lengths in that run, in seconds of audio: 4.4 → 8.1 → 11.8 → 15.1 → 16.3 →
4.8 → 5.8 → 9.0 → 8.8. The ramp at the start is the duration cap at work. The
short 4.8 s chunk after the 16.3 s one is what the buffer is for: by the time the
long chunk finished, the following ones were already on disk, so the short one
never had to be waited for. The dangerous direction is the reverse — a short
chunk followed by a much longer one — and that is exactly what the cap keeps out
of the list. Absolute numbers are machine-specific; `first_audio_s` and
`max_gap_s` in the `speak` response are the two figures to compare on your own
hardware.

### `tts_status`

Readiness without loading anything: whether the worker is running, the models
available, the configured default voice (`CHATTERBOX_VOICE`), the multilingual
language list, and the tags turbo understands. When the worker is up it also
reports its resolved device, free VRAM per GPU and the speech queue. When it is
down, `worker_running` is `false` and `worker_note` says what to do — the rest of
the response is still accurate, which is what you want when the missing piece is
the worker itself.

### `tts_unload`

Asks the worker to release the resident model and return its VRAM. Needed when
another process wants the card. Only one model is resident at a time — asking
for a different one evicts the previous automatically — but this frees it
entirely. With no worker running there is nothing to release, and the reply says
so rather than pretending it worked.

### `list_voices`

The reference clips available for cloning, with names to pass as `voice=`.

```
{
  "status": "ok",
  "voices_dir": "<absolute path to the voices directory>",
  "count": 3,
  "voices": ["<clip name>", "<clip name>", "<clip name>"],
  "accepted_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
  "requires_reference_clip": ["multilingual", "original"],
  "note": "Pass the name as voice= to speak()."
}
```

`requires_reference_clip` lists the models that **cannot speak without** a clip
(`multilingual`, `original` — the two 500M models, which ship no built-in
voice). `turbo` and `nano` are absent because they have a stock voice and
accept an optional clip on top of it.

**Renamed from `usable_with`.** The old field name read as "the models that
accept a clip", which is every model — cloning works on turbo and nano too, they
just fall back to the built-in voice when none is given. So the name described
the opposite of what the list contained, and only happened to look correct
because the models that *require* a clip are a subset of those that *accept*
one. Any client keying off `usable_with` now reads `null` and silently gets an
empty list — if you have a saved prompt, script or test matching that key, update
it to `requires_reference_clip`.

### `stop_speech`

Asks the worker to cut off the sentence that is playing and cancel every
utterance still queued.

## Voice cloning

Put reference clips in `voices/` (create it; it is not tracked by git). Then
refer to a clip by its filename stem:

```
voices/<your-clip-name>.wav   →   speak(text, voice="<your-clip-name>")
```

Names resolve case-insensitively, and a unique prefix works (`voice="sev"`).
An ambiguous prefix returns the candidates rather than picking one.

### How much reference audio to use

**Give it as much clean single-speaker speech as you have.** Do not trim a good
recording down to a short excerpt.

This tool's own documentation used to say "5-15 seconds". That is wrong, and it
was wrong in the direction that costs quality, so it was replaced with what an
A/B test actually found. Comparing a 40.1 s reference against an 11.8 s excerpt
of that same recording — five texts per arm, plus a same-reference re-render
arm as the noise floor — gave this for impulsive discontinuities per second
(adaptive Laplacian impulse detector):

| Threshold | 40.1 s reference | 11.8 s excerpt | noise floor |
| --------- | --------------- | -------------- | ----------- |
| k=6 fine      | **283.6** | 459.0 | 5.1 |
| k=10 moderate | **108.8** | 173.6 | 7.7 |
| k=20 large    | **72.5**  | 80.1  | 5.2 |

The short reference produced ~36 % more moderate and ~44 % more fine impulses,
several times the model's own sampling variance, and the effect survived
controlling for gain. Worst-spike magnitude, HF energy and zero-crossing
irregularity were indistinguishable, so what degrades is fine crackle rather
than loud pops. The failure mode is too *little* audio, not too much.

We did not test 20 s, 60 s or longer, so this does not establish an optimum —
only that truncating a good recording to "5-15 s" measurably degrades it.
Under about 5 s is genuinely too little.

Three more things that measurement contradicted:

- **A processed or mastered source is fine.** The 40 s reference above was an
  Audacity-mastered file and was the better of the two. Compression is not the
  problem; shortness is.
- **Do not normalise, gain-match or limit the reference.** Applying +23 dB of
  peak normalisation to the excerpt moved rendered output level by 12.1 dB
  (RMS −23.1 vs −35.2 dBFS, against a 0.35 dB level noise floor), and the hotter
  output was measurably grainier. Leave the source level alone.
- **Sample rate and channel count need not match anything.** The 40 s reference
  was 48 kHz stereo and needed no preparation; Chatterbox resamples and
  downmixes internally.

What you still control: no music, no background noise, no second speaker, and
prefer continuous speech over silence-padded audio.

Caveat on all of the above: n=5 per arm on a single speaker, and the detector
measures discontinuities, not timbre, prosody or speaker similarity. It
quantifies one artefact class, not overall quality — your own listening is
still the arbiter.

`list_voices` reports each clip's duration, sample rate and channel count, and
flags clips short enough to be worth warning about, precisely because duration
is what predicted quality here.

## CLI

```bash
# the synthesis worker: start this once, leave it running
uv run mcp-agent-chatterbox worker
uv run mcp-agent-chatterbox worker --port 8765
uv run mcp-agent-chatterbox worker-status
uv run mcp-agent-chatterbox worker-stop

# diagnostics: device, per-GPU VRAM, models, voices
uv run mcp-agent-chatterbox doctor

# synthesize without going through MCP
uv run mcp-agent-chatterbox speak "Build finished." --model turbo
uv run mcp-agent-chatterbox speak "Guten Morgen." --model multilingual --voice <your-clip-name> --language de
uv run mcp-agent-chatterbox speak "..." --no-play

# the control panel (stop, replay, speak by hand) on http://127.0.0.1:8766/
uv run mcp-agent-chatterbox panel

# run the server directly
uv run mcp-agent-chatterbox serve
```

The console script is only on `PATH` inside the project (or via `uv run`). To call
it from anywhere, install a tiny shim — see
[Install the `mac` shim](#install-the-mac-shim-to-run-it-from-anywhere).

`worker` runs the resident process in the foreground; it loads the model on the
first utterance and keeps it loaded. `speak` shares the same code path and blocks
until the queue is drained, so it is the fastest way to check a fresh install
without involving MCP at all. `panel` is the only command that needs no GPU: it
renders nothing and forwards every action to the worker, so it is also safe to
leave running while another job owns the VRAM.

### Install the `mac` shim (to run it from anywhere)

`pip`/`uv` installs the console script **inside this project's environment only**,
so from any other directory `mcp-agent-chatterbox` is not a command. That is the
one thing that bites in practice: you start a resident worker once, and you want
to do it from wherever you happen to be, not from the repo. A one-line `.cmd`
shim on `PATH` closes that gap.

Nothing extra is shipped for this — the shim is three branches and belongs in
*your* bin directory, not in the repo (it embeds an absolute path that differs
per machine). Create the file yourself:

**1. Pick a directory that is already on `PATH`.** `%USERPROFILE%\bin` is the
convention this stack uses; confirm it resolves:

```powershell
$env:PATH -split ';' | Select-String -SimpleMatch "$env:USERPROFILE\bin"
```

If that prints nothing, either use a directory that *is* on `PATH`, or add this
one permanently (then open a new terminal):

```powershell
[Environment]::SetEnvironmentVariable(
  'Path',
  [Environment]::GetEnvironmentVariable('Path', 'User') + ";$env:USERPROFILE\bin",
  'User'
)
```

**2. Create `%USERPROFILE%\bin\mac.cmd`** with your own repo path in `REPO`:

```bat
@echo off
REM mac - run mcp-agent-chatterbox from any directory.
setlocal
set "REPO=<absolute path to this repo>"

REM Preferred: uv resolves the project environment itself.
where uv >nul 2>&1
if %ERRORLEVEL% equ 0 goto :use_uv

REM Fallback: the project venv, for machines without uv.
if exist "%REPO%\.venv\Scripts\python.exe" goto :use_venv

echo mac: mcp-agent-chatterbox not found.
echo   tried: uv run --directory "%REPO%" mcp-agent-chatterbox
echo   tried: "%REPO%\.venv\Scripts\python.exe" -m mcp_agent_chatterbox
echo Is the repo path right, and has the environment been synced?
exit /b 1

:use_uv
uv run --directory "%REPO%" mcp-agent-chatterbox %*
exit /b %ERRORLEVEL%

:use_venv
"%REPO%\.venv\Scripts\python.exe" -m mcp_agent_chatterbox %*
exit /b %ERRORLEVEL%
```

**3. Verify** from a directory that is *not* the repo — this is the whole point:

```powershell
cd C:\
mac worker-status
```

Expect either `worker on 127.0.0.1:8765: idle` or a `no worker on ...` line. Both
prove the shim reached the package; only the first proves a worker is running.

Why two branches: `uv run` is the fast path and cannot drift from the lockfile,
but it is useless on a machine without `uv`. The venv fallback needs no tooling
beyond the synced environment. `%*` forwards every argument and `exit /b
%ERRORLEVEL%` keeps the exit code, so `mac worker-status` still exits non-zero
when no worker is up — which is what makes it usable in a script.

**Do not wrap the two commands in `if (...)` blocks.** In batch, `%ERRORLEVEL%`
inside a parenthesised block is substituted when the block is *parsed*, not when
each line runs, so `exit /b %ERRORLEVEL%` would return the status from before
the command — 0 every time. That silently turns a failing `mac speak` into a
successful one. Labelled `goto` branches keep each `%ERRORLEVEL%` on a line of
its own, read after the command it belongs to.

Then from anywhere: `mac worker`, `mac worker-status`, `mac worker-stop`,
`mac doctor`, `mac speak "..."`.

## Web surface (HTTP transport)

Yes, there is one, and it is the least-comfortable part of this server. Read
this before binding it to anything but loopback.

```bash
# streamable-http on loopback (the default host)
uv run mcp-agent-chatterbox serve --http --port 8123
# endpoint: http://127.0.0.1:8123/mcp
```

`serve` takes `--http` (a flag), not `--transport <name>`. The underlying
`server.run()` also accepts `"sse"`, but **no CLI path reaches it** — it is
library-only and deprecated in the MCP spec. Prefer `streamable-http`.

**What was verified.** A real MCP client over real uvicorn: 5 tools advertised,
`tts_status`, `speak` and `tts_unload` all correct, plus a 4-client concurrent
burst on a cold engine that produced exactly **one** weight load. That last one
is the load-bearing test — see the concurrency note below.

**No authentication. None.** There is no token, no TLS, no per-client identity.
`--host` defaults to `127.0.0.1`, which is the only thing keeping this off your
network. `--host 0.0.0.0` publishes an unauthenticated text-to-speech endpoint
to every host that can route to you. Do not do that on an untrusted network.

**One process, one engine, all clients share it.** The HTTP server builds a
single `AppContext`, so:

- `stop_speech()` stops playback for *every* connected client, not just yours.
- `tts_unload()` frees the GPU out from under any in-flight `speak`.
- There is no per-client output directory; all sessions write to the same
  `output_dir`.

**Concurrent clients are a queue, not a parallel server.** Measured with 4
simultaneous clients: per-call wall times of 1.63s / 3.34s / 4.72s / 6.22s — a
staircase, because `synthesize()` holds an engine-wide lock (see below). Four
clients took 6.4s where a parallel server would take ~1.6s. That is the correct
trade for one GPU and one resident model: correctness over throughput. `tts_status`
deliberately takes no lock and stays instant even mid-queue.

**The concurrency bug this surface would have had.** The engine was written for
stdio, where the transport serialises one client's calls, so `load()` was an
unlocked check-then-act over a process-global model cache. Under HTTP that is a
live race: two threads both miss the cache check and both call
`from_pretrained()` — 4 copies of turbo's weights on a 12 GB card that the VRAM
preflight had already cleared — and a request for a different model could
`gc.collect()` + `torch.cuda.empty_cache()` while another request was still
generating. `load()`, `unload()` and `synthesize()` are now serialised behind a
re-entrant lock, and `tests/test_concurrency.py` fails if that lock is removed.

**Treat it as single-operator.** It is fine for a local dashboard, a second
harness on the same box, or testing. It is not an authenticated service.

## Configuration

Settings live in a **`.env` file at the repo root**, read once at startup.
Copy `.env.example` and edit it:

```bash
cp .env.example .env
```

The file documents every key inline. `.env` is git-ignored — it holds
machine-specific values (paths, GPU choice, the model you have cached) — while
`.env.example` is tracked and is the reference list.

**Why a file and not just the harness's `env` block?** Because `env` blocks are
not reliably delivered. LM Studio's `mcp.json` and Hermes both accept one, both
are spec-correct, and neither was observed forwarding it to the server
subprocess — the server came up on its compiled defaults while the config asked
for something else, silently. A file the server opens itself has no such
dependency: the same settings apply under LM Studio, Hermes and opencode, and
there is one place to edit. See
[The `environment` block](#the-environment-block-and-why-it-is-not-the-source-of-truth)
for using `env` as a per-harness override on top of this.

**Precedence is environment-over-file.** A variable already present in the
process environment is never overwritten by `.env`. So `CHATTERBOX_MODEL=turbo
uv run …`, a CI job, or a test's monkeypatch all win over the file; the file
holds the durable value.

No secrets are involved.

| Variable | Default | Meaning |
|---|---|---|
| `CHATTERBOX_MODEL` | `nano` | Model used when `speak` names none |
| `CHATTERBOX_VOICE` | unset | Default reference clip (name in the voices dir) used when a `speak` call names neither `voice=` nor `reference_clip=`. Skips the tool-level `reference_clip_required` for multilingual/original, and applies to turbo/nano too. A name that does not exist fails fast with `voice_not_found` — never a silent substitution |
| `CHATTERBOX_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:N` |
| `CHATTERBOX_GPU_INDEX` | unset | Pin a CUDA ordinal. Overrides the `auto` heuristic |
| `CHATTERBOX_OUTPUT_DIR` | `tts_output` | Where WAVs are written |
| `CHATTERBOX_VOICES_DIR` | `voices` | Where reference clips live |
| `CHATTERBOX_LOGS_DIR` | `logs` | Rotating log file location |
| `CHATTERBOX_AUTOPLAY` | `1` | Play each chunk as it renders |
| `CHATTERBOX_MAX_CHARS` | `4000` | Reject longer single requests |
| `CHATTERBOX_MAX_CHUNK_CHARS` | `500` | Hard cap for one rendered chunk — one sentence, or a whole batch. A single sentence longer than this is split on word boundaries |
| `CHATTERBOX_SENTENCES_PER_CHUNK` | `3` | Sentences batched into one chunk. Every chunk is a separate `generate()` call costing a pause and a fresh prosody start, so `1` makes short-sentence speech staccato. The first sentence always stays alone (fast start); a batch never exceeds `CHATTERBOX_MAX_CHUNK_CHARS`, and it is capped by speaking time too, so it can never grow past ~2.2× the playback length of the chunk before it |
| `CHATTERBOX_CHUNK_PAUSE_MS` | `250` | Silence between played chunks — between sentence groups when batching, not between every sentence |
| `CHATTERBOX_WORKER_PORT` | `8765` | Loopback port the worker listens on. The server and the worker read this same value; point them at different ones and `speak` reports `worker_not_running` while a healthy worker sits on the other port |
| `CHATTERBOX_WORKER_CONNECT_TIMEOUT` | `5` | Seconds to wait for the **connection** to the worker. Bounds the connect only, never the read — the dispatch loop answers one caller at a time, so a request may wait behind another connection, and cutting the read off would turn that wait into a false failure |
| `CHATTERBOX_VRAM_MB` | `4096` | Free-VRAM floor for a load attempt |
| `CHATTERBOX_STRICT_VRAM` | `0` | `1` refuses a load below the floor instead of warning |
| `CHATTERBOX_HF_HOME` | unset | Where the Hugging Face cache lives. Does not compete with `HF_HOME` — it **sets** it, at load time, before `huggingface_hub` seals the value on import. Set it in `.env` so the cache root cannot be lost to a harness that does not forward your environment. See [Model weights](#model-weights) |

Relative paths resolve against the project root, never the process CWD, since
MCP harnesses spawn servers with an unpredictable working directory.

**Windows note:** use forward slashes in path values. Backslashes are not
reliably handled by `python-dotenv`.

The server's first log line states exactly which config was resolved, so a
setting that failed to arrive is visible without issuing a tool call:

```
[server] mcp-agent-chatterbox v0.1.0 ready (output=…, voices=…, model=nano, voice=seven, config=I:\…\.env)
```

### Choosing a GPU

`auto` picks the CUDA device with the **most free memory**, which matters when
one card is occupied by something else — a loaded LLM, a rendering job. On a
box where LM Studio holds both GPUs, `doctor` shows the headroom per card and
`CHATTERBOX_GPU_INDEX` pins a specific one.

## Design notes

**Synthesis lives in its own process, and that is not incidental.** The server
used to render in-process and reliably stalled after two sentences — over both
transports, with the client pipe open and closed. A plain process running the
same code finished the reply. So the worker owns the model and the playback; the
server owns the protocol and carries text across a loopback socket. Requests are
one JSON line each way, one request per connection, and the worker is strictly
single-threaded: one GPU, one resident model, and a Chatterbox model instance
carries mutable state. Nothing may end that loop — a handler exception is logged
and returned as `worker_error` rather than allowed to kill a resident process
that the next utterance also depends on.

**A card that cannot be queried is skipped, not fatal.** If
`torch.cuda.mem_get_info()` throws for every ordinal — a driver that has not
finished initialising, a card held by another process — `auto` falls back to
CPU instead of raising. A `gpu_index` pin outside the visible range is the one
case that errors, because there the user asked for a specific card and silently
using another would be worse than failing.

**One resident model.** `MAX_RESIDENT = 1`. A 350M turbo model and a 500M
multilingual model do not both fit on a 12 GB card that already has an LLM
resident. Asking for a different model evicts the previous one and empties the
CUDA allocator cache.

**The engine is locked; the status reads are not.** Inside the worker,
`load()`, `unload()` and `synthesize()` serialise behind one re-entrant lock,
because the model cache is process-global and a single model instance is not safe
to call `generate()` on from two threads. The worker's accept loop is
single-threaded on top of that, so only one speak is ever in flight. `tts_status`,
`gpu_report` and `free_vram_mb()` deliberately take no lock, so a ten-second cold
weight load never stalls a diagnostic call — which is the call you need while
staring at a slow first request. The lock matters for HTTP, where the engine
would otherwise be reached concurrently; it was the difference between working
and OOM-ing there. The reasoning and the measured numbers are in the web-surface
section.

**The device string goes straight to `from_pretrained()`.** Chatterbox's own
`.to(device)` is incomplete upstream: in `ChatterboxTTS` and
`ChatterboxMultilingualTTS` it moves only `t3` and `gen`, leaving `ve`,
`s3gen` and `conds` on the old device and never updating `self.device` — so
`generate()` would then move tensors to the wrong place. In
`ChatterboxTurboTTS` it iterates a `gen` attribute that `__init__` never
sets, so it raises `AttributeError`. There is therefore no
load-on-CPU-then-move path, which is why free VRAM is checked *before* a load
instead of after a failure.

**VRAM is checked before, not during.** Because weights land directly on the
device, a load started with too little headroom dies part way through reading
the checkpoint with a bare `torch.cuda.OutOfMemoryError`, after the download.
The engine queries free VRAM first, warns (or refuses under
`CHATTERBOX_STRICT_VRAM`), and maps a genuine OOM onto an actionable
`cuda_out_of_memory` payload.

**Weights are never loaded at startup.** The server builds no engine at all, and
a status or voice-listing call must not pay the multi-second torch import plus a
multi-gigabyte fetch. The worker, which is started deliberately and lives for
hours, loads the model on its first utterance and keeps it resident.

## Dependency advisories

Two of the dependencies GitHub flags here cannot both be fixed, and the reason is
worth writing down so nobody "fixes" it the wrong way.

**`certifi` — fixed.** The lock held `2026.7.22`, whose bundle included emSign
root CAs with compromised intermediate keys; it is now `2026.10.19`. This one
mattered because it is the CA bundle used for HTTPS when downloading model
weights from Hugging Face.

**`Pillow` — cannot be fixed while `gradio` is present.** Pillow 12.3.0 carries
nine advisories (2 critical, 6 high, 2 moderate), all fixed in 13.0.0. But
`gradio` requires `pillow<13.0,>=8.0` — including the latest release (6.30.0) —
and `chatterbox-tts` requires `gradio`. So `pillow 13.0.0` is unresolvable, and
`uv lock --upgrade-package pillow==13.0.0` fails with a genuine conflict, not a
stale lock. Verified against PyPI metadata rather than the resolver message
alone:

```
gradio 6.30.0     -> pillow<13.0,>=8.0
diffusers 0.38.0  -> Pillow            (unconstrained)
pillow (PyPI)     -> 13.0.0
```

**Why the exposure is close to zero.** This server takes text and returns WAV. It
has no image input path at all, and every one of those advisories lives in an
image decoder — JPEG, TIFF, PDF, EPS, GIF, `_imagingmorph`. Pillow is in the tree
only because `chatterbox-tts` ships a Gradio demo that this project never runs.
Nothing here hands Pillow a file.

**What not to do.** Do not add `[tool.uv] override-dependencies = ["pillow>=13.0.0"]`.
It silences the alerts by overriding a constraint `gradio` declares outright, so
Gradio's image code would run against a Pillow release it was never tested with —
trading a warning about code you never execute for an untested runtime you cannot
see. The correct fix arrives when Gradio raises its cap; re-check with:

```bash
uv lock --upgrade-package pillow --dry-run
```

## Troubleshooting

**`cuda_out_of_memory` / "not enough free VRAM".** Something else is holding
the card. Check `doctor` for per-GPU free memory, unload the model in LM
Studio, or set `CHATTERBOX_GPU_INDEX` to a freer card.

**`reference_clip_required`.** `multilingual` and `original` have no stock
voice. Add a clip to `voices/` and pass `voice=`, or use `model="turbo"`.

**`voice_not_found`.** Run `list_voices` for the exact names.

**`unsupported_language`.** `language` takes an ISO 639-1 code and only
applies to `multilingual`. `tts_status` lists all 23.

**First call is very slow.** That is the Hugging Face download. Later calls
reuse the cache — **unless the cache root moved.** If the first call is slow
*again* after you changed a config or restarted a harness, the server is
probably re-downloading to a different location: check the model's
registration for `CHATTERBOX_HF_HOME` and look for a second copy under
`C:\Users\<you>\.cache\huggingface\hub`. See
[Model weights](#model-weights) for why `HF_HOME` in your shell is not always
enough.

**Audio does not come out.** First check the worker is up: `tts_status` ->
`worker_running`, or `mcp-agent-chatterbox worker-status`. If it is down, start
it (`mcp-agent-chatterbox worker`) and try again — the server never synthesizes
on its own. With a worker running, `tts_status` -> `worker` shows the queue:
`pending` above zero means it is still rendering, and `last` carries `written`,
`played_chunks`, `errors`, `first_audio_s` and `max_gap_s` for the most recent
utterance — a `max_gap_s` that keeps coming back well above the configured chunk
pause means the renderer is falling behind the player on this machine. Playback
needs Windows; on other platforms the chunk files are still written. `stop_speech`
clears anything queued.

**`worker_not_running`.** Nothing is listening on `CHATTERBOX_WORKER_PORT`. The
MCP server does not synthesize; it only forwards. Start the worker once and leave
it running. Double-check both sides read the same port value.

**`turbo` and the watermarker.** Turbo's output carries Resemble AI's
watermarking mark, added by the vendored Perth library. Turning it off would be
a licence question, not a technical one, so it stays on.

**Logs.** `logs/mcp_agent_chatterbox.log` (rotating, 5 MB × 5). Unhandled
tracebacks go there, never into the model's context. The console handler writes
to stderr with `errors="replace"`, so a character the Windows console code page
cannot represent degrades to `?` instead of raising or garbling the line —
Chatterbox logs a `✅` of its own, so this is not hypothetical.

**`uv sync` fails with file-lock/`Access denied` errors (Windows).** The running
MCP server holds `Scripts/mcp-agent-chatterbox.exe` and the site-packages files
it imported, so a mid-session sync cannot uninstall them. Worse, a sync that
starts anyway and dies halfway can leave a **husked package** — the metadata
looks installed but the package directory lost its `__init__.py` and most files
(a real incident gutted `coverage` this way, and every model then failed to
load with `AttributeError: module 'coverage' has no attribute 'types'`, because
`librosa` → `numba` imports it for tracing). Restart opencode so the server
process exits, then redo `uv sync`. If the husk is already there, repair it in
place: `uv pip install --force-reinstall coverage==7.16.1` (or the husked
package's own version) before relying on the environment. `import librosa`
succeeding is the cheapest canary for this whole chain.

## Development

```bash
uv sync --extra dev
uv run pytest              # unit tests, no model download
uv run ruff check .
uv run ruff format .
```

Tests stub torch and the Chatterbox classes, so the suite runs in seconds and
does not touch the network or the GPU. For a real end-to-end check use
`mcp-agent-chatterbox speak`.

## License

MIT — see [LICENSE](LICENSE). Chatterbox itself is MIT, © Resemble AI.
