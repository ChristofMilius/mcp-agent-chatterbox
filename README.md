# mcp-agent-chatterbox

MCP server for local, GPU-accelerated text-to-speech with
[Chatterbox](https://github.com/resemble-ai/chatterbox) (Resemble AI, MIT).
Text goes in, spoken audio comes out — no cloud service, no API key, no
per-character cost.

Ships two things:

- **An MCP tool surface** (`speak`, `tts_status`, `tts_unload`, `list_voices`,
  `stop_speech`) that any MCP client can call.
- **An opencode plugin** (`.opencode/plugins/tts.js`) that adds a `speak` tool
  and a `/speak` command to opencode itself, and installs itself globally so
  it works in every project, not just this repo.

## The three models

| Key | Params | Languages | Reference clip | Notes |
|---|---|---|---|---|
| `turbo` | 350M | English | **not needed** | Default. Fastest, lowest VRAM. Built-in voice. Understands paralinguistic tags inline: `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`, `[whisper]`. |
| `multilingual` | 500M | 23 (incl. German) | **required** | `language="de"`, `en`, `fr`, … `t3_model="v2"` (default) or `"v3"`. |
| `original` | 500M | English | **required** | The CFG / exaggeration-tuning model, for delivery style control. |

Only `turbo` ships a stock voice. The other two raise
`AssertionError: Please prepare_conditionals first or specify audio_prompt_path`
without a reference clip, so the tool checks this up front and returns a
`reference_clip_required` payload naming the fix instead of a stack trace.

## Requirements

- Python 3.13 (see [Why 3.13 only](#why-313-only))
- An NVIDIA GPU with a driver of 525+ for the CUDA 12.4 wheels. CPU works but
  is slow.
- ~6 GB of free VRAM for a comfortable margin (turbo needs ~2-3 GB, the 500M
  models ~3-4 GB)
- Windows for audio playback. On other platforms the WAV is still written; the
  `played` field comes back `false` with a reason.

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

Expect `2.6.0+cu124 True`.

### Why `setuptools<81` is pinned

This is the one dependency in `pyproject.toml` that looks arbitrary, so here is
the whole story.

The **turbo** model embeds a copy of [Perth](https://github.com/resemble-ai/perth),
Resemble AI's audio watermarking library. That copy is vendored inside the
`chatterbox-tts` sdist — it ships as loose files with no package metadata, and
there is no `perth-watermarker` distribution on PyPI to replace it with.

`perth/perth_net/__init__.py` starts with:

```python
from pkg_resources import resource_filename
```

`pkg_resources` was part of setuptools until setuptools 81 dropped it. So on a
current setuptools the import raises `ImportError`, which `perth/__init__.py`
swallows and turns into `PerthImplicitWatermarker = None` — and
`ChatterboxTurboTTS.__init__` then calls that `None`:

```
TypeError: 'NoneType' object is not callable
```

The failure is silent by design upstream (the `try/except ImportError` is
intentional, so Perth works without the neural watermarker), which is exactly
why it surfaces as a confusing `TypeError` at load time rather than a clear
message. Pinning `setuptools<81` restores `pkg_resources` and the turbo model
loads.

Consequences worth knowing:

- The pin is a **runtime** requirement, not a build-time one. The dependency
  graph still works without it; turbo just fails to load.
- Only the **turbo** model is affected. `multilingual` and `original` never
  touch Perth and load fine on any setuptools.
- If a future `chatterbox-tts` release fixes the vendored import, drop the pin.
  The symptom to watch for is the `TypeError` above returning.

## Model weights

Weights are **not** bundled. They download from Hugging Face on the first
`speak()` call and are cached in the standard location
(`%USERPROFILE%\.cache\huggingface\hub` on Windows, override with `HF_HOME`).
Subsequent calls reuse the cache.

- `turbo` → `ResembleAI/chatterbox-turbo`
- `multilingual` / `original` → `ResembleAI/chatterbox`

Nothing is downloaded by `doctor` or `tts_status`, so those stay fast and work
offline.

## Registering the MCP server

Add to `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "mcp": {
    "mcp-agent-chatterbox": {
      "type": "local",
      "command": [
        "uv", "run", "--directory", "<absolute path to this repo>",
        "mcp-agent-chatterbox", "serve"
      ],
      "environment": {
        "CHATTERBOX_MODEL": "turbo"
      },
      "enabled": true
    }
  }
}
```

Restart opencode. The server speaks stdio, loads no weights at startup, and
stays responsive while the model is idle.

## The opencode plugin

`.opencode/plugins/tts.js` adds a `speak` tool and a `/speak` command to
opencode. It **self-replicates** into `~/.config/opencode/plugins/` the first
time opencode loads it from this repo, so the tool is available in every
project rather than only sessions started inside this folder.

The plugin deliberately does not auto-speak assistant replies: loading a model
costs seconds and VRAM, and unsolicited audio on every turn would be awful.
Speaking stays an explicit choice, by tool call or by `/speak`.

## Tools

### `speak`

The main call. Renders text, writes a WAV to the output dir, plays it.

| Parameter | Default | Meaning |
|---|---|---|
| `text` | — | What to say. Required. |
| `model` | `turbo` | `turbo` \| `multilingual` \| `original` |
| `voice` | — | Reference clip name in the voices dir (filename without extension) |
| `reference_clip` | — | Path to a wav/mp3/flac, as an alternative to `voice` |
| `language` | — | ISO 639-1 code, multilingual only |
| `t3_model` | `v2` | `v2` \| `v3`, multilingual only |
| `exaggeration` | per model | Emotional range |
| `cfg_weight` | per model | Guidance strength. turbo `0.0`, 500M models `0.5` |
| `temperature` | per model | Sampling temperature |
| `play_audio` | server setting | `false` writes the file only |
| `wait` | `false` | Block until playback finishes |
| `filename` | generated | Output basename |

Style parameters left unset keep each model's own tuned values rather than a
single global default — turbo is tuned for latency at `cfg_weight=0.0`, the
500M models for fidelity at `0.5`, and forcing one number onto all three
audibly degrades two of them.

```jsonc
// a quick aside in the stock voice
{ "text": "Build finished. All tests pass.", "model": "turbo" }

// German, cloned voice
{ "text": "Der Build ist fertig.", "model": "multilingual", "voice": "seven", "language": "de" }

// expressive narration
{ "text": "And then [chuckle] it compiled on the first try.", "model": "turbo",
  "exaggeration": 0.4 }
```

### `tts_status`

Runtime readiness without loading anything: packages, torch/CUDA versions,
resolved device, free VRAM per GPU, resident model, available models, the
multilingual language list, and the tags turbo understands. Use it as a
preflight.

### `tts_unload`

Releases the resident model and returns its VRAM. Needed when another process
wants the card. Only one model is resident at a time — asking for a different
one evicts the previous automatically — but this frees it entirely.

### `list_voices`

The reference clips available for cloning, with names to pass as `voice=`.

### `stop_speech`

Cuts off playback started with `wait=false`.

## Voice cloning

Put reference clips in `voices/` (create it; it is not tracked by git). For
best results use **5-15 seconds** of clean single-speaker audio — no music, no
background noise, no second voice. Then refer to a clip by its filename stem:

```
voices/seven.wav   →   speak(text, voice="seven")
```

Names resolve case-insensitively, and a unique prefix works (`voice="sev"`).
An ambiguous prefix returns the candidates rather than picking one.

## CLI

```bash
# diagnostics: device, per-GPU VRAM, models, voices
uv run mcp-agent-chatterbox doctor

# synthesize without going through MCP
uv run mcp-agent-chatterbox speak "Build finished." --model turbo
uv run mcp-agent-chatterbox speak "Guten Morgen." --model multilingual --voice seven --language de
uv run mcp-agent-chatterbox speak "..." --no-play --wait

# run the server directly
uv run mcp-agent-chatterbox serve
```

The CLI `speak` subcommand shares its code path with the MCP tool
(`mcp_agent_chatterbox.speak.speak_once`), so it is the fastest way to check a
fresh install.

## Configuration

All settings are environment variables, read at startup. No secrets.

| Variable | Default | Meaning |
|---|---|---|
| `CHATTERBOX_MODEL` | `turbo` | Model used when `speak` names none |
| `CHATTERBOX_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:N` |
| `CHATTERBOX_GPU_INDEX` | unset | Pin a CUDA ordinal. Overrides the `auto` heuristic |
| `CHATTERBOX_OUTPUT_DIR` | `tts_output` | Where WAVs are written |
| `CHATTERBOX_VOICES_DIR` | `voices` | Where reference clips live |
| `CHATTERBOX_LOGS_DIR` | `logs` | Rotating log file location |
| `CHATTERBOX_AUTOPLAY` | `1` | Play audio after writing |
| `CHATTERBOX_MAX_CHARS` | `4000` | Reject longer single requests |
| `CHATTERBOX_VRAM_MB` | `4096` | Free-VRAM floor for a load attempt |
| `CHATTERBOX_STRICT_VRAM` | `0` | `1` refuses a load below the floor instead of warning |
| `HF_HOME` | — | Override the Hugging Face cache location |

Relative paths resolve against the project root, never the process CWD, since
MCP harnesses spawn servers with an unpredictable working directory.

### Choosing a GPU

`auto` picks the CUDA device with the **most free memory**, which matters when
one card is occupied by something else — a loaded LLM, a rendering job. On a
box where LM Studio holds both GPUs, `doctor` shows the headroom per card and
`CHATTERBOX_GPU_INDEX` pins a specific one.

## Design notes

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

**Weights are never loaded at startup.** A status or voice-listing call must
not pay the multi-second torch import plus a multi-gigabyte fetch.

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
reuse the cache.

**Audio does not come out.** Check `played` and `playback_note` in the
response. Playback needs Windows; on other platforms the WAV is still written.
`stop_speech` clears anything queued.

**`turbo` and the watermarker.** Turbo's output carries Resemble AI's
watermarking mark, added by the vendored Perth library. Turning it off would be
a licence question, not a technical one, so it stays on.

**Logs.** `logs/mcp_agent_chatterbox.log` (rotating, 5 MB × 5). Unhandled
tracebacks go there, never into the model's context. The console handler writes
to stderr with `errors="replace"`, so a character the Windows console code page
cannot represent degrades to `?` instead of raising or garbling the line —
Chatterbox logs a `✅` of its own, so this is not hypothetical.

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
