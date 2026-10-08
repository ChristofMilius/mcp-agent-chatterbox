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

## The four models

| Key | Params | Languages | Reference clip | Notes |
|---|---|---|---|---|
| `turbo` | 350M | English | **not needed** | Default. Fastest, lowest VRAM. Built-in voice. Understands paralinguistic tags inline: `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`, `[whisper]`. |
| `nano` | 110M | English | **not needed** | Turbo's small sibling: same architecture, single-step decoder, tags and built-in voice, tightest latency/memory budget (also runs on CPU). |
| `multilingual` | 500M | 23 (incl. German) | **required** | `language="de"`, `en`, `fr`, … `t3_model="v2"` (default) or `"v3"`. |
| `original` | 500M | English | **required** | The CFG / exaggeration-tuning model, for delivery style control. |

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
        // Which model speak() uses when a call names none.
        // No reference clip is needed for turbo/nano, which both have a
        // built-in voice and are faster to load than the 500M models.
        "CHATTERBOX_MODEL": "turbo",

        // Default reference clip when a call names neither voice= nor
        // reference_clip= (optional). Resolved like any other voice name;
        // a typo fails fast with voice_not_found, it never silently
        // renders in another voice.
        "CHATTERBOX_VOICE": "<a name from list_voices>",

        // The multi-GPU box: pin which card TTS uses. Doctor shows which
        // one LM Studio is not sitting on.
        "CHATTERBOX_GPU_INDEX": "0",

        // Play the audio after writing it (1/0/true/false; default on).
        "CHATTERBOX_AUTOPLAY": "1",

        // Free-VRAM floor (MiB) for a model-load attempt.
        "CHATTERBOX_VRAM_MB": "4096",

        // WAVs and logs; relative paths resolve against the project root,
        // never the process CWD.
        "CHATTERBOX_OUTPUT_DIR": "tts_output",
        "CHATTERBOX_LOGS_DIR": "logs"
      },
      "enabled": true
    }
  }
}
```

Every setting from the [Configuration](#configuration) table can go in that
`environment` block. Nothing is loaded at startup, so an `enabled: true`
server stays idle until a tool call arrives.

**LM Studio Bionic / Claude Desktop / other clients:** the same command and
environment work in their own config format — strict JSON (no comments),
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
        "CHATTERBOX_VOICE": "<a name from list_voices>",
        "CHATTERBOX_GPU_INDEX": "1"
      }
    }
  }
}
```

Restart the client. The server speaks stdio, loads no weights at startup,
and stays responsive while the model is idle.

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
| `model` | `turbo` | `turbo` \| `nano` \| `multilingual` \| `original` |
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
| `play_audio` | server setting | `false` writes the file only |
| `wait` | `false` | Block until playback finishes |
| `progressive` | server setting | With playback on, play each long-text chunk the moment it renders instead of waiting for the whole utterance |
| `filename` | generated | Output basename |

Style parameters left unset keep each model's own tuned values rather than a
single global default — turbo is tuned for latency at `cfg_weight=0.0`, the
500M models for fidelity at `0.5`, and forcing one number onto all three
audibly degrades two of them. A knob a model does not support (turbo ignores
`exaggeration`/`cfg_weight`; the 500M models take no `top_k`/`norm_loudness`)
is dropped with a warning rather than crashing.

Text longer than `CHATTERBOX_MAX_CHUNK_CHARS` is spoken as one file but
rendered in sentence-aligned chunks: Chatterbox's own `generate()` truncates
long inputs (turbo degrades past roughly 600 characters and comes back shorter
than a shorter prompt), so a single oversized render would be garbled. The
chunks share the same voice, style knobs and reference clip, and are stitched
together with `CHATTERBOX_CHUNK_PAUSE_MS` of silence between them. With
`progressive` (or `CHATTERBOX_PROGRESSIVE=1`) and playback enabled, chunk 1
starts playing the moment it is rendered while the later chunks are still being
generated — the first audio lands in about the time of one chunk instead of the
whole utterance. Each chunk is staged to its own `*.progressive-N.wav` in the
output dir during playback and removed afterwards; the stitched final file is
still written as usual.

```jsonc
// a quick aside in the stock voice
{ "text": "Build finished. All tests pass.", "model": "turbo" }

// German, cloned voice
{ "text": "Der Build ist fertig.", "model": "multilingual", "language": "de" }

// expressive narration — exaggeration tunes the 500M models; turbo ignores it
{ "text": "And then [chuckle] it compiled on the first try.", "model": "original",
  "exaggeration": 0.4 }
```

### `tts_status`

Runtime readiness without loading anything: packages, torch/CUDA versions,
resolved device, free VRAM per GPU, resident model, available models, the
configured default voice (`CHATTERBOX_VOICE`), the multilingual language
list, and the tags turbo understands. Use it as a preflight.

### `tts_unload`

Releases the resident model and returns its VRAM. Needed when another process
wants the card. Only one model is resident at a time — asking for a different
one evicts the previous automatically — but this frees it entirely.

### `list_voices`

The reference clips available for cloning, with names to pass as `voice=`.

### `stop_speech`

Cuts off playback started with `wait=false`.

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
# diagnostics: device, per-GPU VRAM, models, voices
uv run mcp-agent-chatterbox doctor

# synthesize without going through MCP
uv run mcp-agent-chatterbox speak "Build finished." --model turbo
uv run mcp-agent-chatterbox speak "Guten Morgen." --model multilingual --voice <your-clip-name> --language de
uv run mcp-agent-chatterbox speak "..." --no-play --wait

# run the server directly
uv run mcp-agent-chatterbox serve
```

The CLI `speak` subcommand shares its code path with the MCP tool
(`mcp_agent_chatterbox.speak.speak_once`), so it is the fastest way to check a
fresh install.

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

All settings are environment variables, read at startup. No secrets.

| Variable | Default | Meaning |
|---|---|---|
| `CHATTERBOX_MODEL` | `turbo` | Model used when `speak` names none |
| `CHATTERBOX_VOICE` | unset | Default reference clip (name in the voices dir) used when a `speak` call names neither `voice=` nor `reference_clip=`. Skips the tool-level `reference_clip_required` for multilingual/original, and applies to turbo/nano too. A name that does not exist fails fast with `voice_not_found` — never a silent substitution |
| `CHATTERBOX_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:N` |
| `CHATTERBOX_GPU_INDEX` | unset | Pin a CUDA ordinal. Overrides the `auto` heuristic |
| `CHATTERBOX_OUTPUT_DIR` | `tts_output` | Where WAVs are written |
| `CHATTERBOX_VOICES_DIR` | `voices` | Where reference clips live |
| `CHATTERBOX_LOGS_DIR` | `logs` | Rotating log file location |
| `CHATTERBOX_AUTOPLAY` | `1` | Play audio after writing |
| `CHATTERBOX_PROGRESSIVE` | `0` | With playback on, play each long-text chunk the moment it renders instead of waiting for the whole utterance |
| `CHATTERBOX_MAX_CHARS` | `4000` | Reject longer single requests |
| `CHATTERBOX_MAX_CHUNK_CHARS` | `500` | Long text auto-splits into sentence-aligned chunks of up to this many chars |
| `CHATTERBOX_CHUNK_PAUSE_MS` | `250` | Silence inserted between concatenated chunks |
| `CHATTERBOX_VRAM_MB` | `4096` | Free-VRAM floor for a load attempt |
| `CHATTERBOX_STRICT_VRAM` | `0` | `1` refuses a load below the floor instead of warning |
| `CHATTERBOX_HF_HOME` | unset | Home for the Hugging Face cache (sets `HF_HOME` before the model loads). Handy when the weights live on a big drive or a path the default `~/.cache/huggingface` cannot cover |

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

**The engine is locked; the status reads are not.** `load()`, `unload()` and
`synthesize()` serialise behind one re-entrant lock, because the model cache is
process-global and a single model instance is not safe to call `generate()` on
from two threads. `tts_status`, `gpu_report` and `free_vram_mb()` deliberately
take no lock, so a ten-second cold weight load never stalls a diagnostic call —
which is the call you need while staring at a slow first request. The lock is
invisible over stdio and is the difference between working and OOM-ing over
HTTP; the reasoning and the measured numbers are in the web-surface section.

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
