---
name: voice-output
description: Use when the user asks you to speak, answer aloud, reply by voice, use the voice/speech mode, or says anything like "say it", "tell me out loud", or "respond with voice". Teaches the agent to always route the spoken reply through the chatterbox MCP speak() tool instead of only printing text — and to speak its own generated answer, never a repetition of the user's message. Applies to the mcp-agent-chatterbox MCP server (speak, tts_status, list_voices, stop_speech, tts_unload).
---

# Voice output with Chatterbox

When the user wants your reply **spoken**, text alone is a failed turn. The
spoken reply is the deliverable; plain chat text is a fallback, not an
alternative.

## The core rule

**If the user asks for a spoken reply, call `speak()` with your reply as
`text` — every time, no exceptions.** Do not describe what you would say, do
not ask whether to speak, do not print the text and offer to read it. Render
it.

- The `text` argument **is your reply**. Write it the way it should sound
  aloud: complete sentences, no markdown, no bullet lists, no code blocks,
  no URLs to spell out unless asked.
- You may still write a short text reply in chat (one line, e.g.
  "Speaking now:"), but the substance goes into `speak()`.
- If the user's instruction is ambiguous ("go on", "tell me more") **and**
  voice mode was established earlier in the conversation, stay in voice
  mode. Mode persists until the user says stop.
- One instruction to speak covers the whole exchange, not just one turn.

## Voice mode: ON means every reply is spoken

"Reply by voice", "speak from now on", "voice mode", "answer aloud" — any
of these turns voice mode **ON**. While it is on, **every substantive
reply is a `speak()` call**. There is nothing else to decide: voice mode
ON = the answer comes out of the speaker.

- The only text-only turns allowed while voice mode is on are a one-line
  prelude ("Speaking now:") and the single "which voice?" question.
- Voice mode stays ON until the user says stop. Do not drop back to
  plain text after a few turns "because the request only applied once".
- If the user never turned voice mode on, reply in text as usual. If
  they turned it on, speak. There is no middle state to judge.

## The most important rule: speak YOUR answer, never the user's words

`text=` carries the reply **you** generated. The user's message is the
question; the text you speak is the answer. **Never pass the user's own
words, never copy their message, never echo the question back.**

- **Wrong:** user says *"Reply by voice: what time is it?"* and you call
  `speak(text="Reply by voice: what time is it?")` — the speaker reads the
  question back. That is a failed turn.
- **Right:** `speak(text="It is just past four thirty in the afternoon.")`
  — your own answer, in your own words.

**The one exception:** the user quotes exact text and asks you to read it
aloud ("say exactly: …", "read this out loud: …"). Then `text=` is that
quoted wording, verbatim, with the quotes removed. If the user's words look
like a question, an order, or a prompt, they are the input you answer — not
the output you speak.

**Pre-flight check before every `speak()` call — run through all three:**

1. Did I write this text myself? (Yes → go on. It is a copy of the user's
   message → **stop and rewrite it as your answer.**)
2. Does it answer what the user asked? (It should be what you would have
   typed as your reply, just spoken.)
3. Would it sound natural read aloud? (Full sentences, no markdown.)

## Choosing what to say

| Reply length | What to pass to `speak()` |
|---|---|
| Short answer (a few sentences) | Verbatim, as your reply |
| Medium answer | Lightly tightened prose — same content, spoken register |
| Long answer (a wall of text) | A spoken summary: lead with the answer, then the two or three points that matter. Say "the details are in my message" if the full version is also on screen |

Never read out file paths, stack traces, JSON, or long enumerations unless
the user explicitly asks to hear them.

## Picking the model — prefer nano for English

`model` falls back to the server setting (turbo) when omitted, but the
deliberate choice for **English replies is `nano`**: same architecture as
turbo, same built-in voice and inline tags, but 110M parameters instead of
350M — quicker to load, lighter on VRAM, and on a box where an LLM already
occupies the GPU that headroom matters. Pass `"model": "nano"` explicitly so
the preference does not depend on the server's default.

- English reply → `nano`, unless the user asked for turbo by name.
- Other languages → `multilingual` (+ `language=`); no English model speaks
  them.
- Voice cloning or delivery style on the 500M models → `original` /
  `multilingual`, which require a reference clip.

## Picking the voice — ask once, then remember

**If the server has a configured default voice** (`CHATTERBOX_VOICE`,
reported as `default_voice` by `tts_status()`), skip the question entirely:
use it silently on every `speak()` call. `list_voices()` is only needed when
no default is configured.

Otherwise, unless the user already named a voice in this conversation ("use
the voice called <name>"), **ask which voice they want before the first
spoken reply** of the exchange:

1. Call `list_voices()` to get the real names (do not invent them).
2. Ask one short text question listing the options — this question is the
   only turn that stays fully in text; that is expected and correct.
3. From the user's answer onward, pass `voice=` on every `speak()` call and
   do not ask again for the rest of the exchange.

If `list_voices()` comes back empty, do not ask — turbo/nano ship a built-in
voice and need no clip. Never block a spoken reply on this question if the
user is waiting for content: give them the answer in the built-in voice and
offer the voice choice alongside it.

## Making the call

Minimal call — text plus the model preference from above:

```json
{ "text": "The tests all pass, and the build finished in about forty seconds.", "model": "nano" }
```

Audio plays automatically; everything else keeps its default.

Useful variations:

```json
{ "text": "Der Build ist fertig.", "model": "multilingual", "language": "de" }
{ "text": "All tests pass.", "voice": "<a name from list_voices()>" }
{ "text": "Long reply...", "wait": true }
{ "text": "Another line.", "play_audio": false }
```

- **German / other languages:** `model="multilingual"` + `language="de"` and
  a `voice=` from `list_voices()` (multilingual requires a reference clip;
  the stock turbo/nano voices are English-only).
- **Cut off playback:** `stop_speech()`.
- **Long replies:** the server auto-splits text at sentence boundaries and
  stitches the chunks; pass `wait: true` if the next tool call must not
  overlap the audio, or `progressive: true` to start playing each chunk as it
  renders instead of waiting for the whole utterance.

## Preflight and failure handling

- First-ever `speak()` on a machine downloads weights from Hugging Face and
  can take minutes; later calls are fast. If a call times out, the download
  is likely still running — retry once before reporting failure.
- If `speak()` returns `cuda_out_of_memory`, report it in one line and
  suggest `tts_unload()` — do not retry in a loop.
- If `reference_clip_required`, the chosen model needs a voice: for an
  English reply fall back to `model="nano"` (no clip needed); if the
  requested language is not English, report the missing clip instead of
  silently switching languages.
- `tts_status()` shows free VRAM and the resident model without loading
  anything; use it only when something already looks wrong.

## When NOT to use this skill

- The user asked a normal question with no voice request — answer in text.
- The user said stop / "enough talking" — call `stop_speech()` once and drop
  voice mode.
- Voice transcription (speech-to-text input) is a different feature; this
  skill is only about text-to-speech output.
