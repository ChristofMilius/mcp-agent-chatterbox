"""
speak_tools — the main MCP call
===============================
speak: accept text for speech, chunk it per sentence, and hand it to the
       process-local queue. The call itself returns in milliseconds; a daemon
       worker renders and plays the chunks in the background.

The tool is a thin shell over speak.Speaker.submit(), which the CLI shares. All
       it adds is JSON framing and the error boundary.

Model selection:
  * model="turbo" (default) speaks with no reference clip and understands
    paralinguistic tags inline in the text ([laugh], [chuckle], ...).
  * model="nano" is turbo's 110M sibling — same architecture, tags and
    built-in voice, tightest latency/memory budget (also CPU-capable).
  * model="multilingual" needs voice=/reference_clip= and accepts language=
    as an ISO 639-1 code.
  * model="original" needs voice=/reference_clip= and is the English
    CFG/exaggeration-tuning model.
"""

from __future__ import annotations

import json

from mcp_agent_chatterbox.errors import ToolFault, tool_error


def register(server, ctx) -> None:
    speaker = ctx.speaker

    @server.tool()
    def speak(
        text: str,
        model: str | None = None,
        voice: str | None = None,
        reference_clip: str | None = None,
        language: str | None = None,
        t3_model: str | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        norm_loudness: bool | None = None,
        seed: int | None = None,
        play_audio: bool | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Speak text aloud with Chatterbox on the local GPU. The call returns as
        soon as the text is accepted; synthesis and playback continue in the
        background, one sentence chunk after another.

        WHEN TO CALL: whenever the user asks for a spoken reply -- "speak",
        "answer aloud", "reply by voice", "say it out loud", and the like --
        put your ENTIRE reply into text= and call this tool ONCE. Do not reply
        in plain text alone while the user is expecting voice; the spoken
        output is the deliverable. text= is YOUR generated reply -- never copy
        the user's message into it; the only exception is when they quoted
        exact words to read aloud. Keep the text in spoken register: full
        sentences, no markdown, no lists, no paths or code. One instruction to
        speak stays in effect for the whole exchange until the user says stop.
        Voice mode ON means EVERY reply is spoken, not just the first.

        IMPORTANT -- DO NOT RETRY: the result says "queued", not "spoken". The
        audio starts a few seconds later and plays through to the end even
        though this call has already returned. Calling speak() again because
        you did not hear anything yet replays the whole utterance a second
        time. Send the complete reply in ONE call and then continue with your
        turn; tts_status() reports queue depth if you need to check progress.
        Leaving out the tail of a reply ("...") is the same mistake: there is
        no length limit that requires it, the text budget is generous and long
        text is chunked automatically.

        MODEL: "nano" is the house default for English (110M, same built-in
        voice and inline tags as turbo, lighter and quicker). Do NOT pass
        model= to "upgrade" it or to express a preference of your own — omit it
        and the server's configured default stands. Pass model= only when the
        user asked for a specific model, or to experiment when they asked you
        to try one ("use turbo", "try the multilingual one"). Overriding
        without being asked is a bug, not an optimisation: the configured
        default is a deliberate local choice about VRAM and latency.
        VOICE: if the user has not named a voice, ask once which voice to
        use before the first spoken reply, then keep that voice for the
        rest of the exchange — unless tts_status reports a configured
        default_voice (use it silently, no question), or the user is
        waiting on the content (speak in the built-in voice and offer the
        voice choice alongside the answer).

        model:   omit it to use the server's configured default (nano by
                 house preference). "nano" (110M, turbo's smaller sibling with
                 competitive quality, built-in voice, tightest budget),
                 "turbo" (350M, English, no reference clip needed, supports
                 inline [laugh]/[chuckle] tags),
                 "multilingual" (500M, 23 languages, requires a reference clip)
                 or "original" (500M, English, requires a reference clip).
                 Pass this only on request — see MODEL above.
        voice:   name of a reference clip in the voices directory (its filename
                 without extension). Cloning works on every model — turbo/nano
                 fall back to their built-in voice only when no clip is given.
                 Required for multilingual/original, which ship no built-in
                 voice.
        reference_clip: path to a wav/mp3/flac clip, as an alternative to voice=.
                 Supply as much clean continuous speech as you have. Short
                 references measurably increase fine clicks and crackle: a 40 s
                 reference beat an 11.8 s excerpt of the same recording by a
                 wide margin, several times the model's own run-to-run variance.
                 Under ~5 s is genuinely too little. Do not normalise or limit
                 the clip -- that shifts output level and adds artefacts. A
                 mastered or compressed recording is fine. Sample rate and
                 channel count need not match anything.
        language: ISO 639-1 code, multilingual only (e.g. "de", "en", "fr").
        t3_model: "v2" (default) or "v3" — multilingual checkpoint.
        temperature: sampling temperature (0.05-5.0). Honored by all models.
        top_p: nucleus-sampling cutoff (0.0-1.0). Honored by all models.
        top_k: top-k sampling size (0-1000). Honored by turbo/nano only — the
                 500M models have no such parameter.
        repetition_penalty: penalise repeated tokens (1.0-2.0). Honored by all
                 models.
        norm_loudness: normalize output to -27 LUFS. Honored by turbo/nano only
                 — the 500M models have no such parameter.
        exaggeration/cfg_weight: style controls (0.0-2.0 / 0.0-1.0). Honored
                 ONLY by the 500M models (multilingual/original); turbo/nano
                 ignore both, so setting them with model="turbo" is dropped
                 with a warning, not an error. cfg_weight>0 doubles the text
                 tokens for CFG guidance.
        seed: reseed torch (CPU + CUDA) so re-renders are reproducible within
                 the resident session; 0 (the upstream convention) or unset
                 keeps random sampling. Byte-identical output across a server
                 restart is not guaranteed — Chatterbox is nondeterministic
                 across CUDA kernel choices.
                 Left unset, every knob keeps the model's own tuned default
                 (e.g. turbo runs cfg_weight 0.0 with top_k 1000; the 500M
                 models run cfg_weight 0.5).
        play_audio: set false to only write the chunk files. Defaults to the
                 server setting (on).
        filename: base name for the chunk files; they are written as
                 <filename>_1.wav, <filename>_2.wav, ... A timestamped message
                 id is used when omitted.

        Each sentence is one chunk: it is rendered, saved and played, then the
        next one follows, so the reply starts within a few seconds and plays
        continuously. Files land in the output directory and replay in filename
        order. Returns JSON with the message id, the planned chunk filenames
        and the queue position -- not durations, which are only known once the
        background rendering is done (see tts_status). The first call downloads
        the model from Hugging Face (hundreds of MB to a few GB) and can take a
        while; later calls reuse the cached weights.
        """
        try:
            payload = speaker.submit(
                text,
                model=model,
                voice=voice,
                reference_clip=reference_clip,
                language=language,
                t3_model=t3_model,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
                seed=seed,
                play_audio=play_audio,
                filename=filename,
            )
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except ToolFault as fault:
            return fault.to_json()
        except Exception as e:
            return tool_error("speak", e)


__all__ = ["register"]
