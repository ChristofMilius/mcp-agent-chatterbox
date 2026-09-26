---
description: Speak text aloud with local GPU text-to-speech (Chatterbox)
---
Speak the following text out loud with the `mcp-agent-chatterbox_speak` tool from
the mcp-agent-chatterbox MCP server:

$ARGUMENTS

Call that tool with `text` set to the text above.

- Leave `model`, `voice`, `language` and the style parameters at their defaults
  unless the text itself implies otherwise (a non-English passage needs
  `model="multilingual"` and a `language` code; a request for a particular
  speaker needs `voice`).
- If the first word of the arguments looks like an option rather than speech
  (for example `--model multilingual` or `--language de`), treat it as a tool
  parameter instead of part of the text.
- The first call on a fresh machine downloads the model and can take a while.
  Say so once, up front, rather than appearing to hang.

Then report in one short line: the output filename, the audio duration, and
whether it played. Do not repeat or quote the spoken text back.
