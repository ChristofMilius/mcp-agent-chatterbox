---
description: Report Chatterbox TTS status: worker, queue, device, VRAM, last utterance
---
Call the `mcp-agent-chatterbox_tts_status` tool from the mcp-agent-chatterbox
MCP server and summarize the result for the user in at most four short lines:

1. whether the resident worker is up, and what it is doing right now (idle or
   busy, how many utterances are queued);
2. which model and voice are in use, and whether a default voice is configured;
3. free VRAM per GPU, and whether that is enough for a model to load;
4. if the last utterance has finished, how many chunks it wrote and played, and
   its `max_gap_s` (the worst unavoidable silence).

Rules:
- Report only what the tool result says. Never estimate VRAM, model size or
  timing from memory.
- If the worker is not reachable, say so plainly and give the command to start
  it: `mcp-agent-chatterbox worker`.
- Do not propose optimisations, unloads or model changes unless the user asks;
  this command answers a question, it does not start a project.
- Keep paths out unless the user asks where the audio is; they are long and
  rarely what the question was about.
