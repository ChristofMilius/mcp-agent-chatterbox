---
description: Stop Chatterbox speech now and cancel everything still queued
---
Call the `mcp-agent-chatterbox_stop_speech` tool from the mcp-agent-chatterbox
MCP server, then report in one short line whether audio was actually cut off and
how many queued utterances were cancelled.

- Do not call `speak` in the same turn. The user asked for quiet, and the tool
  result already reports what was cancelled; do not narrate it as a speech.
- The cut is immediate for the sentence being spoken, but one chunk that was
  already rendering may still be started: say "stopped" rather than promising
  silence this instant if the result reports a cancelled active job.
- If the result reports `worker_not_running`, nothing can be playing. Say so in
  one line and give the command: `mcp-agent-chatterbox worker`.
- If the user then says "continue" or "keep reading", call `speak` with the text
  from after the point that was heard — never repeat the part they already
  listened to.
