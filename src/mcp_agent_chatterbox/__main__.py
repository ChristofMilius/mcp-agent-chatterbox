"""
`python -m mcp_agent_chatterbox` — runs the MCP server (stdio transport by default).

The package directory is put on sys.path first so this works from a source
checkout that has not been installed; the console script
(`mcp-agent-chatterbox`) is the installed equivalent.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mcp_agent_chatterbox.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
