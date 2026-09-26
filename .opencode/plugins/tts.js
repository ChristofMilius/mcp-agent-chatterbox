/**
 * tts.js — makes /speak available in every opencode project
 * ========================================================
 *
 * Part of the opencode_chatterbox project. This plugin has one job: install the
 * `/speak` command globally.
 *
 * Why this exists:
 *   The synthesis itself needs no plugin. Once the mcp-agent-chatterbox MCP
 *   server is registered, opencode exposes its tools to the model on its own,
 *   prefixed with the server name — `mcp-agent-chatterbox_speak`,
 *   `mcp-agent-chatterbox_tts_status`, and so on. A plugin that defined its own
 *   `speak` tool would be a second, redundant path to the same synthesis, with
 *   its own copy of the argument handling to keep in sync.
 *
 *   What opencode has no built-in mechanism for is a project-local command
 *   becoming available everywhere. Commands are plain markdown files read from
 *   `.opencode/commands/` (project) and `~/.config/opencode/commands/`
 *   (global); nothing copies between the two. Without help, `/speak` would
 *   only work in sessions started inside this repo, and the fix would be a
 *   manual file copy that silently drifts.
 *
 *   So this plugin does for the command what the MCP registration does for the
 *   tools: it self-replicates. When opencode loads this file, it copies
 *   `.opencode/commands/speak.md` into the global commands directory. The repo
 *   copy is the source of truth and the global copy is a deployment replica,
 *   refreshed on every load — edit the repo copy, never the global one.
 *
 * Scope, deliberately narrow:
 *   It does not register the MCP server. That lives in opencode.jsonc, and
 *   rewriting the user's config file from a plugin is not a plugin's job.
 *   Without that registration, `/speak` will still appear in the TUI but the
 *   agent will have no `mcp-agent-chatterbox_speak` tool to call.
 *
 * No dependencies — node builtins only, so no `bun install` step and no
 * package.json is required for this plugin to load.
 */

const { join } = await import("node:path")
const { homedir } = await import("node:os")
const { fileURLToPath } = await import("node:url")

const COMMAND_NAME = "speak.md"
const GLOBAL_COMMANDS_DIR = join(
  process.env.XDG_CONFIG_HOME || join(homedir(), ".config"),
  "opencode",
  "commands",
)

/** Locate this plugin's own file on disk, whichever way it was loaded. */
async function selfPath(input) {
  const selfUrl = import.meta && import.meta.url
  if (typeof selfUrl === "string" && selfUrl.startsWith("file:")) {
    return fileURLToPath(selfUrl)
  }
  if (input && input.project) {
    return join(input.project, ".opencode", "plugins", "tts.js")
  }
  return null
}

/** Resolve the repo copy of speak.md that belongs next to this plugin. */
async function commandSource(input) {
  const self = await selfPath(input)
  if (!self) return null
  const fs = await import("node:fs/promises")
  const candidates = [
    join(self, "..", "..", "commands", COMMAND_NAME),
    input && input.project ? join(input.project, ".opencode", "commands", COMMAND_NAME) : null,
  ].filter(Boolean)
  for (const candidate of candidates) {
    try {
      await fs.access(candidate)
      return candidate
    } catch {}
  }
  return null
}

const samePath = (a, b) =>
  a.toLowerCase().replace(/\\/g, "/") === b.toLowerCase().replace(/\\/g, "/")

async function replicateCommand(input, log) {
  const source = await commandSource(input)
  if (!source) {
    log?.("warn", "speak.md not found next to the plugin; /speak was not installed")
    return
  }
  const target = join(GLOBAL_COMMANDS_DIR, COMMAND_NAME)
  if (samePath(source, target)) return

  const fs = await import("node:fs/promises")
  try {
    const mine = await fs.readFile(source)
    let same = false
    try {
      same = (await fs.readFile(target)).equals(mine)
    } catch {}
    if (same) return
    await fs.mkdir(GLOBAL_COMMANDS_DIR, { recursive: true })
    await fs.writeFile(target, mine)
    log?.("info", `installed /speak -> ${GLOBAL_COMMANDS_DIR}`)
  } catch (err) {
    log?.("error", `could not install /speak: ${String(err)}`)
  }
}

export default async function (input) {
  const client = input && input.client
  const log = (level, message) => {
    client?.app?.log?.({ body: { service: "chatterbox-tts", level, message } }).catch(() => {})
  }

  await replicateCommand(input, log)

  return {}
}
