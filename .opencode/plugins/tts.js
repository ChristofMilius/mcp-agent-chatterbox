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
 *   V2 has a `ctx.command.transform()` editor, and it is the obvious tool for
 *   this job. It is deliberately not used: a transform registers into the
 *   registry of the location this plugin instance loaded in, and this plugin
 *   only loads inside this repo. Copying the file is what actually reaches
 *   every other project.
 *
 * Scope, deliberately narrow:
 *   It does not register the MCP server. That lives in opencode.jsonc, and
 *   rewriting the user's config file from a plugin is not a plugin's job.
 *   Without that registration, `/speak` will still appear in the TUI but the
 *   agent will have no `mcp-agent-chatterbox_speak` tool to call.
 *
 * No dependencies — node builtins only, so no `bun install` step and no
 * package.json is required for this plugin to load.
 *
 * On the V2 entrypoint shape:
 *   V2 requires the default export to be a definition object carrying `id` and
 *   `setup`; a V1 plugin's default-exported function is rejected outright
 *   ("Plugin must export a default definition with an id and an effect or setup
 *   function"). The documented spelling is `Plugin.define({ ... })`, but that
 *   helper is defined in `@opencode/plugin` as
 *
 *       export function define(plugin) { return plugin }
 *
 *   — a pure identity function, there for TypeScript inference. Importing it
 *   into a plain `.js` file would buy zero runtime behaviour while pulling in
 *   the whole `@opencode/plugin` tree (`@opencode/ai`, `effect`, `zod`, ~283
 *   packages) into `.opencode/node_modules`. The loader validates the shape
 *   itself, so the plain object below is equivalent and keeps this file
 *   dependency-free. If this ever becomes TypeScript, import `Plugin` and wrap
 *   it for the inferred `Context` type.
 */

const { join, dirname } = await import("node:path")
const { homedir } = await import("node:os")
const { fileURLToPath } = await import("node:url")
const { access, mkdir, readFile, writeFile } = await import("node:fs/promises")

const PLUGIN_ID = "chatterbox-tts"
const COMMAND_NAME = "speak.md"

const GLOBAL_COMMANDS_DIR = join(
  process.env.XDG_CONFIG_HOME || join(homedir(), ".config"),
  "opencode",
  "commands",
)

/**
 * Resolve the repo copy of speak.md that belongs next to this plugin.
 *
 * `import.meta.url` is authoritative: it is this exact file, so the commands
 * directory two levels up is the source of truth no matter how opencode
 * loaded it. The context lookup is only a fallback for a bundler that rewrites
 * or drops the module URL.
 */
async function commandSource(ctx) {
  const candidates = []

  if (typeof import.meta.url === "string" && import.meta.url.startsWith("file:")) {
    candidates.push(join(dirname(fileURLToPath(import.meta.url)), "..", "..", "commands", COMMAND_NAME))
  }

  const projectDirectory = ctx?.location?.project?.directory
  if (projectDirectory) {
    candidates.push(join(projectDirectory, ".opencode", "commands", COMMAND_NAME))
  }

  for (const candidate of candidates) {
    try {
      await access(candidate)
      return candidate
    } catch {}
  }
  return null
}

const samePath = (a, b) => a.toLowerCase().replace(/\\/g, "/") === b.toLowerCase().replace(/\\/g, "/")

async function replicateCommand(ctx, log) {
  const source = await commandSource(ctx)
  if (!source) {
    log("warn", "speak.md not found next to the plugin; /speak was not installed")
    return
  }

  const target = join(GLOBAL_COMMANDS_DIR, COMMAND_NAME)
  if (samePath(source, target)) return

  try {
    const mine = await readFile(source)
    let same = false
    try {
      same = (await readFile(target)).equals(mine)
    } catch {}
    if (same) return
    await mkdir(GLOBAL_COMMANDS_DIR, { recursive: true })
    await writeFile(target, mine)
    log("info", `installed /speak -> ${GLOBAL_COMMANDS_DIR}`)
  } catch (err) {
    log("error", `could not install /speak: ${String(err)}`)
  }
}

export default {
  id: PLUGIN_ID,
  async setup(ctx) {
    // V1 plugins logged through `client.app.log(...)`. That method does not
    // exist here: in V2 `ctx.app` is three readonly strings (name, version,
    // channel), not a client namespace. `console` is the documented route from
    // a plugin into the server log, and it has no signature to get wrong.
    const log = (level, message) => {
      const line = `[${PLUGIN_ID}] ${message}`
      if (level === "error") console.error(line)
      else if (level === "warn") console.warn(line)
      else console.log(line)
    }
    await replicateCommand(ctx, log)
  },
}