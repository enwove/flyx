// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Client-specific setup for the model the server reports. Keyed examples refer to FLYX_KEY.
import { html, useState, useEffect, useRef } from "../lib/html.js";
import { t } from "../lib/i18n.js";
import { CodeSnippet, defaultShell, ShellPicker, Icon } from "../lib/ui.js";
import { psQuote, shQuote } from "../lib/format.js";
import { call, saved } from "../lib/api.js";
import { live } from "../lib/telemetry.js";

const AGENTS = [
  ["claude", "Claude Code", "https://code.claude.com/docs/en/llm-gateway-connect", ClaudeSetup],
  ["codex", "Codex CLI", "https://developers.openai.com/codex/config-reference/", CodexSetup],
  ["deepseek", "DeepSeek Harness", "https://github.com/deepseek-ai/deepseek-harness", DeepSeekSetup],
  ["opencode", "OpenCode", "https://github.com/anomalyco/opencode/blob/v1.19.6/packages/web/src/content/docs/providers.mdx#custom-provider", OpenCodeSetup],
  ["pi", "Pi", "https://docs.pi.dev/latest/models-configuration", PiSetup],
  ["other", "Others", null, OtherSetup],  // any other agent: a text to hand it, by which it sets itself up
];

// The most Claude Code asks for in a reply (CLAUDE_CODE_MAX_OUTPUT_TOKENS): half the context, at most 128,000, the
// cap Claude Code keeps for a model it does not know (32,000 when nothing is set).  The rest is the conversation's.
// apps/flyx-server/agent_launch.cpp sets the same for a launch from here.
const claudeOutput = (context) => Math.max(1, Math.min(128000, Math.floor(context / 2)));

const claudePermissions = [
  ["bypassPermissions", "Skip permission prompts"],
  ["acceptEdits", "Accept project edits and ask about other commands"],
  ["default", "Ask for permission"],
  ["auto", "Review commands with a model"],
  ["", "Use Claude Code settings"],
];

function claudeCommand(shell, origin, model, context, keyed, permissionMode) {
  const permission = permissionMode ? " --permission-mode " + permissionMode : "";
  if (shell === "powershell") {
    // PowerShell 5.1 can strip literal quotes from native arguments. Both model and credentials travel in the file.
    return [
      "& {",
      "  $ErrorActionPreference = 'Stop'",
      ...(keyed ? ["  if (!$env:FLYX_KEY) { throw 'Set $env:FLYX_KEY to the FLYX API key.' }"] : []),
      "  $modelName = " + psQuote(model),
      "  $settings = @{ model = $modelName; env = @{",
      "    ANTHROPIC_MODEL = $modelName",
      "    ANTHROPIC_BASE_URL = " + psQuote(origin),
      "    ANTHROPIC_AUTH_TOKEN = " + (keyed ? "$env:FLYX_KEY" : "'flyx'"),
      "    CLAUDE_CODE_MAX_CONTEXT_TOKENS = '" + context + "'",
      "    CLAUDE_CODE_MAX_OUTPUT_TOKENS = '" + claudeOutput(context) + "'",
      "    CLAUDE_CODE_MODEL_CAPABILITIES = '-mid_conv_system'",
      "    CLAUDE_CODE_GATEWAY_HINT_HEADERS = '1'",
      "  } }",
      "  $file = [IO.Path]::GetTempFileName()",
      "  try {",
      "    [IO.File]::WriteAllText($file, ($settings | ConvertTo-Json))",
      "    claude --settings $file" + permission,
      "  } finally { [IO.File]::Delete($file) }",
      "}",
    ].join("\n");
  }
  const env = {
    ANTHROPIC_MODEL: model,
    ANTHROPIC_BASE_URL: origin,
    ANTHROPIC_AUTH_TOKEN: keyed ? undefined : "flyx",
    CLAUDE_CODE_MAX_CONTEXT_TOKENS: String(context),
    CLAUDE_CODE_MAX_OUTPUT_TOKENS: String(claudeOutput(context)),
    CLAUDE_CODE_MODEL_CAPABILITIES: "-mid_conv_system",
    CLAUDE_CODE_GATEWAY_HINT_HEADERS: "1",
  };
  const settings = JSON.stringify({ model, env }, null, 2);
  if (!keyed) return "claude --settings " + shQuote(settings) + permission;

  // Keep the configured key out of both the clipboard and the launched client's command line.
  return [
    "(",
    "  set -e",
    '  : "${FLYX_KEY:?Export FLYX_KEY with the FLYX API key}"',
    "  export FLYX_KEY",
    "  file=$(mktemp)",
    "  trap 'rm -f \"$file\"' EXIT",
    "  jq '.env.ANTHROPIC_AUTH_TOKEN = env.FLYX_KEY' > \"$file\" <<'FLYX_JSON'",
    settings,
    "FLYX_JSON",
    "  claude --settings \"$file\"" + permission,
    ")",
  ].join("\n");
}

function ClaudeSetup({ origin, model, context, keyed, shell, permissionMode }) {
  return html`
    <p>${t("Run this in your project directory. It uses the model and context shown by FLYX for this Claude Code session.")}</p>
    ${keyed && html`<p>${t(shell === "powershell" ? "Set $env:FLYX_KEY to the server API key first." : "Export FLYX_KEY with the server API key first. This command uses jq.")}</p>`}
    <${CodeSnippet} text=${claudeCommand(shell, origin, model, context, keyed, permissionMode)}/>
    <p class="lead">${t("Seeing 401 Invalid bearer token? Type /exit in Claude Code, then run the command above. For this launch, --settings overrides the Anthropic URL and token saved in Claude Code.")}</p>
    <details><summary>${t("Claude Code output limit")}</summary>
      <p><code>CLAUDE_CODE_MAX_OUTPUT_TOKENS</code> ${t("caps each reply: half the context, at most 128,000. Lower it if long replies tie up the server.")}</p>
    </details>`;
}

function OpenCodeSetup({ origin, model, context, outputLimit, keyed }) {
  const provider = {
    npm: "@ai-sdk/openai-compatible",
    name: "flyx",
    // OpenCode reads provider credentials from env after parsing its JSONC configuration.
    env: keyed ? ["FLYX_KEY"] : undefined,
    options: { baseURL: origin + "/v1", apiKey: keyed ? undefined : "flyx" },
    models: { [model]: { name: model, limit: { context, output: outputLimit } } },
  };
  const config = { model: "flyx/" + model, small_model: "flyx/" + model, provider: { flyx: provider } };
  return html`
    <p>${t("For OpenCode 1.x, merge these settings into the project configuration file.")} <code>opencode.json</code></p>
    <${CodeSnippet} text=${JSON.stringify(config, null, 2)}/>
    <p>${t("Start the agent in your project.")}</p>
    <${CodeSnippet} text="opencode"/>`;
}

function PiSetup({ origin, model, context, outputLimit, keyed }) {
  const provider = {
    baseUrl: origin + "/v1", api: "openai-completions", apiKey: keyed ? "$FLYX_KEY" : "flyx", authHeader: true,
    models: [{ id: model, name: "flyx-local", contextWindow: context, maxTokens: outputLimit }],
  };
  return html`
    <p>${t("Add the flyx provider to your Pi model configuration.")} <code>~/.pi/agent/models.json</code>
      ${t("If PI_CODING_AGENT_DIR is set, use models.json in that directory.")}</p>
    <${CodeSnippet} text=${JSON.stringify({ providers: { flyx: provider } }, null, 2)}/>
    <p>${t("Start the agent in your project.")}</p>
    <${CodeSnippet} text="pi --provider flyx --model flyx-local"/>`;
}

function DeepSeekSetup({ origin, model, context, outputLimit, keyed, shell }) {
  const config = `- id: llm-pi-ai
  config:
    providers:
      flyx:
        api: openai-completions
        apiKeyEnv: ${keyed ? "FLYX_KEY" : "FLYX_LOCAL_KEY"}
        baseURL: ${JSON.stringify(origin + "/v1")}
        models:
          - id: ${JSON.stringify(model)}
            contextWindow: ${context}
            maxTokens: ${outputLimit}
- id: agent-default-model
  config:
    provider: flyx
    model: ${JSON.stringify(model)}`;
  const key = keyed ? "" : shell === "powershell" ? "$env:FLYX_LOCAL_KEY = 'flyx'\n" : "FLYX_LOCAL_KEY=flyx ";
  return html`
    <p>${t("Merge these plugin settings into your DeepSeek Harness headless profile.")} <code>~/.dsh/profiles/headless/cordis.patch.yml</code>
      ${t("If DSH_HOME is set, look under that directory for profiles/headless/.")}</p>
    <${CodeSnippet} text=${config}/>
    <p>${t("Run a task in your project. Replace the example prompt with your own.")}</p>
    <${CodeSnippet} text=${key + 'dsh headless "Say hello."'}/>`;
}

function CodexSetup({ origin, model, context, keyed }) {
  const config = [
    'model_provider = "flyx"', "model = " + JSON.stringify(model), "model_context_window = " + context,
    'web_search = "disabled"', "", "[model_providers.flyx]", 'name = "flyx"',
    "base_url = " + JSON.stringify(origin + "/v1"), 'wire_api = "responses"',
    ...(keyed ? ['env_key = "FLYX_KEY"'] : []),
  ];
  return html`
    <p>${t("Save this profile as")} <code>~/.codex/flyx.config.toml</code>.
      ${t("If CODEX_HOME is set, save flyx.config.toml in that directory.")}</p>
    <${CodeSnippet} text=${config.join("\n")}/>
    <p>${t("Start the agent in your project.")}</p>
    <${CodeSnippet} text="codex --profile flyx"/>
    <details><summary>${t("Windows shell access")}</summary>
      <p>${t("If the read-only shell reports access denied, compatible Windows 11 devices can use MXC for this session.")}</p>
      <${CodeSnippet} text="codex --profile flyx -c windows.sandbox=mxc"/>
      <a href="https://developers.openai.com/codex/windows" target="_blank" rel="noreferrer">OpenAI · ${t("Windows shell access")}</a>
    </details>`;
}

// What any coding agent needs to know to make this server its model, as a text to paste to it.  In English, as the
// agents read it whatever the page's language.
function otherPrompt(origin, model, context, outputLimit, keyed) {
  return [
    "Set yourself up to use the model served by FLYX, a local server that speaks the OpenAI and the Anthropic APIs.",
    "",
    "The server",
    "- OpenAI-compatible base URL: " + origin + "/v1  (POST /chat/completions, POST /responses, GET /models)",
    "- Anthropic-compatible base URL: " + origin + "  (POST /v1/messages)",
    "- Model ID: " + model,
    "- Context window: " + context + " tokens, shared by the prompt and the reply; keep the maximum output at or under " + outputLimit + " tokens",
    keyed ? "- API key: required, as a Bearer token. It is in the FLYX_KEY environment variable: refer to that variable, never write the key itself into a file"
          : '- API key: none is needed; where a key cannot be left empty, any text will do (use "flyx")',
    "- Streaming (SSE) and tool calls are supported; there is no web search or other hosted tool",
    "",
    "What to do",
    "1. Look up, in your own documentation or configuration reference, how you take a custom or OpenAI-compatible provider. Do not guess the file or its fields.",
    '2. Add a provider named "flyx" with the base URL and the model above, and use it for this project (not for all my projects, unless I say so).',
    "3. Leave my other providers, keys and settings as they are. Show me the change to any configuration file before you write it.",
    "4. Check it: GET " + origin + "/v1/models lists the model ID, and a one-line request to the model gets a reply.",
    "5. Tell me how to switch to this model and back.",
  ].join("\n");
}

function OtherSetup({ origin, model, context, outputLimit, keyed }) {
  return html`
    <p>${t("Using another coding agent? Copy this to it: it says what the agent needs to set itself up for this server.")}</p>
    <${CodeSnippet} id="agent-prompt" text=${otherPrompt(origin, model, context, outputLimit, keyed)}/>
    <p class="lead">${t("The text is in English, which the agents read whatever language you work in. It names the model and the context being served now.")}</p>`;
}

// The agents the server can start itself, in a window of PowerShell on its machine (POST /v1/agents/launch): offered
// to a browser on that machine, where the window opens.  The server writes the command; the page names the agent.
const here = () => /^(localhost|127(\.\d+){3}|\[::1\])$/.test(location.hostname);
const launchable = (agent) => here() && (live.last?.server?.agent_launch || []).includes(agent);

const folderName = (d) => d.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || d;

// Where the agent starts, asked over the page: one folder, the one started in last (this browser's) until another
// is chosen in the desktop's own dialog (POST /v1/hub/dir); then the window opens there.
function ProjectDialog({ agent, permissionMode, close, started }) {
  const [dir, setDir] = useState(() => saved.get("project_dir") || "");
  const [state, setState] = useState(null);  // {picking}, {busy}, {err}
  const go = useRef(null);
  useEffect(() => {
    go.current && go.current.focus();
    const key = (e) => { if (e.key === "Escape" && !state?.picking) { e.stopPropagation(); close(); } };
    document.addEventListener("keydown", key, true);
    return () => document.removeEventListener("keydown", key, true);
  }, [state]);
  async function choose() {
    setState({ picking: true });
    const r = await call("POST", "/v1/hub/dir", { title: t("Choose a project folder"), start: dir });
    if (r.ok && r.json && r.json.path) setDir(r.json.path);
    setState(r.ok ? null : { err: r.error });
  }
  async function start() {
    setState({ busy: true });
    const r = await call("POST", "/v1/agents/launch", { agent, dir, permission_mode: permissionMode || undefined });
    if (!r.ok) return setState({ err: r.error });
    saved.set("project_dir", dir);
    started();
  }
  const picking = !!state?.picking;
  return html`
    <div class="mscrim" onClick=${(e) => { if (e.target === e.currentTarget && !state?.picking) close(); }}>
      <div class="modal pdlg" role="dialog" aria-modal="true" aria-labelledby="pdlg-h" id="project-dialog">
        <h3 id="pdlg-h">${t("Choose a project folder")}</h3>
        <p>${t("The agent starts there in a new PowerShell window.")}</p>
        ${dir ? html`
          <div class="pdlg-item" id="pdlg-dir" title=${dir}>
            <${Icon} name="folder"/><span><b>${folderName(dir)}</b><small>${dir}</small></span>
            <button class="btn sm" id="pdlg-choose" disabled=${picking} onClick=${choose}>
              ${t(picking ? "Choose the folder in the window that opened…" : "Change…")}</button>
          </div>` : html`
          <button class="btn pdlg-choose" id="pdlg-choose" disabled=${picking} onClick=${choose}>
            <${Icon} name="folder"/>${t(picking ? "Choose the folder in the window that opened…" : "Choose a folder…")}</button>`}
        ${state?.err && html`<div class="pdlg-err">${state.err}</div>`}
        <div class="row">
          <button class="btn" onClick=${close}>${t("Cancel")}</button>
          <button class="btn primary" id="pdlg-go" ref=${go} disabled=${!dir || !!state?.busy || picking}
                  onClick=${start}>${t("Open in PowerShell")}</button>
        </div>
      </div>
    </div>`;
}

// The button that starts the agent; the folder is asked for in the dialog it opens.
function Launch({ agent, permissionMode }) {
  const [open, setOpen] = useState(false);
  const [ok, setOk] = useState(false);
  return html`
    <span class="launch" id="agent-launch">
      ${ok && html`<span class="launch-note ok">${t("Started in a new PowerShell window.")}</span>`}
      <button class="btn primary sm" id="agent-launch-go" onClick=${() => { setOk(false); setOpen(true); }}>
        <${Icon} name="play"/>${t("Open in PowerShell")}</button>
      ${open && html`<${ProjectDialog} agent=${agent} permissionMode=${permissionMode} close=${() => setOpen(false)}
        started=${() => { setOpen(false); setOk(true); }}/>`}
    </span>`;
}

export function AgentSetup({ origin, model, context, keyed }) {
  const [agent, setAgent] = useState("claude");
  const [shell, setShell] = useState(defaultShell);
  const [claudeMode, setClaudeMode] = useState("bypassPermissions");
  const permissionMode = agent === "claude" ? claudeMode : undefined;
  const [, name, url, setup] = AGENTS.find(([id]) => id === agent);
  const needsShell = agent === "claude" || (agent === "deepseek" && !keyed);
  // The clients reserve room for a reply. A small server context still needs room for the input.
  const outputLimit = Math.max(1, Math.min(4096, Math.floor(context / 2)));
  // Started from here, the agent runs in PowerShell whatever shell is picked: the picker is the manual command's.
  const launch = launchable(agent);
  const picker = html`<${ShellPicker} shell=${shell} onChange=${setShell}/>`;
  return html`
    <section id="d-agents">
      <h2>${t("Coding agents")}</h2>
      <div class="agent-list pills" role="group" aria-label=${t("Coding agents")}>
        ${AGENTS.map(([id, label]) => html`<button key=${id} data-agent=${id} id=${"guide-" + id}
          aria-pressed=${String(agent === id)} aria-controls="agent-guide" onClick=${() => setAgent(id)}>
          <svg class=${"agent-logo agent-" + id} aria-hidden="true"><use href=${"#i-agent-" + id}/></svg>${id === "other" ? t(label) : label}</button>`)}
      </div>
      <div class="agent-guide" id="agent-guide" role="region" aria-labelledby=${"guide-" + agent}>
        ${agent === "claude" && context > 0 && html`<label class="field">
          <span>${t("Permission mode")}</span>
          <select id="claude-permission" value=${permissionMode} onChange=${(e) => setClaudeMode(e.currentTarget.value)}>
            ${claudePermissions.map(([value, label]) => html`<option key=${value} value=${value}>${t(label)}</option>`)}
          </select>
          <em>${t("Model review can cause a single-slot server to process the conversation again after a command.")}</em>
        </label>`}
        <div class="agent-controls">
          ${url ? html`<a href=${url} target="_blank" rel="noreferrer">
            <svg class=${"agent-logo agent-" + agent} aria-hidden="true"><use href=${"#i-agent-" + agent}/></svg>${name} · ${t("Client documentation")}</a>` : html`<span></span>`}
          ${context && launch ? html`<${Launch} key=${agent} agent=${agent} permissionMode=${permissionMode}/>` : needsShell && context && picker}
        </div>
        ${context ? html`
          ${launch && html`
            <div class="manual-h"><span class="caps">${t("Or set it up yourself")}</span>${needsShell && picker}</div>`}
          ${keyed && agent !== "claude" && agent !== "other" && html`<p>${t("Set the FLYX_KEY environment variable to the server API key before launching.")}</p>`}
          <${setup} origin=${origin} model=${model} context=${context} outputLimit=${outputLimit} keyed=${keyed} shell=${shell} permissionMode=${permissionMode}/>
          ${["opencode", "pi", "deepseek"].includes(agent) && html`<p class="lead">${t("Adjust the output limit in the example to allow longer replies.")}</p>`}
        ` : html`<p>${t("Load a model and connect the dashboard to see its launch command.")}</p>`}
      </div>
    </section>`;
}
