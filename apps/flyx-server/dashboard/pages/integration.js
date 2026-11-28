// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Integration page connects coding agents and SDKs to the running server. The API reference is translated a
// sentence at a time (tx), its fields and code as the client sends them. check_server.py checks its endpoints.
import { html, useState, useEffect } from "../lib/html.js";
import { apiKey } from "../lib/api.js";
import { psQuote, shQuote } from "../lib/format.js";
import { t, para, tx } from "../lib/i18n.js";
import { CodeSnippet, defaultShell } from "../lib/ui.js";
import { live } from "../lib/telemetry.js";
import { AgentSetup } from "../components/agent_setup.js";

const TOC = [["d-agents", "Coding agents"], ["d-quick", "SDKs and HTTP"], ["d-endpoints", "Endpoints"], ["d-fields", "Request fields"], ["d-sampling", "Sampling defaults"],
             ["d-sessions", "Sessions"], ["d-flags", "flyx serve flags"], ["d-telemetry", "Telemetry"], ["d-more", "More"]];
const SNIPS = [["powershell", "PowerShell"], ["curl", "curl · Bash / zsh"], ["python", "OpenAI Python"],
               ["anthropic", "Anthropic Python"], ["stream", "SSE · Bash / zsh"]];

// The snippets, for this server's address and model; with a key, read from $FLYX_KEY.
function snippets(origin, model, keyVar) {
  const auth = keyVar ? " \\\n  -H \"Authorization: Bearer " + keyVar + "\"" : "";
  const messages = [{ role: "user", content: "Hello" }];
  const plain = { chat_template_kwargs: { enable_thinking: false } };
  const curl = (options, fields) => "curl " + options + shQuote(origin + "/v1/chat/completions") + auth +
    ' \\\n  -H "Content-Type: application/json" --data-binary @- <<\'JSON\'\n' +
    JSON.stringify({ model, messages, ...fields }, null, 2) + "\nJSON";
  const pyKey = keyVar ? "os.environ[\"FLYX_KEY\"]" : "\"flyx\"";
  const pyImport = keyVar ? "import os\n" : "";
  return {
    curl: curl("", plain),
    powershell: [
      "$flyxBody = " + psQuote(JSON.stringify({ model, messages, ...plain }, null, 2)),
      "$flyxRequest = @{", "  Method = 'Post'", "  Uri = " + psQuote(origin + "/v1/chat/completions"),
      "  ContentType = 'application/json; charset=utf-8'",
      ...(keyVar ? ['  Headers = @{ Authorization = "Bearer $env:FLYX_KEY" }'] : []),
      "  Body = [Text.Encoding]::UTF8.GetBytes($flyxBody)", "}",
      "(Invoke-RestMethod @flyxRequest).choices[0].message.content",
    ].join("\n"),
    python: pyImport + "from openai import OpenAI\n\nclient = OpenAI(base_url=" + JSON.stringify(origin + "/v1") + ", api_key=" + pyKey + ")\n" +
      "reply = client.chat.completions.create(\n    model=" + JSON.stringify(model) + ",\n    messages=[{\"role\": \"user\", \"content\": \"Hello\"}],\n)\n" +
      "print(reply.choices[0].message.content)",
    anthropic: pyImport + "import anthropic\n\nclient = anthropic.Anthropic(base_url=" + JSON.stringify(origin) + ", api_key=" + pyKey + ")\n" +
      "msg = client.messages.create(\n    model=" + JSON.stringify(model) + ",\n    max_tokens=1024,\n    messages=[{\"role\": \"user\", \"content\": \"Hello\"}],\n)\n" +
      "for block in msg.content:\n    if block.type == \"text\":\n        print(block.text)",
    stream: curl("-N ", { stream: true, stream_options: { include_usage: true } }),
  };
}

// Which pages a browser lets call the API from another origin: the server's --cors, as its telemetry tells it
// (a server that does not tell is one with the default).
function corsNote(cors) {
  if (cors?.any) return t("Any web page may call the API from a browser (--cors '*').");
  const own = t(cors && cors.loopback === false
    ? "Pages on this machine may not call the API from a browser (--cors none); another site's pages need --cors ORIGIN."
    : "Pages on this machine (localhost) may call the API from a browser; another site's pages need --cors ORIGIN.");
  return cors?.origins?.length ? own + " " + t("Also allowed:") + " " + cors.origins.join(", ") + "." : own;
}

// A link to a section: the page scrolls there.
function jump(e) {
  e.preventDefault();
  const target = document.getElementById(e.currentTarget.getAttribute("href").slice(1));
  if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
}
// The same for a link inside a reference sentence (set as HTML, so without a handler of its own).
function jumpInside(e) {
  const a = e.target.closest && e.target.closest('a[href^="#d-"]');
  if (!a) return;
  e.preventDefault();
  const target = document.getElementById(a.getAttribute("href").slice(1));
  if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
}

// A line, cell or note of the reference: one sentence with its code in place, in the page's language (tx).
const Li = (en) => html`<li dangerouslySetInnerHTML=${{ __html: tx(en) }}></li>`;
const Td = (en) => html`<td dangerouslySetInnerHTML=${{ __html: tx(en) }}></td>`;
const Lead = (en) => html`<p class="lead" dangerouslySetInnerHTML=${{ __html: tx(en) }}></p>`;

export function Integration({ active }) {
  const [snip, setSnip] = useState(() => defaultShell() === "powershell" ? "powershell" : "curl");
  const [current, setCurrent] = useState("d-agents");
  // The contents follow the reader: the section at the top of the panel is marked.
  useEffect(() => {
    if (!active) return;
    const main = document.getElementById("main");
    const spy = () => {
      const top = main.getBoundingClientRect().top + 40;
      let cur = "d-agents";
      for (const s of document.querySelectorAll(".doc section")) if (s.getBoundingClientRect().top <= top) cur = s.id;
      setCurrent(cur);
    };
    main.addEventListener("scroll", spy, { passive: true });
    return () => main.removeEventListener("scroll", spy);
  }, [active]);
  const served = live.last && live.last.model;
  const origin = location.origin, model = (served && served.id) || "MODEL";
  const keyed = live.notice?.withKey || (live.last?.server?.api_key_required ?? !!apiKey());
  const keyVar = keyed ? "$FLYX_KEY" : null;
  const S = snippets(origin, model, keyVar);
  const ready = served && served.id && Number.isSafeInteger(served.context) && served.context > 0 && !live.notice;

  const telemetrySnip = "curl -s " + shQuote(origin + "/v1/telemetry") + (keyVar ? " -H \"Authorization: Bearer " + keyVar + "\"" : "") +
    "\nflyx monitor " + shQuote(origin) + (keyVar ? " --key \"" + keyVar + "\"" : "");
  return html`
  <div class="docs">
    <nav class="toc" aria-label=${t("contents")} id="toc">
    <span class="caps">${t("On this page")}</span>
    ${TOC.map(([id, name]) => html`<a href=${"#" + id} class=${current === id ? "on" : ""} onClick=${jump}>${t(name)}</a>`)}
  </nav>
    <div class="doc" onClick=${jumpInside}>
      <${AgentSetup} origin=${origin} model=${model} context=${ready ? served.context : null} keyed=${keyed}/>
      <section id="d-quick">
        <h2>${t("SDKs and HTTP")}</h2>
        <table><tbody>
          <tr><td>${t("OpenAI SDKs and compatible tools")}</td><td class="mono" id="d-base-openai">${origin + "/v1"}</td></tr>
          <tr><td>${t("Anthropic SDKs, Claude Code")}</td><td class="mono" id="d-base-anthropic">${origin}</td></tr>
          <tr><td>${t("Model id")}</td><td class="mono" id="d-model">${model}</td></tr>
          <tr><td>${t("API key")}</td><td id="d-key">${t(keyed ? "required: Authorization: Bearer KEY, or x-api-key" : "none set: any value works")}</td></tr>
          <tr><td>${t("Web pages")}</td><td id="d-cors">${corsNote(live.last?.server?.cors)}</td></tr>
        </tbody></table>
        <div class="pills" role="tablist" id="snip-seg" style="margin: 18px 0 8px">
        ${SNIPS.map(([k, name]) => html`<button role="tab" data-snip=${k} aria-selected=${String(snip === k)} onClick=${() => setSnip(k)}>${name}</button>`)}
      </div>
        <${CodeSnippet} id="sn-main" text=${S[snip]}/>
      </section>

      <section id="d-endpoints">
        <h2>${t("Endpoints")}</h2>
        <p class="lead">${t("Every path is served at this address. Open one for its fields.")}</p>

        <h3>${t("OpenAI-compatible")}</h3>
        <div class="eps">
        <details class="ep"><summary><span class="method post">POST</span><span class="path">/v1/chat/completions</span><span class="what">${t("Chat")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>
            ${Li(String.raw`<code>messages</code>, <code>tools</code> (<code>strict: false</code> lets a tool's arguments be any object), <code>tool_choice</code> (<code>auto</code>, <code>required</code>, <code>none</code>, or a named function), <code>parallel_tool_calls</code>`)}
            ${Li(String.raw`<code>response_format</code>: <code>json_object</code> or <code>json_schema</code>`)}
            ${Li(String.raw`<code>stream</code>, <code>stream_options.include_usage</code>`)}
            ${Li(String.raw`<code>max_tokens</code> / <code>max_completion_tokens</code>, <code>stop</code>, <code>seed</code>`)}
            ${Li(String.raw`<code>temperature</code>, <code>top_p</code>, <code>top_k</code>, <code>presence_penalty</code>, <code>frequency_penalty</code> (-2 to 2)`)}
            ${Li(String.raw`<code>reasoning_effort</code> (<code>none</code>, <code>minimal</code>, <code>low</code>, <code>medium</code>, <code>high</code>, <code>xhigh</code>), <code>enable_thinking</code>, and <code>chat_template_kwargs</code> (every key reaches the template)`)}
            ${Li(String.raw`Thinking comes back as <code>reasoning_content</code>, calls as <code>tool_calls</code> (streamed incrementally as deltas); <code>usage.prompt_tokens_details.cached_tokens</code> counts prompt tokens reused from the session.`)}
          </ul></div></details>
        <details class="ep"><summary><span class="method post">POST</span><span class="path">/v1/responses</span><span class="what">${t("Responses")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>
            ${Li(String.raw`<code>input</code> (a string, or items: messages, <code>reasoning</code>, <code>function_call</code>, <code>function_call_output</code>), <code>instructions</code>`)}
            ${Li(String.raw`<code>previous_response_id</code>, <code>store</code>, function <code>tools</code> (standalone or in a <code>namespace</code>), <code>tool_choice</code>, <code>parallel_tool_calls</code>. Calls to a namespaced function carry its <code>namespace</code> and <code>name</code>.`)}
            ${Li(String.raw`<code>text.format</code> (<code>json_object</code>, <code>json_schema</code>), <code>reasoning.effort</code> (<code>none</code>: no thinking)`)}
            ${Li(String.raw`<code>max_output_tokens</code>, <code>temperature</code>, <code>top_p</code>, <code>top_k</code>, <code>seed</code>, <code>stream</code>`)}
            ${Li(String.raw`A retry that replaces an active response leaves its partial output and usage under <code>status: incomplete</code>. The stream ends with <code>response.incomplete</code>.`)}
            ${Li(String.raw`Refused with a 400: built-in and custom tool types, <code>background</code>, <code>conversation</code>, <code>prompt</code>.`)}
            ${Li(String.raw`Stored responses: <code>GET</code> and <code>DELETE /v1/responses/{id}</code>, <code>GET /v1/responses/{id}/input_items</code>; in memory, 256 MB, oldest dropped first.`)}
          </ul></div></details>
        <details class="ep"><summary><span class="method post">POST</span><span class="path">/v1/completions</span><span class="what">${t("Raw text completion")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`Send <code>prompt</code> as a string. Sampling fields are the same as for chat. <code>max_tokens</code> defaults to 16 when omitted or null.`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/models</span><span class="what">${t("The loaded model")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>
            ${Li(String.raw`Empty while no model is loaded.`)}
            ${Li(String.raw`<code>GET /v1/models/{id}</code> returns the loaded model object, or 404 (<code>model_not_found</code>) when unloaded or unknown.`)}
          </ul></div></details>
        </div>

        <h3>${t("Anthropic-compatible")}</h3>
        <div class="eps">
        <details class="ep"><summary><span class="method post">POST</span><span class="path">/v1/messages</span><span class="what">${t("Messages")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>
            ${Li(String.raw`<code>system</code>, text / <code>tool_use</code> / <code>tool_result</code> / <code>thinking</code> blocks, <code>tools</code>`)}
            ${Li(String.raw`<code>tool_choice</code> (<code>auto</code>, <code>any</code>, <code>tool</code>, <code>none</code>; <code>disable_parallel_tool_use</code>), <code>thinking</code> (<code>enabled</code> / <code>disabled</code>), <code>stop_sequences</code>, <code>stream</code>`)}
            ${Li(String.raw`<code>output_config.format</code> (<code>json_schema</code>), <code>output_config.effort</code> (<code>low</code>, <code>medium</code>, <code>high</code>, <code>max</code>)`)}
            ${Li(String.raw`<code>cache_control</code> marks stable prefixes in system blocks, tool definitions and messages. FLYX can reuse a checkpoint when its rendered tokens match the new prompt exactly.`)}
            ${Li(String.raw`<code>max_tokens: 0</code> prewarms a prompt without generating output. Send a non-streaming request. An edited prefix can reuse only an earlier matching checkpoint.`)}
            ${Li(String.raw`<code>cache_read_input_tokens</code> reports how many input tokens were actually reused.`)}
            ${Li(String.raw`Claude Code's <code>x-anthropic-billing-header</code> line is left out of the system prompt: it changes on every request and would defeat prompt reuse. <code>--keep-billing-header</code> keeps it.`)}
          </ul></div></details>
        <details class="ep"><summary><span class="method post">POST</span><span class="path">/v1/messages/count_tokens</span><span class="what">${t("Prompt size of a Messages request")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`The same body as <code>/v1/messages</code>.`)}</ul></div></details>
        </div>

        <h3>${t("Model management")}</h3>
        ${Lead(String.raw`From this machine only, with <code>--api-key</code> for the key's holder, or with <code>--no-auth</code>.`)}
        <div class="eps">
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/load</span><span class="what">${t("Load state")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>
            ${Li(String.raw`<code>state</code>: <code>none</code>, <code>fetching</code>, <code>loading</code>, <code>ready</code>, <code>failed</code>`)}
            ${Li(String.raw`<code>progress</code> (file, bytes done and in all, bytes a second), <code>error</code>, <code>notes</code>, what was loaded and what the settings file remembers`)}
          </ul></div></details>
        <details class="ep"><summary><span class="method post">POST</span><span class="path">/v1/load</span><span class="what">${t("Load a model")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body">
            <div class="snip"><pre>{"model": "unsloth/Qwen3.8-27B-GGUF:UD-IQ2_S",
 "drafters": ["z-lab/Qwen3.8-27B-DFlash2"],
 "ctx": 32768, "drafter": "all", "kv": "fp8", "kv_ram": "auto"}</pre></div>
            <table><tbody>
              <tr><td><code>model</code></td>${Td(String.raw`required: a path or <code>org/name:QUANT</code>`)}</tr>
              <tr><td><code>drafters</code></td>${Td(String.raw`more drafters, paths or Hub names (as <code>--drafter-file</code>)`)}</tr>
              <tr><td><code>ctx</code></td>${Td(String.raw`context per session, or <code>"max"</code>; left out (auto): the model's own, or the longest the GPU and host memory hold`)}</tr>
              <tr><td><code>drafter</code>, <code>speculative</code></td>${Td(String.raw`<code>mtp</code> | <code>dflash2</code> | <code>dflash</code> | <code>dspark</code> | <code>dspine</code> | <code>eagle4</code> | <code>all</code> | <code>none</code>; <code>false</code> for plain decoding`)}</tr>
              <tr><td><code>kv</code>, <code>kv_ram</code>, <code>kv_gpu</code></td>${Td(String.raw`as <code>--kv</code>, <code>--kv-ram</code> (GB or <code>"auto"</code>), <code>--kv-gpu</code>`)}</tr>
              <tr><td><code>vram_reserve</code>, <code>ram_reserve</code></td>${Td(String.raw`as <code>--vram-reserve</code>, <code>--ram-reserve</code> (GB to leave free, <code>0</code> for minimum/all, or <code>"auto"</code>)`)}</tr>
              <tr><td><code>name</code></td>${Td(String.raw`the model id the API reports`)}</tr>
              <tr><td><code>lora</code></td>${Td(String.raw`a LoRA adapter beside the model, as <code>--lora</code>: a PEFT folder or a Hub name (<code>org/name</code>, <code>org/name:FOLDER</code>); <code>null</code> for none`)}</tr>
            </tbody></table>
            <ul>${Li(String.raw`202 and the state; 409 while another load runs. The current model keeps answering while the new one is fetched.`)}
              ${Li(String.raw`If the new model fails to load, no model is served until another is loaded (<code>"state": "failed"</code>).`)}</ul>
          </div></details>
        <details class="ep"><summary><span class="method delete">DELETE</span><span class="path">/v1/load</span><span class="what">${t("Stop a fetch")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`What came stays in the cache; loading the same name again goes on from there.`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET / POST</span><span class="path">/v1/settings/network</span><span class="what">${t("Serve on local network")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><code>{"lan_access": true, "remote_manage": false}</code><p>${t("lan_access applies on restart; remote_manage applies immediately.")}</p></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/hub/config</span><span class="what">${t("Hub mirror setting")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`Active Hugging Face endpoint (<code>endpoint</code>), default endpoint, <code>source</code> (<code>setting</code>, <code>HF_ENDPOINT</code>, or <code>default</code>), and <code>offline</code> flag.`)}</ul></div></details>
        <details class="ep"><summary><span class="method post">POST</span><span class="path">/v1/hub/config</span><span class="what">${t("Set the Hub mirror")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`Sets or resets the Hugging Face endpoint (<code>{"endpoint": "https://hf-mirror.com"}</code>, or <code>""</code> to reset) and persists it in <code>serve.json</code>.`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/hub/search?q=TEXT</span><span class="what">${t("Repositories on Hugging Face")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`GGUF repositories and drafters matching <code>q</code> (or an exact <code>org/name</code>), ordered by <code>sort</code> (<code>trending</code>, <code>downloads</code>, <code>likes</code>, <code>updated</code>, <code>created</code>), paginated with <code>limit</code> and <code>cursor</code>; without <code>q</code>, supported architectures (optionally narrowed by <code>arch</code>).`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/hub/files?repo=ORG/NAME</span><span class="what">${t("A repository's models")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`Its GGUFs by quantization with their sizes and whether cached, its companion files (<code>companions</code>: drafters and <code>mmproj</code> projectors), the drafters verified with FLYX for it; what Hugging Face says of it, and whether this build runs its architecture.`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/hub/drafter?name=NAME</span><span class="what">${t("A drafter's kind")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`The drafter a path or a Hugging Face name holds, as its own fields say (<code>mtp</code>, <code>dflash2</code>, <code>dflash</code>, <code>dspark</code>, <code>dspine</code>, <code>eagle4</code>; <code>null</code> when they say none); nothing is fetched. 400 for a <code>name</code> that is neither, 404 for a Hub name Hugging Face does not have.`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/hub/loras?base=ORG/NAME</span><span class="what">${t("LoRA adapters for a model")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`The LoRA adapters Hugging Face lists as made for <code>base</code>, most downloaded first, each with whether it is cached; with <code>model</code> instead (a Hub name or a file of the cache), the base its model card names. <code>"base": null</code> when it is not known; 502 when the Hub cannot be asked.`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/hub/lora?repo=ORG/NAME</span><span class="what">${t("A repository's LoRA adapters")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`Its adapters, one a folder with an <code>adapter_config.json</code> (at most 32): rank, scale, the linears and layers they adapt, and the name <code>"lora"</code> takes. <code>why</code> is <code>null</code> for one that loads, else the reason a load would give; only each config and safetensors header is read, into memory.`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/hub/cached</span><span class="what">${t("Models on this machine")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`Every snapshot in the Hugging Face cache, and the folders of <code>--models-dir</code> / <code>FLYX_MODELS_DIR</code>: models, <code>mmproj</code> projectors, LoRA adapters (<code>"kind": "lora"</code>) and local drafters, each by the name that loads it, with its quantization, size, revision and whether it carries an MTP layer.`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/hub/dir?path=PATH</span><span class="what">${t("A folder's model files")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`A folder of the server's machine: its subfolders, and the <code>.gguf</code> and <code>.flyx</code> model files in it and two folders down, each by the name that loads it. Without <code>path</code>: where to start from.`)}</ul></div></details>
        <details class="ep"><summary><span class="method post">POST</span><span class="path">/v1/hub/dir</span><span class="what">${t("Choose a folder with the system's dialog")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`Opens the desktop's folder dialog on the server's machine and answers with the folder chosen, as the GET does; <code>{"path": null}</code> when cancelled. From that machine's browser only (403); 501 where no dialog can be shown.`)}</ul></div></details>
        <details class="ep"><summary><span class="method delete">DELETE</span><span class="path">/v1/hub/cached?name=NAME</span><span class="what">${t("Delete a downloaded model")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`From the Hugging Face cache only (never a <code>--models-dir</code> file); 409 for the model being served or while one loads.`)}</ul></div></details>
        <details class="ep"><summary><span class="method post">POST</span><span class="path">/v1/unload</span><span class="what">${t("Unload the model")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`Its requests end and its memory is freed; requests get 503 until a model is loaded. The settings file still remembers it.`)}</ul></div></details>
        </div>

        <h3>${t("Monitoring")}</h3>
        <div class="eps">
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/telemetry</span><span class="what">${t("Live state and history")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`<code>?recent=N</code> finished requests (default 20). Fields under <a href="#d-telemetry">Telemetry</a>.`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/requests</span><span class="what">${t("Recent prompts and answers")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>
            ${Li(String.raw`The last 50 finished requests, newest first (<code>?limit=N</code>): the request body as sent, the thinking, the answer, tool calls, usage and time to first token.`)}
            ${Li(String.raw`Kept in the server's memory only; readable from this machine, with the API key, or over direct LAN when <code>--no-auth</code> or <code>remote_manage</code> is enabled. <code>--no-request-log</code> keeps none (<code>"enabled": false</code>).`)}
          </ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/v1/logs</span><span class="what">${t("The server's log lines")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>
            ${Li(String.raw`What the server process wrote to standard error, numbered: <code>?after=SEQ</code> gives the lines after one already seen, oldest first (<code>?limit=N</code>, default 500). Answers <code>lines</code> (<code>seq</code>, <code>t</code>, <code>text</code>), <code>next</code> and <code>missed</code>.`)}
            ${Li(String.raw`The last 2,000 lines, in memory only; readable from this machine, with the API key, or over direct LAN when <code>--no-auth</code> or <code>remote_manage</code> is enabled. <code>--no-system-log</code> keeps none.`)}
          </ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/health</span><span class="what">${t("Status, model, queue length")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`<code>{"status": "ok", "model": ..., "queued": 0, "loaded": true}</code>`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/</span><span class="what">${t("This page")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`To a browser; anything else gets the endpoint list as JSON.`)}</ul></div></details>
        <details class="ep"><summary><span class="method get">GET</span><span class="path">/dashboard</span><span class="what">${t("This page, to any client")}</span><svg class="i"><use href="#i-chev"/></svg></summary>
          <div class="body"><ul>${Li(String.raw`The same page as <code>/</code>; its files are under <code>/dashboard/</code>.`)}</ul></div></details>
        </div>
      </section>

      <section id="d-fields">
        <span class="caps">${t("Reference")}</span>
        <h2>${t("Request fields worth knowing")}</h2>
        <table><thead><tr><th>${t("Field")}</th><th>${t("API")}</th><th>${t("Effect")}</th></tr></thead><tbody>
          <tr><td><code>reasoning_effort</code>, <code>enable_thinking</code>, <code>chat_template_kwargs</code></td><td>chat</td>${Td(String.raw`thinking effort (<code>none</code>..<code>xhigh</code>) or toggle for this request`)}</tr>
          <tr><td><code>reasoning.effort</code></td><td>responses</td>${Td(String.raw`thinking effort (<code>none</code> disables thinking)`)}</tr>
          <tr><td><code>thinking.type</code>, <code>output_config.effort</code></td><td>messages</td>${Td(String.raw`<code>enabled</code> / <code>disabled</code>; effort <code>low</code>..<code>max</code>`)}</tr>
          <tr><td><code>tool_choice</code></td><td>${t("all")}</td>${Td(String.raw`calls are held to the tools' JSON schemas whenever tools are offered`)}</tr>
          <tr><td><code>response_format</code>, <code>text.format</code>, <code>output_config.format</code></td><td>${t("all")}</td>${Td(String.raw`a JSON object, or JSON that matches a schema`)}</tr>
          <tr><td><code>accept_min_p</code>, <code>accept_typical</code></td><td>chat</td>${Td(String.raw`FLYX's own, 0 to 1: relaxed acceptance of drafts. Lossy, off by default.`)}</tr>
          <tr><td><code>seed</code></td><td>${t("all")}</td>${Td(String.raw`one seed, the same tokens, with a drafter or without`)}</tr>
        </tbody></table>
        ${Lead(String.raw`Not supported: <code>repetition_penalty</code>; audio, video, and remote image URLs get a 400 (send images as base64 data URIs or Anthropic base64 blocks with a vision model).`)}
      </section>

      <section id="d-sampling">
        <span class="caps">${t("Reference")}</span>
        <h2>${t("Sampling defaults")}</h2>
        <p>${t("Requests that leave sampling unset get the model card's settings:")}</p>
        <table><thead><tr><th></th><th class="n">temperature</th><th class="n">top_p</th><th class="n">top_k</th><th class="n">presence</th></tr></thead><tbody>
          <tr><td>${t("Thinking on")}</td><td class="n">0.6</td><td class="n">0.95</td><td class="n">20</td><td class="n">0</td></tr>
          <tr><td>${t("Thinking off")}</td><td class="n">0.7</td><td class="n">0.8</td><td class="n">20</td><td class="n">1.5</td></tr>
        </tbody></table>
        ${Lead(String.raw`<code>--temperature</code>, <code>--top-p</code>, <code>--top-k</code>, <code>--presence-penalty</code> and <code>--frequency-penalty</code> change them for every request. A client that turns thinking off and has the model write tool calls or edit code may want <code>presence_penalty: 0</code>.`)}
      </section>

      <section id="d-sessions">
        <span class="caps">${t("Behaviour")}</span>
        <h2>${t("Sessions and prompt reuse")}</h2>
        ${para("sessions1") ? html`<p dangerouslySetInnerHTML=${{ __html: para("sessions1") }}></p>` : html`<p>FLYX matches requests to conversations by their history. By default, the server plans for two full-context sessions so a title request can run while the main conversation is active. If only one session fits, the next turn can restore a matching checkpoint from host memory or disk.</p>`}
        ${para("sessionsCompact") ? html`<p dangerouslySetInnerHTML=${{ __html: para("sessionsCompact") }}></p>` : html`<p>Claude Code generates a summary during <code>/compact</code>. If a reply reaches its output limit, the client may ask the model to continue. On the next turn, FLYX prefills the rewritten history after the last reusable prefix. If none matches, it prefills the whole prompt.</p>`}
        ${para("sessionsDisk") ? html`<p dangerouslySetInnerHTML=${{ __html: para("sessionsDisk") }}></p>` : html`<p>For a 40 GB disk cache, start <code>flyx serve</code> with <code>--kv-disk ./flyx-cache:40</code>. As a long conversation grows, FLYX captures incremental checkpoints roughly every 10,240 tokens and writes new KV rows in the background.</p>`}
        ${para("sessions2") ? html`<p dangerouslySetInnerHTML=${{ __html: para("sessions2") }}></p>` : html`<p>Available slots limit how many requests run at once. The rest wait according to <code>--queue</code> and <code>--queue-timeout</code>. A new request that repeats or extends a conversation still generating a reply cancels the older request. Disconnecting also cancels that client's request. Claude Code sessions, subagents and Codex threads with different identifiers can send identical prompts without cancelling each other.</p>`}
        ${para("sessionsInspect") ? html`<p dangerouslySetInnerHTML=${{ __html: para("sessionsInspect") }}></p>` : html`<p>Open <a href="#requests">Requests</a> and select the affected turn to see how many input tokens it reused and how long it spent in the queue, in prefill and in decode. The detail also shows when a newer request replaced it. When Claude Code provides the labels, you can also identify the summary and the first turn after compaction.</p>`}
      </section>

      <section id="d-flags">
        <span class="caps">${t("Reference")}</span>
        <h2>${t("flyx serve flags")}</h2>
        ${Lead(String.raw`<code>flyx serve [MODEL] [flags]</code>. MODEL is a <code>.gguf</code> or <code>.flyx</code> path, or <code>org/name:QUANT</code>. Without one, the model served last, or none.`)}
        <h3>${t("Server")}</h3>
        <table><tbody>
          <tr><td><code>--host H</code>, <code>--port P</code></td>${Td(String.raw`where it listens, default <code>127.0.0.1:8080</code>`)}</tr>
          <tr><td><code>--name NAME</code></td>${Td(String.raw`the model id the API reports (default: the file's name)`)}</tr>
          <tr><td><code>--api-key KEY</code></td>${Td(String.raw`require <code>Authorization: Bearer KEY</code> or <code>x-api-key</code>`)}</tr>
          <tr><td><code>--no-auth</code></td>${Td(String.raw`allow LAN dashboard and management without an API key (trusted networks only; ignores <code>FLYX_API_KEY</code>)`)}</tr>
          <tr><td><code>--cors ORIGIN</code></td>${Td(String.raw`let the pages of ORIGIN (<code>https://host[:port]</code>, an extension's origin, <code>null</code>) call the inference API from a browser; repeatable; <code>'*'</code>: any page; <code>none</code>: not even this machine's own pages, which may by default. The model panel, the request log and the logs are never another origin's`)}</tr>
          <tr><td><code>--threads N</code></td>${Td(String.raw`HTTP worker threads (64): one per request in flight, queued or running`)}</tr>
          <tr><td><code>--models-dir DIR</code></td>${Td(String.raw`also list the <code>.gguf</code> / <code>.flyx</code> files under DIR on the Model page (repeatable; also <code>FLYX_MODELS_DIR</code>)`)}</tr>
          <tr><td><code>--no-request-log</code></td>${Td(String.raw`keep no prompts or answers for the Requests page (kept by default: the last 50, in memory)`)}</tr>
          <tr><td><code>--no-system-log</code></td>${Td(String.raw`keep no log lines for the Logs page (kept by default: the last 2,000, in memory)`)}</tr>
          <tr><td><code>--dashboard-dir DIR</code></td>${Td(String.raw`serve this page's files from DIR where it has them (<code>apps/flyx-server/dashboard</code>): a change shows on a reload, without a build`)}</tr>
          <tr><td><code>--no-monitor</code>, <code>--no-browser</code></td>${Td(String.raw`no dashboard on the terminal; do not open this page`)}</tr>
          <tr><td><code>-v</code></td>${Td(String.raw`log every request`)}</tr>
        </tbody></table>
        <h3>${t("Sessions and memory")}</h3>
        <table><tbody>
          <tr><td><code>--sessions N</code></td><td>${t("Maximum GPU sessions, subject to available memory (default 8).")}</td></tr>
          <tr><td><code>--host-cache GB</code></td><td>${t("RAM budget for saved checkpoints, including system prompts shared across sessions (default 8 GB, limited by memory left after live KV). Set to 0 to disable this RAM cache.")}</td></tr>
          <tr><td><code>--kv-disk DIR[:GB]</code></td><td>${t("Enable incremental checkpoints and spill older RAM checkpoints to this directory. Omitting GB uses up to 64 GB, capped at half the available space.")}</td></tr>
          <tr><td><code>--queue N</code>, <code>--queue-timeout S</code></td>${Td(String.raw`requests allowed to wait (16); the longest wait (600 s)`)}</tr>
          <tr><td><code>--idle-unload S</code></td>${Td(String.raw`free the sessions after S idle seconds (0: never)`)}</tr>
          <tr><td><code>--keep-alive S</code></td>${Td(String.raw`unload the model after S idle seconds, load it again on demand (0: never)`)}</tr>
          <tr><td><code>--no-batch</code></td>${Td(String.raw`requests running at once take turns instead of sharing rounds`)}</tr>
        </tbody></table>
        <h3>${t("Sampling defaults")}</h3>
        <table><tbody>
          <tr><td><code>--temperature T</code>, <code>--top-p P</code>, <code>--top-k K</code></td>${Td(String.raw`for requests that do not set them`)}</tr>
          <tr><td><code>--presence-penalty P</code>, <code>--frequency-penalty F</code></td>${Td(String.raw`-2 to 2`)}</tr>
          <tr><td><code>--no-think</code></td>${Td(String.raw`no thinking unless a request asks for it`)}</tr>
          <tr><td><code>--keep-billing-header</code></td>${Td(String.raw`keep Claude Code's billing line in the system prompt`)}</tr>
        </tbody></table>
        <h3>${t("Engine")}</h3>
        <table><tbody>
          <tr><td><code>--ctx N|max</code></td>${Td(String.raw`context per session (default: <code>max</code> — the model's own, or the longest the GPU and host memory hold)`)}</tr>
          <tr><td><code>--kv fp8|16</code></td>${Td(String.raw`the K/V cache: fp8 holds twice the context; 16 is llama.cpp's binary16 (the default where 32 GB of GPU memory are left after the weights and on Llama models, else fp8)`)}</tr>
          <tr><td><code>--kv-ram GB|auto</code></td>${Td(String.raw`host memory for the K/V past the GPU's part (auto; 0: the context shrinks to fit the GPU)`)}</tr>
          <tr><td><code>--kv-gpu N</code></td>${Td(String.raw`at most N tokens' K/V on the GPU, the rest in host memory`)}</tr>
          <tr><td><code>--vram-reserve GB|auto</code></td>${Td(String.raw`GPU memory kept free outside FLYX (auto; 0: minimum allowed by the backend/OS)`)}</tr>
          <tr><td><code>--ram-reserve GB|auto</code></td>${Td(String.raw`host RAM kept free outside KV (auto; 0: use all available RAM)`)}</tr>
          <tr><td><code>--kv-method M</code></td>${Td(String.raw`<code>auto</code> = <code>auto:2</code> (default), <code>exact</code>, <code>strata</code> = <code>strata:2</code>. <code>auto:3</code> uses a 3-bit scan after the switch.`)}</tr>
          <tr><td><code>--drafter D</code></td><td><code>mtp</code> | <code>dflash2</code> | <code>dflash</code> | <code>dspark</code> | <code>dspine</code> | <code>eagle4</code> | <code>dv</code> | <code>all</code> | <code>none</code></td></tr>
          <tr><td><code>--drafter-file F</code></td>${Td(String.raw`a drafter in a file or Hub repository of its own (repeatable): an MTP GGUF, DFlash2, DFlash, DSpark, DSpine or EAGLE-4`)}</tr>
          <tr><td><code>--no-spec</code>, <code>--no-lookup</code></td>${Td(String.raw`plain decoding; no drafts from the text where it repeats`)}</tr>
          <tr><td><code>--width N</code></td>${Td(String.raw`fix the model drafter's width at N, up to its maximum (default: chosen per round). Prompt lookup can still verify 15 drafts from earlier text. Use <code>--no-lookup</code> to disable it. Not beside a DV drafter: its rounds then run alone, its widest of at most N every round.`)}</tr>
          <tr><td><code>--draft-trees</code> / <code>--no-draft-trees</code></td>${Td(String.raw`drafters also propose trees`)}</tr>
          <tr><td><code>--prefill-chunk N</code></td>${Td(String.raw`prompt rows per prefill pass (1024 to 4096 by free GPU memory)`)}</tr>
          <tr><td><code>--backend cuda|cpu_ref</code></td>${Td(String.raw`the GPU, or the float64 reference; <code>CUDA_VISIBLE_DEVICES</code> picks the GPU`)}</tr>
          <tr><td><code>--backend-opt K=V</code></td>${Td(String.raw`a backend option, repeatable (also <code>FLYX_BACKEND_OPTIONS</code>)`)}</tr>
          <tr><td><code>--activations exact|nvfp4</code></td>${Td(String.raw`exact: every linear rounds its inputs as its format does; nvfp4: NVFP4 weights take NVFP4 inputs in prefill passes whatever the file says (a GPU with FP4 tensor cores)`)}</tr>
          <tr><td><code>--chat-template F</code></td>${Td(String.raw`a Jinja template file instead of the model's`)}</tr>
        </tbody></table>
      </section>

      <section id="d-telemetry">
        <span class="caps">${t("Reference")}</span>
        <h2>${t("Telemetry")}</h2>
        ${Lead(String.raw`<code>GET /v1/telemetry</code>, JSON, <code>schema_version: 1</code>. This page reads it once a second, at no measurable cost to decoding. <code>flyx monitor URL</code> draws the same in a terminal.`)}
        <table><thead><tr><th>${t("Key")}</th><th>${t("Contents")}</th></tr></thead><tbody>
          <tr><td><code>server</code></td>${Td(String.raw`name, version, pid, started_at, uptime_s, api_key_required, agent_launch, cors`)}</tr>
          <tr><td><code>model</code></td>${Td(String.raw`id, loaded, device, backend, drafter, max_width, context, chat_format, max_sessions, memory, load_s`)}</tr>
          <tr><td><code>state</code></td>${Td(String.raw`phase, running, queued. The oldest running request supplies label, session, prompt_tokens, prefilled, reused_tokens, generated, tokens_per_round, rounds, drafted, accepted, decode_tok_s, prefill_tok_s and elapsed_s. Chat completions also supply id.`)}</tr>
          <tr><td><code>rates</code></td>${Td(String.raw`decode_tok_s, tokens_per_round and acceptance (0..1) over the recent window across running requests, plus output_per_second, prefill_per_second (prompt tokens computed), rounds_per_second, drafted_per_second and accepted_per_second over the last 60 seconds`)}</tr>
          <tr><td><code>device</code></td>${Td(String.raw`util, mem_used, mem_total, power_w, power_limit_w, temp_c, fan, sm_clock_mhz, pcie_gen, pcie_width, clock_limits, process_host (the process's page-locked host memory, where the system counts it)`)}</tr>
          <tr><td><code>memory</code></td>${Td(String.raw`ledger_used, ledger_budget, runtime (FLYX's CUDA runtime beside the ledger, where the system counts it), host_mapped (the K/V's host memory the resident figure misses), sessions (id, tokens)`)}</tr>
          <tr><td><code>recent</code></td>${Td(String.raw`finished requests: label, status, prompt, reused and output tokens, queue_ms, ttft_ms (arrival to first output event, null without output), prefill_ms, decode_ms, decode_tok_s, tokens_per_round, drafted, accepted_drafts, tool_calls`)}</tr>
          <tr><td><code>totals</code></td>${Td(String.raw`requests, errors, cancelled, prompt, reused and output tokens, tool_calls, grammar_fixes`)}</tr>
          <tr><td><code>load</code></td>${Td(String.raw`as <code>GET /v1/load</code>`)}</tr>
          <tr><td><code>server</code>, <code>process</code>, <code>system</code></td>${Td(String.raw`pid, uptime; cpu_s, rss, ram (all the host memory FLYX holds); ram_total, ram_available, cpus, cpu_s (the machine's busy seconds, all processors added up)`)}</tr>
        </tbody></table>
        <${CodeSnippet} id="sn-telemetry" text=${telemetrySnip}/>
      </section>

      <section id="d-more">
        <span class="caps">${t("Repository")}</span>
        <h2>${t("More")}</h2>
        ${para("more") ? html`<p dangerouslySetInnerHTML=${{ __html: para("more") }}></p>` : html`<p><code>docs/server.md</code> has every field, <code>docs/models.md</code> the models and drafters, <code>docs/monitor.md</code> this page and the terminal panel, <code>docs/building.md</code> the build; the C API is <code>include/flyx/flyx.h</code>.</p>`}
      </section>
    </div>
  </div>`;
}
