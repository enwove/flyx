// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A logged request as chat apps show a conversation: the system prompt and tools folded, the user's turns on the
// right, the assistant's (thinking, text, tool calls) beside its avatar, tool results folded, then the reply.
// Reads the request as each API writes it (Chat Completions, Messages, Responses, text completions).
import { html } from "../lib/html.js";
import { t } from "../lib/i18n.js";
import { fmt, secs } from "../lib/format.js";
import { md } from "../lib/markdown.js";
import { Icon, CopyButton, statusName } from "../lib/ui.js";

export function figures(e) {
  const timing = e.timing || {};
  return { u: e.usage || {}, total: e.finished_at - e.started_at, ttft: e.ttft_ms == null ? null : e.ttft_ms / 1000,
           tps: timing.decode_tok_s, tpr: timing.tokens_per_round };
}

function partsText(c) {
  if (c === null || c === undefined) return "";
  if (typeof c === "string") return c;
  if (Array.isArray(c)) return c.map((p) => typeof p === "string" ? p : p.text !== undefined ? p.text :
    p.type === "image" || p.type === "image_url" || p.type === "input_image" ? "[image]" : "").filter(Boolean).join("\n");
  return JSON.stringify(c);
}

// A request as the steps of a conversation: system, user, assistant (thinking, text, tool calls), tool results.
export function stepsOf(e) {
  const q = e.request || {}, out = [], names = new Map();
  const asst = () => { if (!out.length || out[out.length - 1].kind !== "assistant") out.push({ kind: "assistant", parts: [] }); return out[out.length - 1].parts; };
  if (e.api === "messages") {
    if (q.system) out.push({ kind: "system", text: partsText(q.system) });
    for (const m of q.messages || []) {
      const blocks = typeof m.content === "string" ? [{ type: "text", text: m.content }] : m.content || [];
      for (const b of blocks) {
        if (m.role === "assistant") {
          if (b.type === "thinking") asst().push({ kind: "think", text: b.thinking || "" });
          else if (b.type === "tool_use") { names.set(b.id, b.name); asst().push({ kind: "call", name: b.name, args: b.input }); }
          else if (b.type === "text") asst().push({ kind: "text", text: b.text });
        } else if (b.type === "tool_result") out.push({ kind: "result", name: names.get(b.tool_use_id), text: partsText(b.content) });
        else if (b.type === "text") out.push({ kind: "user", text: b.text });
        else out.push({ kind: "user", text: "[" + b.type + "]" });
      }
    }
  } else if (e.api === "responses") {
    if (q.instructions) out.push({ kind: "system", text: q.instructions });
    if (typeof q.input === "string") out.push({ kind: "user", text: q.input });
    for (const it of Array.isArray(q.input) ? q.input : []) {
      if (it.type === "function_call") { names.set(it.call_id, it.name); asst().push({ kind: "call", name: it.name, args: it.arguments }); }
      else if (it.type === "function_call_output") out.push({ kind: "result", name: names.get(it.call_id), text: partsText(it.output) });
      else if (it.type === "reasoning") asst().push({ kind: "think", text: partsText(it.content || it.summary) });
      else if (it.role === "assistant") asst().push({ kind: "text", text: partsText(it.content) });
      else if (it.role === "system" || it.role === "developer") out.push({ kind: "system", text: partsText(it.content) });
      else out.push({ kind: "user", text: partsText(it.content) });
    }
  } else if (e.api === "completions") {
    out.push({ kind: "user", text: partsText(q.prompt), code: true });
  } else {
    for (const m of q.messages || []) {
      if (m.role === "system" || m.role === "developer") out.push({ kind: "system", text: partsText(m.content) });
      else if (m.role === "assistant") {
        const p = asst();
        if (m.reasoning_content) p.push({ kind: "think", text: m.reasoning_content });
        if (partsText(m.content)) p.push({ kind: "text", text: partsText(m.content) });
        for (const c of m.tool_calls || []) { const f = c.function || {}; names.set(c.id, f.name); p.push({ kind: "call", name: f.name, args: f.arguments }); }
      } else if (m.role === "tool") out.push({ kind: "result", name: m.name || names.get(m.tool_call_id), text: partsText(m.content) });
      else out.push({ kind: "user", text: partsText(m.content) });
    }
  }
  return out;
}

// Thinking, folded: what the model thought, as it wrote it.
export const Think = ({ text, label, open }) => html`
  <details class="cv-think" open=${open}><summary><${Icon} name="chev"/>${label}</summary><div>${text}</div></details>`;

function Call({ name, args }) {
  if (typeof args === "string") { try { args = JSON.parse(args); } catch (x) { /* shown as written */ } }
  return html`<div class="cv-call"><div><${Icon} name="tool"/><b>${name || "tool"}</b></div>
    <pre>${typeof args === "string" ? args : JSON.stringify(args, null, 2)}</pre></div>`;
}

// One assistant turn: thinking, text and tool calls, the avatar beside them; `children` come after the parts.
export function AsstTurn({ parts, name, current, children }) {
  return html`
    <div class="cv-asst"><div class=${"cv-av" + (current ? "" : " past")}>x</div>
      <div class="cv-col">
        ${name ? html`<div class="cv-name">${name}</div>` : ""}
        ${parts.map((p) => p.kind === "think" && p.text ? html`<${Think} text=${p.text} label=${t("Thought · " + fmt(p.text.length) + " characters")}/>`
          : p.kind === "text" && p.text ? html`<div class="md" dangerouslySetInnerHTML=${{ __html: md(p.text) }}></div>`
          : p.kind === "call" ? html`<${Call} name=${p.name} args=${p.args}/>` : "")}
        ${children}
      </div>
    </div>`;
}

// Something long, folded: a title (bold head, the rest after a dot) over its text.
const Fold = ({ head, rest, body, mono }) => html`
  <details class="cv-fold"><summary><${Icon} name="chev"/><b>${head}</b>${rest.map((r) => html` · ${r}`)}</summary>
    <div>${mono ? html`<pre>${body}</pre>` : body}</div></details>`;

export function Conversation({ e }) {
  const q = e.request || {}, f = figures(e);
  const chips = [e.model || "model", e.id];
  if (e.client_hints) {
    const { request_class, compaction, context_compacted } = e.client_hints;
    const kinds = { main: "Main conversation", subagent: "Subagent", workflow: "Workflow",
                    compaction: "Conversation summary", auxiliary: "Auxiliary request" };
    const triggers = { manual: "Manual compaction", auto: "Automatic compaction", reactive: "Prompt exceeded the context" };
    if (kinds[request_class]) chips.push("Claude Code · " + t(kinds[request_class]));
    if (triggers[compaction]) chips.push(t(triggers[compaction]));
    if (triggers[context_compacted]) chips.push(t("First turn after compaction") + " · " + t(triggers[context_compacted]));
  }
  if (q.temperature !== undefined) chips.push("temperature " + q.temperature);
  const mt = q.max_tokens || q.max_completion_tokens || q.max_output_tokens;
  if (mt) chips.push("max " + fmt(mt) + " tokens");
  if (q.stream) chips.push("stream");
  const kwargs = q.chat_template_kwargs && typeof q.chat_template_kwargs === "object" ? q.chat_template_kwargs : null;
  const effort = (kwargs && typeof kwargs.reasoning_effort === "string" && kwargs.reasoning_effort)
    || (typeof q.reasoning_effort === "string" && q.reasoning_effort)
    || (q.reasoning && typeof q.reasoning.effort === "string" && q.reasoning.effort)
    || (q.output_config && typeof q.output_config.effort === "string" && q.output_config.effort)
    || null;
  let thinkingOff = false;
  if (q.thinking && q.thinking.type === "disabled") thinkingOff = true;
  else if (kwargs && typeof kwargs.enable_thinking === "boolean") thinkingOff = !kwargs.enable_thinking;
  else if (typeof q.enable_thinking === "boolean") thinkingOff = !q.enable_thinking;
  else if (effort === "none") thinkingOff = true;
  if (thinkingOff) chips.push("thinking off");
  if (effort && effort !== "none") chips.push("effort " + effort);
  if (q.response_format || (q.text && q.text.format) || (q.output_config && q.output_config.format)) chips.push("JSON answer");
  const tools = q.tools || [];
  const toolNames = tools.map((x) => (x.function && x.function.name) || x.name).filter(Boolean);
  const parts = [];
  if (e.reasoning) parts.push({ kind: "think", text: e.reasoning });
  if (e.content) parts.push({ kind: "text", text: e.content });
  for (const c of e.tool_calls || []) parts.push({ kind: "call", name: c.name, args: c.arguments });
  const chars = (s) => t(fmt((s || "").length) + " characters");
  const meta = [fmt(f.u.output_tokens) + " tokens", f.tps ? fmt(f.tps, 1) + " tok/s" : null, f.ttft !== null ? "TTFT " + secs(f.ttft) : null,
                f.tpr ? fmt(f.tpr, 2) + " tok/round" : null, "total " + secs(f.total)].filter(Boolean);
  return html`
    <div class="cv">
      <div class="cv-chips">${chips.map((c) => html`<span>${c}</span>`)}</div>
      ${tools.length ? html`<${Fold} head=${t(tools.length + (tools.length === 1 ? " tool" : " tools"))} rest=${[toolNames.join(", ")]}
                                    body=${JSON.stringify(tools, null, 2)} mono/>` : ""}
      ${stepsOf(e).map((st) =>
        st.kind === "system" ? html`<${Fold} head=${t("System prompt")} rest=${[chars(st.text)]} body=${st.text}/>`
        : st.kind === "user" ? html`<div class=${"cv-user" + (st.code ? " code" : "")}>${st.text || "—"}</div>`
        : st.kind === "assistant" ? html`<${AsstTurn} parts=${st.parts}/>`
        : st.kind === "result" ? html`<${Fold} head=${t("Tool result")} rest=${(st.name ? [st.name] : []).concat([chars(st.text)])} body=${st.text || ""} mono/>`
        : "")}
      <${AsstTurn} parts=${parts} name=${e.model} current>
        ${e.error ? html`<div class="cv-err">error: ${e.error}</div>` : ""}
        ${e.status === "length" && !e.content && !(e.tool_calls || []).length
          ? html`<div class="empty">${t(e.reasoning ? "Stopped at the token limit while thinking: no answer yet." : "Stopped at the token limit.")}</div>`
          : !parts.length && !e.error ? html`<div class="empty">${t("No output.")}</div>` : ""}
        <div class="cv-meta"><${CopyButton} get=${() => e.content || ""}/>
          ${meta.map((v) => html`<span>${t(v)}</span>`)}<span>${statusName(e)}</span></div>
        <div class="cv-meta">
          ${f.u.prompt_tokens > 0 ? html`<span>${t("Prompt reused")} ${fmt(f.u.cached_tokens)} / ${fmt(f.u.prompt_tokens)}</span>` : ""}
          ${e.timing ? [["Queue time", e.timing.queue_ms], ["Prefill time", e.timing.prefill_ms], ["Decode time", e.timing.decode_ms]]
            .map(([label, ms]) => ms == null ? "" : html`<span>${t(label)} ${secs(ms / 1000)}</span>`) : ""}
        </div>
      <//>
    </div>`;
}
