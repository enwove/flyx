// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Chat page: a conversation with the model served, streamed from POST /v1/chat/completions, with the figures of
// the reply being written (from telemetry) and of each finished one (the server's record of it).  Conversations are
// kept in this browser (lib/chats.js); the history button at the top right lists them.
import { html, useState, useEffect, useRef, useLayoutEffect } from "../lib/html.js";
import { headers } from "../lib/api.js";
import { t } from "../lib/i18n.js";
import { fmt, ktok, secs, ago } from "../lib/format.js";
import { Icon, CopyButton, confirmBox } from "../lib/ui.js";
import { loadChats, storeChats, keptMessage, newChatId } from "../lib/chats.js";
import { md } from "../lib/markdown.js";
import { live, poll, phaseOf } from "../lib/telemetry.js";
import { AsstTurn, Think } from "../components/conversation.js";

// The line under a reply being written: reading the prompt, then speed, tokens a round, tokens, time.
function LiveLine({ m }) {
  const tm = live.last || {}, state = tm.state || {}, phase = phaseOf(tm);
  const put = (k, v) => html`<span><b>${v}</b>${k ? " " + t(k) : ""}</span>`;
  let body;
  if (!m.id || state.id !== m.id)
    body = html`<span>${t(m.content || m.thinking ? "Streaming" : "Waiting for a reply")}</span>`;
  else if (phase === "prefill" && state.prompt_tokens)
    body = html`<span>${t("reading the prompt")}</span>${put("tokens", fmt(state.prefilled) + " / " + fmt(state.prompt_tokens))}`;
  else if (phase === "decode")
    body = html`${Number.isFinite(state.decode_tok_s) && put("tok/s", fmt(state.decode_tok_s, 1))}
      ${put("tok/round", fmt(state.tokens_per_round, 2))}${put("tokens", fmt(state.generated))}
      ${state.elapsed_s ? put("", secs(state.elapsed_s)) : ""}`;
  else body = html`<span>${t("starting")}</span>`;
  return html`<div class="cv-live"><span class="pulse"></span>${body}</div>`;
}

// The browser measures its own latency. Decode speed and tokens per round come from this completion's engine record.
function ReplyMeta({ m }) {
  const last = useRef(null);
  if (m.stopped) return html`<div class="cv-meta">${t("stopped")}</div>`;
  if (m.error) return html`<div class="cv-meta">${t("Error") + ": " + m.error}</div>`;
  if (!m.done) return html`<div class="cv-meta"></div>`;
  if (m.id && last.current?.id !== m.id)
    last.current = live.last?.recent?.find((row) => row.id === m.id) || null;
  const r = m.id && last.current?.id === m.id ? last.current : null;
  const out = m.usage ? m.usage.completion_tokens : r ? r.output_tokens : null;
  const ttft = m.clientTtft;
  const vals = [out !== null ? fmt(out) + " tokens" : null, r ? fmt(r.decode_tok_s, 1) + " tok/s" : null, ttft !== null ? "TTFT " + secs(ttft) : null,
                r ? fmt(r.tokens_per_round, 2) + " tok/round" : null, "total " + secs(m.elapsed)];
  return html`<div class="cv-meta"><${CopyButton} get=${() => m.content}/>${vals.filter(Boolean).map((v) => html`<span>${t(v)}</span>`)}</div>`;
}

export function Chat({ active }) {
  // Messages as the page shows them: {role: "user", content} or {role: "assistant", content, thinking, done, ...}.
  const [msgs, setMsgs] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [think, setThink] = useState(true);
  // The history: the conversations kept in this browser, the one shown among them ("" until its first message),
  // whether their list is open.
  const [chats, setChats] = useState(loadChats);
  const [chatId, setChatId] = useState("");
  const [histOpen, setHistOpen] = useState(false);
  const controller = useRef(null), box = useRef(null), input = useRef(null);
  const streaming = !!controller.current;
  // The conversation `id` in the history as `ms` has it now, at the top; untouched when nothing in it changed (one
  // only opened and read stays where it was).
  const keep = (id, ms) => setChats((old) => {
    const turns = ms.map(keptMessage), first = turns.find((m) => m.role === "user");
    if (!id || !first) return old;
    const was = old.find((c) => c.id === id);
    if (was && JSON.stringify(was.msgs) === JSON.stringify(turns)) return old;
    const title = first.content.replace(/\s+/g, " ").trim().slice(0, 80);
    return storeChats([{ id, at: Date.now(), title, msgs: turns }, ...old.filter((c) => c.id !== id)]);
  });
  // Kept whenever it changes and no reply is being written: after each reply, and after one is stopped or fails.
  useEffect(() => { if (!streaming && chatId && msgs.length) keep(chatId, msgs); }, [msgs, streaming, chatId]);
  useEffect(() => {
    if (!histOpen) return;
    const click = (e) => { if (!e.target.closest(".histwrap")) setHistOpen(false); };
    const esc = (e) => { if (e.key === "Escape") setHistOpen(false); };
    document.addEventListener("click", click);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("click", click); document.removeEventListener("keydown", esc); };
  }, [histOpen]);
  useEffect(() => { if (active && input.current) input.current.focus(); }, [active]);
  useLayoutEffect(() => { if (box.current) box.current.scrollTop = box.current.scrollHeight; }, [msgs]);

  const update = (changes) => setMsgs((ms) => {
    const i = ms.length - 1;
    if (ms[i]?.role !== "assistant") return ms;  // The user can clear the conversation before an aborted read settles.
    const next = ms.slice();
    next[i] = { ...ms[i], ...changes };
    return next;
  });

  async function send() {
    const content = prompt.trim();
    if (!content || controller.current) return;
    setPrompt("");
    // What the model is sent: a turn that failed is left out with its question; a stopped reply goes as far as it got.
    const history = [];
    for (const m of msgs) {
      if (m.role === "user") history.push({ role: "user", content: m.content });
      else if (m.error) history.pop();
      else if (m.done || (m.stopped && m.content)) history.push({ role: "assistant", content: m.content });
    }
    const turns = history.concat([{ role: "user", content }]);
    // The question is kept at once: a page closed before the reply came still has it.
    const chat = chatId || newChatId();
    if (!chatId) setChatId(chat);
    keep(chat, msgs.concat([{ role: "user", content }]));
    setMsgs((ms) => ms.concat([{ role: "user", content }, { role: "assistant", content: "", thinking: "", model: (live.last && live.last.model && live.last.model.id) || "Model" }]));
    controller.current = new AbortController();
    const t0 = performance.now();
    let first = 0, reply = "", thinking = "", usage = null, id = "";
    try {
      const r = await fetch("/v1/chat/completions", {
        method: "POST", signal: controller.current.signal, headers: headers({ "Content-Type": "application/json" }),
        body: JSON.stringify({ model: (live.last && live.last.model && live.last.model.id) || "", messages: turns, stream: true,
                               stream_options: { include_usage: true }, chat_template_kwargs: { enable_thinking: think } }),
      });
      if (r.status === 401) { live.notice = { msg: "The server wants its API key.", withKey: true }; throw new Error("the server wants its API key"); }
      if (!r.ok) {
        const e = await r.json().catch(() => ({}));
        throw new Error((e.error && e.error.message) || "HTTP " + r.status);
      }
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = "", doneEvent = false;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let end;
        while ((end = /\r?\n\r?\n/.exec(buf))) {
          const event = buf.slice(0, end.index); buf = buf.slice(end.index + end[0].length);
          for (const line of event.split(/\r?\n/)) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trimStart();
            if (data === "[DONE]") { doneEvent = true; continue; }
            let j;
            try { j = JSON.parse(data); } catch { throw new Error(t("Invalid reply stream.")); }
            if (j.error) throw new Error(j.error.message || t("Error"));
            if (j.id) id = j.id;
            if (j.usage) usage = j.usage;
            const delta = j.choices && j.choices[0] && j.choices[0].delta;
            if (!delta) continue;
            if (!first && (delta.content || delta.reasoning_content)) first = performance.now();
            if (delta.reasoning_content) thinking += delta.reasoning_content;
            if (delta.content) reply += delta.content;
          }
        }
        update({ content: reply, thinking, id });
      }
      if (!doneEvent) throw new Error(t("Reply stream ended early."));
      update({ content: reply, thinking, done: true, usage, id,
               clientTtft: first ? (first - t0) / 1000 : null, elapsed: (performance.now() - t0) / 1000 });
      // The server's record of the reply comes with the next telemetry: the figures fill in from it.
      setTimeout(poll, 300);
    } catch (e) {
      if (e.name === "AbortError") update({ content: reply, thinking, stopped: true });
      else {
        controller.current.abort();
        update({ content: reply, thinking, error: e.message });
      }
    } finally {
      controller.current = null;
      setMsgs((ms) => ms.slice());  // the buttons follow
      poll();
    }
  }
  const clear = () => {
    if (controller.current) controller.current.abort();
    setMsgs([]);
    setChatId("");
  };
  const openChat = (c) => {
    if (controller.current) controller.current.abort();
    setMsgs(c.msgs.map((m) => Object.assign({}, m)));
    setChatId(c.id);
    setHistOpen(false);
    if (input.current) input.current.focus();
  };
  const forget = (id) => {
    setChats((old) => storeChats(old.filter((c) => c.id !== id)));
    if (id === chatId) clear();
  };
  const forgetAll = async () => {
    setHistOpen(false);
    const n = chats.length;
    if (!await confirmBox("Delete the whole history?", [n + (n === 1 ? " conversation" : " conversations") + " kept in this browser will be deleted."], "Delete", true)) return;
    setChats(() => storeChats([]));
    clear();
  };
  const key = (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } };

  const tm = live.last || {}, model = tm.model || {}, sl = (tm.memory && tm.memory.sessions) || [];
  const lr = (tm.recent || []).find((r) => r.label === "chat");
  const stat = [];
  if (model.context) stat.push(["context", ktok(sl.length ? sl[0].tokens : 0) + " / " + ktok(model.context)]);
  if (lr) stat.push(["last reply", fmt(lr.decode_tok_s, 1) + " tok/s"]);
  if (model.drafter) stat.push(["drafter", model.drafter]);
  return html`
    <div class="chat">
      <div class="histwrap">
        <button class=${"iconbtn" + (histOpen ? " on" : "")} id="chat-history" title=${t("History")} aria-label=${t("History")} aria-haspopup="menu"
                aria-expanded=${String(histOpen)} onClick=${() => setHistOpen(!histOpen)}><${Icon} name="history"/></button>
        <div class="hist" id="chat-hist" role="menu" hidden=${!histOpen}>
          <div class="hist-h"><b>${t("History")}</b><span>${t("kept in this browser")}</span></div>
          <div class="hist-b">
            ${chats.length ? "" : html`<div class="empty pad">${t("No conversations yet. They are kept here as you chat.")}</div>`}
            ${chats.map((c) => html`
              <div class=${"hist-row" + (c.id === chatId ? " sel" : "")} role="menuitem" tabindex="0" data-chat=${c.id} onClick=${() => openChat(c)}
                   onKeyDown=${(e) => { if (e.key === "Enter" && e.target === e.currentTarget) openChat(c); }}>
                <span class="ht">${c.title}</span>
                <span class="hm">${t(ago(c.at / 1000))} · ${t(c.msgs.length + (c.msgs.length === 1 ? " message" : " messages"))}</span>
                <button class="iconbtn del" title=${t("Delete")} aria-label=${t("Delete")}
                        onClick=${(e) => { e.stopPropagation(); forget(c.id); }}><${Icon} name="trash"/></button>
              </div>`)}
          </div>
          ${chats.length ? html`<div class="hist-f"><button class="linkbtn" id="chat-hist-clear" onClick=${forgetAll}>${t("Delete all")}</button></div>` : ""}
        </div>
      </div>
      <div class="msgs" aria-live="polite" ref=${box}><div class="cv" id="msgs">
        ${!msgs.length ? html`<div class="empty" id="chat-empty" style="text-align: center; padding-top: 18vh">
          ${t("Talk to the model on this server. Replies stream from POST /v1/chat/completions.")}</div>` : ""}
        ${msgs.map((m, i) => m.role === "user" ? html`<div class="cv-user">${m.content}</div>` : html`
          <${AsstTurn} parts=${[]} name=${m.model} current>
            ${m.thinking ? html`<${Think} text=${m.thinking} label=${m.done || m.stopped || m.error ? t("Thought · " + fmt(m.thinking.length) + " characters") : t("Thinking")}/>` : ""}
            <div class="md" dangerouslySetInnerHTML=${{ __html: md(m.content || "") }}></div>
            ${streaming && i === msgs.length - 1 && !m.done ? html`<${LiveLine} m=${m}/>` : ""}
            <${ReplyMeta} m=${m}/>
          <//>`)}
      </div></div>
      <div class="composer">
        <div class="compose">
          <textarea id="prompt" rows="1" ref=${input} placeholder=${t("Message FLYX. Enter sends, Shift+Enter for a new line.")}
                    value=${prompt} onInput=${(e) => setPrompt(e.target.value)} onKeyDown=${key}></textarea>
          <div class="btns">
            <button class="sendbtn stop" id="stop" disabled=${!streaming} title=${t("Stop")} aria-label=${t("Stop the reply")}
                    onClick=${() => controller.current && controller.current.abort()}><${Icon} name="stopsq"/></button>
            <button class=${"sendbtn" + (prompt.trim() ? "" : " dim")} id="send" disabled=${streaming} title=${t("Send (Enter)")} aria-label=${t("Send")}
                    onClick=${send}><${Icon} name="arrowup"/></button>
          </div>
        </div>
        <div class="chatopts">
          <label><input type="checkbox" id="think" checked=${think} onChange=${(e) => setThink(e.target.checked)}/> ${t("Thinking")}</label>
          <button class="btn ghost sm" id="clear" onClick=${clear}>${t("New conversation")}</button>
          <div class="chatstat" id="chatstat">${stat.map(([k, v]) => html`<span>${t(k)} <b>${v}</b></span>`)}</div>
        </div>
      </div>
    </div>`;
}
