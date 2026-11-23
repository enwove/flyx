// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What the page asks of the model being served, from the top bar's menu or the Model page: unload it, load a model
// with given settings (POST /v1/load), stop a fetch. Each asks first where it stops requests.
import { call } from "./api.js";
import { confirmBox } from "./ui.js";
import { live, poll } from "./telemetry.js";

const running = () => { const st = (live.last && live.last.state) || {}; return (st.running || 0) + (st.queued || 0); };
const served = () => live.last && live.last.model && live.last.model.id;

// Unload the model; an error comes back as a message for the page to show, else null.
export async function unloadServed() {
  const name = served(), n = running();
  const lines = [name + " leaves the GPU; requests get 503 until a model is loaded again."];
  if (n) lines.unshift(n + " running or waiting request(s) are stopped.");
  lines.push("flyx serve still remembers it: started again without a model, it loads it.");
  if (!await confirmBox("Unload " + name + "?", lines, "Unload", true)) return null;
  const r = await call("POST", "/v1/unload");
  poll();
  return r.ok ? null : r.status === 404 ? "This server cannot unload yet." : r.status ? "Unloading failed: HTTP " + r.status : "Unloading failed.";
}

// Load `want` ({model, ctx, drafters, ...}), after a confirmation when `title` is given.
export async function loadWant(want, title, lines, ok) {
  if (title && !await confirmBox(title, lines, ok)) return;
  const r = await call("POST", "/v1/load", want);
  if (!r.ok) await confirmBox("The model did not start loading", [r.error], "OK");
  poll();
}

// The model served, unloaded and loaded again with the settings it has: its memory back to a clean start.
export function reloadServed() {
  const L = (live.last && live.last.load) || {}, name = served(), n = running();
  if (!L.loaded) return;
  const lines = [name + " is unloaded and loaded again with the same settings."];
  if (n) lines.unshift(n + " running or waiting request(s) are stopped.");
  loadWant(L.loaded, "Reload " + name + " with these settings?", lines, "Reload");
}

export async function stopFetch() {
  await call("DELETE", "/v1/load");
  poll();
}

// The Requests page, on request `id` when there is one.
export function openRequest(go, id) {
  if (id) location.hash = "#requests:" + encodeURIComponent(id); else go("requests");
}
