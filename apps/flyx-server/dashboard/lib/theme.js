// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The page's theme, light or dark: as chosen in this browser, else as the system's setting says, following it when it
// changes.  index.html sets <html data-theme> the same way before the first paint, so a dark page never shows white
// first; style.css's tokens follow the attribute, and the charts read them as they draw.
import { saved } from "./api.js";

export const THEMES = ["system", "light", "dark"];

const systemDark = window.matchMedia("(prefers-color-scheme: dark)");
const listeners = new Set();
let choice = saved.get("theme");
if (!THEMES.includes(choice)) choice = "system";

export const themeChoice = () => choice;
export const shownTheme = () => document.documentElement.dataset.theme === "dark" ? "dark" : "light";

function apply() {
  document.documentElement.dataset.theme = choice === "system" ? (systemDark.matches ? "dark" : "light") : choice;
  for (const f of listeners) f();
}
export function setTheme(c) {
  choice = THEMES.includes(c) ? c : "system";
  saved.set("theme", choice === "system" ? null : choice);
  apply();
}
export function onTheme(f) {
  listeners.add(f);
  return () => listeners.delete(f);
}
systemDark.addEventListener("change", () => { if (choice === "system") apply(); });
