// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The page's languages.  The page is written in English; every string it shows goes through t(), which gives the
// chosen language's words for it: the whole string from the language's dict, else the first of its rules (patterns
// for strings with figures in them) that matches.  A string neither knows stays English.  What the model or a user
// wrote (replies, prompts, code, names, log lines) is never passed to t().
//
// To add a language: a file in i18n/ with the same keys as zh.js, imported below; tools/dashboard_sweep.py lists
// what it leaves in English.
import { saved } from "./api.js";
import zh from "../i18n/zh.js";
import es from "../i18n/es.js";
import fr from "../i18n/fr.js";
import de from "../i18n/de.js";
import ja from "../i18n/ja.js";
import ko from "../i18n/ko.js";

export const LANGS = { zh, es, fr, de, ja, ko };

const browser = (navigator.language || "en").toLowerCase().split("-")[0];
let lang = saved.get("lang") || (LANGS[browser] ? browser : "en");
if (!LANGS[lang]) lang = "en";
const listeners = new Set();

export const currentLang = () => lang;
export function setLang(l) {
  lang = LANGS[l] ? l : "en";
  saved.set("lang", lang);
  document.documentElement.lang = LANGS[lang] ? LANGS[lang].tag : "en";
  for (const f of listeners) f();
}
export function onLang(f) {
  listeners.add(f);
  return () => listeners.delete(f);
}
document.documentElement.lang = LANGS[lang] ? LANGS[lang].tag : "en";

const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

// The shown language's words for an English string; the spaces around it are kept, so pieces join as they did.
export function t(str) {
  const L = LANGS[lang];
  if (!L || typeof str !== "string") return str;
  const k = str.trim().replace(/\s+/g, " ");  // as the eye reads it: a line break or an indent is one space
  if (!k || !/[A-Za-z]/.test(k)) return str;
  let v = own(L.dict, k) ? L.dict[k] : undefined;
  if (v === undefined)
    for (const [re, to] of L.rules) {
      const m = re.exec(k);
      if (m) { v = typeof to === "function" ? to(m) : k.replace(re, to); break; }
    }
  return v === undefined ? str : str.match(/^\s*/)[0] + v + str.match(/\s*$/)[0];
}

// A backend error string (possibly several sentences): translate whole first, else sentence by sentence, collapsing
// spaces after CJK punctuation.
export function tErr(str) {
  if (typeof str !== "string" || !str) return "";
  const whole = t(str);
  if (whole !== str) return whole;
  const parts = str.split(/(?<=\.)\s+/);
  return parts.length > 1 ? parts.map((s) => t(s)).join(" ").replace(/([。！？]) /g, "$1") : whole;
}

// A paragraph translated whole (its code kept in place), as HTML, or null for English.
export const para = (key) => (LANGS[lang] && LANGS[lang].paras[key]) || null;

// A sentence of the API reference with its code in place, as HTML, in the shown language: the language's html map
// has it by its English (spaces as the eye reads them), else it stays English.  Only the page's own text, never
// anything a user or the model wrote: it is set as HTML.
export function tx(en) {
  const L = LANGS[lang];
  const v = L && L.html && L.html[en.trim().replace(/\s+/g, " ")];
  return v || en;
}
