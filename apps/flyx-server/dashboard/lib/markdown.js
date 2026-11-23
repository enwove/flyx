// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Markdown, as replies write it: escaped first, then a few block and inline forms.  md() returns HTML safe to insert:
// every character of the source is escaped before any tag is added.
export const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
function inline(t) {  // t is escaped already
  const codes = [];
  t = t.replace(/`([^`\n]+)`/g, (m, c) => { codes.push(c); return "\u0000" + (codes.length - 1) + "\u0000"; });
  t = t.replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>").replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<i>$2</i>")
       .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, (m, a, u) => "<a href=\"" + u + "\" target=\"_blank\" rel=\"noopener noreferrer\">" + a + "</a>");
  return t.replace(/\u0000(\d+)\u0000/g, (m, i) => "<code>" + codes[+i] + "</code>");
}
export function md(src) {
  const lines = esc(src || "").split("\n"), out = [];
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    const fence = /^\s*```(\S*)/.exec(l);
    if (fence) {
      const code = [];
      for (++i; i < lines.length && !/^\s*```/.test(lines[i]); ++i) code.push(lines[i]);
      ++i;
      out.push("<pre><code>" + code.join("\n") + "</code></pre>");
      continue;
    }
    if (/^\s*$/.test(l)) { ++i; continue; }
    const h = /^(#{1,4})\s+(.*)$/.exec(l);
    if (h) { out.push("<h" + h[1].length + ">" + inline(h[2]) + "</h" + h[1].length + ">"); ++i; continue; }
    if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(l)) { out.push("<hr>"); ++i; continue; }
    if (/^\s*&gt;\s?/.test(l)) {
      const q = [];
      for (; i < lines.length && /^\s*&gt;\s?/.test(lines[i]); ++i) q.push(lines[i].replace(/^\s*&gt;\s?/, ""));
      out.push("<blockquote>" + inline(q.join("<br>")) + "</blockquote>");
      continue;
    }
    const ul = /^\s*[-*+]\s+/, ol = /^\s*\d+[.)]\s+/;
    if (ul.test(l) || ol.test(l)) {
      const tag = ul.test(l) ? "ul" : "ol", re = tag === "ul" ? ul : ol, items = [];
      for (; i < lines.length && (re.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length)); ++i) {
        if (re.test(lines[i])) items.push(lines[i].replace(re, "")); else items[items.length - 1] += " " + lines[i].trim();
      }
      out.push("<" + tag + ">" + items.map((x) => "<li>" + inline(x) + "</li>").join("") + "</" + tag + ">");
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      const row = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const head = row(l), body = [];
      for (i += 2; i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i]); ++i) body.push(row(lines[i]));
      out.push("<table><thead><tr>" + head.map((c) => "<th>" + c + "</th>").join("") + "</tr></thead><tbody>" +
               body.map((r) => "<tr>" + r.map((c) => "<td>" + c + "</td>").join("") + "</tr>").join("") + "</tbody></table>");
      continue;
    }
    const para = [];
    for (; i < lines.length && !/^\s*$/.test(lines[i]) && !/^\s*```/.test(lines[i]) && !/^#{1,4}\s/.test(lines[i]) &&
           !ul.test(lines[i]) && !ol.test(lines[i]) && !/^\s*&gt;/.test(lines[i]); ++i) para.push(lines[i]);
    if (!para.length) { para.push(lines[i]); ++i; }
    out.push("<p>" + inline(para.join("<br>")) + "</p>");
  }
  return out.join("");
}
