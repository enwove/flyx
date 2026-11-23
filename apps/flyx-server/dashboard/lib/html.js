// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Markup in components: htm reads a tagged template as Preact elements, so the page needs no compiler.
//   html`<p class="lead">${t("Waiting for a request")}</p>`
import { h, render, Fragment, Component } from "preact";
import htm from "htm";

export const html = htm.bind(h);
export { h, render, Fragment };

// A component drawn again only when one of its props changed (by identity): for long lists redrawn every second.
export function memo(draw) {
  class Memo extends Component {
    shouldComponentUpdate(next) {
      for (const k in next) if (next[k] !== this.props[k]) return true;
      for (const k in this.props) if (!(k in next)) return true;
      return false;
    }
    render(props) { return draw(props); }
  }
  return Memo;
}
export { useState, useEffect, useRef, useMemo, useCallback, useLayoutEffect } from "preact/hooks";
