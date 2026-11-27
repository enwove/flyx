// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
import { html, useEffect, useState } from "../lib/html.js";
import { call } from "../lib/api.js";
import { t, tErr } from "../lib/i18n.js";
import { CodeSnippet } from "../lib/ui.js";

const path = "/v1/settings/network";

export function NetworkSettings() {
  const [network, setNetwork] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let current = true;
    call("GET", path).then(r => {
      if (current) { setNetwork(r.ok ? r.json : null); setError(r.error); }
    });
    return () => { current = false; };
  }, []);

  const saveSwitch = (field) => async (e) => {
    const enabled = e.currentTarget.checked;
    e.currentTarget.checked = !!network[field];  // reflect the saved value once the write succeeds
    setBusy(true);
    const r = await call("POST", path, { [field]: enabled });
    if (r.ok) setNetwork(r.json);
    setError(r.error);
    setBusy(false);
  };
  const locked = network?.host_from_cli;
  const authLocked = !!(network?.no_auth_from_cli || network?.api_key_set);
  const remoteEffective = network
    ? ("remote_manage_effective" in network ? !!network.remote_manage_effective
       : !!(network.serving_lan && !network.api_key_set && network.remote_manage))
    : false;
  const urls = network?.api_urls.length ? network.api_urls : [`http://<this-PC-IP>:${network?.port}/v1`];
  return html`
    <section class="pane network-settings">
      <div class="pane-h"><h2>${t("LAN access")}</h2></div>
      ${error && html`<p role="alert">${tErr(error)}</p>`}
      ${network && html`
        <label class="switch">
          <input type="checkbox" role="switch" checked=${locked ? network.serving_lan : network.lan_access}
                 disabled=${busy || locked} onChange=${saveSwitch("lan_access")}/>
          <i aria-hidden="true"></i>${t("Serve on local network")}${locked ? "" : " (0.0.0.0)"}
        </label>
        <p class="lead" role="status">${t(locked ? "The --host flag controls this launch."
          : network.restart_required ? "Saved. Close and reopen flyx to apply this change." : "Restart flyx to apply changes.")}</p>
        <p class="lead">${t("Devices that can reach this PC can use the API. Set FLYX_API_KEY or --api-key to require a key.")}</p>
        <div class="pane-h next"><h2>${t("LAN dashboard & management")}</h2></div>
        <label class="switch">
          <input type="checkbox" id="net-remote-manage" role="switch" checked=${!!network.remote_manage}
                 disabled=${busy || authLocked} onChange=${saveSwitch("remote_manage")}/>
          <i aria-hidden="true"></i>${t("Allow LAN management without an API key")}
        </label>
        <p class="lead" role="status">${t(network.no_auth_from_cli ? "The --no-auth flag controls this launch."
          : network.api_key_set ? "An API key is configured (--api-key or FLYX_API_KEY)."
          : remoteEffective ? "Enabled immediately. Anyone on this network can load models, browse folders, and view logs or requests."
          : network.remote_manage ? "Saved. Takes effect once LAN access is enabled and flyx is restarted."
          : "Only this machine can manage models, logs, and settings unless --api-key is set or this option is enabled on a trusted LAN.")}</p>
        <div class="pane-h next"><h2>${t("Active API address")}</h2></div>
        <p>${t("Listening on")} <code>${network.listen_host.includes(":") ? "[" + network.listen_host + "]" : network.listen_host}:${network.port}</code></p>
        ${urls.map(url => html`<${CodeSnippet} key=${url} text=${url}/>`)}
      `}
    </section>`;
}
