// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// agent_powershell: the commands POST /v1/agents/launch runs.  On Windows each runs in Windows PowerShell with the
// agent replaced by a function that prints what it was given: Claude Code's settings file, Codex's -c overrides, for a
// model id with quotes of every kind in it, with and without an API key.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "apps/flyx-server/agent_launch.h"
#include "nlohmann/json.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

using flyx::server::AgentTarget;
using flyx::server::agent_powershell;
using flyx::server::ps_quote;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

#ifdef _WIN32
// What `script` prints between the markers, run by Windows PowerShell from a file (UTF-8 with a BOM, as 5.1 reads it),
// with FLYX_KEY set to `key` when there is one.
std::string run(const std::string& script, const std::string& key) {
  char dir[512];
  const size_t n = GetEnvironmentVariableA("TEMP", dir, sizeof dir);
  const std::string file = std::string(dir, n) + "\\flyx_agent_launch_test.ps1";
  std::ofstream(file, std::ios::binary) << "\xEF\xBB\xBF[Console]::OutputEncoding = [Text.Encoding]::UTF8\n"
                                        << "function claude { [Console]::Out.Write('<<<' + [IO.File]::ReadAllText($args[1]) + '>>>') }\n"
                                        << "function codex { [Console]::Out.Write('<<<' + ($args -join \"`n\") + '>>>') }\n"
                                        << (key.empty() ? "Remove-Item Env:FLYX_KEY -ErrorAction SilentlyContinue\n"
                                                        : "$env:FLYX_KEY = " + ps_quote(key) + "\n")
                                        << script
                                        << "[Console]::Out.Write('<<<' + [string]$env:FLYX_KEY + '>>>')\n";
  FILE* p = _popen(("powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File \"" + file + "\"").c_str(), "rb");
  std::string out;
  char buf[4096];
  for (size_t got; p && (got = std::fread(buf, 1, sizeof buf, p)) > 0;) out.append(buf, got);
  if (p) _pclose(p);
  std::remove(file.c_str());
  return out;
}

// The text between the i-th pair of markers.
std::string part(const std::string& out, int i) {
  size_t at = 0;
  for (int k = 0; k <= i; ++k) {
    const size_t open = out.find("<<<", at);
    if (open == std::string::npos) return "<missing>";
    const size_t close = out.find(">>>", open);
    if (close == std::string::npos) return "<missing>";
    if (k == i) return out.substr(open + 3, close - open - 3);
    at = close + 3;
  }
  return "<missing>";
}
#endif

}  // namespace

int main() {
  check(ps_quote("it's") == "'it''s'", "a quote is doubled");
  check(ps_quote("\xE2\x80\x98x\xE2\x80\x99") == "'\xE2\x80\x98\xE2\x80\x98x\xE2\x80\x99\xE2\x80\x99'",
        "typographic quotes are doubled");
  check(agent_powershell("pi", {}).empty(), "no command for an agent not listed");

  const std::string model = "lab/\xCE\xB2 \"Local\" & 'One' \xE2\x80\x98" "demo\xE2\x80\x99 $HOME";
  const std::string key = "sample 'key' \"with\" spaces & $symbols";
  for (bool keyed : {false, true}) {
    AgentTarget t{"http://127.0.0.1:18473", model, 18000, keyed};
    // Codex: the same id without its double quotes and &, which it cannot be given (agent_launch.h)
    AgentTarget tc = t;
    tc.model = "lab/\xCE\xB2 Local 'One' \xE2\x80\x98" "demo\xE2\x80\x99 $HOME";
    const std::string tag = keyed ? " (keyed)" : "";
    check(agent_powershell("codex", t).empty(), "no codex command for an id with a double quote" + tag);
    for (const char* bad : {"a&b", "a|b", "a<b", "a>b", "a^b", "50%"}) {
      AgentTarget tb = t;
      tb.model = bad;
      check(agent_powershell("codex", tb).empty(), std::string("no codex command for the id ") + bad + tag);
    }
    const std::string claude = agent_powershell("claude", t), codex = agent_powershell("codex", tc);
    check(claude.find("claude --settings $file") != std::string::npos, "claude runs with a settings file" + tag);
    check(claude.find(key) == std::string::npos && codex.find(key) == std::string::npos, "no key in a command" + tag);
    check(codex.find('"') == std::string::npos, "no double quote reaches codex's arguments" + tag);
#ifdef _WIN32
    const std::string k = keyed ? key : "";
    {
      const std::string out = run(claude, k);
      const auto settings = nlohmann::json::parse(part(out, 0), nullptr, false);
      check(!settings.is_discarded(), "claude settings are JSON" + tag + ": " + out);
      if (!settings.is_discarded()) {
        const auto& env = settings["env"];
        check(settings["model"] == model && env["ANTHROPIC_MODEL"] == model, "claude model" + tag);
        check(env["ANTHROPIC_BASE_URL"] == t.origin, "claude base URL" + tag);
        check(env["ANTHROPIC_AUTH_TOKEN"] == (keyed ? key : "flyx"), "claude token" + tag);
        check(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "18000", "claude context" + tag);
        check(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "9000", "claude output limit: half the context" + tag);
      }
      check(part(out, 1).empty(), "FLYX_KEY is gone after claude" + tag);
    }
    {
      const std::string out = run(codex, k);
      std::vector<std::string> want{"-c", "model_provider=flyx", "-c", "model=" + tc.model, "-c",
                                    "model_context_window=18000", "-c", "web_search=disabled", "-c",
                                    "model_providers.flyx.name=flyx", "-c",
                                    "model_providers.flyx.base_url='http://127.0.0.1:18473/v1'", "-c",
                                    "model_providers.flyx.wire_api=responses"};
      if (keyed) want.insert(want.end(), {"-c", "model_providers.flyx.env_key=FLYX_KEY"});
      std::string joined;
      for (const auto& w : want) joined += (joined.empty() ? "" : "\n") + w;
      check(part(out, 0) == joined, "codex arguments" + tag + ": " + part(out, 0));
      check(part(out, 1).empty(), "FLYX_KEY is gone after codex" + tag);
    }
#endif
  }
  {
    AgentTarget t{"http://127.0.0.1:8080", "Qwen3.8-27B-UD-IQ2_S", 262144, false};
    check(agent_powershell("codex", t).find("'model=''Qwen3.8-27B-UD-IQ2_S'''") != std::string::npos,
          "a plain model id is a TOML literal string");
    check(agent_powershell("claude", t).find("CLAUDE_CODE_MAX_OUTPUT_TOKENS = '128000'") != std::string::npos,
          "claude output limit: at most 128,000");
  }
  if (failures) return 1;
  std::printf("PASS: agent launch commands\n");
  return 0;
}
