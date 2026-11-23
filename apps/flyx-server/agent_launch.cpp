// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/flyx-server/agent_launch.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace flyx::server {

const std::vector<std::string>& launch_agents() {
  static const std::vector<std::string> agents{"claude", "codex"};
  return agents;
}

std::string ps_quote(const std::string& s) {
  std::string out = "'";
  for (size_t i = 0; i < s.size(); ++i) {
    out += s[i];
    if (s[i] == '\'') out += '\'';
    // U+2018..U+201B in UTF-8: E2 80 98..9B
    if (i + 2 < s.size() && uint8_t(s[i]) == 0xE2 && uint8_t(s[i + 1]) == 0x80 && uint8_t(s[i + 2]) >= 0x98 &&
        uint8_t(s[i + 2]) <= 0x9B) {
      out += s.substr(i + 1, 2);
      out += s.substr(i, 3);
      i += 2;
    }
  }
  return out + "'";
}

namespace {

std::string lines(const std::vector<std::string>& v) {
  std::string out;
  for (const auto& l : v) out += l + "\n";
  return out;
}

// A gray line first: what the window runs, before the agent takes it over.
std::string banner(const char* agent, const AgentTarget& t) {
  return "  Write-Host " + ps_quote(std::string("flyx: ") + agent + " with " + t.model + ", context " +
                                    std::to_string(t.context) + " tokens, at " + t.origin) +
         " -ForegroundColor DarkGray";
}

// The most Claude Code asks for in a reply: half the context, at most 128,000, the cap Claude Code keeps for a model it
// does not know (32,000 when nothing is set).  As the Integration page's claudeOutput.
int64_t claude_output(int64_t context) { return std::max<int64_t>(1, std::min<int64_t>(128000, context / 2)); }

// Claude Code, as the Integration page's PowerShell command: its settings in a file for this launch (--settings
// overrides an Anthropic URL and token saved in Claude Code), the key read from FLYX_KEY into it.
std::string claude(const AgentTarget& t) {
  return lines({
      "& {",
      "  $ErrorActionPreference = 'Stop'",
      banner("Claude Code", t),
      "  $modelName = " + ps_quote(t.model),
      "  $settings = @{ model = $modelName; env = @{",
      "    ANTHROPIC_MODEL = $modelName",
      "    ANTHROPIC_BASE_URL = " + ps_quote(t.origin),
      std::string("    ANTHROPIC_AUTH_TOKEN = ") + (t.keyed ? "$env:FLYX_KEY" : "'flyx'"),
      "    CLAUDE_CODE_MAX_CONTEXT_TOKENS = '" + std::to_string(t.context) + "'",
      "    CLAUDE_CODE_MAX_OUTPUT_TOKENS = '" + std::to_string(claude_output(t.context)) + "'",
      "    CLAUDE_CODE_MODEL_CAPABILITIES = '-mid_conv_system'",
      "    CLAUDE_CODE_GATEWAY_HINT_HEADERS = '1'",
      "  } }",
      "  $file = [IO.Path]::GetTempFileName()",
      "  try {",
      "    [IO.File]::WriteAllText($file, ($settings | ConvertTo-Json))",
      "    claude --settings $file",
      std::string("  } finally { [IO.File]::Delete($file)") + (t.keyed ? "; Remove-Item Env:FLYX_KEY -ErrorAction SilentlyContinue" : "") + " }",
      "}",
  });
}

// A TOML value for codex -c: a literal string where one can hold it.  Codex takes a value that is not TOML as the
// text itself, which is the way left for the rest (an id with a quote in it).
std::string toml_text(const std::string& s) {
  for (char c : s)
    if (c == '\'' || (uint8_t(c) < 0x20 && c != '\t') || c == 0x7f) return s;
  return "'" + s + "'";
}

// Codex CLI with the Integration page's profile given as -c overrides: no profile file needed.  The program, not
// npm's codex.ps1, which a system that runs no scripts (PowerShell's default execution policy) refuses: codex.cmd
// (npm) or codex.exe.  Each argument is a PowerShell literal with no double quote in it, which Windows PowerShell 5.1
// would drop on its way to the program, nor a character cmd.exe takes for its own (codex.cmd is run by it).
std::string codex(const AgentTarget& t) {
  if (t.model.find_first_of("\"&|<>^%") != std::string::npos) return "";  // no way through: see agent_launch.h
  std::vector<std::string> sets{
      "model_provider=flyx",
      "model=" + toml_text(t.model),
      "model_context_window=" + std::to_string(t.context),
      "web_search=disabled",
      "model_providers.flyx.name=flyx",
      "model_providers.flyx.base_url=" + toml_text(t.origin + "/v1"),
      "model_providers.flyx.wire_api=responses",
  };
  if (t.keyed) sets.push_back("model_providers.flyx.env_key=FLYX_KEY");
  std::string call = "    & $codex";
  for (const auto& s : sets) call += " -c " + ps_quote(s);
  return lines({
      "& {",
      banner("Codex CLI", t),
      "  try {",
      "    $codex = Get-Command codex -CommandType Function, Application -ErrorAction SilentlyContinue | Select-Object -First 1",
      "    if (!$codex) { Write-Host 'Codex CLI was not found. Install it: npm.cmd install -g @openai/codex' -ForegroundColor Red; return }",
      call,
      std::string("  } finally {") + (t.keyed ? " Remove-Item Env:FLYX_KEY -ErrorAction SilentlyContinue" : "") + " }",
      "}",
  });
}

}  // namespace

std::string agent_powershell(const std::string& agent, const AgentTarget& t) {
  if (agent == "claude") return claude(t);
  if (agent == "codex") return codex(t);
  return "";
}

}  // namespace flyx::server
