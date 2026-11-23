// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Coding agents the dashboard starts in a window of their own on the server's machine (POST /v1/agents/launch).  The
// server writes their command from what it serves, never from the request: the page names an agent and a folder.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flyx::server {

// The agents there is a command for, by the ids the dashboard's Integration page uses.
const std::vector<std::string>& launch_agents();

struct AgentTarget {
  std::string origin;  // the server as the agent reaches it: "http://127.0.0.1:8080"
  std::string model;   // the API's model id
  int64_t context = 0;
  bool keyed = false;  // the server takes an API key: the window has it in FLYX_KEY, the command refers to that
};

// The Windows PowerShell command that starts `agent` against the server: the launch commands the Integration page shows
// (components/agent_setup.js), with no file of the user's written.  "" for an agent there is none for, and for Codex
// with one of " & | < > ^ % in the model id: Windows PowerShell 5.1 drops a double quote from a program's arguments,
// and cmd.exe, which runs npm's codex.cmd, takes the others for its own.
std::string agent_powershell(const std::string& agent, const AgentTarget& t);

// `s` as a PowerShell literal: in single quotes, each quote doubled (PowerShell takes the typographic ones, U+2018 to
// U+201B, as quotes too).
std::string ps_quote(const std::string& s);

}  // namespace flyx::server
