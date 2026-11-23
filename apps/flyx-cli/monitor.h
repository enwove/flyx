// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The live panel of a server (monitor.cpp): `flyx monitor URL` for a server elsewhere, and the dashboard flyx serve
// draws on its own terminal while it runs.
#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace flyx::monitor {

// What flyx serve shares with the dashboard it draws on its terminal.
struct ServeView {
  std::string model;                                        // the file being served
  std::string url;                                          // server address for GET /v1/telemetry
  std::string key;                                          // server API key, if configured
  std::string page;                                         // web page address, shown and opened with `o`
  std::function<std::vector<std::string>(size_t)> recent_logs;  // newest N stderr lines, oldest first
  std::function<bool()> open_page;                          // false: there was no browser to open it in
  std::atomic<bool> listening{false};                       // the model is loaded and the server answers
  std::atomic<bool> quit{false};                            // a signal or server exit; also set when user quits
};

// flyx serve's dashboard: the model loading (the log as it comes), then flyx monitor's panel with the newest log lines
// under it.  Returns once the user quits (q twice: that stops the server) or v->quit is set.
void serve_dashboard(ServeView* v);

// flyx monitor [URL] [options]: argv[1] is "monitor".
int run(int argc, char** argv);

}  // namespace flyx::monitor
