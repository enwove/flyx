// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Unit tests for GPU process classification, display naming, group IDs, and live listing safety invariants.
#include "apps/flyx-server/gpu_apps.h"
#include "tests/common/check.h"

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace {

#define CHECK(x) check(static_cast<bool>(x), #x)

void test_system_classification() {
  using flyx::server::is_system_gpu_process;
  // Windows system & desktop compositor processes must never be closable.
  CHECK(is_system_gpu_process("dwm.exe", "C:\\Windows\\System32\\dwm.exe"));
  CHECK(is_system_gpu_process("DWM", ""));
  CHECK(is_system_gpu_process("explorer.exe", "C:\\Windows\\explorer.exe"));
  CHECK(is_system_gpu_process("csrss.exe", "C:\\Windows\\System32\\csrss.exe"));
  CHECK(is_system_gpu_process("StartMenuExperienceHost.exe",
                              "C:\\Windows\\SystemApps\\Microsoft.Windows.StartMenuExperienceHost\\StartMenuExperienceHost.exe"));
  CHECK(is_system_gpu_process("customhost.exe", "C:\\Windows\\System32\\customhost.exe"));
  CHECK(is_system_gpu_process("nvcontainer.exe", "C:\\Program Files\\NVIDIA Corporation\\NvContainer\\nvcontainer.exe"));
  // Linux compositors must never be closable.
  CHECK(is_system_gpu_process("Xorg", "/usr/lib/xorg/Xorg"));
  CHECK(is_system_gpu_process("gnome-shell", "/usr/bin/gnome-shell"));
  CHECK(is_system_gpu_process("kwin_wayland", "/usr/bin/kwin_wayland"));
  // Regular desktop user apps must not be flagged as system.
  CHECK(!is_system_gpu_process("Discord.exe", "C:\\Users\\me\\AppData\\Local\\Discord\\app-1.0\\Discord.exe"));
  CHECK(!is_system_gpu_process("chrome.exe", "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"));
  CHECK(!is_system_gpu_process("Code.exe", "C:\\Users\\me\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"));
  CHECK(!is_system_gpu_process("Slack.exe", "C:\\Program Files\\WindowsApps\\Slack\\Slack.exe"));
  CHECK(!is_system_gpu_process("RadeonSoftware.exe", "C:\\Program Files\\WindowsApps\\AMD\\RadeonSoftware.exe"));
  CHECK(!is_system_gpu_process("python3", "/usr/bin/python3"));
}

void test_clean_app_names() {
  using flyx::server::clean_gpu_app_name;
  // Specific ProductName wins.
  CHECK(clean_gpu_app_name("chrome", "Google Chrome", "Google Chrome") == "Google Chrome");
  CHECK(clean_gpu_app_name("ChatGPT", "Codex", "Codex") == "Codex");
  CHECK(clean_gpu_app_name("wetype_update", "WeType IME", "WeType IME") == "WeType IME");
  // Generic OS/Electron ProductName falls back cleanly.
  CHECK(clean_gpu_app_name("WidgetBoard.exe", "Microsoft® Windows® Operating System", "") == "Windows Widgets");
  CHECK(clean_gpu_app_name("Code.exe", "Electron", "Electron") == "Visual Studio Code");
  CHECK(clean_gpu_app_name("WhatsApp.Root.exe", "", "") == "WhatsApp");
  CHECK(clean_gpu_app_name("RadeonSoftware.exe", "", "") == "AMD Software");
  CHECK(clean_gpu_app_name("MyApp.Root", "", "") == "MyApp");
}

void test_group_ids() {
  using flyx::server::gpu_app_group_id;
  const std::string a = gpu_app_group_id("c:\\program files\\google\\chrome\\application\\chrome.exe");
  const std::string b = gpu_app_group_id("c:\\users\\me\\appdata\\local\\discord\\discord.exe");
  CHECK(a.size() == 16);
  CHECK(b.size() == 16);
  CHECK(a != b);
  CHECK(a == gpu_app_group_id("c:\\program files\\google\\chrome\\application\\chrome.exe"));
}

void test_live_listing_safety() {
  using namespace flyx::server;
  const GpuAppSnapshot snap = list_gpu_apps("");
  for (const GpuApp& app : snap.apps) {
    CHECK(!app.id.empty());
    CHECK(!app.name.empty());
    CHECK(!is_system_gpu_process(app.process_name, ""));
    if (!app.closable) CHECK(!app.reason.empty());
  }
  // Closing a bogus id must never succeed.
  const GpuCloseResult r = close_gpu_app("", "0000000000000000", false);
  CHECK(r.status == CloseStatus::kNotFound || r.status == CloseStatus::kUnsupported);

  // Live RAM listing must populate total memory and enforce the same safety invariants.
  const GpuAppSnapshot ram = list_ram_apps();
  CHECK(ram.supported);
  CHECK(ram.mem_total > 0);
  for (const GpuApp& app : ram.apps) {
    CHECK(!app.id.empty());
    CHECK(!app.name.empty());
    CHECK(!is_system_gpu_process(app.process_name, ""));
    if (!app.closable) CHECK(!app.reason.empty());
  }
  const GpuCloseResult rr = close_ram_app("0000000000000000", false);
  CHECK(rr.status == CloseStatus::kNotFound);
}

}  // namespace

// Which NVML device a hint names, and whether its readings are surely that device's: never where a GPU's handle or
// name was not read (it could be the hinted one's twin), nor where two of that name are listed.
void test_device_match() {
  using flyx::server::match_device;
  using Names = std::vector<std::optional<std::string>>;
  const auto one = match_device(Names{"NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090"}, "RTX 4090");
  CHECK(one.index == 0 && one.unique);
  const auto twins = match_device(Names{"NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"}, "RTX 4090");
  CHECK(twins.index == 0 && !twins.unique);
  const auto unread = match_device(Names{std::nullopt, "NVIDIA GeForce RTX 4090"}, "RTX 4090");
  CHECK(unread.index == 1 && !unread.unique);  // the device not read could be the other 4090
  const auto unread_after = match_device(Names{"NVIDIA GeForce RTX 4090", std::nullopt}, "RTX 4090");
  CHECK(unread_after.index == 0 && !unread_after.unique);
  const auto alone = match_device(Names{"NVIDIA GeForce RTX 5070 Ti"}, "");
  CHECK(alone.index == 0 && alone.unique);  // the one GPU there is
  const auto none_named = match_device(Names{"NVIDIA A100", "NVIDIA H100"}, "RTX 4090");
  CHECK(none_named.index == 0 && !none_named.unique);  // the first read, not known to be the hinted device
  const auto nothing = match_device(Names{std::nullopt}, "RTX 4090");
  CHECK(nothing.index == -1 && !nothing.unique);
}

int main() {
  test_device_match();
  test_system_classification();
  test_clean_app_names();
  test_group_ids();
  test_live_listing_safety();
  return flyx::test::finish("gpu_apps_test");
}
