// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "core/exec/engine.h"
#include "core/os/path.h"

#ifdef _WIN32
#include <windows.h>
#endif

extern "C" void flyx_test_delegate(const flyx_backend_api*);

int main() {
  using namespace flyx;
  namespace fs = std::filesystem;
  std::unique_ptr<os::DynLib> lib;
  const flyx_backend_api* api = nullptr;
  flyx_device_info info{};
  EngineOptions opt;
  int failures = 0;
  auto expect_error = [&](const std::string& expected) {
    std::string err;
    if (open_backend(opt, &lib, &api, &info, &err) || err != expected) {
      std::fprintf(stderr, "expected %s, got %s\n", expected.c_str(), err.c_str());
      ++failures;
    }
  };

  // An installed backend's probe supplies the reason, even with the other three candidates absent.
  flyx_backend_api mock{};
  mock.abi_version = FLYX_BACKEND_ABI_VERSION;
  mock.struct_size = sizeof(mock);
  mock.probe = [](flyx_device_info*) { return -1; };
  mock.last_error = [] { return "no NVIDIA driver found"; };
  flyx_test_delegate(&mock);
  opt.backend = "test-host-kv";
  expect_error("no usable backend\n  " + os::executable_dir() + "/libflyx-test-host-kv" + os::kDynLibSuffix +
               ": no NVIDIA driver found");

  const auto tmp = fs::temp_directory_path() / ("flyx-backend-errors-" + std::to_string(os::process_id()));
  fs::create_directories(tmp / "bin");
  fs::create_directories(tmp / "lib/flyx");
  opt.backend_dir = os::from_path(tmp / "bin");
  opt.backend = "fixture";
  const std::string file = "libflyx-fixture" + std::string(os::kDynLibSuffix);
  expect_error("no usable backend\n  " + file + " not found in backend paths relative to " + opt.backend_dir);
  // The package layout puts the broken library between absent candidates. Keep its platform-specific loader error.
  std::ofstream(tmp / "lib/flyx" / file) << "invalid library";
  const std::string path = opt.backend_dir + "/../lib/flyx/" + file;
  os::DynLib broken;
  std::string loader_error;
#ifdef _WIN32
  // Exercise the load even when the parent process has suppressed Windows error dialogs.
  DWORD previous;
  if (!SetThreadErrorMode(0, &previous)) return 1;
  struct RestoreErrorMode {
    DWORD mode;
    ~RestoreErrorMode() { SetThreadErrorMode(mode, nullptr); }
  } restore{previous};
#endif
  if (broken.open(path, &loader_error) || loader_error.empty()) return 1;
#ifdef _WIN32
  if (GetThreadErrorMode() != 0) return 1;
#endif
  expect_error("no usable backend\n  " + path + ": " + loader_error);
  fs::remove_all(tmp);
  return failures ? 1 : 0;
}
