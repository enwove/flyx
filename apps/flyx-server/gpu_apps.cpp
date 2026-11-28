// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/flyx-server/gpu_apps.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "core/os/os.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <winver.h>
#else
#include <csignal>
#include <dlfcn.h>
#include <errno.h>
#include <filesystem>
#include <fstream>
#include <limits.h>
#include <sstream>
#include <unistd.h>
#endif

namespace flyx::server {

namespace {

std::string to_lower_ascii(std::string s) {
  for (char& c : s)
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
  return s;
}

std::string trim_copy(const std::string& s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

std::string strip_exe_suffix(const std::string& name) {
  if (name.size() > 4 && to_lower_ascii(name.substr(name.size() - 4)) == ".exe") return name.substr(0, name.size() - 4);
  return name;
}

std::string basename_of(const std::string& path) {
  const size_t slash = path.find_last_of("\\/");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

bool contains_ci(const std::string& haystack, const std::string& needle) {
  if (needle.empty()) return true;
  return to_lower_ascii(haystack).find(to_lower_ascii(needle)) != std::string::npos;
}

bool generic_product_name(const std::string& s) {
  const std::string lo = to_lower_ascii(s);
  if (lo.empty()) return true;
  return lo.find("operating system") != std::string::npos || lo == "electron" || lo == "chromium" ||
         lo == "host process" || lo == "windows";
}

// Minimum memory thresholds for listing a non-system user application.
constexpr uint64_t kMinAppVramBytes = 4ull * 1024 * 1024;
constexpr uint64_t kMinAppRamBytes = 32ull * 1024 * 1024;

struct NvmlDevice;
struct NvmlMemory {
  unsigned long long total, free, used;
};

#if !defined(_WIN32)
struct NvmlProcessInfoV2 {
  unsigned int pid;
  unsigned long long usedGpuMemory;
  unsigned int gpuInstanceId;
  unsigned int computeInstanceId;
};
struct NvmlProcessInfoV1 {
  unsigned int pid;
  unsigned long long usedGpuMemory;
};
#endif

}  // namespace

bool is_system_gpu_process(const std::string& proc_name, const std::string& exe_path) {
  const std::string base = to_lower_ascii(strip_exe_suffix(proc_name));
  static const std::unordered_set<std::string> kSystemNames = {
      // Windows desktop, shell, kernel, display driver, and core host services.
      "system", "registry", "memory compression", "dwm", "csrss", "winlogon", "wininit", "services", "lsass",
      "svchost", "sihost", "explorer", "fontdrvhost", "shellexperiencehost", "startmenuexperiencehost", "searchhost",
      "searchapp", "textinputhost", "lockapp", "applicationframehost", "shellhost", "crossdeviceresume",
      "phoneexperiencehost", "systemsettings", "runtimebroker", "ctfmon", "dllhost", "audiodg", "conhost",
      "openconsole", "wudfhost", "nvcontainer", "nvdisplay.container", "nvwmi64", "igfxem", "igfxhk", "amdrsserv",
      "amddvr", "taskmgr", "taskhostw", "smartscreen", "securityhealthsystray", "msmpeng", "spoolsv",
      "smss", "lsaiso", "tiworker", "trustedinstaller", "wmiprvse", "searchindexer", "nissrv",
      "securityhealthservice", "aggregatorhost", "vmmem", "vmmemwsl", "wslservice", "vmcompute", "vmms",
      "dashost", "sppsvc", "backgroundtaskhost", "mousocoreworker", "usocoreworker", "werfault",
      // Linux display servers, desktop compositors, and core daemons.
      "xorg", "xwayland", "gnome-shell", "kwin_wayland", "kwin_x11", "sway", "hyprland", "mutter", "weston",
      "plasmashell", "sddm", "gdm", "gdm3", "lightdm", "systemd", "dbus-daemon", "systemd-journald", "sshd",
      "pipewire", "pulseaudio", "networkmanager", "dockerd", "containerd",
  };
  if (base.empty() || kSystemNames.count(base) > 0) return true;

  std::string path = to_lower_ascii(exe_path);
  std::replace(path.begin(), path.end(), '/', '\\');
  if (path.find("\\windows\\system32\\") != std::string::npos ||
      path.find("\\windows\\syswow64\\") != std::string::npos ||
      path.find("\\windows\\systemapps\\") != std::string::npos ||
      path.find("\\windows\\immersivecontrolpanel\\") != std::string::npos)
    return true;
  if (path.size() >= 21 && path.rfind("\\windows\\explorer.exe") == path.size() - 21) return true;
  return false;
}

std::string clean_gpu_app_name(const std::string& proc_name, const std::string& product_name,
                               const std::string& file_desc) {
  const std::string prod = trim_copy(product_name);
  if (!generic_product_name(prod)) return prod;
  const std::string desc = trim_copy(file_desc);
  if (!generic_product_name(desc)) return desc;

  const std::string base = strip_exe_suffix(trim_copy(proc_name));
  const std::string lo = to_lower_ascii(base);
  if (lo == "chrome") return "Google Chrome";
  if (lo == "msedge") return "Microsoft Edge";
  if (lo == "msedgewebview2") return "Edge WebView2";
  if (lo == "firefox") return "Firefox";
  if (lo == "brave") return "Brave";
  if (lo == "code") return "Visual Studio Code";
  if (lo == "radeonsoftware") return "AMD Software";
  if (lo == "whatsapp.root") return "WhatsApp";
  if (lo == "widgetboard") return "Windows Widgets";
  if (lo == "lghub_system_tray" || lo == "lghub_agent") return "Logitech G HUB";
  if (lo == "wetype_update" || lo == "wetype") return "WeType";
  if (base.size() > 5 && to_lower_ascii(base.substr(base.size() - 5)) == ".root") return base.substr(0, base.size() - 5);
  return base.empty() ? "Unknown app" : base;
}

std::string gpu_app_group_id(const std::string& normalized_key) {
  uint64_t h = 14695981039346656037ull;  // FNV-1a 64-bit
  for (unsigned char c : normalized_key) {
    h ^= c;
    h *= 1099511628211ull;
  }
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return std::string(buf);
}

#if defined(_WIN32)

namespace {

std::string utf8_from_wide(const wchar_t* w, int len = -1) {
  if (!w || (len == 0)) return "";
  const int n = WideCharToMultiByte(CP_UTF8, 0, w, len, nullptr, 0, nullptr, nullptr);
  if (n <= 0) return "";
  std::string out(static_cast<size_t>(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w, len, out.data(), n, nullptr, nullptr);
  if (len < 0 && !out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

std::wstring wide_from_utf8(const std::string& s) {
  if (s.empty()) return L"";
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
  if (n <= 0) return L"";
  std::wstring out(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), out.data(), n);
  return out;
}

// Dynamic loaders for System32 libraries: no extra link dependencies.
struct PdhApi {
  using OpenQueryFn = PDH_STATUS(WINAPI*)(LPCWSTR, DWORD_PTR, PDH_HQUERY*);
  using AddEnglishCounterFn = PDH_STATUS(WINAPI*)(PDH_HQUERY, LPCWSTR, DWORD_PTR, PDH_HCOUNTER*);
  using CollectFn = PDH_STATUS(WINAPI*)(PDH_HQUERY);
  using GetArrayFn = PDH_STATUS(WINAPI*)(PDH_HCOUNTER, DWORD, LPDWORD, LPDWORD, PPDH_FMT_COUNTERVALUE_ITEM_W);
  using CloseQueryFn = PDH_STATUS(WINAPI*)(PDH_HQUERY);

  OpenQueryFn open_query = nullptr;
  AddEnglishCounterFn add_english_counter = nullptr;
  CollectFn collect = nullptr;
  GetArrayFn get_array = nullptr;
  CloseQueryFn close_query = nullptr;
  bool ok = false;
};

const PdhApi& pdh_api() {
  static PdhApi api = [] {
    PdhApi a;
    HMODULE m = LoadLibraryExA("pdh.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!m) return a;
    a.open_query = reinterpret_cast<PdhApi::OpenQueryFn>(GetProcAddress(m, "PdhOpenQueryW"));
    a.add_english_counter =
        reinterpret_cast<PdhApi::AddEnglishCounterFn>(GetProcAddress(m, "PdhAddEnglishCounterW"));
    a.collect = reinterpret_cast<PdhApi::CollectFn>(GetProcAddress(m, "PdhCollectQueryData"));
    a.get_array = reinterpret_cast<PdhApi::GetArrayFn>(GetProcAddress(m, "PdhGetFormattedCounterArrayW"));
    a.close_query = reinterpret_cast<PdhApi::CloseQueryFn>(GetProcAddress(m, "PdhCloseQuery"));
    a.ok = a.open_query && a.add_english_counter && a.collect && a.get_array && a.close_query;
    return a;
  }();
  return api;
}

struct VersionApi {
  using SizeFn = DWORD(WINAPI*)(LPCWSTR, LPDWORD);
  using InfoFn = BOOL(WINAPI*)(LPCWSTR, DWORD, DWORD, LPVOID);
  using QueryFn = BOOL(WINAPI*)(LPCVOID, LPCWSTR, LPVOID*, PUINT);
  SizeFn size = nullptr;
  InfoFn info = nullptr;
  QueryFn query = nullptr;
  bool ok = false;
};

const VersionApi& version_api() {
  static VersionApi api = [] {
    VersionApi a;
    HMODULE m = LoadLibraryExA("version.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!m) return a;
    a.size = reinterpret_cast<VersionApi::SizeFn>(GetProcAddress(m, "GetFileVersionInfoSizeW"));
    a.info = reinterpret_cast<VersionApi::InfoFn>(GetProcAddress(m, "GetFileVersionInfoW"));
    a.query = reinterpret_cast<VersionApi::QueryFn>(GetProcAddress(m, "VerQueryValueW"));
    a.ok = a.size && a.info && a.query;
    return a;
  }();
  return api;
}

struct NvmlWinApi {
  int (*init)() = nullptr;
  int (*count)(unsigned int*) = nullptr;
  int (*by_index)(unsigned int, NvmlDevice**) = nullptr;
  int (*name)(NvmlDevice*, char*, unsigned int) = nullptr;
  int (*memory)(NvmlDevice*, NvmlMemory*) = nullptr;
  bool ok = false;
};

const NvmlWinApi& nvml_win_api() {
  static NvmlWinApi api = [] {
    NvmlWinApi a;
    HMODULE m = LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!m) return a;
    a.init = reinterpret_cast<decltype(a.init)>(GetProcAddress(m, "nvmlInit_v2"));
    a.count = reinterpret_cast<decltype(a.count)>(GetProcAddress(m, "nvmlDeviceGetCount_v2"));
    a.by_index = reinterpret_cast<decltype(a.by_index)>(GetProcAddress(m, "nvmlDeviceGetHandleByIndex_v2"));
    a.name = reinterpret_cast<decltype(a.name)>(GetProcAddress(m, "nvmlDeviceGetName"));
    a.memory = reinterpret_cast<decltype(a.memory)>(GetProcAddress(m, "nvmlDeviceGetMemoryInfo"));
    a.ok = a.init && a.count && a.by_index && a.name && a.memory && a.init() == 0;
    return a;
  }();
  return api;
}

bool query_nvml_memory(const std::string& device_name, uint64_t* total, uint64_t* used, uint64_t* free_b,
                       bool* unique) {
  *unique = false;
  const auto& nv = nvml_win_api();
  if (!nv.ok) return false;
  unsigned int n = 0;
  if (nv.count(&n) != 0 || n == 0) return false;
  NvmlDevice* picked = nullptr;
  NvmlDevice* first = nullptr;
  int matches = 0;
  for (unsigned int i = 0; i < n; ++i) {
    NvmlDevice* d = nullptr;
    if (nv.by_index(i, &d) != 0 || !d) continue;
    if (!first) first = d;
    char buf[128] = {};
    if (nv.name(d, buf, sizeof(buf)) == 0 && !device_name.empty() && contains_ci(buf, device_name)) {
      if (!picked) picked = d;
      ++matches;
    }
  }
  *unique = matches == 1 || (n == 1 && first);
  if (!picked) picked = first;
  if (!picked) return false;
  NvmlMemory mem{};
  if (nv.memory(picked, &mem) != 0 || mem.total == 0) return false;
  *total = uint64_t(mem.total);
  *used = uint64_t(mem.used);
  *free_b = uint64_t(mem.free);
  return true;
}

struct AdapterChoice {
  bool found = false;
  bool unique = false;  // the only adapter the hint names (the only one, without a hint)
  LUID luid{};
  std::string name;
  uint64_t dedicated_total = 0;
  std::string idle_igpu;
};

AdapterChoice find_target_adapter(const std::string& device_hint) {
  AdapterChoice out;
  using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
  static CreateFactoryFn create_factory = [] {
    HMODULE m = LoadLibraryExA("dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return m ? reinterpret_cast<CreateFactoryFn>(GetProcAddress(m, "CreateDXGIFactory1")) : nullptr;
  }();
  if (!create_factory) return out;
  IDXGIFactory1* factory = nullptr;
  if (FAILED(create_factory(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) || !factory) return out;

  struct Cand {
    LUID luid{};
    std::string name;
    UINT vendor = 0;
    uint64_t mem = 0;
  };
  std::vector<Cand> cands;
  IDXGIAdapter1* a = nullptr;
  for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 d{};
    if (SUCCEEDED(a->GetDesc1(&d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
      cands.push_back({d.AdapterLuid, utf8_from_wide(d.Description), d.VendorId, uint64_t(d.DedicatedVideoMemory)});
    }
    a->Release();
  }
  factory->Release();
  if (cands.empty()) return out;

  int best = -1, named = 0;
  if (!device_hint.empty()) {
    for (int i = 0; i < int(cands.size()); ++i) {
      if (contains_ci(cands[i].name, device_hint) || contains_ci(device_hint, cands[i].name)) {
        if (best < 0) best = i;
        ++named;
      }
    }
  }
  out.unique = named == 1 || cands.size() == 1;
  if (best < 0) {
    // Prefer discrete NVIDIA GPU with largest dedicated VRAM.
    uint64_t best_mem = 0;
    for (int i = 0; i < int(cands.size()); ++i) {
      if (cands[i].vendor == 0x10DE && cands[i].mem >= best_mem) {
        best = i;
        best_mem = cands[i].mem;
      }
    }
  }
  if (best < 0) {
    uint64_t best_mem = 0;
    for (int i = 0; i < int(cands.size()); ++i) {
      if (cands[i].mem >= best_mem) {
        best = i;
        best_mem = cands[i].mem;
      }
    }
  }
  if (best >= 0) {
    out.found = true;
    out.luid = cands[best].luid;
    out.name = cands[best].name;
    out.dedicated_total = cands[best].mem;
    // Detect a secondary integrated GPU (AMD or Intel with <= 2 GiB dedicated memory, separate from the target).
    for (int i = 0; i < int(cands.size()); ++i) {
      if (i == best) continue;
      if ((cands[i].vendor == 0x1002 || cands[i].vendor == 0x8086) && cands[i].mem <= (2ull << 30)) {
        out.idle_igpu = cands[i].name;
        break;
      }
    }
  }
  return out;
}

// Parse "pid_<PID>_luid_0x<HIGH>_0x<LOW>_phys_..." from PDH instance names.
bool parse_pdh_gpu_instance(const wchar_t* inst, DWORD* pid_out, uint32_t* hi_out, uint32_t* lo_out) {
  if (!inst || wcsncmp(inst, L"pid_", 4) != 0) return false;
  wchar_t* end = nullptr;
  const unsigned long pid = wcstoul(inst + 4, &end, 10);
  if (!end || pid == 0 || wcsncmp(end, L"_luid_", 6) != 0) return false;
  const wchar_t* p = end + 6;
  const unsigned long hi = wcstoul(p, &end, 16);
  if (!end || *end != L'_') return false;
  const unsigned long lo = wcstoul(end + 1, &end, 16);
  if (!end || (*end != L'_' && *end != L'\0')) return false;
  *pid_out = DWORD(pid);
  *hi_out = uint32_t(hi);
  *lo_out = uint32_t(lo);
  return true;
}

bool read_pdh_counter_array(const PdhApi& pdh, PDH_HCOUNTER counter,
                            const std::function<void(const wchar_t*, uint64_t)>& fn) {
  DWORD buf_size = 0, count = 0;
  std::vector<uint8_t> buf;
  for (int attempt = 0; attempt < 3; ++attempt) {
    const PDH_STATUS st = pdh.get_array(
        counter, PDH_FMT_LARGE, &buf_size, &count,
        buf.empty() ? nullptr : reinterpret_cast<PPDH_FMT_COUNTERVALUE_ITEM_W>(buf.data()));
    if (st == PDH_STATUS(PDH_MORE_DATA)) {
      buf.resize(buf_size + 256);
      continue;
    }
    if (st != ERROR_SUCCESS) return false;
    const auto* items = reinterpret_cast<const PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
    for (DWORD i = 0; i < count; ++i) {
      if (items[i].FmtValue.CStatus == PDH_CSTATUS_VALID_DATA || items[i].FmtValue.CStatus == PDH_CSTATUS_NEW_DATA) {
        if (items[i].FmtValue.largeValue > 0 && items[i].szName)
          fn(items[i].szName, uint64_t(items[i].FmtValue.largeValue));
      }
    }
    return true;
  }
  return false;
}

struct ProcMem {
  uint64_t local = 0;
  uint64_t dedicated = 0;
};

bool query_pdh_gpu_processes(LUID target_luid, std::unordered_map<DWORD, ProcMem>* by_pid,
                             uint64_t* adapter_dedicated_out) {
  const auto& pdh = pdh_api();
  if (!pdh.ok) return false;
  PDH_HQUERY query = nullptr;
  if (pdh.open_query(nullptr, 0, &query) != ERROR_SUCCESS || !query) return false;
  struct Guard {
    const PdhApi& p;
    PDH_HQUERY q;
    ~Guard() { p.close_query(q); }
  } guard{pdh, query};

  PDH_HCOUNTER c_local = nullptr, c_ded = nullptr, c_ad = nullptr;
  const bool has_local =
      pdh.add_english_counter(query, L"\\GPU Process Memory(*)\\Local Usage", 0, &c_local) == ERROR_SUCCESS;
  const bool has_ded =
      pdh.add_english_counter(query, L"\\GPU Process Memory(*)\\Dedicated Usage", 0, &c_ded) == ERROR_SUCCESS;
  const bool has_ad =
      pdh.add_english_counter(query, L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0, &c_ad) == ERROR_SUCCESS;
  if (!has_local && !has_ded) return false;
  if (pdh.collect(query) != ERROR_SUCCESS) return false;

  const uint32_t want_hi = uint32_t(target_luid.HighPart);
  const uint32_t want_lo = uint32_t(target_luid.LowPart);
  if (has_local) {
    read_pdh_counter_array(pdh, c_local, [&](const wchar_t* name, uint64_t val) {
      DWORD pid = 0;
      uint32_t hi = 0, lo = 0;
      if (parse_pdh_gpu_instance(name, &pid, &hi, &lo) && hi == want_hi && lo == want_lo) (*by_pid)[pid].local += val;
    });
  }
  if (has_ded) {
    read_pdh_counter_array(pdh, c_ded, [&](const wchar_t* name, uint64_t val) {
      DWORD pid = 0;
      uint32_t hi = 0, lo = 0;
      if (parse_pdh_gpu_instance(name, &pid, &hi, &lo) && hi == want_hi && lo == want_lo)
        (*by_pid)[pid].dedicated += val;
    });
  }
  if (has_ad && adapter_dedicated_out) {
    wchar_t match[64];
    std::swprintf(match, sizeof(match) / sizeof(match[0]), L"luid_0x%08x_0x%08x", want_hi, want_lo);
    read_pdh_counter_array(pdh, c_ad, [&](const wchar_t* name, uint64_t val) {
      if (wcsstr(name, match)) *adapter_dedicated_out += val;
    });
  }
  return true;
}

// Current user's SID for same-user verification.
std::vector<uint8_t> current_user_sid() {
  static std::vector<uint8_t> sid = [] {
    std::vector<uint8_t> out;
    HANDLE tok = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &tok)) return out;
    DWORD needed = 0;
    GetTokenInformation(tok, TokenUser, nullptr, 0, &needed);
    if (needed > 0) {
      std::vector<uint8_t> buf(needed);
      if (GetTokenInformation(tok, TokenUser, buf.data(), needed, &needed)) {
        auto* tu = reinterpret_cast<TOKEN_USER*>(buf.data());
        const DWORD len = GetLengthSid(tu->User.Sid);
        out.resize(len);
        std::memcpy(out.data(), tu->User.Sid, len);
      }
    }
    CloseHandle(tok);
    return out;
  }();
  return sid;
}

bool process_is_same_user(HANDLE proc) {
  const auto& mine = current_user_sid();
  if (mine.empty()) return false;
  HANDLE tok = nullptr;
  if (!OpenProcessToken(proc, TOKEN_QUERY, &tok)) return false;
  bool same = false;
  DWORD needed = 0;
  GetTokenInformation(tok, TokenUser, nullptr, 0, &needed);
  if (needed > 0) {
    std::vector<uint8_t> buf(needed);
    if (GetTokenInformation(tok, TokenUser, buf.data(), needed, &needed)) {
      auto* tu = reinterpret_cast<TOKEN_USER*>(buf.data());
      same = EqualSid(tu->User.Sid, const_cast<uint8_t*>(mine.data())) != FALSE;
    }
  }
  CloseHandle(tok);
  return same;
}

std::string query_exe_path(HANDLE proc) {
  wchar_t buf[1024];
  DWORD len = DWORD(sizeof(buf) / sizeof(buf[0]));
  if (!QueryFullProcessImageNameW(proc, 0, buf, &len)) return "";
  return utf8_from_wide(buf, int(len));
}

struct ProcTreeInfo {
  std::unordered_map<DWORD, DWORD> parent_of;
  std::unordered_map<DWORD, std::string> name_of;
  std::unordered_set<DWORD> ancestors;
};

ProcTreeInfo snapshot_process_tree() {
  ProcTreeInfo info;
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snap == INVALID_HANDLE_VALUE) return info;
  PROCESSENTRY32W pe{};
  pe.dwSize = sizeof(pe);
  if (Process32FirstW(snap, &pe)) {
    do {
      info.parent_of[pe.th32ProcessID] = pe.th32ParentProcessID;
      info.name_of[pe.th32ProcessID] = utf8_from_wide(pe.szExeFile);
    } while (Process32NextW(snap, &pe));
  }
  CloseHandle(snap);

  DWORD cur = GetCurrentProcessId();
  for (int depth = 0; depth < 32; ++depth) {
    const auto it = info.parent_of.find(cur);
    if (it == info.parent_of.end() || it->second == 0 || it->second == cur) break;
    if (!info.ancestors.insert(it->second).second) break;
    cur = it->second;
  }
  return info;
}

std::string friendly_name_from_exe(const std::string& exe_path, const std::string& proc_name) {
  static std::mutex mu;
  static std::unordered_map<std::string, std::string> cache;
  const std::string key = to_lower_ascii(exe_path.empty() ? proc_name : exe_path);
  {
    std::lock_guard<std::mutex> lock(mu);
    const auto it = cache.find(key);
    if (it != cache.end()) return it->second;
  }

  std::string prod, desc;
  const auto& ver = version_api();
  if (ver.ok && !exe_path.empty()) {
    const std::wstring wpath = wide_from_utf8(exe_path);
    DWORD dummy = 0;
    const DWORD sz = ver.size(wpath.c_str(), &dummy);
    if (sz > 0 && sz < (4u << 20)) {
      std::vector<uint8_t> data(sz);
      if (ver.info(wpath.c_str(), 0, sz, data.data())) {
        struct LangCode {
          WORD lang, cp;
        };
        LangCode* trans = nullptr;
        UINT tlen = 0;
        std::vector<LangCode> langs;
        if (ver.query(data.data(), L"\\VarFileInfo\\Translation", reinterpret_cast<void**>(&trans), &tlen) &&
            trans && tlen >= sizeof(LangCode)) {
          const UINT n = tlen / sizeof(LangCode);
          for (UINT i = 0; i < n && i < 4; ++i) langs.push_back(trans[i]);
        }
        langs.push_back({0x0409, 0x04B0});  // US English Unicode fallback
        auto read_field = [&](const wchar_t* field) -> std::string {
          for (const auto& lc : langs) {
            wchar_t sub[96];
            std::swprintf(sub, sizeof(sub) / sizeof(sub[0]), L"\\StringFileInfo\\%04x%04x\\%ls", lc.lang, lc.cp,
                          field);
            wchar_t* val = nullptr;
            UINT vlen = 0;
            if (ver.query(data.data(), sub, reinterpret_cast<void**>(&val), &vlen) && val && vlen > 0) {
              const std::string s = trim_copy(utf8_from_wide(val));
              if (!s.empty()) return s;
            }
          }
          return "";
        };
        prod = read_field(L"ProductName");
        desc = read_field(L"FileDescription");
      }
    }
  }

  std::string result = clean_gpu_app_name(proc_name, prod, desc);
  {
    std::lock_guard<std::mutex> lock(mu);
    if (cache.size() < 512) cache[key] = result;
  }
  return result;
}

struct LiveGroup {
  GpuApp app;
  std::string exe_path_lower;
  std::vector<DWORD> gpu_pids;
};

struct LiveScan {
  GpuAppSnapshot snap;
  std::unordered_map<std::string, LiveGroup> groups_by_id;
  ProcTreeInfo tree;
};

LiveScan scan_windows_gpu_apps(const std::string& device_hint) {
  LiveScan out;
  const AdapterChoice ad = find_target_adapter(device_hint);
  if (!ad.found) return out;

  out.snap.device_name = ad.name;
  out.snap.idle_igpu = ad.idle_igpu;
  out.snap.mem_total = ad.dedicated_total;

  std::unordered_map<DWORD, ProcMem> by_pid;
  uint64_t ad_dedicated = 0;
  if (!query_pdh_gpu_processes(ad.luid, &by_pid, &ad_dedicated)) return out;
  out.snap.supported = true;

  uint64_t nv_total = 0, nv_used = 0, nv_free = 0;
  bool nv_unique = false;
  if (query_nvml_memory(ad.name, &nv_total, &nv_used, &nv_free, &nv_unique)) {
    out.snap.mem_total = nv_total;
    out.snap.mem_used = nv_used;
    out.snap.mem_free = nv_free;
    out.snap.device_unique = ad.unique && nv_unique;
  } else {
    out.snap.device_unique = ad.unique;
    out.snap.mem_used = ad_dedicated;
    out.snap.mem_free = out.snap.mem_total > ad_dedicated ? out.snap.mem_total - ad_dedicated : 0;
  }

  out.tree = snapshot_process_tree();
  const DWORD self_pid = GetCurrentProcessId();

  // Resolve exe paths of ancestor processes so multi-process IDEs/terminals (where the GPU process is a sibling child
  // of the ancestor main process) are recognized as running FLYX and never offered for closing.
  std::unordered_set<std::string> ancestor_exes;
  for (DWORD apid : out.tree.ancestors) {
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, apid);
    if (!h) continue;
    const std::string p = to_lower_ascii(query_exe_path(h));
    if (!p.empty()) ancestor_exes.insert(p);
    CloseHandle(h);
  }

  struct Acc {
    std::string key;
    std::string exe_path;
    std::string proc_name;
    uint64_t dedicated = 0;
    uint64_t local = 0;
    int procs = 0;
    bool system = false;
    bool runs_flyx = false;
    bool any_same_user = false;
    bool all_same_user = true;
    std::vector<DWORD> pids;
  };
  std::unordered_map<std::string, Acc> groups;

  for (const auto& [pid, pm] : by_pid) {
    if (pm.local == 0 && pm.dedicated == 0) continue;
    if (pid == self_pid) {
      out.snap.own_bytes += pm.local > 0 ? pm.local : pm.dedicated;
      continue;
    }

    std::string proc_name;
    if (auto it = out.tree.name_of.find(pid); it != out.tree.name_of.end()) proc_name = it->second;

    std::string exe_path;
    bool same_user = false;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (h) {
      exe_path = query_exe_path(h);
      same_user = process_is_same_user(h);
      CloseHandle(h);
    }
    if (proc_name.empty() && !exe_path.empty()) proc_name = basename_of(exe_path);
    const std::string exe_lo = to_lower_ascii(exe_path);

    if (is_system_gpu_process(proc_name, exe_path)) {
      // For desktop compositor (dwm.exe) and system hosts, sum Local Usage so shared window surfaces are not
      // double-counted on top of user applications.
      out.snap.system_bytes += pm.local;
      continue;
    }

    const std::string key = !exe_lo.empty() ? exe_lo : ("name:" + to_lower_ascii(strip_exe_suffix(proc_name)));
    auto& g = groups[key];
    if (g.key.empty()) {
      g.key = key;
      g.exe_path = exe_path;
      g.proc_name = strip_exe_suffix(proc_name);
    }
    g.dedicated += pm.dedicated;
    g.local += pm.local;
    g.procs += 1;
    g.pids.push_back(pid);
    if (out.tree.ancestors.count(pid) > 0 || (!exe_lo.empty() && ancestor_exes.count(exe_lo) > 0)) g.runs_flyx = true;
    if (same_user) g.any_same_user = true;
    else g.all_same_user = false;
  }

  out.snap.others_bytes =
      out.snap.mem_used > out.snap.own_bytes ? out.snap.mem_used - out.snap.own_bytes : 0;

  std::vector<LiveGroup> list;
  list.reserve(groups.size());
  for (auto& [k, g] : groups) {
    const uint64_t shown = g.dedicated > 0 ? g.dedicated : g.local;
    if (shown < kMinAppVramBytes) continue;
    LiveGroup lg;
    lg.app.id = gpu_app_group_id(g.key);
    lg.app.process_name = g.proc_name;
    lg.app.name = friendly_name_from_exe(g.exe_path, g.proc_name);
    lg.app.bytes = shown;
    lg.app.local_bytes = g.local;
    lg.app.processes = g.procs;
    if (g.runs_flyx) {
      lg.app.closable = false;
      lg.app.reason = "runs_flyx";
    } else if (!g.any_same_user || g.exe_path.empty()) {
      lg.app.closable = false;
      lg.app.reason = "other_user";
    } else {
      lg.app.closable = true;
    }
    lg.exe_path_lower = to_lower_ascii(g.exe_path);
    lg.gpu_pids = std::move(g.pids);
    list.push_back(std::move(lg));
  }

  std::sort(list.begin(), list.end(), [](const LiveGroup& a, const LiveGroup& b) {
    if (a.app.bytes != b.app.bytes) return a.app.bytes > b.app.bytes;
    if (a.app.local_bytes != b.app.local_bytes) return a.app.local_bytes > b.app.local_bytes;
    return a.app.name < b.app.name;
  });

  for (auto& lg : list) {
    out.snap.apps.push_back(lg.app);
    out.groups_by_id[lg.app.id] = std::move(lg);
  }
  return out;
}

std::vector<DWORD> find_same_user_pids_for_exe(const std::string& exe_path_lower,
                                               const std::unordered_set<DWORD>& ancestors) {
  std::vector<DWORD> pids;
  if (exe_path_lower.empty()) return pids;
  const DWORD self_pid = GetCurrentProcessId();
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snap == INVALID_HANDLE_VALUE) return pids;
  PROCESSENTRY32W pe{};
  pe.dwSize = sizeof(pe);
  if (Process32FirstW(snap, &pe)) {
    do {
      const DWORD pid = pe.th32ProcessID;
      if (pid == 0 || pid == self_pid || ancestors.count(pid) > 0) continue;
      HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
      if (!h) continue;
      const std::string p = to_lower_ascii(query_exe_path(h));
      const bool same = (p == exe_path_lower) && process_is_same_user(h);
      CloseHandle(h);
      if (same) pids.push_back(pid);
    } while (Process32NextW(snap, &pe));
  }
  CloseHandle(snap);
  return pids;
}

int post_wm_close_to_pids(const std::unordered_set<DWORD>& pids) {
  struct Ctx {
    const std::unordered_set<DWORD>* pids;
    std::vector<HWND> visible;
  } ctx{&pids, {}};
  EnumWindows(
      [](HWND hwnd, LPARAM lp) -> BOOL {
        auto* c = reinterpret_cast<Ctx*>(lp);
        DWORD pid = 0;
        GetWindowThreadProcessId(hwnd, &pid);
        if (c->pids->count(pid) == 0) return TRUE;
        if (!IsWindowVisible(hwnd)) return TRUE;
        if (GetWindow(hwnd, GW_OWNER) != nullptr) return TRUE;
        c->visible.push_back(hwnd);
        return TRUE;
      },
      reinterpret_cast<LPARAM>(&ctx));
  int sent = 0;
  for (HWND hwnd : ctx.visible) {
    if (PostMessageW(hwnd, WM_CLOSE, 0, 0)) ++sent;
  }
  return sent;
}

bool any_pid_alive(const std::vector<DWORD>& pids) {
  for (DWORD pid : pids) {
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (!h) continue;
    const DWORD w = WaitForSingleObject(h, 0);
    CloseHandle(h);
    if (w == WAIT_TIMEOUT) return true;
  }
  return false;
}

uint64_t query_process_ram_bytes(HANDLE proc) {
  using GetMemInfoFn = BOOL(WINAPI*)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
  static GetMemInfoFn fn = [] {
    HMODULE k32 = GetModuleHandleA("kernel32.dll");
    return k32 ? reinterpret_cast<GetMemInfoFn>(GetProcAddress(k32, "K32GetProcessMemoryInfo")) : nullptr;
  }();
  if (!fn) return 0;
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  pmc.cb = sizeof(pmc);
  if (!fn(proc, reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&pmc), sizeof(pmc))) return 0;
  return uint64_t(pmc.WorkingSetSize);
}

LiveScan scan_windows_ram_apps() {
  LiveScan out;
  const os::SystemMemory sm = os::system_memory();
  if (sm.total == 0) return out;
  out.snap.supported = true;
  out.snap.device_name = "System RAM";
  out.snap.mem_total = sm.total;
  out.snap.mem_free = sm.available;
  out.snap.mem_used = sm.total > sm.available ? sm.total - sm.available : 0;

  out.tree = snapshot_process_tree();
  const DWORD self_pid = GetCurrentProcessId();

  std::unordered_set<std::string> ancestor_exes;
  for (DWORD apid : out.tree.ancestors) {
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, apid);
    if (!h) continue;
    const std::string p = to_lower_ascii(query_exe_path(h));
    if (!p.empty()) ancestor_exes.insert(p);
    CloseHandle(h);
  }

  struct Acc {
    std::string key;
    std::string exe_path;
    std::string proc_name;
    uint64_t bytes = 0;
    int procs = 0;
    bool runs_flyx = false;
    bool any_same_user = false;
    std::vector<DWORD> pids;
  };
  std::unordered_map<std::string, Acc> groups;

  for (const auto& [pid, name_raw] : out.tree.name_of) {
    if (pid == 0) continue;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h) continue;
    const uint64_t rss = query_process_ram_bytes(h);
    const std::string exe_path = query_exe_path(h);
    const bool same_user = process_is_same_user(h);
    CloseHandle(h);
    if (rss == 0) continue;

    if (pid == self_pid) {
      out.snap.own_bytes += rss;
      continue;
    }

    const std::string proc_name = !name_raw.empty() ? name_raw : basename_of(exe_path);
    const std::string exe_lo = to_lower_ascii(exe_path);
    if (is_system_gpu_process(proc_name, exe_path)) {
      out.snap.system_bytes += rss;
      continue;
    }

    const std::string key = !exe_lo.empty() ? exe_lo : ("name:" + to_lower_ascii(strip_exe_suffix(proc_name)));
    auto& g = groups[key];
    if (g.key.empty()) {
      g.key = key;
      g.exe_path = exe_path;
      g.proc_name = strip_exe_suffix(proc_name);
    }
    g.bytes += rss;
    g.procs += 1;
    g.pids.push_back(pid);
    if (out.tree.ancestors.count(pid) > 0 || (!exe_lo.empty() && ancestor_exes.count(exe_lo) > 0)) g.runs_flyx = true;
    if (same_user) g.any_same_user = true;
  }

  out.snap.others_bytes = out.snap.mem_used > out.snap.own_bytes ? out.snap.mem_used - out.snap.own_bytes : 0;

  std::vector<LiveGroup> list;
  list.reserve(groups.size());
  for (auto& [k, g] : groups) {
    if (g.bytes < kMinAppRamBytes) continue;
    LiveGroup lg;
    lg.app.id = gpu_app_group_id(g.key);
    lg.app.process_name = g.proc_name;
    lg.app.name = friendly_name_from_exe(g.exe_path, g.proc_name);
    lg.app.bytes = g.bytes;
    lg.app.local_bytes = g.bytes;
    lg.app.processes = g.procs;
    if (g.runs_flyx) {
      lg.app.closable = false;
      lg.app.reason = "runs_flyx";
    } else if (!g.any_same_user || g.exe_path.empty()) {
      lg.app.closable = false;
      lg.app.reason = "other_user";
    } else {
      lg.app.closable = true;
    }
    lg.exe_path_lower = to_lower_ascii(g.exe_path);
    lg.gpu_pids = std::move(g.pids);
    list.push_back(std::move(lg));
  }

  std::sort(list.begin(), list.end(), [](const LiveGroup& a, const LiveGroup& b) {
    if (a.app.bytes != b.app.bytes) return a.app.bytes > b.app.bytes;
    return a.app.name < b.app.name;
  });

  for (auto& lg : list) {
    out.snap.apps.push_back(lg.app);
    out.groups_by_id[lg.app.id] = std::move(lg);
  }
  return out;
}

GpuCloseResult close_windows_live(const std::function<LiveScan()>& scan_fn, const std::string& id, bool force) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);

  GpuCloseResult res;
  LiveScan live = scan_fn();
  res.snapshot = live.snap;
  if (!live.snap.supported) {
    res.status = CloseStatus::kUnsupported;
    return res;
  }
  const auto it = live.groups_by_id.find(id);
  if (it == live.groups_by_id.end()) {
    res.status = CloseStatus::kNotFound;
    return res;
  }
  const LiveGroup& target = it->second;
  res.app_name = target.app.name;
  if (!target.app.closable || target.exe_path_lower.empty()) {
    res.status = CloseStatus::kForbidden;
    return res;
  }

  const std::vector<DWORD> all_pids = find_same_user_pids_for_exe(target.exe_path_lower, live.tree.ancestors);
  if (all_pids.empty()) {
    res.snapshot = scan_fn().snap;
    res.status = CloseStatus::kClosed;
    return res;
  }

  if (!force) {
    const std::unordered_set<DWORD> pid_set(all_pids.begin(), all_pids.end());
    res.windows_signaled = post_wm_close_to_pids(pid_set);
    if (res.windows_signaled == 0) {
      res.status = CloseStatus::kNoWindow;
      return res;
    }
    for (int i = 0; i < 8; ++i) {
      os::sleep_ms(100);
      if (!any_pid_alive(target.gpu_pids)) break;
    }
  } else {
    int terminated = 0, denied = 0;
    for (DWORD pid : all_pids) {
      HANDLE h = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, pid);
      if (!h) {
        if (GetLastError() == ERROR_ACCESS_DENIED) ++denied;
        continue;
      }
      if (TerminateProcess(h, 0)) ++terminated;
      else if (GetLastError() == ERROR_ACCESS_DENIED) ++denied;
      CloseHandle(h);
    }
    if (terminated == 0 && denied > 0) {
      res.status = CloseStatus::kAccessDenied;
      return res;
    }
    for (int i = 0; i < 8; ++i) {
      os::sleep_ms(100);
      if (!any_pid_alive(target.gpu_pids)) break;
    }
  }

  LiveScan after = scan_fn();
  res.snapshot = after.snap;
  res.status = after.groups_by_id.count(id) == 0 ? CloseStatus::kClosed : CloseStatus::kStillRunning;
  return res;
}

}  // namespace

GpuAppSnapshot list_gpu_apps(const std::string& device_hint) { return scan_windows_gpu_apps(device_hint).snap; }

GpuCloseResult close_gpu_app(const std::string& device_hint, const std::string& id, bool force) {
  return close_windows_live([&] { return scan_windows_gpu_apps(device_hint); }, id, force);
}

GpuAppSnapshot list_ram_apps() { return scan_windows_ram_apps().snap; }

GpuCloseResult close_ram_app(const std::string& id, bool force) {
  return close_windows_live([] { return scan_windows_ram_apps(); }, id, force);
}

#else  // POSIX / Linux

namespace {

struct NvmlLinuxApi {
  int (*init)() = nullptr;
  int (*count)(unsigned int*) = nullptr;
  int (*by_index)(unsigned int, NvmlDevice**) = nullptr;
  int (*name)(NvmlDevice*, char*, unsigned int) = nullptr;
  int (*memory)(NvmlDevice*, NvmlMemory*) = nullptr;
  int (*compute_v3)(NvmlDevice*, unsigned int*, NvmlProcessInfoV2*) = nullptr;
  int (*graphics_v3)(NvmlDevice*, unsigned int*, NvmlProcessInfoV2*) = nullptr;
  int (*compute_v1)(NvmlDevice*, unsigned int*, NvmlProcessInfoV1*) = nullptr;
  int (*graphics_v1)(NvmlDevice*, unsigned int*, NvmlProcessInfoV1*) = nullptr;
  bool ok = false;
};

const NvmlLinuxApi& nvml_linux_api() {
  static NvmlLinuxApi api = [] {
    NvmlLinuxApi a;
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) return a;
    a.init = reinterpret_cast<decltype(a.init)>(dlsym(lib, "nvmlInit_v2"));
    a.count = reinterpret_cast<decltype(a.count)>(dlsym(lib, "nvmlDeviceGetCount_v2"));
    a.by_index = reinterpret_cast<decltype(a.by_index)>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
    a.name = reinterpret_cast<decltype(a.name)>(dlsym(lib, "nvmlDeviceGetName"));
    a.memory = reinterpret_cast<decltype(a.memory)>(dlsym(lib, "nvmlDeviceGetMemoryInfo"));
    a.compute_v3 = reinterpret_cast<decltype(a.compute_v3)>(dlsym(lib, "nvmlDeviceGetComputeRunningProcesses_v3"));
    if (!a.compute_v3)
      a.compute_v3 = reinterpret_cast<decltype(a.compute_v3)>(dlsym(lib, "nvmlDeviceGetComputeRunningProcesses_v2"));
    a.graphics_v3 = reinterpret_cast<decltype(a.graphics_v3)>(dlsym(lib, "nvmlDeviceGetGraphicsRunningProcesses_v3"));
    if (!a.graphics_v3)
      a.graphics_v3 =
          reinterpret_cast<decltype(a.graphics_v3)>(dlsym(lib, "nvmlDeviceGetGraphicsRunningProcesses_v2"));
    a.compute_v1 = reinterpret_cast<decltype(a.compute_v1)>(dlsym(lib, "nvmlDeviceGetComputeRunningProcesses"));
    a.graphics_v1 = reinterpret_cast<decltype(a.graphics_v1)>(dlsym(lib, "nvmlDeviceGetGraphicsRunningProcesses"));
    a.ok = a.init && a.count && a.by_index && a.name && a.memory && a.init() == 0;
    return a;
  }();
  return api;
}

template <class Info, class Fn>
void collect_nvml_procs(Fn fn, NvmlDevice* dev, std::unordered_map<int, uint64_t>* out) {
  if (!fn) return;
  unsigned int count = 128;
  std::vector<Info> buf(count);
  int rc = fn(dev, &count, buf.data());
  if (rc == 7 && count > buf.size()) {  // NVML_ERROR_INSUFFICIENT_SIZE
    buf.resize(count);
    rc = fn(dev, &count, buf.data());
  }
  if (rc != 0) return;
  for (unsigned int i = 0; i < count; ++i) {
    const uint64_t used = buf[i].usedGpuMemory == ~0ull ? 0 : uint64_t(buf[i].usedGpuMemory);
    auto& slot = (*out)[int(buf[i].pid)];
    if (used > slot) slot = used;
  }
}

std::string read_proc_comm(int pid) {
  std::ifstream f("/proc/" + std::to_string(pid) + "/comm");
  std::string line;
  std::getline(f, line);
  return trim_copy(line);
}

std::string read_proc_exe(int pid) {
  char buf[PATH_MAX];
  const std::string link = "/proc/" + std::to_string(pid) + "/exe";
  const ssize_t n = readlink(link.c_str(), buf, sizeof(buf) - 1);
  if (n <= 0) return "";
  buf[n] = '\0';
  return std::string(buf);
}

bool proc_same_uid(int pid) {
  std::ifstream f("/proc/" + std::to_string(pid) + "/status");
  std::string line;
  const uid_t me = geteuid();
  while (std::getline(f, line)) {
    if (line.compare(0, 4, "Uid:") == 0) {
      std::istringstream is(line.substr(4));
      unsigned long real_uid = 0;
      if (is >> real_uid) return uid_t(real_uid) == me;
    }
  }
  return false;
}

int proc_ppid(int pid) {
  std::ifstream f("/proc/" + std::to_string(pid) + "/stat");
  std::string line;
  if (!std::getline(f, line)) return 0;
  const size_t rp = line.rfind(')');
  if (rp == std::string::npos || rp + 2 >= line.size()) return 0;
  std::istringstream is(line.substr(rp + 2));
  char state = 0;
  int ppid = 0;
  if (is >> state >> ppid) return ppid;
  return 0;
}

std::unordered_set<int> linux_ancestors() {
  std::unordered_set<int> anc;
  int cur = int(getpid());
  for (int i = 0; i < 32; ++i) {
    const int p = proc_ppid(cur);
    if (p <= 1 || !anc.insert(p).second) break;
    cur = p;
  }
  return anc;
}

struct LinuxGroup {
  GpuApp app;
  std::vector<int> pids;
};

struct LinuxScan {
  GpuAppSnapshot snap;
  std::unordered_map<std::string, LinuxGroup> groups_by_id;
  std::unordered_set<int> ancestors;
};

struct LinuxAppUsage {
  std::string key, proc_name;
  uint64_t bytes = 0;
  int procs = 0;
  bool runs_flyx = false;
  bool any_same_uid = false;
  std::vector<int> pids;
};

// A scan's groups as the listing shows them: those of at least minimum_bytes, the largest first, each with whether
// it can be closed.
void finish_linux_scan(LinuxScan& out, std::unordered_map<std::string, LinuxAppUsage>& groups,
                       uint64_t minimum_bytes) {
  out.snap.others_bytes =
      out.snap.mem_used > out.snap.own_bytes ? out.snap.mem_used - out.snap.own_bytes : 0;

  std::vector<LinuxGroup> list;
  for (auto& [k, g] : groups) {
    if (g.bytes < minimum_bytes) continue;
    LinuxGroup lg;
    lg.app.id = gpu_app_group_id(g.key);
    lg.app.process_name = g.proc_name;
    lg.app.name = clean_gpu_app_name(g.proc_name, "", "");
    lg.app.bytes = g.bytes;
    lg.app.local_bytes = g.bytes;
    lg.app.processes = g.procs;
    if (g.runs_flyx) {
      lg.app.closable = false;
      lg.app.reason = "runs_flyx";
    } else if (!g.any_same_uid) {
      lg.app.closable = false;
      lg.app.reason = "other_user";
    } else {
      lg.app.closable = true;
    }
    lg.pids = std::move(g.pids);
    list.push_back(std::move(lg));
  }
  std::sort(list.begin(), list.end(), [](const LinuxGroup& a, const LinuxGroup& b) {
    if (a.app.bytes != b.app.bytes) return a.app.bytes > b.app.bytes;
    return a.app.name < b.app.name;
  });
  for (auto& lg : list) {
    out.snap.apps.push_back(lg.app);
    out.groups_by_id[lg.app.id] = std::move(lg);
  }
}

LinuxScan scan_linux_gpu_apps(const std::string& device_hint) {
  LinuxScan out;
  const auto& nv = nvml_linux_api();
  if (!nv.ok) return out;
  unsigned int n = 0;
  if (nv.count(&n) != 0 || n == 0) return out;

  NvmlDevice* dev = nullptr;
  std::string dev_name;
  bool named = false;
  int matches = 0;
  for (unsigned int i = 0; i < n; ++i) {
    NvmlDevice* d = nullptr;
    if (nv.by_index(i, &d) != 0 || !d) continue;
    char buf[128] = {};
    nv.name(d, buf, sizeof(buf));
    if (!device_hint.empty() && contains_ci(buf, device_hint)) {
      if (!named) dev = d, dev_name = buf, named = true;
      ++matches;
    }
    if (!dev) {
      dev = d;
      dev_name = buf;
    }
  }
  if (!dev) return out;
  out.snap.device_unique = matches == 1 || n == 1;

  NvmlMemory mem{};
  if (nv.memory(dev, &mem) != 0 || mem.total == 0) return out;
  out.snap.supported = true;
  out.snap.device_name = dev_name;
  out.snap.mem_total = uint64_t(mem.total);
  out.snap.mem_used = uint64_t(mem.used);
  out.snap.mem_free = uint64_t(mem.free);

  std::unordered_map<int, uint64_t> by_pid;
  if (nv.compute_v3) collect_nvml_procs<NvmlProcessInfoV2>(nv.compute_v3, dev, &by_pid);
  else collect_nvml_procs<NvmlProcessInfoV1>(nv.compute_v1, dev, &by_pid);
  if (nv.graphics_v3) collect_nvml_procs<NvmlProcessInfoV2>(nv.graphics_v3, dev, &by_pid);
  else collect_nvml_procs<NvmlProcessInfoV1>(nv.graphics_v1, dev, &by_pid);

  const int self_pid = int(getpid());
  out.ancestors = linux_ancestors();
  std::unordered_set<std::string> ancestor_exes;
  for (int ap : out.ancestors) {
    const std::string e = read_proc_exe(ap);
    if (!e.empty()) ancestor_exes.insert(e);
  }

  std::unordered_map<std::string, LinuxAppUsage> groups;

  for (const auto& [pid, used] : by_pid) {
    if (pid == self_pid) {
      out.snap.own_bytes += used;
      continue;
    }
    const std::string comm = read_proc_comm(pid);
    const std::string exe = read_proc_exe(pid);
    if (is_system_gpu_process(comm, exe)) {
      out.snap.system_bytes += used;
      continue;
    }
    const std::string key = !exe.empty() ? exe : ("name:" + to_lower_ascii(comm));
    auto& g = groups[key];
    if (g.key.empty()) {
      g.key = key;
      g.proc_name = !comm.empty() ? comm : basename_of(exe);
    }
    g.bytes += used;
    g.procs += 1;
    g.pids.push_back(pid);
    if (out.ancestors.count(pid) > 0 || (!exe.empty() && ancestor_exes.count(exe) > 0)) g.runs_flyx = true;
    if (proc_same_uid(pid)) g.any_same_uid = true;
  }

  finish_linux_scan(out, groups, kMinAppVramBytes);
  return out;
}

uint64_t read_proc_rss_bytes(int pid) {
  std::ifstream f("/proc/" + std::to_string(pid) + "/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, 6, "VmRSS:") == 0) {
      return std::strtoull(line.c_str() + 6, nullptr, 10) * 1024ull;
    }
  }
  return 0;
}

LinuxScan scan_linux_ram_apps() {
  LinuxScan out;
  const os::SystemMemory sm = os::system_memory();
  if (sm.total == 0) return out;
  out.snap.supported = true;
  out.snap.device_name = "System RAM";
  out.snap.mem_total = sm.total;
  out.snap.mem_free = sm.available;
  out.snap.mem_used = sm.total > sm.available ? sm.total - sm.available : 0;

  const int self_pid = int(getpid());
  out.ancestors = linux_ancestors();
  std::unordered_set<std::string> ancestor_exes;
  for (int ap : out.ancestors) {
    const std::string e = read_proc_exe(ap);
    if (!e.empty()) ancestor_exes.insert(e);
  }

  std::unordered_map<std::string, LinuxAppUsage> groups;

  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator("/proc", ec)) {
    const std::string fname = entry.path().filename().string();
    if (fname.empty() || !std::all_of(fname.begin(), fname.end(), [](unsigned char c) { return std::isdigit(c); }))
      continue;
    const int pid = std::atoi(fname.c_str());
    if (pid <= 1) continue;
    const uint64_t rss = read_proc_rss_bytes(pid);
    if (rss == 0) continue;
    if (pid == self_pid) {
      out.snap.own_bytes += rss;
      continue;
    }
    const std::string comm = read_proc_comm(pid);
    const std::string exe = read_proc_exe(pid);
    if (is_system_gpu_process(comm, exe)) {
      out.snap.system_bytes += rss;
      continue;
    }
    const std::string key = !exe.empty() ? exe : ("name:" + to_lower_ascii(comm));
    auto& g = groups[key];
    if (g.key.empty()) {
      g.key = key;
      g.proc_name = !comm.empty() ? comm : basename_of(exe);
    }
    g.bytes += rss;
    g.procs += 1;
    g.pids.push_back(pid);
    if (out.ancestors.count(pid) > 0 || (!exe.empty() && ancestor_exes.count(exe) > 0)) g.runs_flyx = true;
    if (proc_same_uid(pid)) g.any_same_uid = true;
  }

  finish_linux_scan(out, groups, kMinAppRamBytes);
  return out;
}

GpuCloseResult close_linux_live(const std::function<LinuxScan()>& scan_fn, const std::string& id, bool force) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);

  GpuCloseResult res;
  LinuxScan live = scan_fn();
  res.snapshot = live.snap;
  if (!live.snap.supported) {
    res.status = CloseStatus::kUnsupported;
    return res;
  }
  const auto it = live.groups_by_id.find(id);
  if (it == live.groups_by_id.end()) {
    res.status = CloseStatus::kNotFound;
    return res;
  }
  const LinuxGroup& target = it->second;
  res.app_name = target.app.name;
  if (!target.app.closable) {
    res.status = CloseStatus::kForbidden;
    return res;
  }
  const int sig = force ? SIGKILL : SIGTERM;
  const int self_pid = int(getpid());
  int sent = 0, denied = 0;
  for (int pid : target.pids) {
    if (pid <= 1 || pid == self_pid || live.ancestors.count(pid) > 0 || !proc_same_uid(pid)) continue;
    if (kill(pid_t(pid), sig) == 0) ++sent;
    else if (errno == EPERM) ++denied;
  }
  if (sent == 0 && denied > 0) {
    res.status = CloseStatus::kAccessDenied;
    return res;
  }
  res.windows_signaled = sent;
  for (int i = 0; i < 8; ++i) {
    os::sleep_ms(100);
    bool alive = false;
    for (int pid : target.pids)
      if (os::process_running(pid)) alive = true;
    if (!alive) break;
  }
  LinuxScan after = scan_fn();
  res.snapshot = after.snap;
  res.status = after.groups_by_id.count(id) == 0 ? CloseStatus::kClosed : CloseStatus::kStillRunning;
  return res;
}

}  // namespace

GpuAppSnapshot list_gpu_apps(const std::string& device_hint) { return scan_linux_gpu_apps(device_hint).snap; }

GpuCloseResult close_gpu_app(const std::string& device_hint, const std::string& id, bool force) {
  return close_linux_live([&] { return scan_linux_gpu_apps(device_hint); }, id, force);
}

GpuAppSnapshot list_ram_apps() { return scan_linux_ram_apps().snap; }

GpuCloseResult close_ram_app(const std::string& id, bool force) {
  return close_linux_live([] { return scan_linux_ram_apps(); }, id, force);
}

#endif

}  // namespace flyx::server
