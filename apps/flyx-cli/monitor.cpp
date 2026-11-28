// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// flyx monitor: a live panel of a running server (flyx serve), drawn in the
// terminal from its GET /v1/telemetry (read-only; quitting leaves the server
// running).  flyx serve draws the same panel on its own terminal while it runs
// (serve_dashboard), with the log the server writes under it.
//
//   flyx monitor [URL] [--key KEY] [--interval SECONDS] [--once [--for SECONDS]]
//
// The model's state and the running request, decode speed, speculation,
// prefill and time to first token, the GPU (load, memory, temperature,
// power), the process's CPU, the resident sessions, system RAM, totals and
// the recent requests. Keys: q quit, space freeze, r refresh.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "apps/common/api_key.h"
#include "apps/common/defaults.h"
#include "apps/common/numbers.h"
#include "apps/flyx-cli/monitor.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "httplib/httplib.h"
#include "nlohmann/json.hpp"

namespace flyx::monitor {
namespace {

using json = nlohmann::json;

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

// ---- text --------------------------------------------------------------------------------------

const char* kAccent = "38;5;42";
const char* kBold = "1";
const char* kDim = "2";
const char* kWarm = "38;5;214";
const char* kHot = "38;5;203";
const char* kCool = "38;5;81";

// Columns a string takes: UTF-8 code points outside escape sequences (every glyph used here is one column).
int width(const std::string& s) {
  int w = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\x1b') {
      while (i < s.size() && s[i] != 'm') ++i;
      continue;
    }
    if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) ++w;
  }
  return w;
}

// Plain text cut or padded to w columns.
std::string fit(const std::string& s, int w, bool right = false) {
  std::string out;
  int cols = 0;
  for (size_t i = 0; i < s.size() && cols < w;) {
    size_t n = 1;
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c >= 0xF0) n = 4;
    else if (c >= 0xE0) n = 3;
    else if (c >= 0xC0) n = 2;
    out += s.substr(i, n);
    i += n;
    ++cols;
  }
  const std::string padding(size_t(std::max(0, w - cols)), ' ');
  return right ? padding + out : out + padding;
}

std::string repeat(const char* glyph, int n) {
  std::string s;
  for (int i = 0; i < n; ++i) s += glyph;
  return s;
}

std::string num(double v, int decimals = 0) {
  char b[64];
  std::snprintf(b, sizeof(b), "%.*f", decimals, v);
  return b;
}

std::string grouped(long long v) {
  std::string s = std::to_string(v < 0 ? -v : v), out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (i && (s.size() - i) % 3 == 0) out += ',';
    out += s[i];
  }
  return v < 0 ? "-" + out : out;
}

std::string gib(double bytes) { return num(bytes / 1073741824.0, bytes >= 1e11 ? 0 : 1); }
// A few MiB to many GiB, in the unit that says it ("0.0 GiB" said nothing of a server's 40 MiB).
std::string mem_text(double bytes) {
  return bytes < 1073741824.0 ? num(bytes / 1048576.0, 0) + " MiB" : gib(bytes) + " GiB";
}

std::string seconds(double s) {
  if (s < 10) return num(s, 2) + " s";
  if (s < 100) return num(s, 1) + " s";
  if (s < 3600) return num(std::floor(s / 60), 0) + "m" + num(std::fmod(s, 60), 0) + "s";
  return num(std::floor(s / 3600), 0) + "h" + num(std::fmod(s, 3600) / 60, 0) + "m";
}

double get(const json& j, const char* key, double dflt = 0) {
  return j.is_object() && j.contains(key) && j[key].is_number() ? j[key].get<double>() : dflt;
}
std::string str(const json& j, const char* key) {
  return j.is_object() && j.contains(key) && j[key].is_string() ? j[key].get<std::string>() : "";
}
bool flag(const json& j, const char* key, bool dflt = false) {
  return j.is_object() && j.contains(key) && j[key].is_boolean() ? j[key].get<bool>() : dflt;
}
bool has(const json& j, const char* key) { return j.is_object() && j.contains(key) && j[key].is_number(); }

std::string clock_time(double at) {
  const std::time_t t = std::time_t(at);
  std::tm tm{};
#if defined(_WIN32)
  if (localtime_s(&tm, &t) != 0) return "--:--:--";
#else
  if (!localtime_r(&t, &tm)) return "--:--:--";
#endif
  char b[16];
  return std::strftime(b, sizeof(b), "%H:%M:%S", &tm) ? std::string(b) : std::string("--:--:--");
}

// ---- terminal & telemetry session --------------------------------------------------------------

struct AltScreen {
  std::vector<char> buf_;
  AltScreen() : buf_(65536) {
    // Buffer a whole frame into one OS write so Windows Console / Terminal never renders a half-written frame.
    std::setvbuf(stdout, buf_.data(), _IOFBF, buf_.size());
    std::fputs("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H", stdout);  // the alternate screen, no cursor
    std::fflush(stdout);
  }
  ~AltScreen() {
    std::fputs("\x1b[?25h\x1b[?1049l", stdout);  // the cursor and the normal screen back
    std::fflush(stdout);
    std::setvbuf(stdout, nullptr, _IONBF, 0);
  }
  AltScreen(const AltScreen&) = delete;
  AltScreen& operator=(const AltScreen&) = delete;
};

struct History {
  std::deque<double> util, power, cpu;
  double last_cpu_s = -1, last_at = 0;
  void push(std::deque<double>* d, double v) {
    d->push_back(v);
    if (d->size() > 200) d->pop_front();
  }
};

std::vector<double> vec(const std::deque<double>& d) { return {d.begin(), d.end()}; }

class TelemetrySampler {
 public:
  TelemetrySampler(std::string url, std::string key) : url_(std::move(url)), key_(std::move(key)), cli_(url_) {
    cli_.set_connection_timeout(2, 0);
    cli_.set_read_timeout(5, 0);
  }

  bool sample() {
    httplib::Headers h;
    if (!key_.empty()) h.emplace("Authorization", "Bearer " + key_);
    auto res = cli_.Get("/v1/telemetry?recent=40", h);
    if (!res) {
      err_ = "cannot reach " + url_ + ": " + httplib::to_string(res.error());
      return false;
    }
    if (res->status != 200) {
      err_ = url_ + "/v1/telemetry: HTTP " + std::to_string(res->status);
      return false;
    }
    json next = json::parse(res->body, nullptr, false);
    if (next.is_discarded() || !next.is_object()) {
      err_ = "the server's telemetry is not JSON";
      return false;
    }
    if (!next.contains("schema_version") || !next["schema_version"].is_number_integer() ||
        next["schema_version"].get<int>() != 1) {
      err_ = "unknown telemetry schema version";
      return false;
    }
    t_ = std::move(next);
    err_.clear();
    const json& d = t_["device"];
    if (d.is_object()) {
      hist_.push(&hist_.util, get(d, "util"));
      hist_.push(&hist_.power, get(d, "power_w"));
    }
    const double cpu_s = get(t_["process"], "cpu_s"), at = get(t_, "sampled_at");
    if (hist_.last_cpu_s >= 0 && at > hist_.last_at && cpu_s >= hist_.last_cpu_s) {
      cpu_pct_ = (cpu_s - hist_.last_cpu_s) / (at - hist_.last_at) * 100;
      hist_.push(&hist_.cpu, cpu_pct_);
    }
    hist_.last_cpu_s = cpu_s;
    hist_.last_at = at;
    return true;
  }

  const json& telemetry() const { return t_; }
  const History& history() const { return hist_; }
  const std::string& error() const { return err_; }
  double cpu_pct() const { return cpu_pct_; }

 private:
  std::string url_, key_, err_;
  httplib::Client cli_;
  json t_;
  History hist_;
  double cpu_pct_ = -1;
};

// ---- the panel ---------------------------------------------------------------------------------

struct Tile {
  std::string title, big, sub, chart;
};

struct TermStyle {
  bool color = true;

  std::string paint(const char* code, const std::string& s) const {
    return color && !s.empty() ? std::string("\x1b[") + code + "m" + s + "\x1b[0m" : s;
  }

  // Painted text cut to w columns (escape sequences kept whole).
  std::string cut(const std::string& s, int w) const {
    if (width(s) <= w) return s;
    std::string out;
    int cols = 0;
    for (size_t i = 0; i < s.size(); ++i) {
      if (s[i] == '\x1b') {
        const size_t m = s.find('m', i);
        out += s.substr(i, m == std::string::npos ? std::string::npos : m - i + 1);
        i = m == std::string::npos ? s.size() : m;
        continue;
      }
      const bool lead = (static_cast<unsigned char>(s[i]) & 0xC0) != 0x80;
      if (lead && cols == w) break;
      cols += lead;
      out += s[i];
    }
    return color ? out + "\x1b[0m" : out;
  }

  // Painted text cut or padded to w columns.
  std::string pad(const std::string& s, int w) const {
    const std::string c = cut(s, w);
    return c + std::string(size_t(std::max(0, w - width(c))), ' ');
  }

  // A bar of w cells, frac of them full.
  std::string bar(double frac, int w, const char* code) const {
    frac = std::max(0.0, std::min(1.0, std::isfinite(frac) ? frac : 0.0));
    const int full = int(std::lround(frac * w));
    return paint(code, repeat("█", full)) + paint(kDim, repeat("░", w - full));
  }

  // The last w values as a sparkline, scaled to top (0: their maximum).
  std::string spark(const std::vector<double>& v, int w, double top, const char* code) const {
    if (w <= 0) return {};
    static const char* levels[] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    std::vector<double> tail(v.end() - std::min<long>(long(v.size()), w), v.end());
    if (top <= 0)
      for (double x : tail) top = std::max(top, x);
    std::string s(size_t(w - int(tail.size())), ' ');
    std::string drawn;
    for (double x : tail) {
      if (x <= 0 || top <= 0) {
        drawn += " ";
        continue;
      }
      drawn += levels[std::max(0, std::min(7, int(x / top * 7.999)))];
    }
    return s + paint(code, drawn);
  }

  // Tiles side by side, each `w` columns wide.
  void draw_tiles(const std::vector<Tile>& row, int w, std::vector<std::string>* out) const {
    const int inner = w - 4;
    std::string top, l1, l2, l3, bottom;
    for (size_t i = 0; i < row.size(); ++i) {
      const std::string gap = i ? " " : "";
      const std::string t = " " + row[i].title + " ";
      top += gap + paint(kDim, "┌─") + t + paint(kDim, repeat("─", w - 3 - width(t)) + "┐");
      l1 += gap + paint(kDim, "│ ") + pad(row[i].big, inner) + paint(kDim, " │");
      l2 += gap + paint(kDim, "│ ") + pad(row[i].sub, inner) + paint(kDim, " │");
      l3 += gap + paint(kDim, "│ ") + pad(row[i].chart, inner) + paint(kDim, " │");
      bottom += gap + paint(kDim, "└" + repeat("─", w - 2) + "┘");
    }
    for (auto* s : {&top, &l1, &l2, &l3, &bottom}) out->push_back(*s);
  }

  void render_header(const json& t, const json& model, const json& load, const std::string& phase,
                     const std::string& load_state, double progress, const std::string& url, bool frozen, int cols,
                     std::vector<std::string>* out) const {
    const bool loaded = flag(model, "loaded");
    const std::string model_id = str(model, "id");
    const std::string target = str(load, "target");
    const std::string id = !model_id.empty() ? model_id : (!target.empty() ? target : "no model");
    std::string where = paint(kBold, paint(kAccent, "FLYX")) + "  " + paint(kBold, id) + paint(kDim, "  " + url);
    static const std::map<std::string, std::string> names = {
        {"none", "No model"},       {"fetching", "Fetching"},   {"unloaded", "Unloaded"},
        {"loading", "Loading the model"}, {"idle", "Idle"},     {"prefill", "Reading the prompt"},
        {"decode", "Generating a reply"}};
    std::string status = "● " + (names.count(phase) ? names.at(phase) : phase);
    if (progress >= 0) status += " · " + num(progress * 100) + "%";
    const std::string live = frozen ? paint(kWarm, "[FROZEN]") : paint(kAccent, "[LIVE]");
    const std::string right =
        paint(phase == "idle" || phase == "unloaded" || phase == "none" ? kDim : kAccent, status) + "  " + live;
    out->push_back(where + std::string(size_t(std::max(1, cols - width(where) - width(right))), ' ') + right);

    std::string info;
    bool degraded = false;
    if (loaded) {
      const std::string dr = str(model, "drafter"), dw = str(model, "draft_weights");
      const std::string dr_str = dr.empty() || dr == "none"
                                     ? "no drafter"
                                     : dr + (dw.empty() ? "" : " (" + dw + ")") + " up to " +
                                           num(get(model, "max_width")) + " drafts";
      degraded = model.contains("degraded") && model["degraded"].is_array() && !model["degraded"].empty();
      info = str(model, "device") + " · " + str(model, "backend") + " · " + dr_str + " · " +
             str(model, "chat_format") + " chat · context " + grouped(static_cast<long long>(get(model, "context")));
    } else if (model_id.empty()) {
      info = load_state == "failed" ? "the last model did not load: " + str(load, "error")
                                    : "no model loaded: name one on the dashboard's page (" + url + "/)";
    } else {
      info = "model not loaded (it loads on the next request)";
    }
    info += " · up " + seconds(get(t["server"], "uptime_s"));
    if (degraded) {
      const std::string tag = " [VRAM-adjusted]";
      out->push_back(paint(kDim, fit(info, std::max(0, cols - int(tag.size())))) + paint(kWarm, tag));
    } else {
      out->push_back(paint(kDim, fit(info, cols)));
    }
    out->push_back(paint(kDim, repeat("─", cols)));
  }

  void render_state_bar(const json& state, const std::string& phase, double progress, const std::string& detail,
                        int cols, std::vector<std::string>* out) const {
    std::string states = paint(kBold, "Model state") + " ";
    static const std::vector<std::pair<std::string, std::string>> phases = {
        {"unloaded", "Unloaded"}, {"loading", "Loading"}, {"idle", "Idle"},
        {"prefill", "Reading"},   {"decode", "Generating"}};
    for (const auto& [key, label] : phases)
      states += " " + (key == phase ? paint(kAccent, "[" + label + "]") : paint(kDim, label));
    // Several requests take turns; the line and the bar are the one that started first.
    const int running = int(get(state, "running"));
    if (running > 1) states += "  " + paint(kAccent, "Running " + std::to_string(running));
    const int queued = int(get(state, "queued"));
    states += "  " + (queued ? paint(kWarm, "Queued " + std::to_string(queued)) : paint(kDim, "Queued 0"));
    std::string job;
    if (!str(state, "label").empty())
      job = str(state, "label") + " · session " + num(get(state, "session")) + " · " + detail + " · " +
            seconds(get(state, "elapsed_s"));
    out->push_back(states + std::string(size_t(std::max(1, cols - width(states) - width(job))), ' ') + job);
    out->push_back(bar(progress < 0 ? 0 : progress, cols, kAccent));
    out->push_back("");
  }

  void render_tiles(const json& t, const json& model, const json& state, const json& dev, const json& recent,
                    const std::string& phase, const History& h, double cpu_pct, int cols,
                    std::vector<std::string>* out) const {
    const bool loaded = flag(model, "loaded");
    const int per_row = cols >= 4 * 24 + 3 ? 4 : 2;
    const int tw = (cols - (per_row - 1)) / per_row, cw = tw - 4;
    std::vector<double> per_second;
    const json& rates = t.contains("rates") && t["rates"].is_object() ? t["rates"] : json::object();
    if (rates.contains("output_per_second") && rates["output_per_second"].is_array())
      for (const auto& v : rates["output_per_second"])
        if (v.is_number()) per_second.push_back(v.get<double>());
    const json last = recent.empty() ? json::object() : recent[0];
    std::vector<double> tpr, ttft;
    for (auto it = recent.rbegin(); it != recent.rend(); ++it) {
      tpr.push_back(get(*it, "tokens_per_round"));
      ttft.push_back(get(*it, "ttft_ms", -1));  // spark leaves a gap for an unobserved request
    }
    std::vector<Tile> tiles;
    const bool decoding = phase == "decode";
    const double speed = decoding ? get(rates, "decode_tok_s") : get(last, "decode_tok_s");
    tiles.push_back({"Speed", paint(kBold, num(speed, 1) + " tok/s"), paint(kDim, decoding ? "now" : "last reply"),
                     spark(per_second, cw, 0, kAccent)});
    const double tpr_now = decoding ? (has(rates, "tokens_per_round") ? get(rates, "tokens_per_round")
                                                                      : get(state, "tokens_per_round"))
                                    : get(last, "tokens_per_round");
    const std::string dr_tile = str(model, "drafter"), dw_tile = str(model, "draft_weights");
    const std::string dr_sub = dr_tile.empty() || dr_tile == "none"
                                   ? "no drafter"
                                   : dr_tile + (dw_tile.empty() ? "" : " · " + dw_tile) + " up to " +
                                         num(get(model, "max_width"));
    tiles.push_back({"Speculation", paint(kBold, num(tpr_now, 2) + " tokens/round"),
                     paint(kDim, (decoding ? "now · " : "last reply · ") + dr_sub),
                     spark(tpr, cw, get(model, "max_width") + 1, kAccent)});
    const double fresh = get(last, "prompt_tokens") - get(last, "reused_tokens");
    const double prefill_ms = get(last, "prefill_ms");
    const double first = get(last, "ttft_ms", -1);
    tiles.push_back({"Time to first token", paint(kBold, first < 0 ? "-" : num(first / 1000, 2) + " s"),
                     paint(kDim, recent.empty()
                                     ? ""
                                     : grouped(static_cast<long long>(fresh)) + " new · " +
                                           (prefill_ms > 0 ? grouped(static_cast<long long>(fresh * 1000 / prefill_ms)) : "-") +
                                           " tok/s"),
                     spark(ttft, cw, 0, kCool)});
    if (dev.is_object()) {
      // The device's own name: the server gives its figures with no model loaded too.
      const std::string name = !str(dev, "name").empty() ? str(dev, "name") : str(model, "device");
      const std::string short_name = name.rfind("NVIDIA GeForce ", 0) == 0 ? name.substr(15) : name;
      const std::string limits = str(dev, "clock_limits");
      tiles.push_back({"GPU load", paint(kBold, num(get(dev, "util")) + " %"),
                       paint(kDim, short_name + " · " + num(get(dev, "sm_clock_mhz")) + " MHz" +
                                       (limits.empty() || limits == "idle" ? "" : " · " + limits)),
                       spark(vec(h.util), cw, 100, kAccent)});
      const double used = get(dev, "mem_used"), total = get(dev, "mem_total");
      tiles.push_back({"VRAM", paint(kBold, gib(used) + " / " + gib(total) + " GiB"),
                       paint(kDim, loaded ? "engine " + gib(get(t["memory"], "ledger_used")) + " of " +
                                                gib(get(t["memory"], "ledger_budget")) + " GiB"
                                          : "model not loaded"),
                       bar(total > 0 ? used / total : 0, cw, kAccent)});
      const double temp = get(dev, "temp_c");
      const char* hot = temp >= 83 ? kHot : temp >= 70 ? kWarm : kAccent;
      tiles.push_back({"GPU temp", paint(kBold, num(temp) + " °C"),
                       paint(kDim, has(dev, "fan") ? "fan " + num(get(dev, "fan")) + " %" : ""),
                       bar(temp / 100, cw, hot)});
      tiles.push_back({"Power", paint(kBold, num(get(dev, "power_w")) + " W"),
                       paint(kDim, "of " + num(get(dev, "power_limit_w")) + " W · PCIe " +
                                       num(get(dev, "pcie_gen")) + " x" + num(get(dev, "pcie_width"))),
                       spark(vec(h.power), cw, get(dev, "power_limit_w"), kWarm)});
    } else {
      for (const char* title : {"GPU load", "VRAM", "GPU temp", "Power"})
        tiles.push_back({title, paint(kDim, "-"), paint(kDim, loaded ? "no device figures" : "model not loaded"), ""});
    }
    tiles.push_back({"CPU", paint(kBold, cpu_pct < 0 ? "-" : num(cpu_pct, 1) + " %"),
                     // flyx's memory as the dashboard counts it: resident, and the page-locked memory the GPU reads
                     // (process.ram), not the working set alone.
                     paint(kDim, "of a core · RAM " + mem_text(get(t["process"], "ram", get(t["process"], "rss")))),
                     spark(vec(h.cpu), cw, 0, kCool)});
    for (size_t i = 0; i < tiles.size(); i += size_t(per_row))
      draw_tiles({tiles.begin() + long(i), tiles.begin() + long(std::min(tiles.size(), i + size_t(per_row)))}, tw, out);
    out->push_back("");
  }

  std::vector<std::string> render_left_column(const json& t, const json& model, int lw) const {
    std::vector<std::string> left;
    const double ctx = get(model, "context");
    left.push_back(paint(kAccent, "Context fill") + paint(kDim, "  per session, latest first"));
    const json& sessions = t["memory"]["sessions"];
    if (sessions.empty()) left.push_back(paint(kDim, "  none"));
    for (const auto& sl : sessions) {
      const double tok = get(sl, "tokens");
      const std::string label = fit("#" + num(get(sl, "id")), 4) + fit(grouped(static_cast<long long>(tok)), 8, true) + " ";
      const std::string pct = " " + fit(num(ctx > 0 ? tok / ctx * 100 : 0) + "%", 4, true);
      left.push_back(label + bar(ctx > 0 ? tok / ctx : 0, std::max(4, lw - width(label) - width(pct) - 1), kAccent) +
                     pct);
    }
    const double ram_total = get(t["system"], "ram_total"), ram_used = ram_total - get(t["system"], "ram_available");
    const std::string ram = "System RAM  " + gib(ram_used) + " / " + gib(ram_total) + " GiB ";
    left.push_back(ram + bar(ram_total > 0 ? ram_used / ram_total : 0, std::max(4, lw - width(ram) - 1), kAccent));
    const json& tt = t["totals"];
    const double prompt_total = get(tt, "prompt_tokens");
    left.push_back(paint(kAccent, "Totals"));
    left.push_back(fit(grouped(static_cast<long long>(get(tt, "requests"))) + " requests · " +
                           grouped(static_cast<long long>(get(tt, "output_tokens"))) + " tokens out",
                       lw));
    left.push_back(fit(num(prompt_total > 0 ? get(tt, "reused_tokens") / prompt_total * 100 : 0) + "% of " +
                           grouped(static_cast<long long>(prompt_total)) + " prompt tokens reused",
                       lw));
    const double errors = get(tt, "errors");
    left.push_back(paint(errors > 0 ? kHot : kDim,
                         fit(grouped(static_cast<long long>(errors)) + " errors · " +
                                 grouped(static_cast<long long>(get(tt, "cancelled"))) + " cancelled",
                             lw)));
    left.push_back(paint(kDim,
                         fit(grouped(static_cast<long long>(get(tt, "tool_calls"))) + " tool calls · " +
                                 grouped(static_cast<long long>(get(tt, "grammar_fixes"))) + " tokens corrected",
                             lw)));
    return left;
  }

  std::vector<std::string> render_recent_table(const json& recent, int rw) const {
    struct Col {
      const char* head;
      int w;
      bool right;
    };
    static const std::vector<Col> all = {{"TIME", 8, false},  {"API", 10, false},  {"STATUS", 7, false},
                                         {"PROMPT", 7, true}, {"REUSED", 7, true}, {"OUTPUT", 6, true},
                                         {"TOK/S", 6, true},  {"TTFT", 7, true},   {"TOTAL", 7, true},
                                         {"T/RND", 5, true},  {"FIXED", 5, true}};
    std::vector<Col> shown;
    int used_w = 0;
    for (const auto& c : all)
      if (used_w + c.w + (shown.empty() ? 0 : 1) <= rw) {
        used_w += c.w + (shown.empty() ? 0 : 1);
        shown.push_back(c);
      }
    std::vector<std::string> right_col;
    right_col.push_back(paint(kAccent, "Recent requests"));
    std::string head;
    for (const auto& c : shown) head += (head.empty() ? "" : " ") + fit(c.head, c.w, c.right);
    right_col.push_back(paint(kDim, head));
    for (const auto& r : recent) {
      std::string row;
      const std::string st = str(r, "status");
      const char* sc = st == "error" ? kHot : st == "length" ? kWarm : st == "cancelled" ? kDim
                     : st == "tool_calls" ? kCool : kAccent;
      const std::string st_name = st == "stop" ? "done" : st == "tool_calls" ? "tools"
                                : st == "cancelled" ? "cancel" : st;
      for (const auto& c : shown) {
        std::string cell;
        const std::string h = c.head;
        if (h == "TIME") cell = clock_time(get(r, "finished_at"));
        else if (h == "API") cell = str(r, "label");
        else if (h == "STATUS") cell = st_name;
        else if (h == "PROMPT") cell = grouped(static_cast<long long>(get(r, "prompt_tokens")));
        else if (h == "REUSED") cell = grouped(static_cast<long long>(get(r, "reused_tokens")));
        else if (h == "OUTPUT") cell = grouped(static_cast<long long>(get(r, "output_tokens")));
        else if (h == "TOK/S") cell = num(get(r, "decode_tok_s"), 1);
        else if (h == "TTFT") {
          const double first = get(r, "ttft_ms", -1);
          cell = first < 0 ? "-" : seconds(first / 1000);
        } else if (h == "TOTAL") cell = seconds((get(r, "queue_ms") + get(r, "prefill_ms") + get(r, "decode_ms")) / 1000);
        else if (h == "T/RND") cell = num(get(r, "tokens_per_round"), 2);
        else if (h == "FIXED") cell = num(get(r, "grammar_fixes"));
        cell = fit(cell, c.w, c.right);
        row += (row.empty() ? "" : " ") + (h == "STATUS" ? paint(sc, cell) : cell);
      }
      right_col.push_back(row);
    }
    if (recent.empty()) right_col.push_back(paint(kDim, "none yet"));
    return right_col;
  }

  // The panel for `cols` x `rows` (the recent requests take the rows left).
  std::vector<std::string> render(const json& t, const std::string& url, bool frozen, const History& h, int cols,
                                  int rows, double cpu_pct) const {
    const json& model = t["model"];
    const json& state = t["state"];
    const json& dev = t["device"];
    const json& recent = t["recent"];
    const json& load = t.contains("load") && t["load"].is_object() ? t["load"] : json::object();
    const std::string load_state = str(load, "state");
    // A model being fetched or loaded, or none named yet, before what the engine does.
    const std::string phase = load_state == "fetching" || load_state == "loading" ? load_state
                              : str(model, "id").empty() ? "none" : str(state, "phase");

    double progress = -1;
    std::string detail;
    const int prompt = int(get(state, "prompt_tokens")), generated = int(get(state, "generated"));
    if (phase == "fetching" && load.contains("progress")) {
      const json& p = load["progress"];
      const double total = get(p, "total"), done = get(p, "done");
      progress = total > 0 ? done / total : 0;
      detail = str(p, "file") + " " + num(done / 1e9) + " / " + num(total / 1e9) + " GB";
    } else if (phase == "prefill" && prompt > 0) {
      progress = get(state, "prefilled") / prompt;
      detail = grouped(static_cast<long long>(get(state, "prefilled"))) + " / " + grouped(prompt) + " prompt tokens";
      if (has(state, "reused_tokens"))
        detail += " · " + grouped(static_cast<long long>(get(state, "reused_tokens"))) + " reused";
    } else if (phase == "decode") {
      const int limit = int(get(state, "max_new_tokens"));
      const int of = limit > 0 ? limit : int(get(model, "context")) - prompt;
      progress = of > 0 ? double(generated) / of : 0;
      detail = grouped(generated) + " / " + grouped(of) + " tokens";
    }

    std::vector<std::string> out;
    render_header(t, model, load, phase, load_state, progress, url, frozen, cols, &out);
    render_state_bar(state, phase, progress, detail, cols, &out);
    render_tiles(t, model, state, dev, recent, phase, h, cpu_pct, cols, &out);

    const bool side = cols >= 120;
    const int lw = side ? cols * 36 / 100 : cols;
    const int rw = side ? cols - lw - 3 : cols;
    std::vector<std::string> left = render_left_column(t, model, lw);
    std::vector<std::string> right_col = render_recent_table(recent, rw);

    // Keep the panel on the screen: the table gets the rows that are left.
    const int room = rows - int(out.size()) - (side ? 0 : int(left.size()) + 1);
    const int keep_rows = std::max(room, side ? int(left.size()) : 3);
    if (int(right_col.size()) > keep_rows) right_col.resize(size_t(keep_rows));
    if (side) {
      for (size_t i = 0; i < std::max(left.size(), right_col.size()); ++i)
        out.push_back(pad(i < left.size() ? left[i] : "", lw) + "   " + (i < right_col.size() ? right_col[i] : ""));
    } else {
      out.insert(out.end(), left.begin(), left.end());
      out.push_back("");
      out.insert(out.end(), right_col.begin(), right_col.end());
    }
    return out;
  }
};

// Diff-based painter: updates only changed rows via explicit cursor positioning (no \r\n scroll, no full-screen erase),
// wrapped in DEC synchronized output (?2026h/l), and writes nothing on unchanged ticks.
class ScreenPainter {
 public:
  void paint(const TermStyle& style, const std::vector<std::string>& lines, int cols, int rows) {
    if (rows <= 0 || cols <= 0) return;
    const bool resized = cols != cols_ || rows != rows_;
    std::vector<std::string> next(static_cast<size_t>(rows));
    for (int i = 0; i < rows && i < int(lines.size()); ++i) {
      // Leave the bottom-right cell unused so legacy Windows conhost never scrolls the viewport on paint.
      const int limit = i + 1 == rows ? std::max(1, cols - 1) : cols;
      next[size_t(i)] = style.cut(lines[size_t(i)], limit);
    }
    std::string delta;
    delta.reserve(4096);
    for (int i = 0; i < rows; ++i) {
      if (!resized && i < int(prev_.size()) && next[size_t(i)] == prev_[size_t(i)]) continue;
      delta += "\x1b[" + std::to_string(i + 1) + ";1H";
      delta += next[size_t(i)];
      delta += "\x1b[0m\x1b[K";
    }
    if (delta.empty()) return;
    std::string out;
    out.reserve(delta.size() + 16);
    out += "\x1b[?2026h";
    out += delta;
    out += "\x1b[?2026l";
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);
    prev_ = std::move(next);
    cols_ = cols;
    rows_ = rows;
  }

 private:
  std::vector<std::string> prev_;
  int cols_ = 0, rows_ = 0;
};

struct Options {
  std::string url = apps::kDefaultServerUrl, key;
  double interval = 1.0;
  bool once = false;
  double once_seconds = 1;  // --once: how long to sample before the frame
};

[[noreturn]] void usage(bool asked = false) {
  std::fprintf(asked ? stdout : stderr,
               "usage: flyx monitor [URL] [options]   (default URL http://127.0.0.1:8080)\n"
               "  --key KEY         the server's API key (default: FLYX_API_KEY)\n"
               "  --interval S      seconds between samples (default 1)\n"
               "  --once            sample for a second (--for S: S seconds), print one plain frame, exit\n"
               "keys: q quit, space freeze, r refresh\n");
  std::exit(asked ? 0 : 2);
}

}  // namespace

int run(int argc, char** argv) {
  Options o;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--key" && i + 1 < argc) o.key = argv[++i];
    else if (a == "--interval" && i + 1 < argc) o.interval = std::max(0.1, apps::number_flag("--interval", argv[++i], 0, 86400));
    else if (a == "--once") o.once = true;
    else if (a == "--for" && i + 1 < argc) o.once_seconds = apps::number_flag("--for", argv[++i], 0, 86400);
    else if (a == "-h" || a == "--help") usage(true);
    else if (!a.empty() && a[0] != '-') o.url = a.find("://") == std::string::npos ? "http://" + a : a;
    else usage();
  }
  // The client reads plain HTTP, which is what flyx serve speaks; given an https URL its library throws.
  if (o.url.rfind("http://", 0) != 0) {
    std::fprintf(stderr, "flyx monitor: %s: only http:// URLs (flyx serve speaks plain HTTP; reach one behind TLS "
                         "through a tunnel to its port)\n", o.url.c_str());
    return 1;
  }
  if (o.key.empty()) o.key = apps::api_key_from_env();
  while (!o.url.empty() && o.url.back() == '/') o.url.pop_back();
  g_stop = false;
  TermStyle style{!os::env_exists(env::kNoColor)};
  TelemetrySampler sampler(o.url, o.key);
  os::Terminal term;

  if (o.once) {
    int cols = 120, rows = 0;
    if (os::stdout_is_terminal()) {
      term.size(&cols, &rows);
      cols = std::max(apps::kMinTerminalCols, cols);
    } else {  // a file or a pipe: plain text, COLUMNS wide
      style.color = false;
      cols = os::env_int(env::kColumns, 120, apps::kMinTerminalCols, 10000);
    }
    // Samples over `seconds` (two at least: the CPU figure is a difference), then one frame.
    const int n = std::max(2, int(o.once_seconds / o.interval) + 1);
    for (int i = 0; i < n; ++i) {
      if (i) os::sleep_ms(int(o.interval * 1000));
      if (!sampler.sample()) return std::fprintf(stderr, "flyx monitor: %s\n", sampler.error().c_str()), 1;
    }
    for (const auto& line :
         style.render(sampler.telemetry(), o.url, false, sampler.history(), cols, 1000, sampler.cpu_pct()))
      std::printf("%s\n", line.c_str());
    return 0;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const bool keys = term.raw();
  AltScreen screen;
  ScreenPainter painter;
  bool frozen = false, ok = false;
  double next = 0;
  while (!g_stop) {
    if (!frozen && os::now_ms() >= next) {
      ok = sampler.sample();
      next = os::now_ms() + o.interval * 1000;
    }
    int cols = 0, rows = 0;
    term.size(&cols, &rows);
    cols = std::max(apps::kMinTerminalCols, cols);
    std::vector<std::string> lines;
    if (ok || !sampler.telemetry().is_null()) {
      try {
        lines = style.render(sampler.telemetry(), o.url, frozen, sampler.history(), cols, rows - 3, sampler.cpu_pct());
      } catch (const std::exception& e) {
        lines = {style.paint(kHot, fit(std::string("monitor render error: ") + e.what(), cols))};
      }
    }
    if (!sampler.error().empty()) lines.insert(lines.begin(), style.paint(kHot, fit(sampler.error() + " (retrying)", cols)));
    lines.push_back("");
    lines.push_back(style.paint(kDim, fit("q quit   space freeze   r refresh   ·   read only: GET " + o.url +
                                              "/v1/telemetry every " + num(o.interval, 1) + " s",
                                          cols)));
    painter.paint(style, lines, cols, rows);
    const int k = keys ? term.key(200) : (os::sleep_ms(200), -1);
    if (k == 'q' || k == 'Q' || k == 3) break;
    if (k == ' ') frozen = !frozen;
    if (k == 'r' || k == 'R') next = 0;
  }
  return 0;
}

void serve_dashboard(ServeView* v) {
  const TermStyle style{!os::env_exists(env::kNoColor)};
  TelemetrySampler sampler(v->url, v->key);
  os::Terminal term;
  const bool keys = term.raw();
  AltScreen screen;
  ScreenPainter painter;

  constexpr int kLogRows = 4;  // the newest log lines under the live panel
  constexpr double kIntervalMs = 1000.0;
  const double started = os::now_ms();
  bool frozen = false, ok = false, whole_log = false;
  double next = 0, quit_armed_until = 0, notice_until = 0;
  std::string notice;
  auto recent_logs = [&](size_t n) { return v->recent_logs ? v->recent_logs(n) : std::vector<std::string>{}; };
  while (!v->quit) {
    const double now = os::now_ms();
    const bool live = v->listening;
    if (live && !frozen && now >= next) {
      ok = sampler.sample();
      next = now + kIntervalMs;
    }
    int cols = 0, rows = 0;
    term.size(&cols, &rows);
    cols = std::max(apps::kMinTerminalCols, cols);
    std::vector<std::string> lines;
    if (!live || whole_log || (!ok && sampler.telemetry().is_null())) {
      // The model loading (or the log asked for): a header, then as much of the log as the screen holds.
      const std::string where = style.paint(kBold, style.paint(kAccent, "FLYX")) + "  " + style.paint(kBold, v->model);
      const std::string status = !live ? style.paint(kAccent, "● Loading the model · " + seconds((now - started) / 1000))
                                       : style.paint(kAccent, whole_log ? "● Log" : "● Starting");
      lines.push_back(where + std::string(size_t(std::max(1, cols - width(where) - width(status))), ' ') + status);
      lines.push_back(style.paint(kDim, fit("serving at " + v->url + " · the web page " + v->page, cols)));
      lines.push_back(style.paint(kDim, repeat("─", cols)));
      for (const auto& l : recent_logs(size_t(std::max(1, rows - 5)))) lines.push_back(fit(l, cols));
    } else {
      try {
        lines = style.render(sampler.telemetry(), v->url, frozen, sampler.history(), cols, rows - 4 - kLogRows,
                             sampler.cpu_pct());
      } catch (const std::exception& e) {
        lines = {style.paint(kHot, fit(std::string("dashboard render error: ") + e.what(), cols))};
      }
      if (!sampler.error().empty()) lines.insert(lines.begin(), style.paint(kHot, fit(sampler.error() + " (retrying)", cols)));
      lines.push_back(style.paint(kAccent, "Log") + style.paint(kDim, "  newest last · l: all of it"));
      for (const auto& l : recent_logs(kLogRows)) lines.push_back(style.paint(kDim, fit(l, cols)));
    }
    // Keep the footer on the screen.
    if (int(lines.size()) > rows - 2) lines.resize(size_t(std::max(0, rows - 2)));
    lines.push_back("");
    if (now < quit_armed_until) {
      lines.push_back(style.paint(kHot, fit("q again stops the server; any other key keeps it running", cols)));
    } else if (now < notice_until) {
      lines.push_back(style.paint(kAccent, fit(notice, cols)));
    } else {
      lines.push_back(style.paint(kDim, fit("q stop the server   o open the web page (" + v->page + ")   l log   space freeze"
                                            "   r refresh",
                                            cols)));
    }
    painter.paint(style, lines, cols, rows);

    const int k = keys ? term.key(200) : (os::sleep_ms(200), -1);
    if (k < 0) continue;
    if (k == 'q' || k == 'Q') {
      if (os::now_ms() < quit_armed_until) break;
      quit_armed_until = os::now_ms() + 3000;
      continue;
    }
    if (k == 3) break;  // Ctrl-C where it arrives as a key
    quit_armed_until = 0;
    if (k == 'o' || k == 'O') {
      notice = v->open_page && v->open_page() ? "opened " + v->page + " in the browser"
                                              : "no browser here: open " + v->page + " in one";
      notice_until = os::now_ms() + 4000;
    }
    if (k == 'l' || k == 'L') whole_log = !whole_log;
    if (k == ' ') frozen = !frozen;
    if (k == 'r' || k == 'R') next = 0;
  }
  v->quit = true;
}

}  // namespace flyx::monitor
