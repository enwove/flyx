// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
//
// CUDA backend: implements the backend ABI on one GPU.
//
// * Weights are uploaded once; projections that share an input are stored
//   side by side so one GEMM computes them (q|k|v, gate|up, GDN in-proj).
// * A round is a fixed sequence of kernels whose varying inputs live in
//   DevParams on the device.  It is captured into a CUDA graph per sequence
//   on first use and replayed afterwards (option graphs=0 disables this).
// * Recurrent state uses FLYX_ROUND_SLOTS slots; each round keeps its per-row inputs, so
//   committing the accepted rows and rolling back inside the last rounds are
//   both folds of a saved state, never a recompute of the prompt.
#include "backends/cuda/backend.h"
#include "backends/cuda/runtime/cold_experts.h"
#include "backends/cuda/regions/layers.h"
#include "core/interfaces/memory_policy.h"

#include <charconv>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace flyx::cuda {

thread_local std::string g_error;

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw CudaError(std::string(what) + ": " + cudaGetErrorString(e));
}

// An ask in MiB, or in KiB below one (a W12 head's last parts are a few KB), a decimal below 10: not "0 MiB".
static std::string asked(size_t bytes) {
  const bool kib = bytes < (size_t(1) << 20);
  const double n = kib ? bytes / 1024.0 : bytes / 1048576.0;
  char s[32];
  std::snprintf(s, sizeof(s), "%.*f %s", n < 10 ? 1 : 0, n, kib ? "KiB" : "MiB");
  return s;
}

#ifdef FLYX_TEST_DEVICE_ALLOC
// Only the loader test executable supplies these. The production module always calls the driver.
void test_device_room(size_t bytes);
cudaError_t test_cuda_malloc(void** ptr, size_t bytes);
#endif

void device_room(size_t bytes, size_t floor) {
#ifdef FLYX_TEST_DEVICE_ALLOC
  test_device_room(bytes);
#endif
#if defined(_WIN32)
  // Windows' driver places an allocation past what its video memory manager keeps free in host memory instead of
  // refusing it: refused here first, against NVML's free memory (gpu_memory) less the floor for allocations that size.
  size_t free = 0, total = 0;
  if (!gpu_memory(&free, &total)) CK(cudaMemGetInfo(&free, &total));
  if (free < floor || bytes > free - floor) {
    char m[240];
    std::snprintf(m, sizeof(m), "out of GPU memory: %s asked, %.0f MiB free, of which flyx keeps %.0f MiB (past "
                  "70 to 630 MiB free, by the allocation's size, Windows puts allocations in host memory)",
                  asked(bytes).c_str(), free / 1048576.0, floor / 1048576.0);
    throw OutOfDeviceMemory(m);
  }
#else
  (void)bytes;
  (void)floor;
#endif
}

void* device_malloc(size_t bytes) {
  device_room(bytes, kPagedFloor);
  char m[240];
  // The driver refuses elsewhere; on Windows it may too (another program between device_room and the allocation).
  void* p = nullptr;
#ifdef FLYX_TEST_DEVICE_ALLOC
  const cudaError_t e = test_cuda_malloc(&p, bytes);
#else
  const cudaError_t e = cudaMalloc(&p, bytes);
#endif
  if (e == cudaErrorMemoryAllocation) {
    cudaGetLastError();  // not sticky: clear it
    std::snprintf(m, sizeof(m), "out of GPU memory: %s asked, the driver refused it", asked(bytes).c_str());
    throw OutOfDeviceMemory(m);
  }
  CK(e);
  return p;
}

int fail(const std::string& m) {
  g_error = m;
  return -1;
}

int fail_oom(const std::string& m) {
  g_error = m;
  return FLYX_ERR_OOM;
}

int no_sequence(const char* call) {
  return fail(std::string(call) + ": no sequence (seq_new failed or was not called)");
}

namespace {

bool parse_flag_val(const std::string& key, const std::string& val, bool* f, std::string* err) {
  if (val != "0" && val != "1") return *err = key + " takes 0 or 1, not \"" + val + "\"", false;
  *f = val == "1";
  return true;
}

bool parse_whole_val(const std::string& key, const std::string& val, int* v, std::string* err) {
  char* end = nullptr;
  const long long n = std::strtoll(val.c_str(), &end, 10);
  if (val.empty() || *end || n < 0 || n > INT_MAX)
    return *err = key + " takes a whole number, not \"" + val + "\"", false;
  *v = int(n);
  return true;
}

bool check_choice(const std::string& key, const std::string& val, std::initializer_list<const char*> allowed,
                  std::string* err) {
  for (const char* a : allowed)
    if (val == a) return true;
  *err = "bad value \"" + val + "\" for " + key;
  return false;
}

struct OptionSpec {
  const char* name;
  std::function<bool(const std::string& key, const std::string& val, BackendOptions* o, std::string* err)> parse;
  std::function<std::string(const BackendOptions& o, const BackendOptions& def)> describe;
};

template <class Member>
OptionSpec flag_opt(const char* name, Member ptr) {
  return {
      name,
      [ptr](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
        return parse_flag_val(key, val, &(o->*ptr), err);
      },
      [name, ptr](const BackendOptions& o, const BackendOptions& def) -> std::string {
        return o.*ptr != def.*ptr ? std::string(name) + (o.*ptr ? "=1" : "=0") : "";
      },
  };
}

OptionSpec kernel_flag_opt(const char* name, bool KernelOptions::*ptr) {
  return {
      name,
      [ptr](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
        return parse_flag_val(key, val, &(o->kernels.*ptr), err);
      },
      [name, ptr](const BackendOptions& o, const BackendOptions& def) -> std::string {
        return o.kernels.*ptr != def.kernels.*ptr ? std::string(name) + (o.kernels.*ptr ? "=1" : "=0") : "";
      },
  };
}

OptionSpec bool_choice_opt(const char* name, bool BackendOptions::*ptr, const char* on_name, const char* off_name) {
  return {
      name,
      [ptr, on_name, off_name](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
        if (!check_choice(key, val, {on_name, off_name}, err)) return false;
        o->*ptr = val == on_name;
        return true;
      },
      [name, ptr, on_name, off_name](const BackendOptions& o, const BackendOptions& def) -> std::string {
        return o.*ptr != def.*ptr ? std::string(name) + "=" + (o.*ptr ? on_name : off_name) : "";
      },
  };
}

template <class T>
OptionSpec kernel_bool_choice_opt(const char* name, T KernelOptions::*ptr, const char* on_name,
                                  const char* off_name) {
  return {
      name,
      [ptr, on_name, off_name](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
        if (!check_choice(key, val, {on_name, off_name}, err)) return false;
        o->kernels.*ptr = static_cast<T>(val == on_name);
        return true;
      },
      [name, ptr, on_name, off_name](const BackendOptions& o, const BackendOptions& def) -> std::string {
        return o.kernels.*ptr != def.kernels.*ptr ? std::string(name) + "=" + (o.kernels.*ptr ? on_name : off_name)
                                                  : "";
      },
  };
}

OptionSpec kernel_whole_opt(const char* name, int KernelOptions::*ptr, int min_val = 0, int max_val = INT_MAX) {
  return {
      name,
      [ptr, min_val, max_val](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
        if (!parse_whole_val(key, val, &(o->kernels.*ptr), err)) return false;
        if (o->kernels.*ptr < min_val || o->kernels.*ptr > max_val) {
          *err = key + " takes " + std::to_string(min_val) +
                 (max_val == INT_MAX ? " or more" : " to " + std::to_string(max_val));
          return false;
        }
        return true;
      },
      [name, ptr](const BackendOptions& o, const BackendOptions& def) -> std::string {
        return o.kernels.*ptr != def.kernels.*ptr ? std::string(name) + "=" + std::to_string(o.kernels.*ptr) : "";
      },
  };
}

OptionSpec kernel_int_choice_opt(const char* name, int KernelOptions::*ptr, std::initializer_list<const char*> allowed) {
  std::vector<const char*> choices(allowed);
  return {
      name,
      [ptr, choices](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
        for (const char* a : choices)
          if (val == a) {
            o->kernels.*ptr = std::stoi(val);
            return true;
          }
        *err = "bad value \"" + val + "\" for " + key;
        return false;
      },
      [name, ptr](const BackendOptions& o, const BackendOptions& def) -> std::string {
        return o.kernels.*ptr != def.kernels.*ptr ? std::string(name) + "=" + std::to_string(o.kernels.*ptr) : "";
      },
  };
}

OptionSpec whole_opt(const char* name, int BackendOptions::*field) {
  return {name,
          [field](const std::string& key, const std::string& value, BackendOptions* options, std::string* error) {
            return parse_whole_val(key, value, &(options->*field), error);
          },
          [name, field](const BackendOptions& options, const BackendOptions& defaults) {
            return options.*field == defaults.*field ? std::string{} : std::string(name) + "=" + std::to_string(options.*field);
          }};
}

const std::vector<OptionSpec>& option_specs() {
  static const std::vector<OptionSpec> specs = {
      whole_opt("gpu_layers", &BackendOptions::gpu_layers),
      whole_opt("gpu_experts", &BackendOptions::gpu_experts),
      whole_opt("expert_prefill_min", &BackendOptions::expert_prefill_min),
      whole_opt("expert_threads", &BackendOptions::expert_threads),
      flag_opt("profile", &BackendOptions::profile),
      flag_opt("graphs", &BackendOptions::graphs),
      flag_opt("dv_conditional", &BackendOptions::dv_conditional),
      flag_opt("fusion", &BackendOptions::fusion),
      kernel_flag_opt("gdn_round", &KernelOptions::gdn_round),
      kernel_flag_opt("stacks", &KernelOptions::stacks),
      kernel_flag_opt("round_glu", &KernelOptions::round_glu),
      {"moe_dp4a",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (val == moe_dp4a_name(-1)) {
           o->kernels.moe_dp4a = -1;
           return true;
         }
         for (int i = 0; i < kMoeDp4aConfigCount; ++i) {
           if (val != moe_dp4a_name(i)) continue;
           // Every type, op kind and row band must compile it: a release build compiles only fused gate|up
           // configurations its tables reach, so non-original overrides require FLYX_TUNE_CANDIDATES=ON.
           for (int dtype : kMoeDp4aTypes)
             for (bool fused : {false, true})
               for (int r : {1, 2})
                 if (!moe_dp4a_compiled(dtype, i, fused, r)) {
                   *err = key + "=" + val + " is not compiled in this build (configure with -DFLYX_TUNE_CANDIDATES=ON)";
                   return false;
                 }
           o->kernels.moe_dp4a = i;
           return true;
         }
         *err = "bad value \"" + val + "\" for " + key;
         return false;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         if (o.kernels.moe_dp4a == def.kernels.moe_dp4a) return "";
         return std::string("moe_dp4a=") + moe_dp4a_name(o.kernels.moe_dp4a);
       }},
      kernel_flag_opt("side_stream", &KernelOptions::side_stream),
      kernel_int_choice_opt("lora_lanes", &KernelOptions::lora_lanes, {"0", "1", "2", "4", "8"}),
      kernel_flag_opt("lora_glu", &KernelOptions::lora_glu),
      {"attention",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (!check_choice(key, val, {"auto", "split", "head"}, err)) return false;
         o->kernels.attention = val == "split" ? kAttnSplit : val == "head" ? kAttnPerHead : kAttnAuto;
         return true;
       },
       [](const BackendOptions& o, const BackendOptions&) -> std::string {
         return o.kernels.attention != kAttnAuto
                    ? (o.kernels.attention == kAttnSplit ? "attention=split" : "attention=head")
                    : "";
       }},
      kernel_bool_choice_opt("gemm", &KernelOptions::row_gemms, "rows", "auto"),
      kernel_bool_choice_opt("rows_attention", &KernelOptions::rows_mma, "mma", "cores"),
      kernel_whole_opt("rows_attention_from", &KernelOptions::rows_mma_from),
      kernel_int_choice_opt("rows_attention_parts", &KernelOptions::rows_mma_parts, {"1", "2", "3"}),
      kernel_bool_choice_opt("rows_attention_q", &KernelOptions::rows_mma_q8, "fp8", "16"),
      {"pdl",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         o->pdl_given = true;
         return parse_flag_val(key, val, &o->kernels.pdl, err);
       },
       [](const BackendOptions& o, const BackendOptions&) -> std::string {
         return (o.kernels.pdl || o.pdl_given) ? (o.kernels.pdl ? "pdl=1" : "pdl=0") : "";
       }},
      kernel_flag_opt("head_fp8", &KernelOptions::head_fp8),
      kernel_flag_opt("head_cert", &KernelOptions::head_cert),
      kernel_flag_opt("head_w12", &KernelOptions::head_w12),
      {"wait",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (!check_choice(key, val, {"poll", "spin", "auto"}, err)) return false;
         o->wait = val == "spin" ? WaitMode::spin : val == "poll" ? WaitMode::poll : WaitMode::automatic;
         return true;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         if (o.wait == def.wait) return "";
         return o.wait == WaitMode::spin ? "wait=spin" : o.wait == WaitMode::poll ? "wait=poll" : "wait=auto";
       }},
      {"draft_weights",
       [](const std::string&, const std::string& val, BackendOptions* o, std::string* err) {
         return interfaces::parse_draft_weights(val, &o->draft_weights, nullptr, err);
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         return o.draft_weights != def.draft_weights
                    ? std::string("draft_weights=") + interfaces::draft_weights_name(o.draft_weights)
                    : "";
       }},
      kernel_int_choice_opt("act_group", &KernelOptions::act_group, {"32", "64", "128", "256"}),
      kernel_int_choice_opt("act_fp4", &KernelOptions::act_fp4, {"0", "1", "2"}),
      bool_choice_opt("draft_head", &BackendOptions::full_draft_head, "full", "pick"),
      {"eagle4_tree",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         // STEPS 0 (by the width) or at least the trunk: a shallower tree would cut the chain its trunk is, whose
         // acceptance the controller counts to its 7th draft (Model::block_drafts)
         int beam = 0, steps = 0;
         const char *p = val.data(), *end = p + val.size();
         auto r = std::from_chars(p, end, beam);
         bool ok = r.ec == std::errc() && r.ptr < end && *r.ptr == 'x';
         if (ok) {
           r = std::from_chars(r.ptr + 1, end, steps);
           ok = r.ec == std::errc() && r.ptr == end;
         }
         if (!ok || beam < 1 || beam > flyx::cuda::kEagleBeamMax || (steps != 0 && steps < flyx::cuda::kEagleTrunk) ||
             int64_t(steps) * beam + 1 > flyx::cuda::kEagleSlots) {
           *err = "bad value \"" + val + "\" for " + key + " (BEAMxSTEPS: BEAM 1.." +
                  std::to_string(flyx::cuda::kEagleBeamMax) + ", STEPS 0 or " +
                  std::to_string(flyx::cuda::kEagleTrunk) + " or more, 1 + BEAM * STEPS <= " +
                  std::to_string(flyx::cuda::kEagleSlots) + ")";
           return false;
         }
         o->eagle4_beam = beam;
         o->eagle4_steps = steps;
         return true;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         if (o.eagle4_beam == def.eagle4_beam && o.eagle4_steps == def.eagle4_steps) return "";
         return "eagle4_tree=" + std::to_string(o.eagle4_beam) + "x" + std::to_string(o.eagle4_steps);
       }},
      bool_choice_opt("draft_kv", &BackendOptions::draft_kv16, "16", "fp8"),
      bool_choice_opt("blocks", &BackendOptions::packed_blocks, "packed", "bf16"),
      {"embed",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (!check_choice(key, val, {"auto", "host", "gpu"}, err)) return false;
         o->embed = val == "host" ? 1 : val == "gpu" ? 2 : 0;
         return true;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         return o.embed != def.embed ? std::string("embed=") + (o.embed == 1 ? "host" : o.embed == 2 ? "gpu" : "auto")
                                     : "";
       }},
      {"vision",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (!check_choice(key, val, {"disk", "file", "host", "cpu", "gpu", "auto"}, err)) return false;
         o->vision = (val == "disk" || val == "file")   ? FLYX_VISION_DEV_DISK
                     : (val == "host" || val == "cpu")  ? FLYX_VISION_DEV_HOST
                     : val == "gpu"                     ? FLYX_VISION_DEV_GPU
                                                        : FLYX_VISION_DEV_AUTO;
         return true;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         if (o.vision == def.vision) return "";
         return std::string("vision=") +
                (o.vision == FLYX_VISION_DEV_DISK   ? "disk"
                 : o.vision == FLYX_VISION_DEV_HOST ? "host"
                 : o.vision == FLYX_VISION_DEV_GPU  ? "gpu"
                                                    : "auto");
       }},
      {"vision_quant",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (!check_choice(key, val, {"exact", "bf16", "q8", "q8_0", "nvfp4", "fp4"}, err)) return false;
         o->vision_quant = (val == "q8" || val == "q8_0")       ? FLYX_VISION_QUANT_Q8
                           : (val == "nvfp4" || val == "fp4")   ? FLYX_VISION_QUANT_NVFP4
                                                                : FLYX_VISION_QUANT_EXACT;
         return true;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         if (o.vision_quant == def.vision_quant) return "";
         return std::string("vision_quant=") +
                (o.vision_quant == FLYX_VISION_QUANT_Q8     ? "q8"
                 : o.vision_quant == FLYX_VISION_QUANT_NVFP4 ? "nvfp4"
                                                             : "exact");
       }},
      {"wide",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (!check_choice(key, val, {"auto", "0", "1"}, err)) return false;
         o->wide = val == "0" ? 1 : val == "1" ? 2 : 0;
         return true;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         return o.wide != def.wide ? std::string("wide=") + (o.wide == 1 ? "0" : o.wide == 2 ? "1" : "auto") : "";
       }},
      {"tuning",
       [](const std::string&, const std::string& val, BackendOptions* o, std::string* err) {
         o->tuning_defaults = val == "defaults";
         o->tuning_table = val == "table" || val == "defaults" ? "" : val;
         if (val.empty()) {
           *err = "tuning: table, defaults or a table's name";
           return false;
         }
         return true;
       },
       [](const BackendOptions& o, const BackendOptions&) -> std::string {
         if (o.tuning_defaults) return "tuning=defaults";
         if (!o.tuning_table.empty()) return "tuning=" + o.tuning_table;
         return "";
       }},
      kernel_flag_opt("log_shapes", &KernelOptions::log_shapes),
      kernel_whole_opt("strata_denom", &KernelOptions::strata_denom),  // 0: no pairs refined (the copy alone)
      kernel_whole_opt("strata_least", &KernelOptions::strata_least, 1),
      kernel_whole_opt("strata_recent", &KernelOptions::strata_recent),
      kernel_whole_opt("strata_most", &KernelOptions::strata_most),
      kernel_whole_opt("strata_chunk", &KernelOptions::strata_chunk, 1),
      {"strata_hot",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (!parse_whole_val(key, val, &o->kernels.strata_hot, err)) return false;
         const int h = o->kernels.strata_hot;
         if (h != 0 && (h < 32 || h > (1 << 20) || (h & (h - 1)))) {
           *err = key + " takes 0 or a power of two from 32";
           return false;
         }
         return true;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         return o.kernels.strata_hot != def.kernels.strata_hot ? "strata_hot=" + std::to_string(o.kernels.strata_hot)
                                                               : "";
       }},
      {"strata_from",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         return parse_whole_val(key, val, &o->kernels.strata_from, err);
       },
       [](const BackendOptions& o, const BackendOptions&) -> std::string {
         return o.kernels.strata_from >= 0 ? "strata_from=" + std::to_string(o.kernels.strata_from) : "";
       }},
      kernel_bool_choice_opt("strata_grid", &KernelOptions::strata_grid, "normal", "range"),
      kernel_bool_choice_opt("strata_scan", &KernelOptions::strata_mma, "mma", "cores"),
      kernel_flag_opt("stage_ahead", &KernelOptions::stage_ahead),
      flag_opt("stage_dma", &BackendOptions::stage_dma),
      flag_opt("strata_compact", &BackendOptions::strata_compact),
      {"strata_copy_slots",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (!parse_whole_val(key, val, &o->strata_copy_slots, err)) return false;
         if (o->strata_copy_slots != 2 && o->strata_copy_slots != 3) {
           *err = key + " takes 2 or 3";
           return false;
         }
         return true;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         return o.strata_copy_slots != def.strata_copy_slots
                    ? "strata_copy_slots=" + std::to_string(o.strata_copy_slots)
                    : "";
       }},
      kernel_whole_opt("stage_blocks", &KernelOptions::stage_blocks, 1),
      {"stage_window",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (!parse_whole_val(key, val, &o->kernels.stage_window, err)) return false;
         if (o->kernels.stage_window < kFlashWindowAlign || o->kernels.stage_window % kFlashWindowAlign) {
           *err = key + " takes a multiple of " + std::to_string(kFlashWindowAlign);
           return false;
         }
         return true;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         return o.kernels.stage_window != def.kernels.stage_window
                    ? "stage_window=" + std::to_string(o.kernels.stage_window)
                    : "";
       }},
      kernel_whole_opt("mtp_window", &KernelOptions::mtp_window),
      kernel_whole_opt("mtp_copy", &KernelOptions::mtp_copy),
      kernel_whole_opt("mtp_sink", &KernelOptions::mtp_sink),
      kernel_flag_opt("mtp_ring", &KernelOptions::mtp_ring),
      kernel_flag_opt("dflash_ring", &KernelOptions::dflash_ring),
      // Inkling-2's rounds (kEnvAliases: the environment spellings they had before they were options).
      kernel_int_choice_opt("sc_cluster", &KernelOptions::sc_cluster, {"1", "4", "8"}),
      kernel_flag_opt("moe_vec", &KernelOptions::moe_vec),
      {"warm",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (!parse_whole_val(key, val, &o->kernels.warm, err)) return false;
         if (o->kernels.warm & ~kWarmAll) {
           *err = key + " takes a sum of 1, 2, 4 and 16";
           return false;
         }
         return true;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         return o.kernels.warm != def.kernels.warm ? "warm=" + std::to_string(o.kernels.warm) : "";
       }},
      kernel_int_choice_opt("expert_prefetch", &KernelOptions::expert_prefetch, {"0", "1", "2"}),
      kernel_whole_opt("guess", &KernelOptions::guess, 0, 15),
      kernel_whole_opt("guess_k", &KernelOptions::guess_k),
      kernel_flag_opt("side_loop_msg", &KernelOptions::side_loop_msg),
      kernel_flag_opt("side_companion", &KernelOptions::side_companion),
      kernel_flag_opt("side_kv", &KernelOptions::side_kv),
      {"pageable_kv",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         if (!check_choice(key, val, {"auto", "on", "off"}, err)) return false;
         o->pageable_kv = val == "on" ? PageableKvMode::on : val == "off" ? PageableKvMode::off : PageableKvMode::automatic;
         return true;
       },
       [](const BackendOptions& o, const BackendOptions& def) -> std::string {
         if (o.pageable_kv == def.pageable_kv) return "";
         return o.pageable_kv == PageableKvMode::on ? "pageable_kv=on"
                : o.pageable_kv == PageableKvMode::off ? "pageable_kv=off"
                                                       : "pageable_kv=auto";
       }},
      {"pinned_host_mb",
       [](const std::string& key, const std::string& val, BackendOptions* o, std::string* err) {
         return parse_whole_val(key, val, &o->pinned_host_mb, err);
       },
       [](const BackendOptions& o, const BackendOptions&) -> std::string {
         return o.pinned_host_mb >= 0 ? "pinned_host_mb=" + std::to_string(o.pinned_host_mb) : "";
       }},
  };
  return specs;
}

// The environment spellings some options had while the kernels that take them read them at first use, FLYX_<NAME>
// for option <name> (docs/server.md).  resolve_options reads them at every load, before the options given, which
// override them; unset, empty or not an integer (the whole value): the option keeps its default.  An integer keeps
// the meaning it had there, among the option's values.
struct EnvAlias {
  const char* env;
  const char* option;
  void (*set)(KernelOptions* k, int v);
};
const EnvAlias kEnvAliases[] = {
    {"FLYX_SC_CLUSTER", "sc_cluster", [](KernelOptions* k, int v) { k->sc_cluster = v == 4 || v == 8 ? v : 1; }},
    {"FLYX_MOE_VEC", "moe_vec", [](KernelOptions* k, int v) { k->moe_vec = v != 0; }},
    {"FLYX_WARM", "warm", [](KernelOptions* k, int v) { k->warm = v & kWarmAll; }},
    {"FLYX_EXPERT_PREFETCH", "expert_prefetch",
     [](KernelOptions* k, int v) { k->expert_prefetch = v == 0 || v == 2 ? v : 1; }},
    {"FLYX_GUESS", "guess", [](KernelOptions* k, int v) { k->guess = v > 0 ? v & 15 : 0; }},
    {"FLYX_GUESS_K", "guess_k", [](KernelOptions* k, int v) { k->guess_k = std::max(v, 0); }},
    {"FLYX_SIDE_LOOP_MSG", "side_loop_msg", [](KernelOptions* k, int v) { k->side_loop_msg = v != 0; }},
    {"FLYX_SIDE_COMPANION", "side_companion", [](KernelOptions* k, int v) { k->side_companion = v != 0; }},
    {"FLYX_SIDE_KV", "side_kv", [](KernelOptions* k, int v) { k->side_kv = v != 0; }},
};

}  // namespace

void apply_env_aliases(BackendOptions* o, EnvLookup lookup) {
  for (const EnvAlias& a : kEnvAliases) {
    const char* text = lookup ? lookup(a.env) : std::getenv(a.env);
    if (!text || !*text) continue;
    const char* end = text + std::strlen(text);
    int v = 0;
    const auto [stop, ec] = std::from_chars(text, end, v);
    if (ec != std::errc{} || stop != end) {
      std::fprintf(stderr, "flyx-cuda: %s=\"%s\" is not an integer: ignored (option %s keeps its default)\n", a.env,
                   text, a.option);
      continue;
    }
    a.set(&o->kernels, v);
  }
}

bool resolve_options(const char* text, BackendOptions* o, std::string* err, EnvLookup lookup) {
  apply_env_aliases(o, lookup);
  return parse_options(text, o, err);
}

bool parse_options(const char* text, BackendOptions* o, std::string* err) {
  const auto& specs = option_specs();
  std::string s = text ? text : "";
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    const std::string kv = s.substr(i, j - i);
    i = j + 1;
    if (kv.empty()) continue;
    const size_t eq = kv.find('=');
    const std::string key = kv.substr(0, eq), val = eq == std::string::npos ? "" : kv.substr(eq + 1);
    const OptionSpec* found = nullptr;
    for (const OptionSpec& spec : specs)
      if (key == spec.name) {
        found = &spec;
        break;
      }
    if (!found) return *err = "unknown option \"" + key + "\"", false;
    if (!found->parse(key, val, o, err)) return false;
  }
  if (o->profile) o->graphs = false;  // a timed node needs its kernels launched one by one
  return true;
}

// Whether the model's NVFP4 weights take NVFP4 inputs in passes of more than 16 rows (FLYX_ACT_NVFP4: the W4A4
// kernels): all of them or none, on a device that has those kernels, every K a multiple of 128 (their tile).
// A contract this backend cannot compute fails here rather than turning into another one.
bool nvfp4_inputs(const flyx_model_desc& desc, bool* on, std::string* err) {
  int with = 0, without = 0;
  for (int i = 0; i < desc.n_weights; ++i) {
    const flyx_weight& w = desc.weights[i];
    if (!w.t || (w.t->dtype != FLYX_DT_NVFP4 && w.t->dtype != FLYX_DT_GGML_NVFP4)) continue;
    if (w.activations != FLYX_ACT_NVFP4) {
      if (w.t->ndim == 2) ++without;  // 3D MoE experts always use bf16 activations (FLYX_ACT_FORMAT)
      continue;
    }
    ++with;
    if (w.t->ndim != 2 || w.t->shape[1] % 128)
      return *err = std::string(w.t->name) + ": NVFP4 inputs need K a multiple of 128 (use --activations exact)", false;
  }
  if (with && without) return *err = "NVFP4 weights that round their inputs apart: one model, one way", false;
  if (with && !gemm_w4a4_supported())
    return *err = "the model's NVFP4 weights take NVFP4 inputs, which this device or build cannot compute (use "
                  "--activations exact)", false;
  *on = with > 0;
  return true;
}

std::string describe(const BackendOptions& o) {
  const BackendOptions def{};
  std::string s;
  for (const OptionSpec& spec : option_specs()) {
    const std::string item = spec.describe(o, def);
    if (!item.empty()) s += (s.empty() ? "" : ",") + item;
  }
  return s.empty() ? "defaults" : s;
}


int probe(flyx_device_info* out) {
  *out = {};
  int driver = 0;
  cudaDriverGetVersion(&driver);
  const int need = CUDART_VERSION / 1000 * 1000;  // any driver of this CUDA major version (minor version compatibility)
  if (driver == 0) return fail("no NVIDIA driver found");
  if (driver < need)
    return fail("the NVIDIA driver supports CUDA " + std::to_string(driver / 1000) + "." +
                std::to_string(driver % 1000 / 10) + "; this build needs CUDA " + std::to_string(need / 1000) +
                (need == 13000 ? " (driver 580 or newer)" : " or newer"));
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  // A CUDA forward-compatibility driver (a container's libcuda newer than the kernel driver: the version above is its)
  // runs on data-center GPUs only: an RTX 4080 SUPER with kernel driver 570 under a CUDA 13 image said only "forward
  // compatibility was attempted on non supported HW".
  if (e == cudaErrorCompatNotSupportedOnDevice)
    return fail("the NVIDIA kernel driver is older than CUDA " + std::to_string(need / 1000) + " needs" +
                (need == 13000 ? " (580 or newer)" : "") +
                ", and the CUDA forward-compatibility driver in use runs on data-center GPUs only: update the driver");
  if (e != cudaSuccess || n == 0) return fail(std::string("no CUDA device: ") + cudaGetErrorString(e));
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return fail("cannot query device 0");
  if (prop.major < 8) return fail(std::string(prop.name) + ": needs compute capability 8.0 or newer");
  if (!device_supported())
    return fail(std::string(prop.name) + " (sm_" + std::to_string(prop.major) + std::to_string(prop.minor) +
                "): this build of the CUDA backend has no code for it (FLYX_CUDA_ARCHS)");
  remember_device(0);
  std::snprintf(out->name, sizeof(out->name), "%.95s", prop.name);
  out->sm_major = prop.major;
  out->sm_minor = prop.minor;
  out->sm_count = prop.multiProcessorCount;
  out->fp4_inputs = gemm_w4a4_supported() ? 1 : 0;  // the FP4 tensor cores of sm_120, and a build with them
  size_t fr = 0, tot = 0;
  if (!gpu_memory(&fr, &tot)) return fail("cannot query device memory");
  out->mem_free = fr;
  out->mem_total = tot;
  out->reserve_floor = device_reserve_floor();
  return 0;
}

int open_model(const flyx_model_desc* desc, const flyx_open_options* opts, flyx_bmodel** out) {
  auto m = std::make_unique<flyx_bmodel>();
  m->verbose = opts ? opts->verbose : 0;
  m->kv_fp8 = opts && opts->kv_fp8;
  {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::string err;
    if (opts) {
      if (opts->lock_auto_from) m->opt.kernels.strata_from = opts->auto_from;
      m->opt.vision = opts->vision_device;
      m->opt.vision_quant = opts->vision_quant;
      if (opts->draft_weights == FLYX_DRAFT_WEIGHTS_NVFP4)
        m->opt.draft_weights = interfaces::DraftWeightsMode::kNvfp4;
      else if (opts->draft_weights == FLYX_DRAFT_WEIGHTS_Q8)
        m->opt.draft_weights = interfaces::DraftWeightsMode::kQ8;
      else if (opts->draft_weights == FLYX_DRAFT_WEIGHTS_EXACT)
        m->opt.draft_weights = interfaces::DraftWeightsMode::kExact;
      else
        m->opt.draft_weights = interfaces::DraftWeightsMode::kAuto;
    }
    if (!resolve_options(opts ? opts->options : nullptr, &m->opt, &err)) return fail("options: " + err);
    if (!apply_weight_caps(opts, &m->opt)) return fail("GPU residency cap exceeds INT_MAX");
    if (!nvfp4_inputs(*desc, &m->opt.kernels.prefill_fp4, &err)) return fail(err);
    if (m->opt.embed == 0) {
      // embed=auto: host memory frees the table's GPU memory for a few kilobytes of reads a token; where 32 GB stay
      // free after the weights the memory is not missed, and a prefill pass reads its rows faster on the GPU (the
      // 27B's 4096 rows: 2.6 ms over PCIe on the RTX PRO 6000).
      uint64_t weights = 0;
      for (int i = 0; i < desc->n_weights; ++i)
        if (desc->weights[i].t) weights += desc->weights[i].t->nbytes;
      size_t free_b = 0, total_b = 0;
      const bool roomy = gpu_memory(&free_b, &total_b) && free_b > weights &&
                         free_b - weights >= size_t(32) * 1000000000;
      cudaGetLastError();
      m->opt.embed = roomy ? 2 : 1;
    }
    if (m->opt.vision == FLYX_VISION_DEV_AUTO) {
      size_t free_b = 0, total_b = 0;
      if (!gpu_memory(&free_b, &total_b) && cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        cudaGetLastError();
        free_b = 0;
      }
      m->opt.vision = resolve_vision_device(*desc, m->opt, free_b, nullptr);
    }
    // FP8 tensor cores whose e4m3 sums are float32's: sm_90 and later.  Ada's (sm_89: L4, L40S) sum in fewer bits:
    // attention with q in e4m3 came out about 50 times past the op test's tolerance (relative error up to 1.5e-3
    // against 2e-5), where H100 and B200 held it.  Without them rows_attention_q=fp8 sums the same e4m3 query
    // vector on the binary16 tensor cores (attention_rows.cu). And the loaded code must have them.
    const int dev_cc = prop.major * 10 + prop.minor;
    const int code_cc = compiled_arch();
    m->graph_prices = graph_prices(dev_cc);
    m->opt.kernels.fp8_mma = dev_cc >= 90 && code_cc >= 89;
    // The GPU's own table, else its architecture's, unless the options name one or none.
    int arch_cc = 0;
    if (m->opt.tuning_defaults) {
      m->opt.kernels.tuning = -1;
    } else if (!m->opt.tuning_table.empty()) {
      m->opt.kernels.tuning = gemm_tuning_named(m->opt.tuning_table.c_str());
      if (m->opt.kernels.tuning < 0)
        return fail("options: no tuning table named \"" + m->opt.tuning_table + "\" (" + gemm_tuning_names() + ")");
      if (!gemm_tuning_compiled_for(m->opt.kernels.tuning, code_cc))
        return fail("options: tuning table \"" + m->opt.tuning_table + "\" is not compiled for loaded sm_" +
                    std::to_string(code_cc) + " code in this release build");
    } else {
      m->opt.kernels.tuning = gemm_tuning_auto(prop.name, dev_cc, code_cc, &arch_cc);
    }
    if (!m->opt.pdl_given) m->opt.kernels.pdl = gemm_tuning_pdl(m->opt.kernels.tuning);
    if (m->verbose) {
      const std::string table = gemm_tuning_label(m->opt.kernels.tuning, arch_cc, dev_cc);
      int q8 = 0, a4 = 0;  // the weights' inputs, as their file says
      for (int i = 0; i < desc->n_weights; ++i) {
        q8 += desc->weights[i].activations == FLYX_ACT_Q8;
        a4 += desc->weights[i].activations == FLYX_ACT_NVFP4;
      }
      std::string inputs = q8 ? std::to_string(q8) + " weights 8-bit (every pass)" : "";
      if (a4) inputs += (q8 ? ", " : "") + std::to_string(a4) + " weights NVFP4 (prefill)";
      if (inputs.empty()) inputs = "exact";
      std::fprintf(stderr, "flyx-cuda: options %s; tuning table %s; inputs %s\n", describe(m->opt).c_str(),
                   table.c_str(), inputs.c_str());
    }
  }
  // The kernels' hard limits, before anything is allocated for the model.
  if (std::string why; !kernel_limits(*desc, &why)) return fail(why);
  double t[6] = {load_clock(), 0, 0, 0, 0, 0};  // the load's steps, for its report
  LoadTimes load_times{};
  try {
    CK(cudaSetDevice(0));
    CK(cudaFree(nullptr));  // the context, here rather than in the first upload
    t[1] = load_clock();
    bind(m.get(), *desc, &load_times);
    init_cold_experts(m.get());
    t[2] = load_clock();
    if (m->verbose)  // the bf16 residual (residual16): a stack of W4A4 residual adds and no adapters
      std::fprintf(stderr, "flyx-cuda: prefill passes of more than 16 rows keep the residual in %s\n",
                   residual16_model(m.get()) ? "bf16" : "float32");
    std::string why;
    for (const Block& b : m->blocks)
      if (!supported(m.get(), b, &why)) return fail(why);
    make_draft_weights(m.get());
    t[3] = load_clock();
    m->need = row_needs(m.get());
    for (int i = 0; i < desc->n_weights; ++i) m->need.q8 = m->need.q8 || desc->weights[i].activations == FLYX_ACT_Q8;
    // Prefill passes must start at multiples of the chunk to reproduce a longer pass (numerics.md: K).
    bool chunked = !m->states.gdn.empty();
    for (const GdnDims& gd : m->states.gdn) chunked = chunked && gdn_chunk_takes(gd);
    m->prefill_align = chunked ? gdn_chunk_align() : 1;
    // Wide lookup rounds (up to FLYX_MAX_WIDE_ROWS rows, every row the 1..16-row bits): wide=1, or wide=auto where
    // every projection of the stack is NVFP4 (17..64-row GEMMs that read the weights once) and 24 GB of the GPU stay
    // free for the sequences' wider round buffers.
    if (m->opt.wide != 1) {
      bool nvfp4 = true;
      std::vector<bool> mtp_parts(m->blocks.size(), false);  // the MTP layer's blocks run no wide round
      for (const Block& b : m->blocks)
        if (const auto* t = std::get_if<MtpBlock>(&b)) {
          if (t->attn >= 0) mtp_parts[size_t(t->attn)] = true;
          if (t->mlp >= 0) mtp_parts[size_t(t->mlp)] = true;
        }
      for (size_t i = 0; i < m->blocks.size(); ++i) {
        if (mtp_parts[i]) continue;
        const Block& b = m->blocks[i];
        if (const auto* g = std::get_if<GdnBlock>(&b)) nvfp4 = nvfp4 && g->w_in.q && g->w_out.q;
        else if (const auto* a = std::get_if<AttnBlock>(&b)) nvfp4 = nvfp4 && a->w_in.q && a->w_out.q;
        else if (const auto* f = std::get_if<MlpBlock>(&b)) nvfp4 = nvfp4 && f->w_gu.q && f->w_down.q;
        else if (std::holds_alternative<RelAttnBlock>(b) || std::holds_alternative<MoeBlock>(b)) nvfp4 = false;
      }
      size_t free_b = 0, total_b = 0;
      const bool roomy = gpu_memory(&free_b, &total_b) && free_b >= size_t(24) * 1000000000;
      if (m->opt.wide == 2 || (nvfp4 && roomy)) m->wide_rows = FLYX_MAX_WIDE_ROWS;
    }
    // A model that holds DV drafts its file's tree every round (docs/design/dv.md): where wide=auto gives no wide
    // rows, it has those of DV's widest round (its tree's drafts and the anchor); wide=0 keeps 16, and DV's rounds of
    // more drafts are refused (said at load).
    if (m->opt.wide == 0 && m->dv_block >= 0)
      m->wide_rows = std::max(m->wide_rows,
                              std::min(FLYX_MAX_WIDE_ROWS, block<DvBlock>(m.get(), m->dv_block).tree_width + 1));
    // The buffers rounds need, before the copies of the head kept only for speed: those take what room is left
    // (device_malloc) and go without where none is.
    if (!m->cold && !m->layer_ring) alloc_batch(m.get());  // rounds of several sequences
    CK(cudaEventCreateWithFlags(&m->prefill_ev, cudaEventDisableTiming));
    t[4] = load_clock();
    make_head_fp8(m.get());
    make_head_w12(m.get());
    CK(cudaDeviceSynchronize());
    t[5] = load_clock();
  } catch (const OutOfDeviceMemory&) {
    // Not the free memory at the start: on Windows that leaves out what other programs hold until the GPU runs short.
    // Past the free memory, what this process holds (the weights so far and its CUDA runtime) where the system says,
    // else the weights so far: the rest is other programs'.
    size_t free = 0, total = 0;
    const bool mem_ok = gpu_memory(&free, &total) && total > 0;
    const size_t process = gpu_process_memory();
    const bool runtime_known = process > 0 && process >= m->bytes;
    const size_t own = runtime_known ? process : m->bytes;
    const size_t others = mem_ok && total > free && total - free > own ? total - free - own : 0;
    const bool has_others =
        others >= (runtime_known ? interfaces::kClosableOthersKnownBytes : interfaces::kClosableOthersUnknownBytes);
    // From the description: a drafter's layers may come before its block, and run out of memory before it is bound.
    bool drafter = false;
    for (int i = 0; i < desc->n_blocks; ++i)
      for (int k : {FLYX_BLOCK_MTP, FLYX_BLOCK_DFLASH2, FLYX_BLOCK_DFLASH2_LAYER, FLYX_BLOCK_DV, FLYX_BLOCK_DV_LAYER,
                    FLYX_BLOCK_DSPARK, FLYX_BLOCK_DSPARK_LAYER, FLYX_BLOCK_DSPINE, FLYX_BLOCK_DSPINE_LAYER,
                    FLYX_BLOCK_EAGLE4})
        drafter = drafter || desc->blocks[i].kind == k;
    char head[160] = "Not enough VRAM while loading weights.";
    if (has_others)
      std::snprintf(head, sizeof(head), "Not enough VRAM while loading weights: %.2f / %.2f GB used by other apps.",
                    others / 1e9, total / 1e9);
    else if (mem_ok)
      std::snprintf(head, sizeof(head), "Not enough VRAM while loading weights (%.2f GB GPU).", total / 1e9);
    const std::string advice =
        has_others
            ? std::string("Close other apps") + (drafter ? ", disable speculative decoding" : "") +
                  ", or use a smaller quantization."
            : (drafter ? "Disable speculative decoding or use a smaller quantization."
                       : "Use a smaller quantization.");
    return fail_oom(std::string(head) + " " + advice);
  } catch (const std::exception& e) {
    return fail(std::string("uploading weights: ") + e.what());
  }
  if (m->verbose) {
    std::fprintf(stderr, "flyx-cuda: weights %.2f GB on the GPU", m->bytes / 1e9);
    if (m->vision_dev_bytes) std::fprintf(stderr, " (vision %.2f GB)", m->vision_dev_bytes / 1e9);
    if (m->host_bytes) {
      std::fprintf(stderr, ", %.2f GB in host memory", m->host_bytes / 1e9);
      if (m->layer_ring) std::fprintf(stderr, " (includes streamed layers)");
      else if (m->cold) std::fprintf(stderr, " (includes cold experts, GPU cap %d per layer)", m->opt.gpu_experts);
      else if (m->vision_host_bytes && m->host_bytes > m->vision_host_bytes)
        std::fprintf(stderr, " (embedding + vision %.2f GB)", m->vision_host_bytes / 1e9);
      else if (m->vision_host_bytes)
        std::fprintf(stderr, " (vision)");
      else
        std::fprintf(stderr, " (the embedding)");
    }
    if (m->vision_disk_bytes) std::fprintf(stderr, ", %.2f GB on disk (vision)", m->vision_disk_bytes / 1e9);
    std::fprintf(stderr, "\n");
    const auto& L = load_times;
    std::fprintf(stderr,
                 "flyx-cuda: load %.2f s: the GPU's context %.2f s; weights %.2f s (GGML blocks: %.2f GB read and laid "
                 "out in %.2f s, waits for their copies %.2f s); drafting copies %.2f s; round buffers %.2f s; "
                 "head copies %.2f s\n",
                 t[5] - t[0], t[1] - t[0], t[2] - t[1], L.file_bytes / 1e9, L.pack_s, L.copy_s, t[3] - t[2],
                 t[4] - t[3], t[5] - t[4]);
  }
  *out = m.release();
  return 0;
}

void close_model(flyx_bmodel* m) { delete m; }


const char* last_error() { return g_error.c_str(); }

int32_t prefill_align(flyx_bmodel* m) { return m->prefill_align; }

// Page-locked, portable (any context may copy through it): a 32K-token snapshot of the 27B (2.3 GB) moves in about
// 0.1 s instead of the second or two that fresh pageable memory took.
void* pinned_alloc(uint64_t bytes) {
  void* p = nullptr;
  if (bytes == 0 || cudaHostAlloc(&p, size_t(bytes), cudaHostAllocPortable) != cudaSuccess) {
    (void)cudaGetLastError();  // not sticky: clear it for the next call
    return nullptr;
  }
  return p;
}

void pinned_free(void* p) {
  if (p) (void)cudaFreeHost(p);
}

const char* draft_weights_active(flyx_bmodel* m) { return m ? m->active_draft_weights.c_str() : ""; }

const flyx_backend_api kApi = {
    FLYX_BACKEND_ABI_VERSION, sizeof(flyx_backend_api), "cuda",
    probe, open_model, close_model, lower, plan_free, seq_new, seq_free,
    prefill, launch, read_result, rewind, checkpoint, restore, set_oracle, last_error, memory, seq_bytes,
    prefill_align, device_stats, ready, set_reply, snapshot_bytes, snapshot_save, snapshot_load, launch_batch,
    pinned_alloc, pinned_free, read_logits, seq_new_in, seq_size, trim, graph_stats, prefill_snapshot,
    encode_vision, prefill_ex, estimate_load, draft_weights_active,
};

}  // namespace flyx::cuda

#ifdef FLYX_STATIC_BACKEND
extern "C" const flyx_backend_api* flyx_backend_get_api_cuda(void) {
  return &flyx::cuda::kApi;
}
#else
extern "C" FLYX_BACKEND_EXPORT const flyx_backend_api* flyx_backend_get_api(void) {
  return &flyx::cuda::kApi;
}
#endif
