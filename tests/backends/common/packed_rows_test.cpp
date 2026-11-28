// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <numeric>
#include <vector>

#include "backends/common/packed_rows.h"
#include "core/os/os.h"
#include "tests/common/check.h"

namespace {
using flyx::test::check;
struct Source {
  flyx::os::MappedFile mapping;
  std::vector<uint64_t> reads;
  bool force_failure = false;
  static int read(const void* ctx, uint64_t at, void* dest, size_t bytes) {
    auto& self = *const_cast<Source*>(static_cast<const Source*>(ctx));
    self.reads.push_back(at);
    return !self.force_failure && self.mapping.read_at(at, dest, bytes) ? 0 : -1;
  }
};
void run(const std::filesystem::path& path, int dt) {
  const size_t stride = dt == FLYX_DT_BF16 ? 64 : flyx_block_bytes(dt);
  constexpr size_t offset = 32, rows = 64;
  std::vector<uint8_t> content(offset + rows * stride);
  std::iota(content.begin(), content.end(), uint8_t(11));
  {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(content.data()), content.size());
  }
  Source src;
  std::string err;
  if (!src.mapping.open(path.string(), &err)) throw std::runtime_error(err);
  flyx_tensor_view table{};
  table.file = &src;
  table.read_at = Source::read;
  table.file_offset = offset;
  table.shape[0] = rows;
  table.shape[1] = 32;
  table.ndim = 2;
  table.dtype = dt;
  table.data = src.mapping.data() + offset;
  table.nbytes = rows * stride;
  const int32_t ids[] = {6, 3, 6, 0};
  std::vector<uint8_t> bytes(4 * stride);
  for (int tier : {FLYX_PLE_GPU, FLYX_PLE_HOST, FLYX_PLE_DISK, FLYX_PLE_HYBRID}) {
    src.reads.clear();
    flyx::backends::PackedRows store(table, tier, 4 * stride + 128, 0);
    store.gather(ids, 4, bytes.data());
    for (int i = 0; i < 4; ++i)
      check(!std::memcmp(bytes.data() + i * stride, src.mapping.data() + offset + ids[i] * stride, stride),
            "every tier preserves packed bytes and batch order");
    if (tier == FLYX_PLE_DISK || tier == FLYX_PLE_HYBRID)
      check(src.reads == std::vector<uint64_t>({offset, offset + 3 * stride, offset + 6 * stride}),
            "sorted, deduplicated reads of the original shard");
    if (tier == FLYX_PLE_HYBRID) {
      src.reads.clear();
      store.gather(ids, 4, bytes.data());
      check(src.reads.empty(), "hot rows served from cache");
      for (int32_t id = 7; id <= 12; ++id) store.gather(&id, 1, bytes.data());
      src.reads.clear();
      store.gather(ids, 4, bytes.data());
      check(!src.reads.empty(), "bounded CLOCK evicts old rows");
    }
  }
  // Independent requests share the table and can replace rows in the same cache
  // set concurrently.
  auto concurrent = table;
  concurrent.file = &src.mapping;
  concurrent.read_at = [](const void *ctx, uint64_t at, void *dest,
                          size_t size) {
    return static_cast<const flyx::os::MappedFile *>(ctx)->read_at(at, dest,
                                                                   size)
               ? 0
               : -1;
  };
  flyx::backends::PackedRows shared(concurrent, FLYX_PLE_HYBRID,
                                    8 * stride + 256, 0);
  std::vector<std::future<bool>> readers;
  for (int request = 0; request < 4; ++request) {
    readers.push_back(std::async(std::launch::async, [&, request] {
      std::vector<uint8_t> row(stride);
      for (int i = 0; i < 128; ++i) {
        const int32_t id = (request + 7 * i) % rows;
        shared.gather(&id, 1, row.data());
        if (std::memcmp(row.data(), content.data() + offset + id * stride,
                        stride))
          return false;
      }
      return true;
    }));
  }
  for (auto &reader : readers)
    check(reader.get(), "concurrent cache eviction preserves row bytes");
  table.shape[0] = INT32_MAX;
  table.nbytes = uint64_t(INT32_MAX) * stride;
  flyx::backends::PackedRows sparse(table, FLYX_PLE_DISK, 0, 0);
  check(sparse.storage_bytes() == 0, "disk reserves no bytes proportional to file size");
  int32_t missing = 64;
  bool failed = false;
  try {
    sparse.gather(&missing, 1, bytes.data());
  } catch (const std::runtime_error&) {
    failed = true;
  }
  check(failed, "short file fails the gather");
  src.force_failure = true;
  failed = false;
  try {
    sparse.gather(ids, 1, bytes.data());
  } catch (const std::runtime_error&) {
    failed = true;
  }
  check(failed, "I/O failure propagates");
}
void sparse_file(const std::filesystem::path &path) {
  constexpr int rows = 1 << 19;
  constexpr int stride = 64;
  {
    std::ofstream f(path, std::ios::binary);
    f.exceptions(std::ios::badbit | std::ios::failbit);
    f.seekp(uint64_t(rows) * stride - 1);
    f.put('Q');
  }
  Source source;
  std::string error;
  if (!source.mapping.open(path.string(), &error))
    throw std::runtime_error(error);
  flyx_tensor_view table{};
  table.file = &source;
  table.read_at = Source::read;
  table.ndim = 2;
  table.shape[0] = rows;
  table.shape[1] = 32;
  table.dtype = FLYX_DT_BF16;
  table.nbytes = uint64_t(rows) * stride;
  flyx::backends::PackedRows disk(table, FLYX_PLE_AUTO, 0, 0);
  std::array<uint8_t, stride> bytes{};
  int32_t last = rows - 1;
  disk.gather(&last, 1, bytes.data());
  check(bytes.back() == 'Q' && disk.storage_bytes() == 0 &&
            source.reads.size() == 1,
        "auto with zero hot budget reads one distant row without resident "
        "table storage");
}
}  // namespace

int main() {
  const auto file =
      std::filesystem::temp_directory_path() / ("flyx-ple-rows-" + std::to_string(flyx::os::process_id()));
  for (int dt : {FLYX_DT_BF16, FLYX_DT_GGML_Q8_0, FLYX_DT_GGML_Q5_0, FLYX_DT_GGML_Q4_0}) run(file, dt);
  sparse_file(file);
  std::filesystem::remove(file);
  return flyx::test::finish("packed_rows");
}
