// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Store policy with small pieces, including failed writes and missing ancestors. The real backends' serialized
// bytes and model outputs after a restore are checked in tests/e2e/golden_test.cpp.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/os/os.h"
#include "core/os/path.h"
#include "core/session/store.h"
#include "tests/common/check.h"

struct flyx_seq {
  int pos = 0;
  std::vector<int32_t> rows;
  int sizes = 0, captures = 0;
};

namespace {
struct Header {
  int begin, end;
};

int bytes(flyx_seq* seq, int32_t, int32_t begin, uint64_t* size) {
  ++seq->sizes;
  *size = sizeof(Header) + sizeof(int32_t) * (seq->pos - begin);
  return 0;
}

int save(flyx_seq* seq, int32_t, int32_t begin, const flyx_host_pieces* dst, uint64_t bytes) {
  ++seq->captures;
  Header h{begin, seq->pos};
  std::vector<uint8_t> data(bytes);
  std::memcpy(data.data(), &h, sizeof h);
  std::memcpy(data.data() + sizeof h, seq->rows.data() + begin, bytes - sizeof h);
  for (int i = 0; i < dst->n; ++i) {
    const size_t start = i * dst->piece_bytes;
    if (dst->pieces[i]) std::memcpy(dst->pieces[i], data.data() + start, std::min(dst->piece_bytes, bytes - start));
  }
  return h.end;
}

int load(flyx_seq* seq, const flyx_host_pieces* src, uint64_t bytes, int32_t) {
  Header h;
  std::memcpy(&h, src->pieces[0], sizeof h);
  if (h.begin && seq->pos != h.begin && seq->pos != h.end) return -1;
  seq->rows.resize(h.end);
  for (int i = 0; i < src->n; ++i) {
    if (!src->pieces[i]) continue;
    const size_t start = i * src->piece_bytes;
    const size_t skip = i ? 0 : sizeof h;
    std::memcpy(reinterpret_cast<uint8_t*>(seq->rows.data() + h.begin) + (start + skip - sizeof h),
                static_cast<const uint8_t*>(src->pieces[i]) + skip, std::min(src->piece_bytes, bytes - start) - skip);
  }
  return seq->pos = h.end;
}

int host_allocations = 0, host_frees = 0;
}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / ("flyx-store-test-" + std::to_string(flyx::os::process_id()));
  flyx_backend_api api{};
  api.snapshot_bytes = bytes;
  api.snapshot_save = save;
  api.snapshot_load = load;
  api.last_error = [] { return "bad test snapshot"; };
  flyx_seq source, restored;
  source.rows.resize(7000);
  for (int i = 0; i < 7000; ++i) source.rows[i] = i * 19 % 997;
  {
    flyx::SnapshotStore store(0, 4096);
    std::string why;
    if (!store.set_disk(flyx::os::from_path(dir), 30000, &why)) return 1;
    uint64_t tip = 0;
    auto put = [&](int pos, bool staging) {
      source.pos = pos;
      const std::vector<int32_t> prefix(source.rows.begin(), source.rows.begin() + pos);
      return store.put(&api, &source, 0, prefix, &why, {.disk_tip = &tip, .staging_only = staging});
    };
    check(put(2500, false), "independent disk checkpoint");
    const uint64_t root = tip;
    check(put(5000, true), "decoded rows captured");
    const auto* first = store.find(source.rows, source.rows.size(), 0);
    check(first && first->disk_id == root, "staging hidden from normal restore");
    check(put(5000, false), "eligible terminal state published over staged rows");
    const auto* terminal = store.find(source.rows, source.rows.size(), 0);
    check(terminal && terminal->bytes == sizeof(Header), "same position captured without rewriting prior rows");
    check(terminal && store.load(&api, &restored, *terminal, 0, &why) == 5000 &&
          std::equal(restored.rows.begin(), restored.rows.end(), source.rows.begin()), "complete chain restored");
    check(store.ancestor(tip, 4999) == root, "rewind removes unproven suffix");
    check(store.disk_bytes() == 5000 * sizeof(int32_t) + 3 * sizeof(Header), "disk charged only for new rows");

    // An independent branch needs 12,008 bytes; this forces eviction starting at the leaf.
    tip = 0;
    source.rows[0] += 1;
    check(put(3000, false), "another branch fits by pruning descendants first");
    check(first && store.ancestor(root, 2500) == root, "shared root outlives its evicted descendants");
    const auto* branch = store.find(source.rows, source.rows.size(), 0);
    if (!branch) return 1;
    fs::resize_file(branch->file, sizeof(Header));
    check(!store.find(source.rows, source.rows.size(), 0), "truncated range is not offered");
    check(store.load(&api, &restored, *branch, 0, &why) < 0 &&
          fs::file_size(branch->file) == sizeof(Header), "failed read never extends damaged file");
    fs::remove(branch->file);
    check(store.load(&api, &restored, *branch, 0, &why) < 0 && !fs::exists(branch->file),
          "failed read never recreates missing file");
  }
  auto failed_write = [&](int end, bool unlinkable) {
    flyx::SnapshotStore store(0, 4096);
    std::string why;
    if (!store.set_disk(flyx::os::from_path(dir), 30000, &why)) return check(false, "a disk tier for the failed write");
    uint64_t tip = 0;
    auto put = [&](int pos) {
      source.pos = pos;
      const std::vector<int32_t> prefix(source.rows.begin(), source.rows.begin() + pos);
      return store.put(&api, &source, 0, prefix, &why, {.disk_tip = &tip});
    };
    check(put(300), "a committed parent before a disk write failure");
    const auto* parent = store.find(source.rows, 300, 0);
    if (!parent) return check(false, "a readable parent before testing a failed write");
    // Only the child's path fails to open. An empty directory can be removed; a nonempty one retains its charge.
    const auto path = fs::path(parent->file).parent_path() / "2.snapshot";
    fs::create_directory(path);
    if (!unlinkable) fs::create_directory(path / "occupied");
    // The small child fits in one batch; the large one must observe the failure before its last capture.
    check(put(end) == (end == 500), "write failures during capture and after it returns");
    store.settle();
    const uint64_t child_bytes = sizeof(Header) + (end - 300) * sizeof(int32_t);
    check(!store.keeps() && !store.disk_enabled() && store.disk_bytes() == parent->bytes + (unlinkable ? 0 : child_bytes),
          "an I/O failure stops disk capture and only reclaimed space is credited");
    check(store.size() == 1 && store.on_disk() == 1 && !parent->children && store.find(source.rows, 500, 0) == parent &&
          store.load(&api, &restored, *parent, 0, &why) == 300 &&
          std::equal(restored.rows.begin(), restored.rows.end(), source.rows.begin()),
          "the failed child is discarded and its committed parent still restores");
    const int sizes = source.sizes, captures = source.captures;
    check(!put(600) && !put(650) && sizes == source.sizes && captures == source.captures,
          "later requests skip sizing and device capture after the write failure");
  };
  for (int end : {500, 6600})
    for (bool unlinkable : {true, false}) failed_write(end, unlinkable);
  {
    // Old processes leave flat directories of numbered snapshot files. Other files in the cache root belong to
    // the user, even if their names resemble a snapshot directory.
    const auto root = dir / "reclaim";
    if (!fs::create_directory(root)) return 1;
    const auto dead = root / "flyx-snapshots-999999999-0123456789abcdef";
    const auto live = root / ("flyx-snapshots-" + std::to_string(flyx::os::process_id()) + "-0123456789abcdef");
    const auto backup = root / "flyx-snapshots-999999999-0123456789abcdef-backup";
    const auto file = root / "flyx-snapshots-999999999-0123456789abcdee";
    const auto link = root / "flyx-snapshots-999999999-0123456789abcded";
    const auto mixed = root / "flyx-snapshots-999999999-0123456789abcdec";
    for (const auto& path : {dead, live, backup, mixed}) fs::create_directory(path);
    std::ofstream(dead / "1.snapshot") << "abc";
    std::ofstream(backup / "1.snapshot") << "backup";
    std::ofstream(mixed / "notes.txt") << "notes";
    std::ofstream(mixed / "0.snapshot") << "unrelated";
    std::ofstream(file) << "regular file";
    std::error_code link_error;
    fs::create_directory_symlink(backup, link, link_error);
    {
      flyx::SnapshotStore store(0, 4096);
      std::string why;
      if (!store.set_disk(flyx::os::from_path(root), 30000, &why)) return 1;
      check(!fs::exists(dead) && store.stale_freed() == 3, "reclaim the dead process's snapshots");
      check(fs::exists(live) && fs::exists(backup / "1.snapshot") && fs::exists(file) &&
            fs::exists(mixed / "notes.txt") && fs::exists(mixed / "0.snapshot"),
            "keep live snapshots, backups and unrelated files");
      check(link_error || fs::is_symlink(fs::symlink_status(link)), "keep directory links and their target");
    }
    fs::remove_all(root);  // the fixture directory created exclusively above
  }
  {
    // This budget holds two whole host pieces. A slightly larger snapshot must use the disk in both directions.
    flyx::SnapshotStore store(10000, 4096);
    std::string why;
    if (!store.set_disk(flyx::os::from_path(dir), 30000, &why)) return 1;
    source.pos = 2300;
    const std::vector<int32_t> prefix(source.rows.begin(), source.rows.begin() + source.pos);
    bool on_disk = false;
    check(store.put(&api, &source, 0, prefix, &why, {.on_disk = &on_disk}) && on_disk, "partial host piece spills to disk");
    const auto* saved = store.find(prefix, prefix.size(), 0);
    check(saved && store.load(&api, &restored, *saved, 0, &why) == source.pos && restored.rows == prefix &&
          store.reserved() == 0, "partial host piece restored directly from disk");
  }
  {
    // A backend may decline an individual allocation. The first piece uses regular RAM; the second is pinned.
    flyx_backend_api mixed = api;
    mixed.host_alloc = [](uint64_t n) -> void* { return ++host_allocations == 1 ? nullptr : new uint8_t[n]; };
    mixed.host_free = [](void* p) { ++host_frees; delete[] static_cast<uint8_t*>(p); };
    flyx::SnapshotStore store(8192, 4096);
    source.pos = 1500;
    const std::vector<int32_t> prefix(source.rows.begin(), source.rows.begin() + source.pos);
    std::string why;
    check(store.put(&mixed, &source, 0, prefix, &why), "snapshot with mixed host allocation");
    const auto* saved = store.find(prefix, prefix.size(), 0);
    if (!saved) return 1;
    check(saved->backend_owned() && host_allocations == 2, "mixed snapshot holds backend memory");
    store.own_memory();
    check(!saved->backend_owned() && host_frees == 1, "unload releases every backend-owned piece");
    check(store.reserved() == 8192 && store.load(&api, &restored, *saved, 0, &why) == source.pos &&
          restored.rows == prefix, "mixed snapshot still restores after unload");
  }
  for (bool disk : {false, true}) {
    flyx::SnapshotStore store(3 * 4096, 4096);
    std::string why;
    if (disk && !store.set_disk(flyx::os::from_path(dir), 65536, &why)) return 1;
    source.pos = 1500;
    const std::vector<int32_t> prefix(source.rows.begin(), source.rows.begin() + source.pos);
    check(store.put(&api, &source, 0, prefix, &why), "checkpoint before the cache budget changes");
    if (!disk)
      check(store.fits_without_eviction(4096) && !store.fits_without_eviction(4097),
            "a marked prefix may use the last free piece without displacing an existing snapshot");
    store.set_budget(4095);  // no whole piece remains
    check(store.reserved() == 0, "a smaller cache budget releases held memory");
    const auto* saved = store.find(prefix, prefix.size(), 0);
    if (disk)
      check(saved && store.load(&api, &restored, *saved, 0, &why) == source.pos && restored.rows == prefix,
            "checkpoint moved to disk still restores after the budget shrinks");
    else check(!saved, "a cache with no disk evicts checkpoints that no longer fit");
    store.set_budget(3 * 4096);
    bool on_disk = true;
    check(store.put(&api, &source, 0, prefix, &why, {.on_disk = &on_disk}) && !on_disk && store.reserved() == 2 * 4096,
          "a larger cache budget can again retain new checkpoints in RAM");
  }
  {
    flyx::SnapshotStore store(8192, 4096);
    auto history = std::make_shared<const flyx::SnapshotStore::History>(source.rows, "the parked conversation");
    std::weak_ptr<const flyx::SnapshotStore::History> saved = history;
    std::string why;
    for (int pos : {400, 900}) {
      source.pos = pos;
      const std::vector<int32_t> prefix(source.rows.begin(), source.rows.begin() + pos);
      check(store.put(&api, &source, 0, prefix, &why, {.history = history}), "retain a checkpoint with shared history");
    }
    history.reset();
    store.set_budget(4096);
    const auto* kept = store.continued_by("the parked conversation continues");
    check(kept && kept == saved.lock().get() && kept->tokens == source.rows,
          "one remaining checkpoint retains the conversation text and original tokenization");
    store.set_budget(0);
    check(saved.expired(), "the last checkpoint releases the conversation history");
  }
  for (bool disk : {false, true}) {
    flyx::SnapshotStore store(4096, 4096);
    std::string why;
    if (disk) check(store.set_disk(flyx::os::from_path(dir), 4096, &why), "open the spill tier for a conversation swap");
    source.pos = 400;
    const std::vector<int32_t> a(source.rows.begin(), source.rows.begin() + source.pos);
    check(store.put(&api, &source, 0, a, &why), "park the incoming conversation");
    const auto* incoming = store.find(a, a.size(), 0);
    if (!incoming) return 1;
    ++source.rows.front();
    const std::vector<int32_t> b(source.rows.begin(), source.rows.begin() + source.pos);
    bool on_disk = false;
    const bool saved = store.put(&api, &source, 0, b, &why, {.on_disk = &on_disk, .restore = incoming});
    check(saved == disk && on_disk == disk, "park the outgoing conversation on disk when the incoming one fills RAM");
    flyx_seq restored{};
    check(store.find(a, a.size(), 0) == incoming && store.load(&api, &restored, *incoming, 0, &why) == source.pos &&
              restored.rows == a, "the returning conversation still restores its exact rows");
    --source.rows.front();
  }
  {
    // A transient backend snapshot_save error during direct-to-disk capture must not create a junk file or
    // permanently disable the disk tier for later snapshots.
    flyx::SnapshotStore store(0, 4096);
    std::string why;
    if (!store.set_disk(flyx::os::from_path(dir), 30000, &why)) return 1;
    flyx_backend_api flaky = api;
    flaky.snapshot_save = [](flyx_seq*, int, int, const flyx_host_pieces*, uint64_t) -> int { return -1; };
    source.pos = 2300;
    const std::vector<int32_t> prefix(source.rows.begin(), source.rows.begin() + source.pos);
    bool on_disk = false;
    check(!store.put(&flaky, &source, 0, prefix, &why, {.on_disk = &on_disk}) && store.disk_enabled() &&
              store.disk_bytes() == 0,
          "failed direct-to-disk capture leaves disk tier healthy and empty");
    check(store.put(&api, &source, 0, prefix, &why, {.on_disk = &on_disk}) && on_disk,
          "subsequent direct-to-disk capture still succeeds after transient backend error");
    const auto* saved = store.find(prefix, prefix.size(), 0);
    check(saved && store.load(&api, &restored, *saved, 0, &why) == source.pos && restored.rows == prefix,
          "and restores its exact rows");
  }
  fs::remove(dir);
  return flyx::test::finish("snapshot_store_test");
}
