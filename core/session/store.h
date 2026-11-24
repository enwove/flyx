// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// SnapshotStore: sessions' checkpoints in host memory. A
// conversation that gives its session to another keeps its state here, and a
// later request that continues it loads it instead of computing it again.
//
// A snapshot is a checkpoint as the backend writes it (snapshot_save: the
// recurrent state and every positional cache before its position) and the
// tokens it holds.  The store keeps snapshots within a budget of host memory,
// the least recently used dropped first.  Only checkpoints are kept, never a
// live state, so a loaded snapshot gives the bits a fresh prefill gives.
// With a disk tier (set_disk), the least recently used
// snapshots go to files before the memory budget runs short, written by a
// thread of the store's own, and come back to memory when loaded; past the
// disk's budget the least recently used go.  A snapshot larger than the
// memory budget (a long conversation's, the budget what its K/V left) goes
// to a file directly and is loaded from there, a few pieces at a time
// (kBounce of them), never whole in memory. Along a known live lineage, successive disk checkpoints
// capture only the new positional rows and the terminal state. A checkpoint can depend on older ranges; eviction
// removes leaves before their ancestors. The writer never references a live model buffer.
// One thread owns it (the scheduler's, or a single caller).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/interfaces/chat.h"
#include "core/interfaces/vision.h"
#include "flyx/backend_abi.h"

namespace flyx {

// A piece of host memory: the backend's (host_alloc: the device copies at the bus's speed) when it gives some, else
// the store's own.
class HostBuffer {
 public:
  HostBuffer() = default;
  HostBuffer(const flyx_backend_api* api, uint64_t bytes);  // null get(): none to be had
  ~HostBuffer() { reset(); }
  HostBuffer(HostBuffer&& o) noexcept { *this = std::move(o); }
  HostBuffer& operator=(HostBuffer&& o) noexcept;
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
  uint8_t* get() const { return p_; }
  uint64_t capacity() const { return cap_; }
  bool backend_owned() const { return api_ != nullptr; }
  void reset();

 private:
  const flyx_backend_api* api_ = nullptr;  // set: p_ came from api_->host_alloc
  uint8_t* p_ = nullptr;
  uint64_t cap_ = 0;
};

class SnapshotStore {
 public:
  // budget 0 and no disk tier: the store keeps nothing.  piece_bytes: kPieceBytes (tests: smaller, 4096 at least).
  explicit SnapshotStore(uint64_t budget, uint64_t piece_bytes = kPieceBytes) : budget_(budget), piece_(piece_bytes) {}
  // Set at model load and idle reload, between requests. Spill or evict old snapshots if fewer pieces now fit.
  void set_budget(uint64_t budget);
  ~SnapshotStore();
  SnapshotStore(const SnapshotStore&) = delete;
  SnapshotStore& operator=(const SnapshotStore&) = delete;
  uint64_t budget() const { return budget_; }
  bool keeps() const { return budget_ > 0 || disk_enabled(); }
  bool disk_enabled() const { return disk_budget_ > 0 && !disk_failed_.load(); }
  uint64_t bytes() const { return bytes_; }  // the snapshots in memory
  int size() const { return static_cast<int>(entries_.size()); }  // in memory and on disk
  // The disk tier: files in a directory of this process's own under `dir`, `budget` bytes of them at most.  False
  // (and *err) when the directory cannot be made.
  // Reclaims numbered snapshot files from directories that ended processes left behind. Other files stay.
  // stale_freed() says how many bytes went.
  bool set_disk(const std::string& dir, uint64_t budget, std::string* err);
  uint64_t stale_freed() const { return stale_freed_; }
  uint64_t disk_bytes() const { return disk_bytes_; }
  int on_disk() const;
  // Its memory comes in pieces of one size, any snapshot's for any other's (flyx_host_pieces), so a snapshot of any
  // length takes what another gave back and page-locked memory is got once.
  static constexpr uint64_t kPieceBytes = uint64_t(32) << 20;
  // A snapshot larger than the budget goes to and from its file through this many pieces (128 MB), held only then.
  static constexpr size_t kBounce = 4;
  // Host memory the store holds: its snapshots' pieces and the spare ones it keeps for the next (within budget()).
  uint64_t reserved() const { return reserved_; }
  // Space available for a new snapshot without displacing any existing checkpoint.
  bool fits_without_eviction(uint64_t bytes) const {
    const uint64_t occupied = reserved_ - spare_.size() * piece_;
    return occupied <= budget_ && bytes <= (budget_ - occupied) / piece_ * piece_;
  }
  // Before the backend that gave its memory goes (an unload): every snapshot moves to the store's own memory and the
  // spare pieces go back, so the snapshots outlive it.
  void own_memory();

  // A conversation's history (its last prompt and the reply its client saw),
  // as the session that held it had it: a request that continues its text
  // keeps its tokens (Session::tokenize), so the snapshot's still match.
  struct History {
    std::vector<int32_t> tokens;
    std::string text;
    std::vector<interfaces::MediaSpan> media{};
    std::vector<interfaces::RenderedImage> images{};
  };
  struct Spill;
  struct Entry {
    std::vector<int32_t> prefix;  // the tokens before the snapshot's position (a prefix of history.tokens)
    std::vector<interfaces::MediaSpan> media;  // multimodal spans within [0, prefix.size())
    std::shared_ptr<const History> history;  // shared by the checkpoints of one parked conversation
    std::vector<HostBuffer> pieces;  // `piece` bytes each, `bytes` in all; none while on disk
    std::vector<void*> at;           // their addresses, for view()
    uint64_t bytes = 0, used = 0, piece = kPieceBytes;
    std::string file;                // set: on disk, its pieces gone back to the store
    std::shared_ptr<Spill> spill;    // set: being written to disk (its pieces read until it is done)
    // A disk range contains the rows after its parent and its own final recurrent state.
    uint64_t disk_id = 0, parent = 0;
    size_t children = 0;  // only leaves can be evicted
    bool staging_only = false;  // decoded rows an eligible descendant can use, never themselves a prefill checkpoint
    std::shared_ptr<Spill> range_write;  // captures own their buffers until the writer finishes
    // The pieces as the backend reads and writes them.
    flyx_host_pieces view() const {
      return {at.data(), static_cast<int32_t>(at.size()), 0, piece};
    }
    bool backend_owned() const;
  };
  struct PutOptions {
    std::shared_ptr<const History> history = nullptr;
    bool* on_disk = nullptr;
    uint64_t* disk_tip = nullptr;
    bool staging_only = false;
    const Entry* restore = nullptr;
    const std::vector<interfaces::MediaSpan>* media = nullptr;
  };
  // Keeps checkpoint `slot` of `seq`, which holds `prefix`, of the
  // conversation `opts.history`.  A snapshot of the same tokens is replaced; the
  // least recently used go while the budget is short.  False (and *why) when
  // it cannot be kept; *opts.on_disk: it went straight to disk (larger than the budget).
  // A session passes its last disk capture in opts.disk_tip. After a rewind it must truncate that lineage to the
  // position it actually retained. Identical tokens in another execution do not establish identical KV bytes.
  uint64_t ancestor(uint64_t tip, size_t retained) const;
  bool put(const flyx_backend_api* api, flyx_seq* seq, int slot, const std::vector<int32_t>& prefix,
           std::string* why, const PutOptions& opts);
  bool put(const flyx_backend_api* api, flyx_seq* seq, int slot, const std::vector<int32_t>& prefix,
           std::string* why) {
    return put(api, seq, slot, prefix, why, PutOptions{});
  }
  // The snapshot holding the most tokens of `prompt`: more than `floor` and at
  // most `limit` of them.  Null: none.
  const Entry* find(const std::vector<int32_t>& prompt, size_t limit, size_t floor,
                    const std::vector<interfaces::MediaSpan>* media = nullptr);
  // The stored conversation whose history `text` continues furthest (the whole
  // history's text a prefix of it).  Null: none.
  const History* continued_by(const std::string& text);
  // Makes `e` the state of `seq`, with `anchor` as its next input (reading it back from disk first when it is
  // there). Returns the position, or -1 with *err. On failure, failed_end receives the end of the range that
  // failed, so a fallback can skip longer checkpoints that would read the same damaged parent again.
  int load(const flyx_backend_api* api, flyx_seq* seq, const Entry& e, int32_t anchor, std::string* err,
           size_t* failed_end = nullptr);
  // Waits for the snapshots being written to disk (tests; the store does when it needs their memory).
  void settle();

  // A snapshot being written to disk: the writer thread's job.
  struct Spill {
    std::vector<const void*> at;
    uint64_t bytes = 0, piece = kPieceBytes;
    std::string path;
    std::string error;
    int state = 0;  // 0 queued or writing, 1 written, 2 failed (under the store's mutex)
  };

 private:
  static size_t pieces_for(uint64_t bytes, uint64_t piece) {
    return static_cast<size_t>((bytes + piece - 1) / piece);
  }
  void recycle_pieces(std::vector<HostBuffer>& pieces);
  static void set_entry_pieces(Entry* e, std::vector<HostBuffer> pieces);
  std::string next_snapshot_path();
  // RAM snapshots occupy whole pieces, even when their last piece holds only a few bytes.
  bool fits_in_memory(uint64_t bytes) const { return bytes <= budget_ - budget_ % piece_; }
  Entry* by_id(uint64_t id) const;
  bool ready(const std::shared_ptr<Spill>& job);
  bool available(const Entry& entry);
  bool reclaim_pieces(const Entry* restore = nullptr);  // make an entry's host pieces available for reuse or release
  void drop(size_t i);  // entry i's pieces become spare ones, its file goes
  // Make room without evicting an incoming restore target or a delta's parent.
  bool disk_room(uint64_t bytes, const Entry* restore, const Entry* parent = nullptr);
  // A snapshot larger than the budget: written to its file kBounce pieces at a time, then loaded from there the same
  // way (the file stays).
  bool put_on_disk(const flyx_backend_api* api, flyx_seq* seq, int slot, int begin, uint64_t bytes, int expect, Entry* e,
                   std::string* why);
  int load_in_parts(const flyx_backend_api* api, flyx_seq* seq, const Entry& e, int32_t anchor, std::string* err);
  // `n` pieces within the budget: spare ones first, then new ones, then those of snapshots written to disk, the
  // least recently used snapshots given up while the budget is short.  False: not that many to be had (what was
  // taken goes back to the spares).
  bool take(const flyx_backend_api* api, size_t n, std::vector<HostBuffer>* out, const Entry* restore);
  // The disk tier: snapshots past three quarters of the memory budget start for the disk, the least recently used
  // first; reap() gives the pieces of those written back to the spares; wait_spill() waits for `e`'s.
  void spill_some(const Entry* restore = nullptr);
  bool start_spill(Entry* e, const Entry* restore);
  void reap();
  void discard_failed_ranges();
  void remove_file(const std::string& path, uint64_t charged);
  void wait_spill(Entry* e);
  void writer();
  // Empty buffers: the immutable in-memory snapshot in Spill::at. Otherwise the job owns just this range
  // of host pieces. At most kBounce pieces are in flight plus the header piece required by each capture.
  struct WriteBatch {
    std::shared_ptr<Spill> target;
    std::vector<HostBuffer> buffers;
    size_t first = 0;
    bool last = true;
    bool cancel = false;
  };
  void enqueue(WriteBatch batch);
  uint64_t budget_ = 0, piece_ = kPieceBytes, bytes_ = 0, reserved_ = 0, clock_ = 0;
  std::vector<std::unique_ptr<Entry>> entries_;
  // Pieces of snapshots that went, kept for the next ones: page-locked memory costs about as much to get as to fill
  // the first time (the kernel faults and pins every page), nothing the next.
  std::vector<HostBuffer> spare_;
  std::string disk_dir_;  // empty: no disk tier
  uint64_t disk_budget_ = 0, disk_bytes_ = 0, files_ = 0, stale_freed_ = 0;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<WriteBatch> queue_;
  size_t staged_ = 0;
  std::atomic<bool> disk_failed_{false};  // the writer stops further captures as soon as a write fails
  bool stop_ = false;
  std::thread writer_;
};

}  // namespace flyx
