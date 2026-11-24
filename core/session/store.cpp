// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/session/store.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <new>
#include <random>
#include <string_view>

#include "core/os/os.h"
#include "core/os/path.h"
#include "core/resources/ledger.h"

namespace flyx {

namespace {

namespace fs = std::filesystem;
constexpr char kDiskFailed[] = "snapshot disk writes disabled after a write failure";
using os::from_path;
using os::to_path;

int snapshot_owner(std::string_view name) {
  constexpr std::string_view prefix = "flyx-snapshots-";
  if (!name.starts_with(prefix)) return 0;
  name.remove_prefix(prefix.size());
  const size_t dash = name.find('-');
  if (dash == 0 || dash == name.npos || name.front() == '0') return 0;
  const auto nonce = name.substr(dash + 1);
  if (nonce.size() != 16 || nonce.find_first_not_of("0123456789abcdef") != nonce.npos) return 0;
  int pid = 0;
  const char* end = name.data() + dash;
  const auto parsed = std::from_chars(name.data(), end, pid);
  return parsed.ec == std::errc{} && parsed.ptr == end ? pid : 0;
}

uint64_t reclaim_snapshots(const fs::path& dir) {
  uint64_t freed = 0;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    std::error_code file_error;
    if (!fs::is_regular_file(it->symlink_status(file_error))) continue;
    const std::string name = from_path(it->path().filename());
    const size_t digits = name.find_first_not_of("0123456789");
    if (digits == 0 || digits == name.npos || name.front() == '0' ||
        name.compare(digits, name.npos, ".snapshot") != 0) continue;
    const auto bytes = it->file_size(file_error);
    if (!file_error && fs::remove(it->path(), file_error)) freed += bytes;
  }
  fs::remove(dir, ec);  // only if empty: unfamiliar files and subdirectories stay
  return freed;
}

}  // namespace

HostBuffer::HostBuffer(const flyx_backend_api* api, uint64_t bytes) {
  if (bytes == 0) return;
  if (api && api->host_alloc) p_ = static_cast<uint8_t*>(api->host_alloc(bytes));
  if (p_) api_ = api;
  else p_ = new (std::nothrow) uint8_t[bytes];
  if (p_) cap_ = bytes;
}

HostBuffer& HostBuffer::operator=(HostBuffer&& o) noexcept {
  if (this != &o) {
    reset();
    api_ = std::exchange(o.api_, nullptr);
    p_ = std::exchange(o.p_, nullptr);
    cap_ = std::exchange(o.cap_, 0);
  }
  return *this;
}

void HostBuffer::reset() {
  if (api_) api_->host_free(p_);
  else delete[] p_;
  api_ = nullptr;
  p_ = nullptr;
  cap_ = 0;
}

bool SnapshotStore::Entry::backend_owned() const {
  return std::any_of(pieces.begin(), pieces.end(), [](const HostBuffer& p) { return p.backend_owned(); });
}

SnapshotStore::~SnapshotStore() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
    queue_.clear();  // not begun: their snapshots go with the store
  }
  cv_.notify_all();
  if (writer_.joinable()) writer_.join();
  if (!disk_dir_.empty()) {
    std::error_code ec;
    fs::remove_all(to_path(disk_dir_), ec);
  }
}

bool SnapshotStore::set_disk(const std::string& dir, uint64_t budget, std::string* err) {
  if (!disk_dir_.empty()) return *err = "the store has a disk tier already", false;
  // The directories of processes that ended without removing theirs (killed, crashed: a destructor that did not run)
  // go first, or every such end leaves gigabytes behind.  A directory names its process.
  std::error_code ec;
  for (fs::directory_iterator it(to_path(dir), ec), end; !ec && it != end; it.increment(ec)) {
    std::error_code entry_error;
    if (!fs::is_directory(it->symlink_status(entry_error))) continue;
    const int pid = snapshot_owner(from_path(it->path().filename()));
    if (pid > 0 && pid != os::process_id() && !os::process_running(pid)) stale_freed_ += reclaim_snapshots(it->path());
  }
  std::random_device rd;
  char name[64];
  std::snprintf(name, sizeof(name), "flyx-snapshots-%d-%08x%08x", os::process_id(), rd(), rd());  // this process's own
  const fs::path d = to_path(dir) / name;
  ec.clear();
  fs::create_directories(d, ec);
  if (ec) return *err = "cannot make " + from_path(d) + ": " + ec.message(), false;
  disk_dir_ = from_path(d);
  disk_budget_ = budget;
  writer_ = std::thread([this] { writer(); });
  return true;
}

int SnapshotStore::on_disk() const {
  int n = 0;
  for (const auto& e : entries_) n += !e->file.empty();
  return n;
}

void SnapshotStore::writer() {
  std::unique_lock<std::mutex> lock(mu_);
  for (;;) {
    cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
    if (stop_) return;
    WriteBatch batch = std::move(queue_.front());
    queue_.pop_front();
    lock.unlock();
    const auto& s = batch.target;
    const size_t staged = batch.buffers.size();
    if (batch.cancel) {
      batch.buffers.clear();
      lock.lock();
      staged_ -= staged;
      if (batch.last && s->state == 0) s->state = 2;
      cv_.notify_all();
      continue;
    }
    os::RandomAccessFile f;
    std::string err;
    const size_t count = staged ? staged : s->at.size();
    if (disk_failed_.load()) err = kDiskFailed;
    bool ok = err.empty() && f.open(s->path, s->bytes, &err);
    for (size_t i = 0; ok && i < count; ++i) {
      const uint64_t off = uint64_t(batch.first + i) * s->piece;
      const void* data = staged ? batch.buffers[i].get() : s->at[i];
      ok = f.write_at(off, data, size_t(std::min(s->piece, s->bytes - off)), &err);
    }
    f.close();
    batch.buffers.clear();
    lock.lock();
    staged_ -= staged;
    if (!ok) {
      disk_failed_.store(true);
      if (s->error.empty()) s->error = std::move(err);
    }
    if (batch.last) s->state = ok ? 1 : 2;
    cv_.notify_all();
  }
}

void SnapshotStore::enqueue(WriteBatch batch) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    staged_ += batch.buffers.size();
    queue_.push_back(std::move(batch));
  }
  cv_.notify_all();
}

bool SnapshotStore::ready(const std::shared_ptr<Spill>& job) {
  if (!job) return true;
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [&] { return job->state != 0; });
  return job->state == 1;
}

SnapshotStore::Entry* SnapshotStore::by_id(uint64_t id) const {
  if (id)
    for (const auto& e : entries_)
      if (e->disk_id == id) return e.get();
  return nullptr;
}

uint64_t SnapshotStore::ancestor(uint64_t tip, size_t retained) const {
  if (!disk_enabled()) return 0;
  for (Entry* e = by_id(tip); e; e = by_id(e->parent))
    if (e->prefix.size() <= retained) return e->disk_id;
  return 0;
}

bool SnapshotStore::available(const Entry& entry) {
  const Entry* e = &entry;
  for (;;) {
    if (!ready(e->range_write)) return false;
    if (!e->file.empty()) {
      std::error_code ec;
      if (fs::file_size(to_path(e->file), ec) != e->bytes || ec) return false;
    }
    if (!e->parent) return true;
    e = by_id(e->parent);
    if (!e) return false;
  }
}

bool SnapshotStore::disk_room(uint64_t bytes, const Entry* restore, const Entry* parent) {
  if (!disk_enabled() || bytes > disk_budget_) return false;
  while (disk_bytes_ > disk_budget_ - bytes) {  // the least recently used there go
    if (!disk_enabled()) return false;
    size_t oldest = entries_.size();
    for (size_t i = 0; i < entries_.size(); ++i) {
      const Entry* x = entries_[i].get();
      if (!x->file.empty() && !x->children && x != restore && x != parent &&
          (oldest == entries_.size() || x->used < entries_[oldest]->used)) oldest = i;
    }
    if (oldest == entries_.size()) return false;  // the rest of the disk's budget is being written
    drop(oldest);
  }
  return true;
}

void SnapshotStore::recycle_pieces(std::vector<HostBuffer>& pieces) {
  for (auto& p : pieces) spare_.push_back(std::move(p));
  pieces.clear();
}

void SnapshotStore::set_entry_pieces(Entry* e, std::vector<HostBuffer> pieces) {
  e->pieces = std::move(pieces);
  e->at.clear();
  e->at.reserve(e->pieces.size());
  for (auto& p : e->pieces) e->at.push_back(p.get());
}

std::string SnapshotStore::next_snapshot_path() {
  return disk_dir_ + "/" + std::to_string(++files_) + ".snapshot";
}

bool SnapshotStore::start_spill(Entry* e, const Entry* restore) {
  if (disk_dir_.empty() || e->spill || !e->file.empty() || e->pieces.empty() || !disk_room(e->bytes, restore)) return false;
  auto s = std::make_shared<Spill>();
  s->at.assign(e->at.begin(), e->at.end());
  s->bytes = e->bytes;
  s->piece = e->piece;
  s->path = next_snapshot_path();
  disk_bytes_ += e->bytes;
  e->spill = s;
  enqueue({s, {}});
  return true;
}

void SnapshotStore::reap() {
  for (auto& x : entries_) {
    Entry* e = x.get();
    if (!e->spill) continue;
    int state = 0;
    {
      std::lock_guard<std::mutex> lock(mu_);
      state = e->spill->state;
    }
    if (state == 0) continue;
    if (state == 1) {  // on disk: its memory goes back to the spares
      e->file = e->spill->path;
      bytes_ -= e->bytes;
      recycle_pieces(e->pieces);
      e->at.clear();
    } else {  // not written (the disk full?): it stays in memory, and the disk takes no more
      remove_file(e->spill->path, e->bytes);
    }
    e->spill.reset();
  }
}

void SnapshotStore::wait_spill(Entry* e) {
  const std::shared_ptr<Spill> s = e->spill;
  if (!s) return;
  ready(s);
  reap();
}

void SnapshotStore::settle() {
  for (auto& e : entries_) {
    wait_spill(e.get());
    ready(e->range_write);
  }
  discard_failed_ranges();
}

void SnapshotStore::remove_file(const std::string& path, uint64_t charged) {
  std::error_code ec;
  fs::remove(to_path(path), ec);
  if (!ec) disk_bytes_ -= charged;  // an inaccessible file may still occupy its full reservation
}

void SnapshotStore::discard_failed_ranges() {
  if (!disk_failed_.load()) return;
  for (size_t i = entries_.size(); i-- > 0;) {
    const auto& job = entries_[i]->range_write;
    if (!job) continue;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (job->state != 2) continue;
    }
    // A child can only attach after this job succeeds. A failed range is always a leaf.
    drop(i);
  }
}

void SnapshotStore::spill_some(const Entry* restore) {
  if (!disk_enabled()) return;
  // The snapshots staying in memory, past three quarters of its budget: the least recently used start for the
  // disk, all but the latest.
  uint64_t staying = 0;
  int count = 0;
  for (const auto& e : entries_)
    if (e->file.empty() && !e->spill) staying += e->bytes, ++count;
  while (staying > budget_ / 4 * 3 && count > 1) {
    Entry* oldest = nullptr;
    for (const auto& e : entries_)
      if (e->file.empty() && !e->spill && e.get() != restore && (!oldest || e->used < oldest->used)) oldest = e.get();
    if (!oldest || !start_spill(oldest, restore)) return;
    staying -= oldest->bytes;
    --count;
  }
}

bool SnapshotStore::put(const flyx_backend_api* api, flyx_seq* seq, int slot, const std::vector<int32_t>& prefix,
                        std::string* why, const PutOptions& opts) {
  std::shared_ptr<const History> history = opts.history;
  bool* on_disk = opts.on_disk;
  uint64_t* disk_tip = opts.disk_tip;
  const bool staging_only = opts.staging_only;
  const Entry* restore = opts.restore;
  static const std::vector<interfaces::MediaSpan> kEmptyMedia;
  const auto& media = opts.media ? *opts.media : kEmptyMedia;
  reap();
  discard_failed_ranges();
  if (on_disk) *on_disk = false;
  if (disk_failed_.load() && (!budget_ || (disk_tip && !history))) return *why = kDiskFailed, false;
  if (staging_only && (!disk_enabled() || !disk_tip))
    return *why = "staging decoded rows requires a disk lineage", false;
  Entry* parent = disk_tip ? by_id(ancestor(*disk_tip, prefix.size())) : nullptr;
  while (parent && (!interfaces::media_prefix_equal(parent->prefix, parent->media, prefix, media) ||
                    !available(*parent))) {
    if (!disk_enabled()) return *why = kDiskFailed, false;
    parent = by_id(parent->parent);
  }
  if (parent && parent->prefix.size() == prefix.size() && parent->media == media &&
      (staging_only || !parent->staging_only)) {
    if (history) parent->history = std::move(history);
    parent->used = ++clock_;
    *disk_tip = parent->disk_id;
    if (on_disk) *on_disk = true;
    return true;
  }
  int begin = parent ? static_cast<int>(parent->prefix.size()) : 0;
  uint64_t bytes = 0;
  if (api->snapshot_bytes(seq, slot, begin, &bytes) != 0) return *why = api->last_error(), false;
  // Proactive anchors go to disk while a conversation is live. Preserve the RAM-first policy for independent
  // parked checkpoints that already fit there.
  const uint64_t held = restore ? restore->pieces.size() * piece_ : 0;
  const uint64_t ram = (budget_ - std::min(held, budget_)) / piece_ * piece_;
  const bool direct = bytes > ram || parent || (disk_enabled() && disk_tip && !history);
  if (direct && restore && !disk_enabled() && fits_in_memory(bytes))
    return *why = "keeping the incoming conversation's checkpoint", false;
  if (direct && !disk_enabled())
    return *why = "a snapshot of " + gb(bytes) + " does not fit the host cache of " + gb(budget_), false;
  // A snapshot of the same tokens and media goes first: this one replaces it.
  for (size_t i = 0; i < entries_.size(); ++i)
    if (!staging_only && entries_[i].get() != restore && !entries_[i]->disk_id &&
        entries_[i]->prefix == prefix && entries_[i]->media == media) {
      drop(i);
      break;
    }
  auto e = std::make_unique<Entry>();
  e->piece = piece_;
  e->staging_only = staging_only;
  if (direct) {
    bool room = disk_room(bytes, restore, parent);
    if (!room && parent && disk_enabled()) {
      // A chain's recurrent checkpoints can fill the disk before the latest full snapshot would. Start a new root.
      begin = 0;
      parent = nullptr;
      if (api->snapshot_bytes(seq, slot, begin, &bytes) != 0) return *why = api->last_error(), false;
      room = disk_room(bytes, restore);
    }
    if (!room) return *why = "no room on disk for a snapshot of " + gb(bytes), false;
    if (!put_on_disk(api, seq, slot, begin, bytes, static_cast<int>(prefix.size()), e.get(), why)) return false;
    if (disk_tip) {
      e->parent = parent ? parent->disk_id : 0;
      if (parent) ++parent->children;
      e->disk_id = ++clock_;
      *disk_tip = e->disk_id;
    }
  } else {
    std::vector<HostBuffer> pieces;
    if (!take(api, pieces_for(bytes, piece_), &pieces, restore))
      return *why = "no host memory for a snapshot of " + gb(bytes), false;
    set_entry_pieces(e.get(), std::move(pieces));
    const flyx_host_pieces dst = e->view();
    const int pos = api->snapshot_save(seq, slot, 0, &dst, bytes);
    if (pos < 0 || pos != static_cast<int>(prefix.size())) {
      recycle_pieces(e->pieces);
      return *why = pos < 0 ? api->last_error() : "the checkpoint is not where its tokens end", false;
    }
    bytes_ += bytes;
  }
  e->prefix = prefix;
  e->media = media;
  e->history = std::move(history);
  e->bytes = bytes;
  e->used = ++clock_;
  entries_.push_back(std::move(e));
  if (on_disk) *on_disk = direct;
  spill_some(restore);
  return true;
}

// The pieces a snapshot goes through to or from its file: the first (the header, given with every call) and kBounce
// for the others, of the backend's page-locked memory where it gives some; false when there is none to be had.
namespace {
bool bounce_pieces(const flyx_backend_api* api, uint64_t piece, size_t n, HostBuffer* head,
                   std::vector<HostBuffer>* rest) {
  *head = HostBuffer(api, piece);
  for (size_t i = 0; head->get() && i < std::min(SnapshotStore::kBounce, n - 1); ++i) {
    rest->emplace_back(api, piece);
    if (!rest->back().get()) return false;
  }
  return head->get() != nullptr;
}
}  // namespace

bool SnapshotStore::put_on_disk(const flyx_backend_api* api, flyx_seq* seq, int slot, int begin, uint64_t bytes,
                                int expect, Entry* e, std::string* why) {
  const size_t n = pieces_for(bytes, piece_);
  {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return disk_failed_.load() || staged_ <= kBounce; });
    if (disk_failed_.load()) return *why = kDiskFailed, false;
  }
  HostBuffer head(api, std::min(piece_, bytes));
  if (!head.get()) return *why = "no host memory for the snapshot header", false;
  auto job = std::make_shared<Spill>();
  job->path = next_snapshot_path();
  job->bytes = bytes;
  job->piece = piece_;
  disk_bytes_ += bytes;
  // Keep the header for all captures. Transfer ownership of it last, so current model rows may change immediately
  // after the call. Two batches fit in the staging limit, overlapping device capture with the previous write.
  std::vector<void*> at(n, nullptr);
  at[0] = head.get();
  std::string error;
  for (size_t first = 1;;) {
    const size_t count = std::min(n - first, kBounce / 2);
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [&] { return disk_failed_.load() || staged_ + count <= kBounce; });
      if (!job->error.empty()) error = job->error;
      else if (disk_failed_.load()) error = kDiskFailed;
    }
    WriteBatch batch{job, {}, first, false};
    for (size_t i = first; error.empty() && i < first + count; ++i) {
      batch.buffers.emplace_back(api, std::min(piece_, bytes - i * piece_));
      at[i] = batch.buffers.back().get();
      if (!at[i]) error = "no host memory for snapshot staging";
    }
    if (error.empty()) {
      const flyx_host_pieces dst{at.data(), static_cast<int32_t>(n), 0, piece_};
      const int pos = api->snapshot_save(seq, slot, begin, &dst, bytes);
      if (pos != expect) error = pos < 0 ? api->last_error() : "the checkpoint is not where its tokens end";
    }
    if (!error.empty()) break;
    if (count) enqueue(std::move(batch));
    std::fill(at.begin() + first, at.begin() + first + count, nullptr);
    first += count;
    if (first == n) break;
  }
  WriteBatch last{job, {}, 0, true, !error.empty()};
  if (error.empty()) last.buffers.push_back(std::move(head));
  enqueue(std::move(last));
  if (!error.empty()) {
    ready(job);
    remove_file(job->path, bytes);
    return *why = "a snapshot to disk: " + error, false;
  }
  e->file = job->path;
  e->range_write = job;
  return true;
}

int SnapshotStore::load_in_parts(const flyx_backend_api* api, flyx_seq* seq, const Entry& e, int32_t anchor,
                                 std::string* err) {
  const size_t n = pieces_for(e.bytes, e.piece);
  {
    // An unrelated write may still hold the store's transfer budget. Reads use the same bounded allowance.
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return staged_ == 0; });
  }
  HostBuffer head;
  std::vector<HostBuffer> bounce;
  if (!bounce_pieces(api, e.piece, n, &head, &bounce))
    return *err = "no host memory to read a snapshot back from disk", -1;
  os::RandomAccessFile f;
  if (!f.open(e.file, e.bytes, err, os::RandomAccessFile::kReadExisting)) return -1;
  auto read = [&](size_t i, void* p) {
    const uint64_t off = uint64_t(i) * e.piece;
    return f.read_at(off, p, size_t(std::min(e.piece, e.bytes - off)), err);
  };
  // Each call reads the header and sets the state the pieces it is given hold; the last one completes it.
  std::vector<void*> at(n, nullptr);
  at[0] = head.get();
  bool ok = read(0, head.get());
  int pos = -1;
  for (size_t c0 = 1; ok;) {
    const size_t c1 = std::min(n, c0 + bounce.size());
    for (size_t i = c0; ok && i < c1; ++i) ok = read(i, at[i] = bounce[i - c0].get());
    if (!ok) break;
    const flyx_host_pieces src{at.data(), static_cast<int32_t>(n), 0, e.piece};
    pos = api->snapshot_load(seq, &src, e.bytes, anchor);
    if (pos < 0) {
      *err = api->last_error();
      ok = false;
      break;
    }
    for (size_t i = c0; i < c1; ++i) at[i] = nullptr;
    if (c1 >= n) break;
    c0 = c1;
  }
  f.close();
  return ok ? pos : -1;
}

bool SnapshotStore::take(const flyx_backend_api* api, size_t n, std::vector<HostBuffer>* out, const Entry* restore) {
  if (n > budget_ / piece_) return false;
  while (out->size() < n) {
    if (!spare_.empty()) {
      out->push_back(std::move(spare_.back()));
      spare_.pop_back();
    } else if (reserved_ + piece_ <= budget_) {
      HostBuffer b(api, piece_);
      if (!b.get()) break;
      reserved_ += piece_;
      out->push_back(std::move(b));
    } else if (!reclaim_pieces(restore)) {
      break;
    }
  }
  if (out->size() == n) return true;
  recycle_pieces(*out);
  return false;
}

void SnapshotStore::own_memory() {
  settle();  // a snapshot being written reads its pieces
  for (HostBuffer& b : spare_)
    if (b.backend_owned()) {
      reserved_ -= b.capacity();
      b.reset();
    }
  spare_.erase(std::remove_if(spare_.begin(), spare_.end(), [](const auto& b) { return !b.get(); }), spare_.end());
  for (auto& e : entries_) {
    for (size_t i = 0; i < e->pieces.size(); ++i) {
      HostBuffer& p = e->pieces[i];
      if (!p.backend_owned()) continue;
      HostBuffer own(nullptr, piece_);
      if (!own.get()) {  // no memory of its own for it: the snapshot goes
        reserved_ -= uint64_t(e->pieces.size()) * piece_;
        e->pieces.clear();
        bytes_ -= e->bytes;
        e->bytes = 0;
        break;
      }
      std::copy_n(p.get(), std::min(piece_, e->bytes - i * piece_), own.get());
      p = std::move(own);
      e->at[i] = p.get();
    }
  }
  entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [](const auto& e) { return e->bytes == 0; }),
                 entries_.end());
}

const SnapshotStore::Entry* SnapshotStore::find(const std::vector<int32_t>& prompt, size_t limit, size_t floor,
                                                const std::vector<interfaces::MediaSpan>* media) {
  static const std::vector<interfaces::MediaSpan> kEmptyMedia;
  const auto& pm = media ? *media : kEmptyMedia;
  const Entry* best = nullptr;
  for (const auto& e : entries_) {
    const size_t n = e->prefix.size();
    if (e->staging_only || n <= floor || n > limit || n > prompt.size() || (best && n < best->prefix.size())) continue;
    if (interfaces::media_prefix_equal(e->prefix, e->media, prompt, pm) && available(*e)) best = e.get();
  }
  discard_failed_ranges();
  return best;
}

const SnapshotStore::History* SnapshotStore::continued_by(const std::string& text) {
  const History* best = nullptr;
  for (const auto& e : entries_) {
    if (e->staging_only || !e->history || !e->history->media.empty()) continue;
    const std::string& h = e->history->text;
    if (h.empty() || h.size() > text.size() || (best && h.size() < best->text.size())) continue;
    if (text.compare(0, h.size(), h) == 0 && available(*e)) best = e->history.get();
  }
  discard_failed_ranges();
  return best;
}

int SnapshotStore::load(const flyx_backend_api* api, flyx_seq* seq, const Entry& entry, int32_t anchor,
                        std::string* err, size_t* failed_end) {
  Entry* e = nullptr;
  for (auto& x : entries_)
    if (x.get() == &entry) e = x.get();
  if (!e) return *err = "the snapshot is no longer kept", -1;
  if (failed_end) *failed_end = e->prefix.size();
  reap();
  if (e->staging_only) return *err = "a staging range needs a terminal checkpoint", -1;
  if (!e->file.empty() && (e->disk_id || !fits_in_memory(e->bytes))) {
    std::vector<Entry*> chain{e};
    while (chain.back()->parent) {
      Entry* parent = by_id(chain.back()->parent);
      if (!parent) return *err = "the parent snapshot is no longer kept", -1;
      chain.push_back(parent);
    }
    int pos = -1;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
      Entry& part = **it;
      const size_t end = part.prefix.size();
      if (failed_end) *failed_end = end;
      if (!ready(part.range_write)) return *err = part.range_write->error, -1;
      pos = load_in_parts(api, seq, part, end < e->prefix.size() ? e->prefix[end] : anchor, err);
      if (pos < 0) return -1;
      if (size_t(pos) != end) return *err = "the restored range did not end at its checkpoint", -1;
      part.used = ++clock_;
    }
    return pos;
  }
  if (!e->file.empty()) {  // back from disk into memory first
    std::vector<HostBuffer> pieces;
    if (!take(api, pieces_for(e->bytes, e->piece), &pieces, e))
      return *err = "no host memory to read a snapshot back from disk", -1;
    os::RandomAccessFile f;
    bool ok = f.open(e->file, e->bytes, err, os::RandomAccessFile::kReadExisting);
    for (size_t i = 0; ok && i < pieces.size(); ++i) {
      const uint64_t off = uint64_t(i) * e->piece;
      ok = f.read_at(off, pieces[i].get(), size_t(std::min(e->piece, e->bytes - off)), err);
    }
    f.close();
    if (!ok) {
      recycle_pieces(pieces);
      return -1;
    }
    remove_file(e->file, e->bytes);
    e->file.clear();
    bytes_ += e->bytes;
    set_entry_pieces(e, std::move(pieces));
  }
  const flyx_host_pieces src = e->view();
  const int pos = api->snapshot_load(seq, &src, e->bytes, anchor);
  if (pos < 0) return *err = api->last_error(), -1;
  e->used = ++clock_;
  spill_some();
  return pos;
}

bool SnapshotStore::reclaim_pieces(const Entry* restore) {
  // Wait for a pending write, else spill or evict the least recently used in-memory entry.
  Entry* writing = nullptr;
  size_t oldest = entries_.size();
  for (size_t i = 0; i < entries_.size(); ++i) {
    Entry* e = entries_[i].get();
    if (e->spill && (!writing || e->used < writing->used)) writing = e;
    if (!e->pieces.empty() && !e->spill && e != restore &&
        (oldest == entries_.size() || e->used < entries_[oldest]->used)) oldest = i;
  }
  if (writing) wait_spill(writing);
  else if (oldest == entries_.size()) return false;
  else {
    Entry* e = entries_[oldest].get();
    if (start_spill(e, restore)) wait_spill(e);
    else {
      // Trying to make disk room may have removed earlier entries.
      const auto it = std::find_if(entries_.begin(), entries_.end(), [e](const auto& x) { return x.get() == e; });
      drop(static_cast<size_t>(it - entries_.begin()));
    }
  }
  return true;
}

void SnapshotStore::set_budget(uint64_t budget) {
  budget_ = budget;
  while (reserved_ > budget_) {
    if (spare_.empty()) {
      if (!reclaim_pieces()) break;
    } else {
      reserved_ -= spare_.back().capacity();
      spare_.pop_back();
    }
  }
}

void SnapshotStore::drop(size_t i) {
  Entry* e = entries_[i].get();
  if (auto* parent = by_id(e->parent)) --parent->children;
  wait_spill(e);  // its pieces are being read
  ready(e->range_write);  // never remove a file while its writes are queued
  if (!e->file.empty()) {
    remove_file(e->file, e->bytes);
  } else {
    bytes_ -= e->bytes;
  }
  recycle_pieces(e->pieces);
  entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(i));
}

}  // namespace flyx
