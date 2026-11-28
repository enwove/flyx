// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/modelfile/modelfile.h"

#include "core/modelfile/readers.h"

namespace flyx {

namespace {

bool read(std::vector<std::unique_ptr<os::MappedFile>>* maps, const std::string& path, ModelFile::Contents* c,
          std::string* err, bool need_tokenizer, const PathAllowFn& allow_path) {
  if (allow_path && !allow_path(path))
    return *err = path + ": path is not allowed by remote path policy", false;
  maps->clear();
  maps->push_back(std::make_unique<os::MappedFile>());
  os::MappedFile* first = maps->back().get();
  if (!first->open(path, err)) return false;
  *c = {};
  if (modelfile::is_flyx(first->data(), first->size()))
    return modelfile::read_flyx(first->data(), first->size(), path, c, err);
  if (modelfile::is_gguf(first->data(), first->size()))
    return modelfile::read_gguf(maps, path, c, err, need_tokenizer, allow_path);
  *err = path + ": not a model file this build reads (.flyx or .gguf)";
  return false;
}

}  // namespace

std::string drafter_source_kind(const std::string& path, const PathAllowFn& allow_path) {
  if (allow_path && !allow_path(path)) return "";
  return modelfile::is_drafter_dir(path, allow_path) ? modelfile::drafter_dir_kind(path, allow_path)
                                                     : modelfile::flyx_draft_kind(path);
}

bool ModelFile::open(const std::string& path, std::string* err, const PathAllowFn& allow_path) {
  companions_.clear();
  maps_.clear();
  if (allow_path && !allow_path(path))
    return *err = path + ": path is not allowed by remote path policy", false;
  if (modelfile::is_drafter_dir(path, allow_path))
    return *err = path + " holds a drafter alone, not a model: give it beside the model (--drafter-file)", false;
  if (!read(&maps_, path, &c_, err, /*need_tokenizer=*/true, allow_path)) return false;
  // The header is the file's word: a key of another type than expected is read as absent, never thrown on.
  const auto kind = c_.header.find("kind");
  if (kind != c_.header.end() && kind->is_string() && kind->get_ref<const std::string&>() == "drafter")
    return *err = path + " holds a drafter alone, not a model: give it beside the model (--drafter-file)", false;
  if (c_.header.is_object()) c_.header.erase("companions");  // add() lists them; what a file put there is not one
  return true;
}

bool ModelFile::add(const std::string& path, std::string* err, const PathAllowFn& allow_path) {
  return add(path, err, allow_path, CompanionRole::Any);
}

bool ModelFile::add(const std::string& path, std::string* err, const PathAllowFn& allow_path, CompanionRole role) {
  if (allow_path && !allow_path(path))
    return *err = path + ": path is not allowed by remote path policy", false;
  auto more = std::make_unique<Companion>();
  if (modelfile::is_peft_dir(path, allow_path)) {  // before is_drafter_dir: some adapter directories carry a config.json too
    if (role == CompanionRole::Drafter || role == CompanionRole::Projector)
      return *err = path + ": LoRA adapter requires --lora", false;
    if (!modelfile::read_peft_dir(path, &more->maps, &more->c, err, allow_path)) return false;
  } else if (modelfile::is_drafter_dir(path, allow_path)) {
    if (!modelfile::read_drafter_dir(path, &more->maps, &more->c, err, allow_path)) return false;
  } else {
    if (!read(&more->maps, path, &more->c, err, /*need_tokenizer=*/false, allow_path)) return false;
  }
  const auto kind = more->c.header.find("kind");
  const bool lora = kind != more->c.header.end() && *kind == "lora";
  if (role == CompanionRole::Lora && !lora) return *err = path + ": not a LoRA adapter", false;
  if (lora && (role == CompanionRole::Drafter || role == CompanionRole::Projector))
    return *err = path + ": LoRA adapter requires --lora", false;
  more->role = lora ? CompanionRole::Lora : role;
  c_.header["companions"].push_back(more->c.header);
  companions_.push_back(std::move(more));
  return true;
}

void ModelFile::remove_last_companion() {
  c_.header.at("companions").erase(companions_.size() - 1);
  companions_.pop_back();
}

const flyx_tensor_view* ModelFile::tensor(const std::string& name) const {
  const Contents* c = &c_;
  std::string_view key = name;
  if (key.compare(0, std::char_traits<char>::length(kCompanion), kCompanion) == 0) {  // "companion/<i>/<name>"
    key.remove_prefix(std::char_traits<char>::length(kCompanion));
    const size_t slash = key.find('/');
    if (slash == std::string_view::npos || slash == 0 || slash > 6) return nullptr;
    int i = 0;
    for (char ch : key.substr(0, slash)) {
      if (ch < '0' || ch > '9') return nullptr;
      i = 10 * i + (ch - '0');
    }
    if (i >= companions()) return nullptr;
    c = &companions_[size_t(i)]->c;
    key.remove_prefix(slash + 1);
  }
  auto it = c->index.find(std::string(key));
  return it == c->index.end() ? nullptr : &c->tensors[it->second];
}

std::string_view ModelFile::blob(const std::string& name) const {
  auto it = c_.blobs.find(name);
  return it == c_.blobs.end() ? std::string_view() : it->second;
}

uint64_t ModelFile::bytes() const {
  uint64_t n = 0;
  for (const auto& m : maps_) n += m->size();
  for (const auto& c : companions_)
    for (const auto& m : c->maps) n += m->size();
  return n;
}

void ModelFile::read_ahead(const std::atomic<bool>& stop) const {
  constexpr size_t kPiece = size_t(64) << 20;
  auto ahead = [&](const os::MappedFile& m) {
    for (size_t o = 0; o < m.size() && !stop; o += kPiece) m.read_ahead(o, kPiece);
  };
  for (const auto& m : maps_) ahead(*m);
  for (const auto& c : companions_)
    for (const auto& m : c->maps) ahead(*m);
}

std::vector<int32_t> ModelFile::eos_ids() const {
  std::vector<int32_t> out;
  if (c_.header.is_object() && c_.header.contains("generation") && c_.header["generation"].is_object()) {
    const auto& g = c_.header["generation"];
    if (g.contains("eos_ids") && g["eos_ids"].is_array()) {
      for (const auto& id : g["eos_ids"])
        if (id.is_number_integer()) out.push_back(id.get<int32_t>());
    }
  }
  return out;
}

ModelOrigin model_origin(const std::string& path, const PathAllowFn& allow_path) {
  ModelOrigin out;
  if (allow_path && !allow_path(path)) return out;
  if (const GgufPeek p = gguf_peek(path, allow_path); p.valid) return out.base_models = p.base_models, out;
  nlohmann::json h;
  if (!modelfile::peek_flyx_header(path, &h) || !h.contains("source") || !h["source"].is_object()) return out;
  const nlohmann::json& src = h["source"];
  if (src.contains("repo") && src["repo"].is_string()) out.repo = src["repo"].get<std::string>();
  if (src.contains("base_models") && src["base_models"].is_array())
    for (const auto& r : src["base_models"])
      if (r.is_string()) out.base_models.push_back(r.get<std::string>());
  return out;
}

}  // namespace flyx
