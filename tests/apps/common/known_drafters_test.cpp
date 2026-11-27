// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The verified drafters list (apps/common/model_source.h, drafters.json): the entries made for a model's repository,
// then those of its base models, whose kind this build runs, in the list's order; a list not version 1 throughout is
// refused whole.
#include <string>
#include <vector>

#include "apps/common/model_source.h"
#include "tests/common/check.h"

using namespace flyx::apps;

namespace {

std::string repos(const std::vector<KnownDrafter>& v) {
  std::string s;
  for (const KnownDrafter& k : v) s += (s.empty() ? "" : " ") + k.repo + ":" + k.kind;
  return s;
}

}  // namespace

int main() {
  const std::string list = R"({"version": 1, "drafters": [
    {"base": "Qwen/Qwen3-8B", "repo": "a/next", "kind": "dnext"},
    {"base": "Qwen/Qwen3-8B", "repo": "a/spark", "kind": "dspark"},
    {"base": "Qwen/Qwen3-8B-Base", "repo": "a/flash", "kind": "dflash2"},
    {"base": "Qwen/Qwen3.8-27B", "repo": "z/flash", "kind": "dflash2"}]})";
  // The model is Qwen/Qwen3-8B itself, its card naming Qwen3-8B-Base: DSpark made for it, then DFlash2 inferred; a
  // kind this build does not run left out.
  auto k = known_drafters_in(list, "Qwen/Qwen3-8B", {"Qwen/Qwen3-8B-Base"}, "finetune");
  check(repos(k) == "a/spark:dspark a/flash:dflash2" && k[0].exact && k[0].relation.empty() && !k[1].exact &&
            k[1].relation == "finetune" && k[1].base == "Qwen/Qwen3-8B-Base",
        "its own repository first (exact), then its base models' (inferred, with the relation): " + repos(k));
  // A quantization of it (unsloth's GGUF): only inferred.
  k = known_drafters_in(list, "unsloth/Qwen3-8B-GGUF", {"Qwen/Qwen3-8B"}, "quantized");
  check(repos(k) == "a/spark:dspark" && !k[0].exact && k[0].relation == "quantized", "a quantization: inferred");
  check(known_drafters_in(list, "", {"Qwen/Qwen3.5-4B"}, "").empty(), "a base model the list has none for");
  // A list that is not version 1 throughout is refused whole.
  const std::string good = R"({"base": "Qwen/Qwen3-8B", "repo": "a/spark", "kind": "dspark"})";
  const auto refused = [&](const std::string& entries, const std::string& version = "1") {
    return known_drafters_in(R"({"version": )" + version + R"(, "drafters": [)" + good + entries + "]}",
                             "Qwen/Qwen3-8B", {}, "")
        .empty();
  };
  check(!refused(""), "a well-formed list read");
  check(refused("", "2") && refused("", "\"1\""), "another version");
  check(refused(R"(, {"base": "Qwen/Qwen3-8B", "repo": 7, "kind": "dspark"})"), "an entry's field no string");
  check(refused(R"(, "a/loose")"), "an entry no object");
  check(refused(R"(, {"base": "Qwen/Qwen3-8B", "repo": "a/all", "kind": "all"})"), "a kind that names no drafter");
  check(refused(", " + good), "a repository twice");
  check(known_drafters_in("not json", "Qwen/Qwen3-8B", {}, "").empty(), "a list that is no JSON");
  check(known_drafters_in(R"({"version": 1, "drafters": 3})", "Qwen/Qwen3-8B", {}, "").empty(),
        "a list without an array");
  check(known_drafters("", {}, false).empty(), "nothing known of the model: none");
  return flyx::test::finish("known_drafters_test");
}
