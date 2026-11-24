// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/exec/request.h"

#include <algorithm>
#include <memory>

#include "core/os/os.h"

namespace flyx {
namespace {

int image_token_budget(const std::string& detail, int max_ctx, size_t n_images) {
  const int avail = std::max(256, prompt_limit(max_ctx) - 256);
  if (detail == "low") return std::min(256, avail);
  if (detail == "high") return std::min(4096, avail);
  const int per = std::max(256, (avail / 2) / static_cast<int>(std::max<size_t>(1, n_images)));
  return std::min(2048, std::min(per, avail));
}

struct ByteTokSeg {
  size_t byte_begin = 0, byte_end = 0;
  int32_t tok_begin = 0, tok_end = 0;
  bool is_media = false;
};

std::vector<int32_t> map_byte_offsets(const Tokenizer& tok, const std::vector<int32_t>& prompt,
                                      const std::vector<ByteTokSeg>& segs, const std::vector<size_t>& offsets) {
  std::vector<int32_t> out;
  out.reserve(offsets.size());
  for (size_t off : offsets) {
    int32_t mapped = static_cast<int32_t>(prompt.size());
    for (const auto& seg : segs) {
      if (off < seg.byte_begin) {
        mapped = seg.tok_begin;
        break;
      }
      if (off >= seg.byte_end) continue;
      if (seg.is_media) {
        mapped = off == seg.byte_begin ? seg.tok_begin : seg.tok_end;
      } else {
        size_t b = seg.byte_begin;
        mapped = seg.tok_end;
        for (int32_t t = seg.tok_begin; t < seg.tok_end; ++t) {
          if (b >= off) {
            mapped = t;
            break;
          }
          b += tok.token_bytes(prompt[size_t(t)]).size();
        }
      }
      break;
    }
    out.push_back(mapped);
  }
  return out;
}

std::vector<TextSpan> clip_literal(const std::vector<TextSpan>& lit, size_t begin, size_t end) {
  std::vector<TextSpan> out;
  for (const auto& sp : lit) {
    if (sp.end <= begin) continue;
    if (sp.begin >= end) break;
    out.push_back({std::max(sp.begin, begin) - begin, std::min(sp.end, end) - begin});
  }
  return out;
}

void append_media_embeds(const vision::CachedVision& cv, int32_t token_pos, const interfaces::ImageInput& img,
                         GenerateRequest* gen) {
  interfaces::MediaSpan span;
  span.token_pos = token_pos;
  span.n_tokens = static_cast<int32_t>(cv.tokens.size());
  span.sha256 = img.sha256;
  for (int i = 0; i < 3 && size_t(i) < cv.grid_thw.size(); ++i) span.grid_thw[i] = cv.grid_thw[size_t(i)];
  span.n_embeds = cv.n_embeds;
  span.embed_offsets = cv.embed_offsets;
  gen->media_spans.push_back(std::move(span));
  gen->prompt_embeds.insert(gen->prompt_embeds.end(), cv.embeds.begin(), cv.embeds.end());
  if (cv.embed_offsets.empty()) {
    for (int32_t j = 0; j < cv.n_embeds; ++j) gen->prompt_embed_rows.push_back(token_pos + j);
  } else {
    for (int32_t off : cv.embed_offsets) gen->prompt_embed_rows.push_back(token_pos + off);
  }
}

bool prepare_multimodal_prompt(Engine& e, const Session& s, const RenderedChat& rc, PreparedRequest* out,
                               std::string* err) {
  if (!e.has_vision()) {
    out->invalid = true;
    *err = "this model has no vision encoder loaded (pass --mmproj <mmproj.gguf>)";
    return false;
  }
  const size_t n_imgs = rc.images.size();
  std::vector<std::shared_ptr<const vision::CachedVision>> encoded(n_imgs);
  for (size_t i = 0; i < n_imgs; ++i) {
    if (!rc.images[i].image) {
      out->invalid = true;
      *err = "missing image data in multimodal message";
      return false;
    }
    const int budget = image_token_budget(rc.images[i].image->detail, e.options().max_ctx, n_imgs);
    if (!e.encode_image(*rc.images[i].image, budget, &encoded[i], err)) {
      out->invalid = true;
      return false;
    }
  }

  GenerateRequest& gen = out->gen;
  const Tokenizer& tok = e.tokenizer();
  std::vector<ByteTokSeg> segs;

  bool exact_offsets = true;
  for (const auto& ri : rc.images)
    if (ri.byte_offset == std::string::npos || ri.byte_end == std::string::npos || ri.byte_end < ri.byte_offset ||
        ri.byte_end > rc.text.size())
      exact_offsets = false;

  if (exact_offsets) {
    // Reuse session history tokens when rc continues history_text_ with matching leading images.
    size_t start_byte = 0;
    size_t start_img = 0;
    const auto& htext = s.history_text();
    const auto& himgs = s.history_images();
    const auto& hmed = s.history_media();
    bool reuse_history = !htext.empty() && rc.text.size() >= htext.size() &&
                         rc.text.compare(0, htext.size(), htext) == 0 && himgs.size() <= n_imgs &&
                         hmed.size() == himgs.size();
    for (size_t k = 0; reuse_history && k < himgs.size(); ++k) {
      const auto &ha = himgs[k], &rb = rc.images[k];
      if (!ha.image || !rb.image || ha.byte_offset != rb.byte_offset || ha.byte_end != rb.byte_end ||
          ha.image->sha256 != rb.image->sha256 || ha.image->detail != rb.image->detail ||
          hmed[k].n_tokens != static_cast<int32_t>(encoded[k]->tokens.size()) ||
          hmed[k].n_embeds != encoded[k]->n_embeds)
        reuse_history = false;
    }
    if (reuse_history) {
      gen.prompt = s.history();
      out->tokenized = "history kept its tokens";
      size_t bcur = 0;
      int32_t tcur = 0;
      for (size_t k = 0; k < himgs.size(); ++k) {
        const auto& ri = rc.images[k];
        const auto& ms = hmed[k];
        if (ri.byte_offset > bcur || ms.token_pos > tcur)
          segs.push_back({bcur, ri.byte_offset, tcur, ms.token_pos, false});
        segs.push_back({ri.byte_offset, ri.byte_end, ms.token_pos, ms.token_pos + ms.n_tokens, true});
        append_media_embeds(*encoded[k], ms.token_pos, *ri.image, &gen);
        bcur = ri.byte_end;
        tcur = ms.token_pos + ms.n_tokens;
      }
      const int32_t hist_toks = static_cast<int32_t>(gen.prompt.size());
      if (htext.size() > bcur || hist_toks > tcur) segs.push_back({bcur, htext.size(), tcur, hist_toks, false});
      start_byte = htext.size();
      start_img = himgs.size();
    } else if (!htext.empty()) {
      out->tokenized = "history text changed: re-encoded";
    }

    size_t bcur = start_byte;
    for (size_t k = start_img; k < n_imgs; ++k) {
      const auto& ri = rc.images[k];
      if (ri.byte_offset < bcur) {
        out->invalid = true;
        *err = "overlapping image placeholder offsets";
        return false;
      }
      if (ri.byte_offset > bcur) {
        const int32_t t0 = static_cast<int32_t>(gen.prompt.size());
        const auto lit = clip_literal(rc.literal, bcur, ri.byte_offset);
        auto ids = tok.encode(std::string_view(rc.text).substr(bcur, ri.byte_offset - bcur), &lit);
        gen.prompt.insert(gen.prompt.end(), ids.begin(), ids.end());
        segs.push_back({bcur, ri.byte_offset, t0, static_cast<int32_t>(gen.prompt.size()), false});
      }
      const int32_t t0 = static_cast<int32_t>(gen.prompt.size());
      gen.prompt.insert(gen.prompt.end(), encoded[k]->tokens.begin(), encoded[k]->tokens.end());
      const int32_t t1 = static_cast<int32_t>(gen.prompt.size());
      segs.push_back({ri.byte_offset, ri.byte_end, t0, t1, true});
      append_media_embeds(*encoded[k], t0, *ri.image, &gen);
      bcur = ri.byte_end;
    }
    if (bcur < rc.text.size()) {
      const int32_t t0 = static_cast<int32_t>(gen.prompt.size());
      const auto lit = clip_literal(rc.literal, bcur, rc.text.size());
      auto ids = tok.encode(std::string_view(rc.text).substr(bcur), &lit);
      gen.prompt.insert(gen.prompt.end(), ids.begin(), ids.end());
      segs.push_back({bcur, rc.text.size(), t0, static_cast<int32_t>(gen.prompt.size()), false});
    }
  } else {
    // Non-native Jinja template fallback: match occurrences of the adapter's literal placeholder token ID.
    const int32_t ph = e.model().vision()->placeholder_token();
    const auto base = tok.encode(rc.text, &rc.literal);
    size_t found = 0;
    for (int32_t id : base)
      if (id == ph) ++found;
    if (found != n_imgs) {
      out->invalid = true;
      *err = "chat template emitted " + std::to_string(found) + " image placeholders for " + std::to_string(n_imgs) +
             " images";
      return false;
    }
    size_t bcur = 0;
    size_t img_i = 0;
    int32_t seg_t0 = 0;
    size_t seg_b0 = 0;
    for (int32_t id : base) {
      const size_t tb = tok.token_bytes(id).size();
      if (id == ph) {
        const int32_t cur_t = static_cast<int32_t>(gen.prompt.size());
        if (cur_t > seg_t0 || bcur > seg_b0) segs.push_back({seg_b0, bcur, seg_t0, cur_t, false});
        const auto& cv = *encoded[img_i];
        gen.prompt.insert(gen.prompt.end(), cv.tokens.begin(), cv.tokens.end());
        const int32_t end_t = static_cast<int32_t>(gen.prompt.size());
        segs.push_back({bcur, bcur + tb, cur_t, end_t, true});
        append_media_embeds(cv, cur_t, *rc.images[img_i].image, &gen);
        ++img_i;
        bcur += tb;
        seg_b0 = bcur;
        seg_t0 = end_t;
      } else {
        gen.prompt.push_back(id);
        bcur += tb;
      }
    }
    if (static_cast<int32_t>(gen.prompt.size()) > seg_t0 || bcur > seg_b0)
      segs.push_back({seg_b0, bcur, seg_t0, static_cast<int32_t>(gen.prompt.size()), false});
  }

  gen.checkpoints = map_byte_offsets(tok, gen.prompt, segs, rc.boundaries);
  gen.cache_prefixes = map_byte_offsets(tok, gen.prompt, segs, rc.cache_prefixes);
  if (rc.system_end) gen.system_end = map_byte_offsets(tok, gen.prompt, segs, {rc.system_end}).front();

  auto pp = e.model().vision()->build_positions(gen.prompt, gen.media_spans);
  gen.has_mrope = pp.is_3d;
  gen.pos_3d = std::move(pp.pos_3d);
  gen.rope_delta = pp.rope_delta;
  gen.rendered_text = rc.text;
  gen.rendered_images = rc.images;
  return true;
}

}  // namespace

bool render_request(const ChatTemplate& chat, const Tokenizer& tok, const Request& rq, RenderedChat* out,
                    std::string* err, bool branch_points) {
  *out = {};
  if (rq.chat) {
    if (chat.render(rq.messages, rq.tools, rq.chat_options, out, err, branch_points, rq.cache_points)) return true;
    if (!rq.join_leading_system || !rq.cache_points.empty() || rq.messages.size() < 2 ||
        rq.messages[0].role != "system" || rq.messages[1].role != "system") return false;
    // Some templates accept only one system message. Preserve the order of its instructions and leave later turns alone.
    std::vector<ChatMessage> messages{rq.messages.front()};
    auto next = rq.messages.begin() + 1;
    for (; next != rq.messages.end() && next->role == "system"; ++next) {
      messages.front().content += "\n\n";
      messages.front().content += next->content;
    }
    messages.insert(messages.end(), next, rq.messages.end());
    std::string retry_error;
    if (!chat.render(messages, rq.tools, rq.chat_options, out, &retry_error, branch_points)) return false;
    err->clear();
    return true;
  }
  if (!rq.tokens.empty()) return true;
  if (rq.text.empty()) return *err = "empty prompt", false;
  // The prefix as text, as a template writes its <s>: a session's history (decoded text) then continues it.  Each
  // token in a span of its own, so that a longer special token cannot take it together with the caller's text.
  for (int32_t id : tok.prefix()) {
    const size_t at = out->text.size();
    out->text += tok.token_bytes(id);
    out->literal.push_back({at, out->text.size()});
  }
  const size_t at = out->text.size();
  out->text += rq.text;
  out->literal.push_back({at, out->text.size()});
  out->conversation_end = out->text.size();
  return true;
}

bool prepare_request(Engine& e, const Session& s, const Request& rq, const RenderedChat& rc, PreparedRequest* out,
                     std::string* err) {
  *out = {};
  out->gen = rq.gen;
  GenerateRequest& gen = out->gen;
  if (!rq.chat && !rq.tokens.empty()) {
    gen.prompt = rq.tokens;
    gen.checkpoints = token_positions(e.tokenizer(), gen.prompt, rc.boundaries);
    gen.cache_prefixes = token_positions(e.tokenizer(), gen.prompt, rc.cache_prefixes);
    if (rc.system_end) gen.system_end = token_positions(e.tokenizer(), gen.prompt, {rc.system_end}).front();
  } else if (!rc.images.empty()) {
    if (!prepare_multimodal_prompt(e, s, rc, out, err)) return false;
  } else {
    gen.prompt = s.tokenize(rc.text, &rc.literal, &out->tokenized);
    gen.checkpoints = token_positions(e.tokenizer(), gen.prompt, rc.boundaries);
    gen.cache_prefixes = token_positions(e.tokenizer(), gen.prompt, rc.cache_prefixes);
    if (rc.system_end) gen.system_end = token_positions(e.tokenizer(), gen.prompt, {rc.system_end}).front();
  }
  // A question: keep the checkpoint where it ends through the tool calls that follow.
  if (rq.chat && !rq.messages.empty() && rq.messages.back().role == "user" && !gen.checkpoints.empty())
    gen.keep_checkpoint = gen.checkpoints.back();
  if (!rq.chat || gen.prefill_only) return true;

  // The grammar the reply must match: tool calls in the format's syntax with
  // arguments that fit their schemas, tool_choice, a JSON answer (core/grammar).
  const std::string tag = e.chat().reply_grammar(rq.tools, rq.rules, rc);
  if (!tag.empty()) {
    const double t0 = os::now_ms();
    std::string why;
    out->grammar = e.grammar().matcher(tag, &why, &out->grammar_cached);
    out->grammar_ms = os::now_ms() - t0;
    if (!out->grammar) {
      const bool must = rq.rules.tools == ReplyRules::kRequired || rq.rules.tools == ReplyRules::kForced ||
                        rq.rules.json;
      if (must) {
        out->invalid = e.grammar().ready();  // the compiler is there: what it was given is at fault
        return *err = why, false;
      }
      out->unconstrained = why;
    }
  }
  gen.grammar = out->grammar.get();
  if (rq.split_output)
    out->parser = e.chat().parser(rc, rq.rules.tools == ReplyRules::kNone ? nullptr : &rq.tools);
  return true;
}

void warm_request(Engine& e, const Request& rq, const RenderedChat& rc) {
  if (!rq.chat) return;
  const std::string tag = e.chat().reply_grammar(rq.tools, rq.rules, rc);
  std::string why;
  if (!tag.empty()) (void)e.grammar().matcher(tag, &why);
}

}  // namespace flyx
