// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Self-contained multimodal vision unit + golden test (no Python/torch assets required):
//   1. Image decoding (stb_image), base64 data URIs, SHA-256 identity, VisionCache LRU
//   2. OpenAI / Responses / Anthropic multimodal JSON parsing & validation
//   3. Qwen35VisionAdapter preprocessing (smart resize, spatial-merge patches, 3D MRoPE positions)
//   4. MediaSpan prefix safety (same-resolution different images never collide)
//   5. Tiny in-memory VLM (VisionStem + VitAttn + VitMlp + VisionMerge + MRoPE Attn + MTP)
//      on cpu_ref and cuda: encode_vision parity, prefill_ex + decode parity, Gate 4 (MTP == plain),
//      cross-chunk multimodal prefill parity, checkpoint & SnapshotStore rope_delta persistence.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/exec/engine.h"
#include "core/interfaces/chat.h"
#include "core/interfaces/vision.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "core/render/chat_json.h"
#include "core/session/store.h"
#include "core/vision/image.h"
#include "core/vision/media_span.h"
#include "flyx/backend_abi.h"
#include "models/vision/preprocess.h"
#include "ops/dtype/ref.h"
#include "tests/common/gpu.h"

namespace {

using namespace flyx;

#define CHECK(cond, msg)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, std::string(msg).c_str());   \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

// Build an uncompressed 24-bit BMP image in memory (bottom-up BGR, 4-byte padded rows).
std::vector<uint8_t> make_bmp(int w, int h, uint8_t r, uint8_t g, uint8_t b) {
  const int row_bytes = ((w * 3 + 3) / 4) * 4;
  const uint32_t img_bytes = uint32_t(row_bytes * h);
  const uint32_t file_bytes = 54 + img_bytes;
  std::vector<uint8_t> out(file_bytes, 0);
  auto put16 = [&](size_t off, uint16_t v) { out[off] = uint8_t(v); out[off + 1] = uint8_t(v >> 8); };
  auto put32 = [&](size_t off, uint32_t v) {
    out[off] = uint8_t(v);
    out[off + 1] = uint8_t(v >> 8);
    out[off + 2] = uint8_t(v >> 16);
    out[off + 3] = uint8_t(v >> 24);
  };
  out[0] = 'B';
  out[1] = 'M';
  put32(2, file_bytes);
  put32(10, 54);
  put32(14, 40);
  put32(18, uint32_t(w));
  put32(22, uint32_t(h));
  put16(26, 1);
  put16(28, 24);
  put32(34, img_bytes);
  for (int y = 0; y < h; ++y) {
    uint8_t* row = out.data() + 54 + size_t(y) * size_t(row_bytes);
    for (int x = 0; x < w; ++x) {
      row[size_t(x) * 3 + 0] = b;
      row[size_t(x) * 3 + 1] = g;
      row[size_t(x) * 3 + 2] = r;
    }
  }
  return out;
}

std::string encode_b64(const std::vector<uint8_t>& in) {
  static const char kTbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((in.size() + 2) / 3) * 4);
  for (size_t i = 0; i < in.size(); i += 3) {
    const uint32_t a = in[i];
    const uint32_t b = i + 1 < in.size() ? in[i + 1] : 0;
    const uint32_t c = i + 2 < in.size() ? in[i + 2] : 0;
    const uint32_t v = (a << 16) | (b << 8) | c;
    out.push_back(kTbl[(v >> 18) & 63]);
    out.push_back(kTbl[(v >> 12) & 63]);
    out.push_back(i + 1 < in.size() ? kTbl[(v >> 6) & 63] : '=');
    out.push_back(i + 2 < in.size() ? kTbl[v & 63] : '=');
  }
  return out;
}

uint16_t f32_to_bf16(float x) {
  uint32_t u = 0;
  std::memcpy(&u, &x, 4);
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return uint16_t(u >> 16);
}

// Build an uncompressed 32-bit BGRA TGA image in memory (top-left origin).
std::vector<uint8_t> make_rgba_tga(int w, int h, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  std::vector<uint8_t> out(18 + size_t(w) * size_t(h) * 4, 0);
  out[2] = 2;  // uncompressed true-color
  out[12] = uint8_t(w & 0xff);
  out[13] = uint8_t((w >> 8) & 0xff);
  out[14] = uint8_t(h & 0xff);
  out[15] = uint8_t((h >> 8) & 0xff);
  out[16] = 32;  // 32 bpp BGRA
  out[17] = 0x28;  // top-left origin, 8-bit alpha
  for (int i = 0; i < w * h; ++i) {
    out[18 + size_t(i) * 4 + 0] = b;
    out[18 + size_t(i) * 4 + 1] = g;
    out[18 + size_t(i) * 4 + 2] = r;
    out[18 + size_t(i) * 4 + 3] = a;
  }
  return out;
}

void test_image_and_cache() {
  const auto red_bmp = make_bmp(8, 8, 220, 30, 10);
  const auto blue_bmp = make_bmp(8, 8, 10, 40, 230);
  ImageInput img_r, img_b;
  std::string err;
  CHECK(vision::make_image_from_bytes(red_bmp, "image/bmp", "HIGH", &img_r, &err), err);
  CHECK(img_r.detail == "high", "make_image_from_bytes must canonicalize detail case-insensitively");
  CHECK(vision::make_image_from_bytes(blue_bmp, "image/bmp", "", &img_b, &err), err);
  CHECK(img_r.sha256 != img_b.sha256, "distinct images must produce distinct SHA-256 hashes");

  interfaces::RgbImage rgb;
  CHECK(vision::decode_image_bytes(img_r.bytes.data(), img_r.bytes.size(), &rgb, &err), err);
  CHECK(rgb.width == 8 && rgb.height == 8 && rgb.rgb.size() == 8 * 8 * 3, "decoded BMP size mismatch");
  CHECK(rgb.rgb[0] == 220 && rgb.rgb[1] == 30 && rgb.rgb[2] == 10, "decoded RGB values mismatch");

  // Alpha compositing onto white background (prevents transparent PNG/TGA black-halo regression).
  {
    const auto clear_tga = make_rgba_tga(4, 4, 200, 10, 20, 0);
    interfaces::RgbImage alpha0;
    CHECK(vision::decode_image_bytes(clear_tga.data(), clear_tga.size(), &alpha0, &err), err);
    CHECK(alpha0.rgb[0] == 255 && alpha0.rgb[1] == 255 && alpha0.rgb[2] == 255,
          "fully transparent pixel must composite to white");
    const auto half_tga = make_rgba_tga(4, 4, 200, 0, 0, 128);
    interfaces::RgbImage alpha50;
    CHECK(vision::decode_image_bytes(half_tga.data(), half_tga.size(), &alpha50, &err), err);
    // (200 * 128 + 255 * 127 + 127) / 255 = 227; (0 * 128 + 255 * 127 + 127) / 255 = 127
    CHECK(std::abs(int(alpha50.rgb[0]) - 227) <= 1 && std::abs(int(alpha50.rgb[1]) - 127) <= 1 &&
              std::abs(int(alpha50.rgb[2]) - 127) <= 1,
          "semi-transparent pixel must alpha-composite onto white");
  }

  // Strict base64 validation (stray char, bad padding, non-zero trailing bits).
  {
    std::vector<uint8_t> raw;
    CHECK(vision::decode_base64("AA==", &raw, &err) && raw.size() == 1 && raw[0] == 0, "valid AA== base64");
    CHECK(!vision::decode_base64("A", &raw, &err), "stray single base64 char must fail");
    CHECK(!vision::decode_base64("===", &raw, &err), "excess padding must fail");
    CHECK(!vision::decode_base64("AA=A", &raw, &err), "data after '=' padding must fail");
    CHECK(!vision::decode_base64("AB==", &raw, &err), "non-zero trailing 4 bits in == must fail");
    CHECK(!vision::decode_base64("ABD=", &raw, &err), "non-zero trailing 2 bits in = must fail");
  }

  // Data URI parsing (RFC 2397 parameters) & malformed/remote URI rejection.
  ImageInput parsed;
  const std::string uri = "data:image/bmp;base64," + encode_b64(red_bmp);
  CHECK(vision::parse_image_data_uri(uri, &parsed, &err), err);
  CHECK(parsed.sha256 == img_r.sha256, "data URI round-trip SHA-256 mismatch");
  CHECK(vision::parse_image_data_uri("data:image/bmp;charset=utf-8;BASE64," + encode_b64(red_bmp), &parsed, &err),
        "RFC 2397 parameter before ;base64 must succeed");
  CHECK(parsed.mime == "image/bmp" && parsed.sha256 == img_r.sha256, "RFC 2397 MIME extraction mismatch");
  CHECK(!vision::parse_image_data_uri("data:image/bmp;base64foo," + encode_b64(red_bmp), &parsed, &err),
        "substring ;base64foo must be rejected");
  CHECK(!vision::parse_image_data_uri("data:image/bmp;base64;extra=1," + encode_b64(red_bmp), &parsed, &err),
        ";base64 not as final parameter must be rejected");
  CHECK(!vision::parse_image_data_uri("https://example.com/a.png", &parsed, &err), "remote URL must be rejected");
  CHECK(!vision::parse_image_data_uri("data:text/plain;base64,aGk=", &parsed, &err), "non-image MIME must be rejected");

  // VisionCache LRU hit & eviction.
  vision::VisionCache cache(1024);
  auto v1 = std::make_shared<vision::CachedVision>();
  v1->tokens = {1, 2, 3, 4};
  v1->embeds.assign(160, 1.0f);  // 640 bytes
  auto v2 = std::make_shared<vision::CachedVision>();
  v2->tokens = {1, 2, 3, 4};
  v2->embeds.assign(160, 2.0f);  // 640 bytes -> evicts v1 under 1024B cap
  const vision::VisionCacheKey k1{img_r.sha256, "", 1024}, k2{img_b.sha256, "", 1024};
  cache.put(k1, v1);
  std::shared_ptr<const vision::CachedVision> got;
  CHECK(cache.get(k1, &got) && got == v1, "VisionCache hit failed");
  cache.put(k2, v2);
  CHECK(!cache.get(k1, &got), "VisionCache LRU eviction did not evict oldest entry");
  CHECK(cache.get(k2, &got) && got == v2, "VisionCache newest entry missing");
  std::printf("  ok   image decode, data URI, VisionCache\n");
}

void test_chat_json_and_media_prefix() {
  const auto red_bmp = make_bmp(8, 8, 220, 30, 10);
  const auto blue_bmp = make_bmp(8, 8, 10, 40, 230);
  const std::string b64_r = encode_b64(red_bmp);
  const std::string b64_b = encode_b64(blue_bmp);
  std::string err;

  // OpenAI image_url (valid detail canonicalization, invalid detail, disallowed roles)
  {
    ojson msgs = ojson::array({{{"role", "user"},
                                {"content", ojson::array({
                                                {{"type", "image_url"},
                                                 {"image_url", {{"url", "data:image/bmp;base64," + b64_r},
                                                                {"detail", "LOW"}}}},
                                                {{"type", "text"}, {"text", "describe"}},
                                            })}}});
    std::vector<ChatMessage> out;
    CHECK(messages_from_openai(msgs, &out, &err), err);
    CHECK(out.size() == 1 && out[0].parts.size() == 2, "OpenAI multimodal parts count");
    CHECK(out[0].parts[0].kind == ContentPart::kImage && out[0].parts[0].image->detail == "low", "OpenAI image detail");

    ojson bad_detail = ojson::array({{{"role", "user"},
                                      {"content", ojson::array({{{"type", "image_url"},
                                                                 {"image_url", {{"url", "data:image/bmp;base64," + b64_r},
                                                                                {"detail", "ultra"}}}}})}}});
    CHECK(!messages_from_openai(bad_detail, &out, &err), "OpenAI invalid detail must be rejected");

    for (const char* role : {"system", "assistant"}) {
      ojson bad_role = ojson::array({{{"role", role},
                                      {"content", ojson::array({{{"type", "image_url"},
                                                                 {"image_url", {{"url", "data:image/bmp;base64," + b64_r}}}}})}}});
      CHECK(!messages_from_openai(bad_role, &out, &err), std::string("OpenAI image in ") + role + " must be rejected");
    }
  }
  // OpenAI Responses input_image & role/detail validation
  {
    ojson items = ojson::array({{{"type", "message"},
                                 {"role", "user"},
                                 {"content", ojson::array({
                                                 {{"type", "input_image"},
                                                  {"image_url", "data:image/bmp;base64," + b64_b},
                                                  {"detail", "High"}},
                                                 {{"type", "input_text"}, {"text", "hi"}},
                                             })}}});
    std::vector<ChatMessage> out;
    CHECK(messages_from_responses("", items, &out, &err), err);
    CHECK(out.size() == 1 && out[0].parts.size() == 2 && out[0].parts[0].kind == ContentPart::kImage &&
              out[0].parts[0].image->detail == "high",
          "Responses input_image parse");
    ojson bad_role = ojson::array({{{"type", "message"},
                                    {"role", "assistant"},
                                    {"content", ojson::array({{{"type", "input_image"},
                                                               {"image_url", "data:image/bmp;base64," + b64_b}}})}}});
    CHECK(!messages_from_responses("", bad_role, &out, &err), "Responses image in assistant must be rejected");
    ojson bad_detail = ojson::array({{{"type", "message"},
                                      {"role", "user"},
                                      {"content", ojson::array({{{"type", "input_image"},
                                                                 {"image_url", "data:image/bmp;base64," + b64_b},
                                                                 {"detail", "huge"}}})}}});
    CHECK(!messages_from_responses("", bad_detail, &out, &err), "Responses invalid detail must be rejected");
  }
  // Anthropic image, cache_control on image block, non-image media_type & empty source rejection
  {
    ojson req = {{"messages", ojson::array({{{"role", "user"},
                                             {"content", ojson::array({
                                                             {{"type", "image"},
                                                              {"source", {{"type", "base64"},
                                                                          {"media_type", "image/bmp"},
                                                                          {"data", b64_r}}},
                                                              {"cache_control", {{"type", "ephemeral"}}}},
                                                             {{"type", "text"}, {"text", "what?"}},
                                                         })}}})}};
    std::vector<ChatMessage> out;
    std::vector<ojson> tools;
    std::vector<ChatCachePoint> cps;
    CHECK(chat_from_anthropic(req, &out, &tools, &err, false, nullptr, &cps), err);
    CHECK(out.size() == 1 && out[0].parts.size() == 2 && out[0].parts[0].kind == ContentPart::kImage,
          "Anthropic image parse");
    CHECK(cps.size() == 1 && cps[0].messages == 1 && cps[0].parts == 1,
          "Anthropic cache_control on image block must record part prefix");

    ojson bad = {{"messages", ojson::array({{{"role", "user"},
                                             {"content", ojson::array({{{"type", "image"}, {"source", ojson::object()}}})}}})}};
    CHECK(!chat_from_anthropic(bad, &out, &tools, &err), "Anthropic empty image source must be rejected");

    ojson bad_mime = {{"messages", ojson::array({{{"role", "user"},
                                                  {"content", ojson::array({{{"type", "image"},
                                                                             {"source", {{"type", "base64"},
                                                                                         {"media_type", "application/pdf"},
                                                                                         {"data", b64_r}}}}})}}})}};
    CHECK(!chat_from_anthropic(bad_mime, &out, &tools, &err), "Anthropic non-image media_type must be rejected");
  }
  // MediaSpan prefix collision safety
  {
    const std::vector<int32_t> toks = {1, 2, 100, 100, 100, 100, 3, 4};
    interfaces::MediaSpan m1{2, 4, vision::sha256_bytes(red_bmp.data(), red_bmp.size()), {1, 4, 4}, 4, {}};
    interfaces::MediaSpan m2{2, 4, vision::sha256_bytes(blue_bmp.data(), blue_bmp.size()), {1, 4, 4}, 4, {}};
    CHECK(vision::media_prefix_match(toks, {m1}, toks, {m1}) == 8, "identical media spans must match full length");
    CHECK(vision::media_prefix_match(toks, {m1}, toks, {m2}) == 2,
          "different images with identical placeholder tokens must clamp before span");
    CHECK(!vision::media_prefix_equal(toks, {m1}, toks, {m2}), "different images must not compare equal");
    // Truncating mid-span must clamp before span start.
    const std::vector<int32_t> half = {1, 2, 100, 100};
    CHECK(vision::media_prefix_match(toks, {m1}, half, {}) == 2, "must never split inside MediaSpan");
  }
  // Qwen35VisionAdapter preprocessing & 3D MRoPE positions
  {
    models::vision::Qwen35VisionParams vp;
    vp.patch_size = 16;
    vp.merge_size = 2;
    vp.temporal_patch_size = 2;
    vp.placeholder_token = 99;
    vp.out_hidden = 128;
    // Default min_pixels=65536 upscales 64x64 -> 256x256 (256 patches -> 64 merged tokens).
    interfaces::RgbImage rgb;
    rgb.width = 64;
    rgb.height = 64;
    rgb.rgb.assign(64 * 64 * 3, 128);
    {
      models::vision::Qwen35VisionAdapter default_adapter(vp);
      interfaces::PreparedMedia pm_up;
      CHECK(default_adapter.preprocess(rgb, 256, &pm_up, &err), err);
      CHECK(pm_up.n_patches == 256 && pm_up.n_embeds == 64 && pm_up.grid_thw == std::vector<int32_t>({1, 16, 16}),
            "default min_pixels upscale 64x64 -> 256x256 mismatch");
    }
    vp.min_pixels = 32 * 32;
    models::vision::Qwen35VisionAdapter adapter(vp);
    interfaces::PreparedMedia pm;
    CHECK(adapter.preprocess(rgb, 256, &pm, &err), err);
    CHECK(pm.n_patches == 16 && pm.patch_dim == 2 * 3 * 16 * 16, "64x64 patches mismatch");
    CHECK(pm.n_embeds == 4 && pm.tokens.size() == 4 && pm.grid_thw == std::vector<int32_t>({1, 4, 4}),
          "64x64 merged tokens mismatch");
    std::vector<int32_t> prompt = {10, 99, 99, 99, 99, 11, 12};
    interfaces::MediaSpan sp{1, 4, {}, {1, 4, 4}, 4, {}};
    auto pp = adapter.build_positions(prompt, std::span<const interfaces::MediaSpan>(&sp, 1));
    CHECK(pp.is_3d && pp.pos_3d.size() == prompt.size() * 3, "build_positions size");
    // Token 0: (0,0,0); image tokens 1..4: t=1, (h,w) in (1..2,1..2); token 5: (3,3,3); token 6: (4,4,4); cur=5 -> delta=5-7=-2
    CHECK(pp.pos_3d[0] == 0 && pp.pos_3d[3] == 1 && pp.pos_3d[4] == 1 && pp.pos_3d[5] == 1, "pos_3d start");
    CHECK(pp.pos_3d[12] == 1 && pp.pos_3d[13] == 2 && pp.pos_3d[14] == 2, "pos_3d last patch");
    CHECK(pp.pos_3d[15] == 3 && pp.pos_3d[16] == 3 && pp.pos_3d[17] == 3, "pos_3d post-image");
    CHECK(pp.rope_delta == -2, "rope_delta expected -2");
    CHECK(pp.pos_rope_deltas.size() == prompt.size() + 1 && pp.pos_rope_deltas[0] == 0 && pp.pos_rope_deltas[1] == 0 &&
              pp.pos_rope_deltas[5] == -2 && pp.pos_rope_deltas[7] == -2,
          "pos_rope_deltas prefix map mismatch");

    // Two-image prompt: intermediate pos_rope_deltas after 1st image must be -2 while final is -4.
    std::vector<int32_t> prompt2 = {10, 99, 99, 99, 99, 11, 99, 99, 99, 99, 12};
    std::vector<interfaces::MediaSpan> sps = {{1, 4, {}, {1, 4, 4}, 4, {}}, {6, 4, {}, {1, 4, 4}, 4, {}}};
    auto pp2 = adapter.build_positions(prompt2, sps);
    CHECK(pp2.rope_delta == -4 && pp2.pos_rope_deltas.size() == prompt2.size() + 1 && pp2.pos_rope_deltas[5] == -2 &&
              pp2.pos_rope_deltas[6] == -2 && pp2.pos_rope_deltas[10] == -4 && pp2.pos_rope_deltas[11] == -4,
          "two-image pos_rope_deltas mismatch");

    // smart_resize_dims strict budget enforcement under extreme aspect ratio and small max_tokens (< 64).
    {
      interfaces::RgbImage wide;
      wide.width = 2000;
      wide.height = 20;
      wide.rgb.assign(size_t(wide.width) * size_t(wide.height) * 3, 64);
      interfaces::PreparedMedia pm_wide;
      CHECK(adapter.preprocess(wide, 16, &pm_wide, &err), err);
      CHECK(pm_wide.n_embeds <= 16, "extreme aspect ratio + small max_tokens must not exceed budget");

      interfaces::RgbImage too_wide;
      too_wide.width = 2010;
      too_wide.height = 10;
      too_wide.rgb.assign(size_t(too_wide.width) * size_t(too_wide.height) * 3, 64);
      CHECK(!adapter.preprocess(too_wide, 256, &pm_wide, &err), "aspect ratio > 200:1 must be rejected");
    }
  }
  std::printf("  ok   multimodal JSON parsing, MediaSpan prefix safety, Qwen35VisionAdapter\n");
}

// Build an in-memory tiny multimodal model (Vision encoder + MRoPE LLM + MTP drafter).
struct TinyVlm {
  static constexpr int kH = 128;
  static constexpr int kV = 64;
  static constexpr int kNh = 2;
  static constexpr int kNkv = 2;
  static constexpr int kHd = 64;
  static constexpr int kRot = 32;
  static constexpr int kInter = 256;
  // Vision dims: kVInter = 144 (divisible by 16, NOT 32 or 64 — exercises padded ffn_down like Qwen3.8-27B's 4304!).
  static constexpr int kVH = 64;
  static constexpr int kVNh = 2;
  static constexpr int kVHd = 32;
  static constexpr int kVInter = 144;
  static constexpr int kPatchDim = 48;  // 2 * 3 * 2 * 2 * 2 = 48 (arbitrary non-64!)
  static constexpr int kMerge = 2;
  static constexpr int kMergeH = 256;   // kVH * kMerge * kMerge = 64 * 4

  std::vector<std::vector<uint16_t>> bf16_bufs;
  std::vector<std::vector<float>> f32_bufs;
  std::vector<std::vector<uint8_t>> q8_bufs;
  std::vector<flyx_tensor_view> views;
  std::vector<flyx_weight> weights;
  std::vector<flyx_state> states;
  std::vector<flyx_block> blocks;
  flyx_model_desc desc{};

  std::mt19937 rng{42};

  // Add a pre-quantized GGML Q8_0 2D tensor [r, c] (c % 32 == 0, 34 bytes per 32-element block: f16 d + 32 int8s).
  int add_q8_0(int64_t r, int64_t c, float scale = 0.08f) {
    std::normal_distribution<float> dist(0.f, scale);
    const size_t bpr = size_t(c / 32);
    q8_bufs.emplace_back(size_t(r) * bpr * 34);
    std::vector<float> row(static_cast<size_t>(c));
    std::vector<uint8_t> pay(static_cast<size_t>(c));
    std::vector<uint16_t> sc(bpr);
    for (int64_t i = 0; i < r; ++i) {
      for (int64_t k = 0; k < c; ++k) row[size_t(k)] = dist(rng);
      ops::quantize_row_q8_0(row.data(), int(c), int(c), pay.data(), sc.data());
      uint8_t* dst = q8_bufs.back().data() + size_t(i) * bpr * 34;
      for (size_t b = 0; b < bpr; ++b) {
        std::memcpy(dst + b * 34, &sc[b], 2);
        std::memcpy(dst + b * 34 + 2, pay.data() + b * 32, 32);
      }
    }
    flyx_tensor_view tv{};
    tv.name = "v.q8_0";
    tv.data = q8_bufs.back().data();
    tv.nbytes = q8_bufs.back().size();
    tv.dtype = FLYX_DT_GGML_Q8_0;
    tv.ndim = 2;
    tv.shape[0] = r;
    tv.shape[1] = c;
    views.push_back(tv);
    return int(views.size() - 1);
  }

  int add_bf16(int64_t r, int64_t c, float scale = 0.08f) {
    std::normal_distribution<float> dist(0.f, scale);
    bf16_bufs.emplace_back(size_t(r * c));
    for (auto& v : bf16_bufs.back()) v = f32_to_bf16(dist(rng));
    flyx_tensor_view tv{};
    tv.data = bf16_bufs.back().data();
    tv.nbytes = bf16_bufs.back().size() * 2;
    tv.dtype = FLYX_DT_BF16;
    tv.ndim = c > 0 ? 2 : 1;
    tv.shape[0] = r;
    tv.shape[1] = c > 0 ? c : 1;
    views.push_back(tv);
    return int(views.size() - 1);
  }

  int add_f32(int64_t r, int64_t c, float scale = 0.08f, float bias = 0.f) {
    std::normal_distribution<float> dist(bias, scale);
    f32_bufs.emplace_back(size_t(r * (c > 0 ? c : 1)));
    for (auto& v : f32_bufs.back()) v = dist(rng);
    flyx_tensor_view tv{};
    tv.data = f32_bufs.back().data();
    tv.nbytes = f32_bufs.back().size() * 4;
    tv.dtype = FLYX_DT_F32;
    tv.ndim = c > 0 ? 2 : 1;
    tv.shape[0] = r;
    tv.shape[1] = c > 0 ? c : 1;
    views.push_back(tv);
    return int(views.size() - 1);
  }

  void finalize_views() {
    bf16_bufs.reserve(64);
    f32_bufs.reserve(64);
    q8_bufs.reserve(16);
    views.reserve(64);
  }

  bool has_gdn = false;

  explicit TinyVlm(bool prequant_q8_vision = false, bool include_vision = true, bool include_gdn = false)
      : has_gdn(include_gdn) {
    finalize_views();
    // LLM weights
    const int w_emb = add_bf16(kV, kH, 0.25f);
    const int w_anorm = add_f32(kH, 0, 0.05f, 0.f);
    const int w_q = add_bf16(kNh * kHd, kH, 0.08f);
    const int w_k = add_bf16(kNkv * kHd, kH, 0.08f);
    const int w_v = add_bf16(kNkv * kHd, kH, 0.08f);
    const int w_o = add_bf16(kH, kNh * kHd, 0.08f);
    const int w_qn = add_f32(kHd, 0, 0.05f, 0.f);
    const int w_kn = add_f32(kHd, 0, 0.05f, 0.f);
    const int w_mnorm = add_f32(kH, 0, 0.05f, 0.f);
    const int w_gate = add_bf16(kInter, kH, 0.08f);
    const int w_up = add_bf16(kInter, kH, 0.08f);
    const int w_down = add_bf16(kH, kInter, 0.08f);
    const int w_fnorm = add_f32(kH, 0, 0.05f, 0.f);
    const int w_head = add_bf16(kV, kH, 0.15f);
    // Optional GDN block weights (nk=1, nv=1, dk=128, dv=128, conv=4 -> triggers prefill_align=64 on CUDA)
    const int w_gnorm = include_gdn ? add_f32(kH, 0, 0.05f, 0.f) : -1;
    const int w_gqkv = include_gdn ? add_bf16(3 * 128, kH, 0.06f) : -1;
    const int w_gz = include_gdn ? add_bf16(128, kH, 0.06f) : -1;
    const int w_gb = include_gdn ? add_bf16(1, kH, 0.06f) : -1;
    const int w_ga = include_gdn ? add_bf16(1, kH, 0.06f) : -1;
    const int w_gout = include_gdn ? add_bf16(kH, 128, 0.06f) : -1;
    const int w_gconv = include_gdn ? add_f32(3 * 128, 4, 0.05f, 0.f) : -1;
    const int w_galog = include_gdn ? add_f32(1, 0, 0.05f, -1.f) : -1;
    const int w_gdt = include_gdn ? add_f32(1, 0, 0.05f, 0.f) : -1;
    const int w_gonorm = include_gdn ? add_f32(128, 0, 0.05f, 1.f) : -1;
    // MTP weights
    const int w_mtp_en = add_f32(kH, 0, 0.05f, 0.f);
    const int w_mtp_hn = add_f32(kH, 0, 0.05f, 0.f);
    const int w_mtp_fc = add_bf16(kH, 2 * kH, 0.06f);
    const int w_mtp_an = add_f32(kH, 0, 0.05f, 0.f);
    const int w_mtp_q = add_bf16(kNh * kHd, kH, 0.08f);
    const int w_mtp_k = add_bf16(kNkv * kHd, kH, 0.08f);
    const int w_mtp_v = add_bf16(kNkv * kHd, kH, 0.08f);
    const int w_mtp_o = add_bf16(kH, kNh * kHd, 0.08f);
    const int w_mtp_qn = add_f32(kHd, 0, 0.05f, 0.f);
    const int w_mtp_kn = add_f32(kHd, 0, 0.05f, 0.f);
    const int w_mtp_mn = add_f32(kH, 0, 0.05f, 0.f);
    const int w_mtp_g = add_bf16(kInter, kH, 0.08f);
    const int w_mtp_u = add_bf16(kInter, kH, 0.08f);
    const int w_mtp_d = add_bf16(kH, kInter, 0.08f);
    const int w_mtp_fn = add_f32(kH, 0, 0.05f, 0.f);
    // Vision encoder weights (optional pre-quantized GGML Q8_0 for K % 32 == 0 matrices, mimicking mmproj-Q8_0.gguf)
    auto add_vmat = [&](int64_t r, int64_t c, float scale) {
      return (prequant_q8_vision && (c % 32 == 0)) ? add_q8_0(r, c, scale) : add_bf16(r, c, scale);
    };
    const int w_vpatch = include_vision ? add_f32(kVH, kPatchDim, 0.1f) : -1;
    const int w_vpatch_b = include_vision ? add_f32(kVH, 0, 0.05f) : -1;
    const int w_vpos = include_vision ? add_f32(16, kVH, 0.05f) : -1;
    const int w_va_nw = include_vision ? add_f32(kVH, 0, 0.05f, 1.f) : -1;
    const int w_va_nb = include_vision ? add_f32(kVH, 0, 0.02f) : -1;
    const int w_va_qkv = include_vision ? add_vmat(3 * kVNh * kVHd, kVH, 0.08f) : -1;
    const int w_va_qkv_b = include_vision ? add_f32(3 * kVNh * kVHd, 0, 0.02f) : -1;
    const int w_va_o = include_vision ? add_vmat(kVH, kVNh * kVHd, 0.08f) : -1;
    const int w_va_o_b = include_vision ? add_f32(kVH, 0, 0.02f) : -1;
    const int w_vf_nw = include_vision ? add_f32(kVH, 0, 0.05f, 1.f) : -1;
    const int w_vf_nb = include_vision ? add_f32(kVH, 0, 0.02f) : -1;
    const int w_vf_fc1 = include_vision ? add_vmat(kVInter, kVH, 0.08f) : -1;
    const int w_vf_fc1_b = include_vision ? add_f32(kVInter, 0, 0.02f) : -1;
    const int w_vf_fc2 = include_vision ? add_vmat(kVH, kVInter, 0.08f) : -1;
    const int w_vf_fc2_b = include_vision ? add_f32(kVH, 0, 0.02f) : -1;
    const int w_vm_nw = include_vision ? add_f32(kVH, 0, 0.05f, 1.f) : -1;
    const int w_vm_nb = include_vision ? add_f32(kVH, 0, 0.02f) : -1;
    const int w_vm_fc1 = include_vision ? add_vmat(kMergeH, kMergeH, 0.08f) : -1;
    const int w_vm_fc1_b = include_vision ? add_f32(kMergeH, 0, 0.02f) : -1;
    const int w_vm_fc2 = include_vision ? add_vmat(kH, kMergeH, 0.08f) : -1;
    const int w_vm_fc2_b = include_vision ? add_f32(kH, 0, 0.02f) : -1;

    weights.resize(views.size());
    for (size_t i = 0; i < views.size(); ++i) {
      const int act = views[i].dtype == FLYX_DT_GGML_Q8_0 ? FLYX_ACT_Q8 : FLYX_ACT_FORMAT;
      weights[i] = flyx_weight{&views[i], nullptr, nullptr, act, 0};
    }

    // States: 0 = target KV, 1 = MTP KV, 2 = hidden ring
    flyx_state s_kv{};
    s_kv.kind = FLYX_STATE_KV;
    s_kv.layers = 1;
    s_kv.n_kv = kNkv;
    s_kv.head_dim = kHd;
    flyx_state s_mkv = s_kv;
    s_mkv.draft = 1;
    flyx_state s_ring{};
    s_ring.kind = FLYX_STATE_RING;
    s_ring.width = kH;
    states = {s_kv, s_mkv, s_ring};
    if (include_gdn) {
      flyx_state s_gdn{};
      s_gdn.kind = FLYX_STATE_GDN;
      s_gdn.nk = 1;
      s_gdn.nv = 1;
      s_gdn.dk = 128;
      s_gdn.dv = 128;
      s_gdn.conv = 4;
      s_gdn.head_order = FLYX_GDN_HEADS_GROUPED;
      states.push_back(s_gdn);
    }

    auto make_norm = [](int32_t w, double offset = 1.0, double eps = 1e-6) {
      flyx_norm n{};
      n.w = w;
      n.offset = offset;
      n.eps = eps;
      return n;
    };
    auto make_vnorm = [](int32_t w, int32_t b_w, double eps = 1e-6) {
      flyx_vnorm n{};
      n.kind = FLYX_NORM_LAYER;
      n.w = w;
      n.b = b_w;
      n.offset = 0.0;
      n.eps = eps;
      return n;
    };

    // Blocks:
    // 0: Embed, 1: Attn (MRoPE), 2: MLP, 3: Head, 4: MTP Attn, 5: MTP MLP, 6: MTP
    // 7: VisionStem, 8: VitAttn, 9: VitMlp, 10: VisionMerge
    flyx_block b{};
    b = {};
    b.kind = FLYX_BLOCK_EMBED;
    b.embed.hidden = kH;
    b.embed.vocab = kV;
    b.embed.table = w_emb;
    blocks.push_back(b);

    b = {};
    b.kind = FLYX_BLOCK_ATTN;
    b.attn.hidden = kH;
    b.attn.n_heads = kNh;
    b.attn.n_kv = kNkv;
    b.attn.head_dim = kHd;
    b.attn.rot_dim = kRot;
    b.attn.gate = FLYX_GATE_NONE;
    b.attn.theta = 10000.0;
    b.attn.norm = make_norm(w_anorm);
    b.attn.q_norm = make_norm(w_qn);
    b.attn.k_norm = make_norm(w_kn);
    b.attn.q = w_q;
    b.attn.k = w_k;
    b.attn.v = w_v;
    b.attn.o = w_o;
    b.attn.kv = 0;
    b.attn.mrope_section[0] = 4;
    b.attn.mrope_section[1] = 6;
    b.attn.mrope_section[2] = 6;
    b.attn.mrope_interleaved = 1;
    blocks.push_back(b);

    b = {};
    b.kind = FLYX_BLOCK_MLP;
    b.mlp.hidden = kH;
    b.mlp.inter = kInter;
    b.mlp.norm = make_norm(w_mnorm);
    b.mlp.gate = w_gate;
    b.mlp.up = w_up;
    b.mlp.down = w_down;
    blocks.push_back(b);

    b = {};
    b.kind = FLYX_BLOCK_HEAD;
    b.head.hidden = kH;
    b.head.vocab = kV;
    b.head.norm = make_norm(w_fnorm);
    b.head.w = w_head;
    blocks.push_back(b);

    b = {};
    b.kind = FLYX_BLOCK_ATTN;
    b.attn.hidden = kH;
    b.attn.n_heads = kNh;
    b.attn.n_kv = kNkv;
    b.attn.head_dim = kHd;
    b.attn.rot_dim = kRot;
    b.attn.gate = FLYX_GATE_NONE;
    b.attn.theta = 10000.0;
    b.attn.norm = make_norm(w_mtp_an);
    b.attn.q_norm = make_norm(w_mtp_qn);
    b.attn.k_norm = make_norm(w_mtp_kn);
    b.attn.q = w_mtp_q;
    b.attn.k = w_mtp_k;
    b.attn.v = w_mtp_v;
    b.attn.o = w_mtp_o;
    b.attn.kv = 1;
    b.attn.mrope_section[0] = 4;
    b.attn.mrope_section[1] = 6;
    b.attn.mrope_section[2] = 6;
    b.attn.mrope_interleaved = 1;
    blocks.push_back(b);

    b = {};
    b.kind = FLYX_BLOCK_MLP;
    b.mlp.hidden = kH;
    b.mlp.inter = kInter;
    b.mlp.norm = make_norm(w_mtp_mn);
    b.mlp.gate = w_mtp_g;
    b.mlp.up = w_mtp_u;
    b.mlp.down = w_mtp_d;
    blocks.push_back(b);

    b = {};
    b.kind = FLYX_BLOCK_MTP;
    b.mtp.embed = 0;
    b.mtp.head = 3;
    b.mtp.attn = 4;
    b.mtp.mlp = 5;
    b.mtp.norm_emb = make_norm(w_mtp_en);
    b.mtp.norm_hidden = make_norm(w_mtp_hn);
    b.mtp.norm = make_norm(w_mtp_fn);
    b.mtp.fc = w_mtp_fc;
    b.mtp.ring = 2;
    blocks.push_back(b);

    if (include_vision) {
      b = {};
      b.kind = FLYX_BLOCK_VISION_STEM;
      b.vision_stem.in_dim = kPatchDim;
      b.vision_stem.hidden = kVH;
      b.vision_stem.patch_w = w_vpatch;
      b.vision_stem.patch_w2 = -1;
      b.vision_stem.patch_b = w_vpatch_b;
      b.vision_stem.pos_mode = FLYX_VPOS_BILINEAR_2D;
      b.vision_stem.pos_w = w_vpos;
      b.vision_stem.pos_grid_h = 4;
      b.vision_stem.pos_grid_w = 4;
      b.vision_stem.merge_size = kMerge;
      b.vision_stem.cls_token = -1;
      b.vision_stem.pre_norm = make_vnorm(-1, -1);
      blocks.push_back(b);

      b = {};
      b.kind = FLYX_BLOCK_VIT_ATTN;
      b.vit_attn.hidden = kVH;
      b.vit_attn.n_heads = kVNh;
      b.vit_attn.head_dim = kVHd;
      b.vit_attn.rope_mode = FLYX_VROPE_AXIAL_2D;
      b.vit_attn.merge_size = kMerge;
      b.vit_attn.theta = 10000.0;
      b.vit_attn.norm = make_vnorm(w_va_nw, w_va_nb);
      b.vit_attn.q_norm = make_vnorm(-1, -1);
      b.vit_attn.k_norm = make_vnorm(-1, -1);
      b.vit_attn.qkv = w_va_qkv;
      b.vit_attn.qkv_b = w_va_qkv_b;
      b.vit_attn.q = b.vit_attn.k = b.vit_attn.v = -1;
      b.vit_attn.q_b = b.vit_attn.k_b = b.vit_attn.v_b = -1;
      b.vit_attn.o = w_va_o;
      b.vit_attn.o_b = w_va_o_b;
      b.vit_attn.layer_scale = -1;
      blocks.push_back(b);

      b = {};
      b.kind = FLYX_BLOCK_VIT_MLP;
      b.vit_mlp.hidden = kVH;
      b.vit_mlp.inter = kVInter;
      b.vit_mlp.gated = 0;
      b.vit_mlp.act = FLYX_VACT_GELU_TANH;
      b.vit_mlp.norm = make_vnorm(w_vf_nw, w_vf_nb);
      b.vit_mlp.fc1 = w_vf_fc1;
      b.vit_mlp.fc1_b = w_vf_fc1_b;
      b.vit_mlp.gate = b.vit_mlp.gate_b = -1;
      b.vit_mlp.fc2 = w_vf_fc2;
      b.vit_mlp.fc2_b = w_vf_fc2_b;
      b.vit_mlp.layer_scale = -1;
      blocks.push_back(b);

      b = {};
      b.kind = FLYX_BLOCK_VISION_MERGE;
      b.vision_merge.in_hidden = kVH;
      b.vision_merge.merge_size = kMerge;
      b.vision_merge.merge_mode = FLYX_VMERGE_SPATIAL_CONCAT;
      b.vision_merge.post_norm = 0;
      b.vision_merge.fc1_out = kMergeH;
      b.vision_merge.out_hidden = kH;
      b.vision_merge.act = FLYX_VACT_GELU;
      b.vision_merge.norm = make_vnorm(w_vm_nw, w_vm_nb);
      b.vision_merge.fc1 = w_vm_fc1;
      b.vision_merge.fc1_b = w_vm_fc1_b;
      b.vision_merge.fc2 = w_vm_fc2;
      b.vision_merge.fc2_b = w_vm_fc2_b;
      blocks.push_back(b);
    }
    if (include_gdn) {
      b = {};
      b.kind = FLYX_BLOCK_GDN;
      b.gdn.hidden = kH;
      b.gdn.nk = 1;
      b.gdn.nv = 1;
      b.gdn.dk = 128;
      b.gdn.dv = 128;
      b.gdn.conv = 4;
      b.gdn.norm = make_norm(w_gnorm);
      b.gdn.out_norm = make_norm(w_gonorm, 0.0, 1e-6);
      b.gdn.qkv = w_gqkv;
      b.gdn.z = w_gz;
      b.gdn.b = w_gb;
      b.gdn.a = w_ga;
      b.gdn.out = w_gout;
      b.gdn.conv_w = w_gconv;
      b.gdn.A_log = w_galog;
      b.gdn.dt_bias = w_gdt;
      b.gdn.state = 3;
      b.gdn.head_order = FLYX_GDN_HEADS_GROUPED;
      b.gdn.decay = FLYX_GDN_DECAY_LOG;
      blocks.push_back(b);
    }

    desc = {kH, kV, int32_t(weights.size()), int32_t(states.size()), int32_t(blocks.size()),
            weights.data(), states.data(), blocks.data()};
  }
};

struct BackendRun {
  uint64_t estimated_vision = 0;
  uint64_t weights_bytes = 0;
  std::vector<float> vision_embeds;
  std::vector<int32_t> plain_tokens;
  std::vector<int32_t> mtp_tokens;
  std::vector<int32_t> split_tokens;
  std::vector<int32_t> ckpt_followup;
  std::vector<int32_t> snap_followup;
  std::vector<int32_t> direct_followup;
};

uint64_t measure_weights_bytes(const char* name, const TinyVlm& vlm, int32_t vision_dev = FLYX_VISION_DEV_GPU,
                               int32_t vision_quant = FLYX_VISION_QUANT_EXACT) {
  EngineOptions eo;
  eo.backend = name;
  std::unique_ptr<os::DynLib> lib;
  const flyx_backend_api* api = nullptr;
  flyx_device_info info{};
  std::string err;
  CHECK(open_backend(eo, &lib, &api, &info, &err), std::string(name) + ": " + err);
  flyx_open_options oo{};
  oo.max_ctx = 256;
  oo.kv_fp8 = 0;
  oo.vision_device = vision_dev;
  oo.vision_quant = vision_quant;
  flyx_bmodel* bm = nullptr;
  CHECK(api->open(&vlm.desc, &oo, &bm) == 0, std::string(name) + " open: " + api->last_error());
  flyx_memory_info mem{};
  if (api->memory) api->memory(bm, &mem);
  api->close(bm);
  return mem.weights;
}

BackendRun run_backend_vlm(const char* name, const TinyVlm& vlm, const std::vector<float>& patches,
                           const char* backend_opts = "", int32_t vision_dev = FLYX_VISION_DEV_GPU,
                           int32_t vision_quant = FLYX_VISION_QUANT_EXACT) {
  EngineOptions eo;
  eo.backend = name;
  eo.backend_options = backend_opts ? backend_opts : "";
  std::unique_ptr<os::DynLib> lib;
  const flyx_backend_api* api = nullptr;
  flyx_device_info info{};
  std::string err;
  CHECK(open_backend(eo, &lib, &api, &info, &err), std::string(name) + ": " + err);

  flyx_open_options oo{};
  oo.max_ctx = 256;
  oo.kv_fp8 = 0;
  oo.options = backend_opts && *backend_opts ? backend_opts : nullptr;
  oo.vision_device = vision_dev;
  oo.vision_quant = vision_quant;
  flyx_bmodel* bm = nullptr;
  CHECK(api->open(&vlm.desc, &oo, &bm) == 0, std::string(name) + " open: " + api->last_error());
  flyx_memory_info mem{};
  if (api->memory) api->memory(bm, &mem);

  // Programs
  std::vector<flyx_node> v_nodes = {{FLYX_NODE_RUN, 7, 0, 0}, {FLYX_NODE_RUN, 8, 0, 0},
                                    {FLYX_NODE_RUN, 9, 0, 0}, {FLYX_NODE_RUN, 10, 0, 0}};
  flyx_program vp{FLYX_PHASE_VISION, 0, 64, int32_t(v_nodes.size()), v_nodes.data()};
  flyx_plan* vplan = nullptr;
  CHECK(api->lower(bm, &vp, &vplan) == 0, std::string(name) + " lower vision: " + api->last_error());

  std::vector<flyx_node> pre_nodes = {{FLYX_NODE_RUN, 0, 0, 0}};
  if (vlm.has_gdn) pre_nodes.push_back({FLYX_NODE_RUN, 11, 0, 0});
  pre_nodes.push_back({FLYX_NODE_RUN, 1, 0, 0});
  pre_nodes.push_back({FLYX_NODE_RUN, 2, 0, 0});
  pre_nodes.push_back({FLYX_NODE_CONTEXT, 6, 0, 0});
  pre_nodes.push_back({FLYX_NODE_RUN, 3, 1, 0});
  flyx_program pp{FLYX_PHASE_PREFILL, 0, 128, int32_t(pre_nodes.size()), pre_nodes.data()};
  flyx_plan* pplan = nullptr;
  CHECK(api->lower(bm, &pp, &pplan) == 0, std::string(name) + " lower prefill: " + api->last_error());

  auto make_round = [&](int width) -> flyx_plan* {
    std::vector<flyx_node> rn;
    if (width > 0) rn.push_back({FLYX_NODE_DRAFT, 6, width, 0});
    rn.push_back({FLYX_NODE_RUN, 0, 0, 0});
    if (vlm.has_gdn) rn.push_back({FLYX_NODE_RUN, 11, 0, 0});
    rn.push_back({FLYX_NODE_RUN, 1, 0, 0});
    rn.push_back({FLYX_NODE_RUN, 2, 0, 0});
    rn.push_back({FLYX_NODE_RUN, 3, 0, 0});
    rn.push_back({FLYX_NODE_ACCEPT, 0, width, 0});
    rn.push_back({FLYX_NODE_COMMIT, 0, 0, 0});
    rn.push_back({FLYX_NODE_CONTEXT, 6, 0, 0});
    flyx_program rp{FLYX_PHASE_ROUND, width, width + 1, int32_t(rn.size()), rn.data()};
    flyx_plan* pl = nullptr;
    CHECK(api->lower(bm, &rp, &pl) == 0, std::string(name) + " lower round: " + api->last_error());
    return pl;
  };
  flyx_plan* rplan0 = make_round(0);
  flyx_plan* rplan3 = make_round(3);

  BackendRun res;
  res.weights_bytes = mem.weights;
  if (api->estimate_load && vision_dev == FLYX_VISION_DEV_GPU) {
    auto estimate = [&](int placement, uint64_t free) {
      auto opts = oo;
      opts.vision_device = placement;
      flyx_load_estimate e{};
      CHECK(api->estimate_load(&vlm.desc, &opts, free, info.mem_total, &e) == 0, "vision estimate");
      return e;
    };
    res.estimated_vision = estimate(FLYX_VISION_DEV_GPU, info.mem_free).weights_bytes -
                           estimate(FLYX_VISION_DEV_DISK, info.mem_free).weights_bytes;
    CHECK(estimate(FLYX_VISION_DEV_AUTO, 0).vision_device == FLYX_VISION_DEV_DISK &&
              estimate(FLYX_VISION_DEV_AUTO, 16000000000ull).vision_device == FLYX_VISION_DEV_GPU,
          "AUTO placement reflects room after the model and vision weights");
  }
  // 1. Vision encoding: 4x4 patch grid -> 2x2 = 4 output embeddings of dim kH.
  const int32_t grid_thw[3] = {1, 4, 4};
  const int32_t cu[2] = {0, 16};
  flyx_vision_input vi{};
  vi.patches = patches.data();
  vi.n_patches = 16;
  vi.patch_dim = TinyVlm::kPatchDim;
  vi.grid_thw = grid_thw;
  vi.n_grids = 1;
  vi.cu_seqlens = cu;
  vi.n_segs = 1;
  res.vision_embeds.resize(4 * TinyVlm::kH);
  int32_t out_rows = 0;
  CHECK(api->encode_vision(vplan, &vi, res.vision_embeds.data(), 4, &out_rows) == 0,
        std::string(name) + " encode_vision: " + api->last_error());
  CHECK(out_rows == 4, "encode_vision row count mismatch");

  // Multimodal prompt of 64 tokens (multiple of CUDA's prefill_align=64 so checkpoint/snapshot
  // tests run unconditionally on both cpu_ref and cuda):
  //   [t0, t1, IMG0, IMG1, IMG2, IMG3, t2..t59]
  std::vector<int32_t> prompt(64);
  prompt[0] = 5;
  prompt[1] = 7;
  prompt[2] = prompt[3] = prompt[4] = prompt[5] = 63;
  for (int i = 6; i < 64; ++i) prompt[size_t(i)] = (i * 7 + 3) % (TinyVlm::kV - 1);
  const int32_t embed_rows[4] = {2, 3, 4, 5};
  // 3D positions: text (0,0,0),(1,1,1), image (2,2..3,2..3), text (4,4,4)..(61,61,61); cur=62 -> rope_delta=62-64=-2
  std::vector<int32_t> pos_3d = {
      0, 0, 0,  1, 1, 1,
      2, 2, 2,  2, 2, 3,  2, 3, 2,  2, 3, 3,
  };
  for (int i = 0; i < 58; ++i) {
    pos_3d.push_back(4 + i);
    pos_3d.push_back(4 + i);
    pos_3d.push_back(4 + i);
  }
  const int32_t rope_delta = -2;

  auto decode_with = [&](flyx_plan* rplan, bool split_chunk) {
    flyx_seq* seq = nullptr;
    CHECK(api->seq_new(bm, 128, &seq) == 0, "seq_new");
    flyx_gen_params g{};
    g.limit = 128;
    if (!split_chunk) {
      flyx_prefill_input in{};
      in.tokens = prompt.data();
      in.n = int32_t(prompt.size());
      in.embeds = res.vision_embeds.data();
      in.embed_rows = embed_rows;
      in.n_embeds = 4;
      in.pos_3d = pos_3d.data();
      in.rope_delta = rope_delta;
      in.set_rope_delta = 1;
      CHECK(api->prefill_ex(pplan, seq, &in, &g) == 0, std::string(name) + " prefill_ex: " + api->last_error());
    } else {
      // Split RIGHT THROUGH THE MIDDLE OF THE IMAGE (chunk 1: rows 0..3, chunk 2: rows 4..63)!
      const int32_t r1[2] = {2, 3};
      flyx_prefill_input in1{};
      in1.tokens = prompt.data();
      in1.n = 4;
      in1.embeds = res.vision_embeds.data();
      in1.embed_rows = r1;
      in1.n_embeds = 2;
      in1.pos_3d = pos_3d.data();
      CHECK(api->prefill_ex(pplan, seq, &in1, nullptr) == 0, "prefill_ex chunk 1");

      const int32_t r2[2] = {0, 1};
      flyx_prefill_input in2{};
      in2.tokens = prompt.data() + 4;
      in2.n = int32_t(prompt.size()) - 4;
      in2.embeds = res.vision_embeds.data() + 2 * TinyVlm::kH;
      in2.embed_rows = r2;
      in2.n_embeds = 2;
      in2.pos_3d = pos_3d.data() + 4 * 3;
      in2.rope_delta = rope_delta;
      in2.set_rope_delta = 1;
      CHECK(api->prefill_ex(pplan, seq, &in2, &g) == 0, "prefill_ex chunk 2");
    }
    flyx_round_result rr{};
    CHECK(api->read(seq, &rr) == 0, "read prefill");
    std::vector<int32_t> out = {rr.tokens[0]};
    while (out.size() < 16) {
      CHECK(api->launch(rplan, seq, &g) == 0, "launch");
      CHECK(api->read(seq, &rr) == 0, "read round");
      for (int i = 0; i < rr.n_tokens && out.size() < 16; ++i) out.push_back(rr.tokens[i]);
    }
    api->seq_free(seq);
    return out;
  };

  res.plain_tokens = decode_with(rplan0, false);
  res.mtp_tokens = decode_with(rplan3, false);
  CHECK(res.mtp_tokens == res.plain_tokens, std::string(name) + ": Gate 4 failed (MTP != plain)");
  if (!vlm.has_gdn) {
    res.split_tokens = decode_with(rplan3, true);
    CHECK(res.split_tokens == res.plain_tokens, std::string(name) + ": split multimodal prefill != whole prefill");
  }

  // 2. Checkpoint restore + SnapshotStore rope_delta persistence + collision check
  const std::vector<int32_t> followup = {21, 22, 23, 24};
  auto decode_followup = [&](flyx_seq* seq) {
    flyx_gen_params g{};
    g.limit = 128;
    CHECK(api->prefill(pplan, seq, followup.data(), int32_t(followup.size()), &g) == 0, "followup prefill");
    flyx_round_result rr{};
    CHECK(api->read(seq, &rr) == 0, "followup read");
    std::vector<int32_t> out = {rr.tokens[0]};
    while (out.size() < 10) {
      CHECK(api->launch(rplan3, seq, &g) == 0, "followup launch");
      CHECK(api->read(seq, &rr) == 0, "followup read round");
      for (int i = 0; i < rr.n_tokens && out.size() < 10; ++i) out.push_back(rr.tokens[i]);
    }
    return out;
  };

  // Direct uninterrupted run (multimodal prompt + followup text)
  {
    flyx_seq* seq = nullptr;
    CHECK(api->seq_new(bm, 128, &seq) == 0, "seq_new direct");
    flyx_prefill_input in{};
    in.tokens = prompt.data();
    in.n = int32_t(prompt.size());
    in.embeds = res.vision_embeds.data();
    in.embed_rows = embed_rows;
    in.n_embeds = 4;
    in.pos_3d = pos_3d.data();
    in.rope_delta = rope_delta;
    in.set_rope_delta = 1;
    CHECK(api->prefill_ex(pplan, seq, &in, nullptr) == 0, "direct prefill_ex");
    // Take checkpoint at end of the 64-token multimodal prompt (aligned on both cpu_ref and cuda).
    const int ck_pos = api->checkpoint(seq, 0);
    CHECK(ck_pos == int(prompt.size()),
          std::string(name) + ": expected checkpoint at 64, got " + std::to_string(ck_pos));
    res.direct_followup = decode_followup(seq);

    {
      // Save into SnapshotStore with MediaSpan and verify collision safety!
      SnapshotStore store(64ull << 20, 4096);
      interfaces::MediaSpan m_real{2, 4, {1, 2, 3}, {1, 4, 4}, 4, {}};
      interfaces::MediaSpan m_fake{2, 4, {9, 9, 9}, {1, 4, 4}, 4, {}};
      std::vector<interfaces::MediaSpan> mv_real = {m_real}, mv_fake = {m_fake};
      std::string why;
      CHECK(store.put(api, seq, 0, prompt, &why, {.media = &mv_real}), why);
      CHECK(store.find(prompt, prompt.size(), 0, &mv_real) != nullptr, "SnapshotStore must match identical MediaSpan");
      CHECK(store.find(prompt, prompt.size(), 0, &mv_fake) == nullptr,
            "SnapshotStore must reject different MediaSpan with identical tokens");

      // Rewind to 0 (clears rope_delta to 0!), then restore checkpoint 0 -> must restore rope_delta == -2!
      CHECK(api->rewind(seq, 0, prompt[0]) == 1, "rewind 0");
      CHECK(api->restore(seq, 0, followup[0]) == ck_pos, "restore ckpt 0");
      res.ckpt_followup = decode_followup(seq);
      CHECK(res.ckpt_followup == res.direct_followup, std::string(name) + ": checkpoint restore lost rope_delta");

      // Load snapshot into a brand-new sequence (whose rope_delta starts at 0) -> must restore rope_delta == -2!
      flyx_seq* seq2 = nullptr;
      CHECK(api->seq_new(bm, 128, &seq2) == 0, "seq_new snap");
      const auto* ent = store.find(prompt, prompt.size(), 0, &mv_real);
      CHECK(ent != nullptr, "snapshot missing");
      CHECK(store.load(api, seq2, *ent, followup[0], &why) == ck_pos, why);
      res.snap_followup = decode_followup(seq2);
      CHECK(res.snap_followup == res.direct_followup, std::string(name) + ": snapshot load lost rope_delta");
      api->seq_free(seq2);
    }
    api->seq_free(seq);
  }

  // 3. Short-window rewind after multimodal turn preserves rope_delta & matches plain tokens.
  {
    flyx_seq* seq = nullptr;
    CHECK(api->seq_new(bm, 128, &seq) == 0, "seq_new rewind");
    flyx_gen_params g{};
    g.limit = 128;
    flyx_prefill_input in{};
    in.tokens = prompt.data();
    in.n = int32_t(prompt.size());
    in.embeds = res.vision_embeds.data();
    in.embed_rows = embed_rows;
    in.n_embeds = 4;
    in.pos_3d = pos_3d.data();
    in.rope_delta = rope_delta;
    in.set_rope_delta = 1;
    CHECK(api->prefill_ex(pplan, seq, &in, &g) == 0, "rewind prefill_ex");
    flyx_round_result rr{};
    CHECK(api->read(seq, &rr) == 0, "rewind read prefill");
    const int32_t first_tok = rr.tokens[0];
    CHECK(first_tok == res.plain_tokens[0], "rewind first_tok mismatch");
    CHECK(api->launch(rplan3, seq, &g) == 0, "rewind launch 1");
    CHECK(api->read(seq, &rr) == 0, "rewind read 1");
    CHECK(api->launch(rplan3, seq, &g) == 0, "rewind launch 2");
    CHECK(api->read(seq, &rr) == 0, "rewind read 2");
    // Rewind back into the first round's window (position 64 + 1 = 65, anchor = res.plain_tokens[1] after 1 emitted token).
    CHECK(api->rewind(seq, int32_t(prompt.size()) + 1, res.plain_tokens[1]) == 1,
          std::string(name) + ": short-window rewind after multimodal turn failed");
    std::vector<int32_t> after_rewind = {res.plain_tokens[0], res.plain_tokens[1]};
    while (after_rewind.size() < res.plain_tokens.size()) {
      CHECK(api->launch(rplan3, seq, &g) == 0, "rewind re-launch");
      CHECK(api->read(seq, &rr) == 0, "rewind re-read");
      for (int i = 0; i < rr.n_tokens && after_rewind.size() < res.plain_tokens.size(); ++i)
        after_rewind.push_back(rr.tokens[i]);
    }
    CHECK(after_rewind == res.plain_tokens, std::string(name) + ": rewind after multimodal turn changed tokens");
    api->seq_free(seq);
  }

  // 4. Intra-pass snapshot with snap_rope_delta != pass-end rope_delta, plus FLYX_PREFILL_SNAP_NONE.
  //    Active when the backend uses chunked prefill alignment (align > 1, e.g. hybrid GDN+Attn on CUDA).
  if (vlm.has_gdn && std::string(name) == "cuda") {
    const int32_t align = api->prefill_align ? api->prefill_align(bm) : 0;
    if (flyx::test::gdn_chunked_device(api))
      CHECK(align > 1 && 64 % align == 0, "hybrid GDN model on a chunked-GDN GPU must have prefill_align dividing 64");
    else
      CHECK(align == 1, "hybrid GDN model on a float32-recurrence GPU (datacenter parts) must have prefill_align 1");
  }
  if (api->prefill_snapshot && api->prefill_align && api->prefill_align(bm) > 1) {
    // Extend prompt (64 tokens, rope_delta=-2) with a 2nd image at rows 66..69 (total 72 tokens, pass-end rope_delta=-4).
    std::vector<int32_t> prompt2 = prompt;
    prompt2.insert(prompt2.end(), {13, 14, 63, 63, 63, 63, 15, 16});
    std::vector<float> embeds2 = res.vision_embeds;
    embeds2.insert(embeds2.end(), res.vision_embeds.begin(), res.vision_embeds.end());
    const int32_t embed_rows2[8] = {2, 3, 4, 5, 66, 67, 68, 69};
    std::vector<int32_t> pos3d2 = pos_3d;
    // rows 64,65: text at 62, 63; rows 66..69: image t=64, h=64..65, w=64..65; rows 70,71: text at 66, 67 -> cur=68, delta=68-72=-4
    const int32_t tail_pos[] = {62, 62, 62, 63, 63, 63, 64, 64, 64, 64, 64, 65,
                                64, 65, 64, 64, 65, 65, 66, 66, 66, 67, 67, 67};
    pos3d2.insert(pos3d2.end(), std::begin(tail_pos), std::end(tail_pos));

    flyx_seq* seq = nullptr;
    CHECK(api->seq_new(bm, 128, &seq) == 0, "seq_new snap_delta");
    CHECK(api->prefill_snapshot(seq, 64) == 0, "prefill_snapshot 64");
    flyx_prefill_input in{};
    in.tokens = prompt2.data();
    in.n = int32_t(prompt2.size());
    in.embeds = embeds2.data();
    in.embed_rows = embed_rows2;
    in.n_embeds = 8;
    in.pos_3d = pos3d2.data();
    in.rope_delta = -4;
    in.set_rope_delta = 1;
    in.has_snap_rope_delta = 1;
    in.snap_rope_delta = -2;
    CHECK(api->prefill_ex(pplan, seq, &in, nullptr) == 0, "snap_delta prefill_ex");
    CHECK(api->checkpoint(seq, 0) == 64, std::string(name) + ": expected intra-pass checkpoint at 64");
    CHECK(api->restore(seq, 0, followup[0]) == 64, "restore intra-pass checkpoint at 64");
    const auto restored_followup = decode_followup(seq);
    CHECK(restored_followup == res.direct_followup,
          std::string(name) + ": intra-pass snap_rope_delta was overwritten by pass-end rope_delta");
    api->seq_free(seq);

    // FLYX_PREFILL_SNAP_NONE suppresses implicit intra-pass snapshot (checkpoint() captures pass-end pos 72, not 64).
    CHECK(api->seq_new(bm, 128, &seq) == 0, "seq_new snap_none");
    CHECK(api->prefill_snapshot(seq, FLYX_PREFILL_SNAP_NONE) == 0, "prefill_snapshot NONE");
    flyx_prefill_input in_none{};
    in_none.tokens = prompt2.data();
    in_none.n = int32_t(prompt2.size());
    in_none.embeds = embeds2.data();
    in_none.embed_rows = embed_rows2;
    in_none.n_embeds = 8;
    in_none.pos_3d = pos3d2.data();
    in_none.rope_delta = -4;
    in_none.set_rope_delta = 1;
    CHECK(api->prefill_ex(pplan, seq, &in_none, nullptr) == 0, "snap_none prefill_ex");
    CHECK(api->checkpoint(seq, 0) == int(prompt2.size()),
          std::string(name) + ": FLYX_PREFILL_SNAP_NONE must not capture an intra-pass snapshot at 64");
    api->seq_free(seq);
  }

  api->plan_free(vplan);
  api->plan_free(pplan);
  api->plan_free(rplan0);
  api->plan_free(rplan3);
  api->close(bm);
  return res;
}

void test_tiny_vlm_backends() {
  TinyVlm vlm;
  std::mt19937 rng(123);
  std::normal_distribution<float> dist(0.f, 0.5f);
  std::vector<float> patches(16 * TinyVlm::kPatchDim);
  for (float& v : patches) v = dist(rng);

  auto rel_err = [](const std::vector<float>& a, const std::vector<float>& b) {
    CHECK(a.size() == b.size(), "embed size mismatch");
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      const double d = double(a[i]) - double(b[i]);
      num += d * d;
      den += double(b[i]) * double(b[i]);
    }
    return std::sqrt(num / std::max(den, 1e-12));
  };

  const BackendRun ref = run_backend_vlm("cpu_ref", vlm, patches, "", FLYX_VISION_DEV_GPU, FLYX_VISION_QUANT_EXACT);
  const BackendRun ref_q8 = run_backend_vlm("cpu_ref", vlm, patches, "", FLYX_VISION_DEV_GPU, FLYX_VISION_QUANT_Q8);
  const BackendRun ref_fp4 = run_backend_vlm("cpu_ref", vlm, patches, "", FLYX_VISION_DEV_GPU, FLYX_VISION_QUANT_NVFP4);
  // Pre-quantized GGML Q8_0 vision weights (like mmproj-Q8_0.gguf)
  TinyVlm vlm_q8(true, true);
  const BackendRun ref_preq8 =
      run_backend_vlm("cpu_ref", vlm_q8, patches, "", FLYX_VISION_DEV_GPU, FLYX_VISION_QUANT_EXACT);
  std::printf(
      "  ok   cpu_ref tiny VLM (exact/q8/nvfp4 + pre-quantized Q8_0, padded kVInter=%d, MTP == plain, ckpt+snapshot)\n",
      TinyVlm::kVInter);

  // The CUDA half where this build has a CUDA backend (its module beside this program, where the probe looks first):
  // there this test is a GPU test (label gpu), and a backend that does not load or finds no device fails it instead
  // of leaving it to pass on cpu_ref alone.  A build without one runs the cpu_ref half only.
  EngineOptions probe_opt;
  probe_opt.backend = "cuda";
  std::string bname, perr;
  flyx_device_info dinfo{};
  const std::string cuda_module = os::executable_dir() + "/libflyx-cuda" + os::kDynLibSuffix;
  const bool cuda_built = std::filesystem::exists(os::to_path(cuda_module));
  if (!probe_device(probe_opt, &bname, &dinfo, &perr)) {
    CHECK(!cuda_built, "this build has a CUDA backend (" + cuda_module + "), and it is not usable here: " + perr);
    std::printf("  skip cuda tiny VLM: no CUDA backend in this build (%s)\n", perr.c_str());
    return;
  }

  // Verify 0 bytes of resident vision VRAM when vision=host vs text-only model!
  TinyVlm text_only(false, false);
  const uint64_t text_vram = measure_weights_bytes("cuda", text_only);

  const BackendRun gpu = run_backend_vlm("cuda", vlm, patches, "", FLYX_VISION_DEV_GPU, FLYX_VISION_QUANT_EXACT);
  const BackendRun host = run_backend_vlm("cuda", vlm, patches, "", FLYX_VISION_DEV_HOST, FLYX_VISION_QUANT_EXACT);
  const BackendRun disk = run_backend_vlm("cuda", vlm, patches, "", FLYX_VISION_DEV_DISK, FLYX_VISION_QUANT_EXACT);
  const BackendRun gpu_unfused =
      run_backend_vlm("cuda", vlm, patches, "fusion=0", FLYX_VISION_DEV_GPU, FLYX_VISION_QUANT_EXACT);
  const double rel_exact = rel_err(gpu.vision_embeds, ref.vision_embeds);
  CHECK(rel_exact < 2e-5, "cuda vs cpu_ref exact encode_vision rel err: " + std::to_string(rel_exact));
  CHECK(host.vision_embeds == gpu.vision_embeds, "vision=host exact must be bit-identical to vision=gpu");
  CHECK(disk.vision_embeds == gpu.vision_embeds, "vision=disk exact must be bit-identical to vision=gpu");
  CHECK(host.weights_bytes == text_vram,
        "vision=host must use 0 resident vision VRAM (" + std::to_string(host.weights_bytes) + " vs " +
            std::to_string(text_vram) + ")");
  CHECK(disk.weights_bytes == text_vram, "vision=disk must use 0 resident vision VRAM");
  CHECK(gpu.weights_bytes > text_vram, "vision=gpu should allocate vision VRAM");
  CHECK(gpu.plain_tokens == ref.plain_tokens, "cuda vs cpu_ref exact multimodal greedy tokens differ");
  CHECK(gpu.direct_followup == ref.direct_followup, "cuda vs cpu_ref follow-up rope_delta tokens differ");
  CHECK(gpu_unfused.plain_tokens == gpu.plain_tokens, "Gate 2 failed: cuda fusion=0 != fused");
  CHECK(gpu_unfused.direct_followup == gpu.direct_followup, "Gate 2 failed: cuda fusion=0 followup != fused");

  // On-the-fly Q8_0 quantization (GPU + Host + Disk offload, with padded kVInter=144 -> 160)
  const BackendRun gpu_q8 = run_backend_vlm("cuda", vlm, patches, "", FLYX_VISION_DEV_GPU, FLYX_VISION_QUANT_Q8);
  const BackendRun host_q8 = run_backend_vlm("cuda", vlm, patches, "", FLYX_VISION_DEV_HOST, FLYX_VISION_QUANT_Q8);
  const BackendRun disk_q8 = run_backend_vlm("cuda", vlm, patches, "", FLYX_VISION_DEV_DISK, FLYX_VISION_QUANT_Q8);
  const BackendRun def_run = run_backend_vlm("cuda", vlm, patches, "", 0, 0);
  const double rel_q8 = rel_err(gpu_q8.vision_embeds, ref_q8.vision_embeds);
  CHECK(rel_q8 < 2e-5, "cuda vs cpu_ref q8 encode_vision rel err: " + std::to_string(rel_q8));
  CHECK(host_q8.vision_embeds == gpu_q8.vision_embeds, "vision=host q8 must be bit-identical to vision=gpu q8");
  CHECK(disk_q8.vision_embeds == gpu_q8.vision_embeds, "vision=disk q8 must be bit-identical to vision=gpu q8");
  CHECK(host_q8.weights_bytes == text_vram, "vision=host q8 must use 0 resident vision VRAM");
  CHECK(disk_q8.weights_bytes == text_vram, "vision=disk q8 must use 0 resident vision VRAM");
  CHECK(def_run.weights_bytes == text_vram, "default flyx_open_options must use 0 resident vision VRAM (disk)");
  CHECK(def_run.vision_embeds == disk_q8.vision_embeds, "default flyx_open_options must match disk + q8 bit-for-bit");
  CHECK(gpu_q8.weights_bytes < gpu.weights_bytes, "vision_quant=q8 must use less VRAM than exact BF16");
  CHECK(gpu_q8.plain_tokens == ref_q8.plain_tokens, "cuda vs cpu_ref q8 greedy tokens differ");

  // On-the-fly NVFP4 quantization (GPU + Host + Disk offload, with padded kVInter=144 -> 192)
  const BackendRun gpu_fp4 = run_backend_vlm("cuda", vlm, patches, "", FLYX_VISION_DEV_GPU, FLYX_VISION_QUANT_NVFP4);
  const BackendRun host_fp4 = run_backend_vlm("cuda", vlm, patches, "", FLYX_VISION_DEV_HOST, FLYX_VISION_QUANT_NVFP4);
  const BackendRun disk_fp4 = run_backend_vlm("cuda", vlm, patches, "", FLYX_VISION_DEV_DISK, FLYX_VISION_QUANT_NVFP4);
  const double rel_fp4 = rel_err(gpu_fp4.vision_embeds, ref_fp4.vision_embeds);
  CHECK(rel_fp4 < 2e-5, "cuda vs cpu_ref nvfp4 encode_vision rel err: " + std::to_string(rel_fp4));
  CHECK(host_fp4.vision_embeds == gpu_fp4.vision_embeds, "vision=host nvfp4 must be bit-identical to vision=gpu nvfp4");
  CHECK(disk_fp4.vision_embeds == gpu_fp4.vision_embeds, "vision=disk nvfp4 must be bit-identical to vision=gpu nvfp4");
  CHECK(host_fp4.weights_bytes == text_vram, "vision=host nvfp4 must use 0 resident vision VRAM");
  CHECK(disk_fp4.weights_bytes == text_vram, "vision=disk nvfp4 must use 0 resident vision VRAM");
  CHECK(gpu_fp4.weights_bytes < gpu_q8.weights_bytes, "vision_quant=nvfp4 must use less VRAM than q8");
  CHECK(gpu_fp4.plain_tokens == ref_fp4.plain_tokens, "cuda vs cpu_ref nvfp4 greedy tokens differ");

  // Pre-quantized GGML Q8_0 vision weights (mmproj-Q8_0.gguf path)
  const BackendRun gpu_preq8 = run_backend_vlm("cuda", vlm_q8, patches, "", FLYX_VISION_DEV_GPU, FLYX_VISION_QUANT_EXACT);
  const BackendRun host_preq8 =
      run_backend_vlm("cuda", vlm_q8, patches, "", FLYX_VISION_DEV_HOST, FLYX_VISION_QUANT_EXACT);
  const BackendRun disk_preq8 =
      run_backend_vlm("cuda", vlm_q8, patches, "", FLYX_VISION_DEV_DISK, FLYX_VISION_QUANT_EXACT);
  const double rel_preq8 = rel_err(gpu_preq8.vision_embeds, ref_preq8.vision_embeds);
  CHECK(rel_preq8 < 2e-5, "cuda vs cpu_ref pre-quantized Q8_0 rel err: " + std::to_string(rel_preq8));
  CHECK(host_preq8.vision_embeds == gpu_preq8.vision_embeds, "pre-quantized Q8_0 host != gpu");
  CHECK(disk_preq8.vision_embeds == gpu_preq8.vision_embeds, "pre-quantized Q8_0 disk != gpu");
  CHECK(gpu_preq8.plain_tokens == ref_preq8.plain_tokens, "pre-quantized Q8_0 greedy tokens differ");

  for (const BackendRun* run : {&gpu, &gpu_q8, &gpu_fp4, &gpu_preq8})
    CHECK(run->estimated_vision == run->weights_bytes - text_vram,
          "resident vision estimate matches uploaded planes, scales, norms and stem");

  // Hybrid GDN + MRoPE Attn + Vision (exercises prefill_align=64, intra-pass snap_rope_delta, FLYX_PREFILL_SNAP_NONE)
  TinyVlm hybrid(false, true, /*include_gdn=*/true);
  const BackendRun ref_hyb = run_backend_vlm("cpu_ref", hybrid, patches, "", FLYX_VISION_DEV_GPU, FLYX_VISION_QUANT_EXACT);
  const BackendRun gpu_hyb = run_backend_vlm("cuda", hybrid, patches, "", FLYX_VISION_DEV_DISK, FLYX_VISION_QUANT_EXACT);
  CHECK(gpu_hyb.plain_tokens == ref_hyb.plain_tokens, "cuda vs cpu_ref hybrid GDN+MRoPE greedy tokens differ");
  CHECK(gpu_hyb.direct_followup == ref_hyb.direct_followup, "cuda vs cpu_ref hybrid GDN+MRoPE followup differs");

  std::printf(
      "  ok   cuda tiny VLM vs cpu_ref (exact %.2e, q8 %.2e, nvfp4 %.2e, pre-Q8_0 %.2e; disk==host==gpu bit-exact, 0 "
      "vision VRAM, hybrid GDN+MRoPE snap_rope_delta)\n",
      rel_exact, rel_q8, rel_fp4, rel_preq8);
}

}  // namespace

int main() {
  std::printf("vision_test\n");
  test_image_and_cache();
  test_chat_json_and_media_prefix();
  test_tiny_vlm_backends();
  std::printf("PASSED\n");
  return 0;
}
