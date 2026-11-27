// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Numbers given on the command line.  A value that is not a number, or one outside what its option takes, ends the
// program with what was wrong: `--port 808O` does not become port 0, nor `--sessions none` a server no request
// ever starts on.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace flyx::apps {

// `text` as a whole number in [lo, hi], all of it (no sign but a leading '-', no spaces, nothing after the digits);
// false otherwise.
bool parse_whole(const std::string& text, long long lo, long long hi, long long* out);
// `text` as a number in [lo, hi] (decimal or exponent form, all of it; not "nan" or "inf"); false otherwise.
bool parse_number(const std::string& text, double lo, double hi, double* out);
// `text` as digits alone, a value up to 2^64 - 1; false otherwise.
bool parse_seed(const std::string& text, uint64_t* out);
// Converts finite gigabytes in [0, kMaxGb] to bytes; false otherwise.
bool gb_to_bytes(double gb, int64_t* bytes_out);
// Parses a memory size in gigabytes ("auto" -> -1, or finite 0..kMaxGb -> bytes); false otherwise.
bool parse_gb_or_auto(std::string_view text, int64_t* bytes_out);

// The value of option `flag`, or the program ends (exit status 2) saying what the option takes.
int whole_flag(const char* flag, const std::string& text, int lo, int hi);
double number_flag(const char* flag, const std::string& text, double lo, double hi);
uint64_t seed_flag(const char* flag, const std::string& text);
int64_t gb_or_auto_flag(const char* flag, const std::string& text);

// Reads an entire file as binary bytes into *out; false if the file cannot be opened or read.
bool read_text_file(const std::string& path, std::string* out);

// Zero until two output tokens have a measured decode interval (the first token is TTFT/prefill).
inline double decode_rate(int output_tokens, double decode_ms) {
  return decode_ms > 0 && output_tokens > 1 ? (output_tokens - 1) * 1000.0 / decode_ms : 0.0;
}

}  // namespace flyx::apps
