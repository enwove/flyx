// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Numbers given on the command line (apps/common/numbers.h): a value is taken when all of its text is a number
// inside what the option takes, and refused otherwise (what atoi made 0 of, or the digits before a letter).
#include <cstdio>
#include <string>

#include "apps/common/numbers.h"

using namespace flyx::apps;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
  }
}

bool whole(const char* text, long long lo, long long hi, long long want) {
  long long v = -12345;
  return parse_whole(text, lo, hi, &v) && v == want;
}
bool no_whole(const char* text, long long lo = -1000000, long long hi = 1000000) {
  long long v = 0;
  return !parse_whole(text, lo, hi, &v);
}
bool number(const char* text, double lo, double hi, double want) {
  double v = -12345;
  return parse_number(text, lo, hi, &v) && v == want;
}
bool no_number(const char* text, double lo = -1e9, double hi = 1e9) {
  double v = 0;
  return !parse_number(text, lo, hi, &v);
}

}  // namespace

int main() {
  check(whole("8080", 1, 65535, 8080), "a port");
  check(whole("1", 1, 65535, 1) && whole("65535", 1, 65535, 65535), "the ends of a range");
  check(whole("0", 0, 10, 0) && whole("-2", -2, 2, -2), "zero and a negative number");
  check(no_whole("0", 1, 65535) && no_whole("65536", 1, 65535) && no_whole("-1", 0, 10), "outside the range");
  check(no_whole("") && no_whole("abc") && no_whole("808O") && no_whole("80 ") && no_whole(" 80") &&
            no_whole("8.5") && no_whole("+8") && no_whole("0x10") && no_whole("1e3") && no_whole("--1"),
        "text that is not a whole number");
  check(no_whole("99999999999999999999999", -1000000, 1000000), "more digits than a number holds");

  check(number("0.7", 0, 100, 0.7) && number("1", 0, 1, 1.0) && number("0", 0, 1, 0.0), "a temperature, a top-p");
  check(number("-1.5", -2, 2, -1.5) && number("1e-3", 0, 1, 1e-3) && number(".5", 0, 1, 0.5), "forms of a number");
  check(no_number("1.01", 0, 1) && no_number("-0.1", 0, 1) && no_number("2.5", -2, 2), "outside the range");
  check(no_number("") && no_number("abc") && no_number("0.7x") && no_number("0,7") && no_number(" 1") &&
            no_number("nan") && no_number("inf") && no_number("-inf") && no_number("1e999"),
        "text that is not a number");

  uint64_t seed = 1;
  check(parse_seed("0", &seed) && seed == 0, "seed 0");
  check(parse_seed("18446744073709551615", &seed) && seed == 18446744073709551615ull, "the largest seed");
  check(!parse_seed("18446744073709551616", &seed) && !parse_seed("-1", &seed) && !parse_seed("", &seed) &&
            !parse_seed("12ab", &seed) && !parse_seed("1.0", &seed),
        "text that is not a seed");

  std::printf(failures ? "%d checks failed\n" : "numbers on the command line: ok\n", failures);
  return failures ? 1 : 0;
}
