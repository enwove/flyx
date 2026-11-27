// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A test program's checks: each one counted, a failed one printed with where it is in the test, and the program's
// verdict and exit code from the count.
//
//   check(reply.done, "the request finishes");  // FAIL tests/core/exec/request_test.cpp:52: the request finishes
//   return flyx::test::finish("request_test");  // request_test: PASSED (57 checks), exit code 0; FAILED ..., 1
//
// flyx::test::print_passes = true also prints an ok line for each check that holds (a test whose output reads as the
// list of what it checks).  The counts are the program's: one test a process, its checks made from one thread.
#pragma once

#include <cstdio>
#include <source_location>
#include <string_view>

namespace flyx::test {

inline int checks = 0;
inline int failures = 0;  // the checks that failed, and what a test counts as failed itself (++failures)
inline bool print_passes = false;
// The failures printed; the rest are counted (a loop of checks that all fail prints its first ones).
inline constexpr int kPrintedFailures = 20;

// `file`'s path in the repository, from its tests/ directory on (the compiler names it as the build did: absolute).
inline std::string_view repo_path(std::string_view file) {
  for (size_t i = file.size(); i-- > 0;) {
    const bool dir = file.compare(i, 6, "tests/") == 0 || file.compare(i, 6, "tests\\") == 0;
    if (dir && (i == 0 || file[i - 1] == '/' || file[i - 1] == '\\')) return file.substr(i);
  }
  return file;
}

inline void check(bool ok, std::string_view what, std::source_location where = std::source_location::current()) {
  ++checks;
  if (ok) {
    if (print_passes) std::printf("ok   %.*s\n", int(what.size()), what.data());
    return;
  }
  if (++failures > kPrintedFailures) return;
  const std::string_view file = repo_path(where.file_name());
  std::printf("FAIL %.*s:%u: %.*s\n", int(file.size()), file.data(), unsigned(where.line()), int(what.size()),
              what.data());
  std::fflush(stdout);  // seen even if the test then crashes
}

// The verdict line, and the exit code: 0 when nothing failed, else 1.
inline int finish(std::string_view name) {
  const int n = int(name.size());
  if (failures == 0) {
    std::printf("%.*s: PASSED (%d checks)\n", n, name.data(), checks);
  } else if (failures <= kPrintedFailures) {
    std::printf("%.*s: FAILED (%d failures, %d checks)\n", n, name.data(), failures, checks);
  } else {
    std::printf("%.*s: FAILED (%d failures, %d checks; the first %d printed)\n", n, name.data(), failures, checks,
                kPrintedFailures);
  }
  return failures ? 1 : 0;
}

}  // namespace flyx::test

using flyx::test::check;
