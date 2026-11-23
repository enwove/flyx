// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// UTF-8 <-> UTF-16 conversions shared by the Windows OS and HTTP implementations.
#pragma once

#include <string>

namespace flyx::os::win {

std::wstring to_wide(const std::string& s);
std::string to_utf8(const std::wstring& w);

}  // namespace flyx::os::win
