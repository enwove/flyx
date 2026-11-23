// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Python string rules chat templates are written against (transformers
// renders them in Python): what a renderer written for a template must
// reproduce byte for byte, and what reading a reply back undoes.
#pragma once

#include <string>
#include <string_view>

#include "core/interfaces/chat.h"

namespace flyx::models::formats {

using interfaces::ojson;

// str.strip(): the whitespace str.isspace() accepts, Unicode spaces included.
std::string_view py_strip(std::string_view s);
// json.dumps(x, ensure_ascii=False) (Jinja's tojson): ", " and ": "
// separators, keys in their order, non-ASCII kept.
std::string py_json(const ojson& j);
// str() of a JSON value (Jinja's | string): True, False, None; containers as tojson.
std::string py_str(const ojson& j);
// Whether text the template makes from a client's value (tojson, | string) is the client's text, as the Jinja
// engine marks it (third_party/jinja/README.flyx.md): the value holds a string or an object key.
bool holds_client_text(const ojson& j);

}  // namespace flyx::models::formats
