// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The .flyx reader on files that are wrong (core/modelfile/flyx_format.cpp): a tensor or a blob the header
// places outside the file is refused, whatever its offset and size add up to; a header that is not the description
// of a model is an error the caller reads, not an exception; format 2 is read with the parts its header requires, and
// "requires" in any other format, or not a list of names, is refused.
//
//   flyx_test DIR        (the files are written there)
#include <cstdio>
#include <exception>
#include <string>

#include "core/modelfile/modelfile.h"
#include "tests/core/modelfile/write_flyx.h"

using namespace flyx;
using namespace flyx::testing;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
  }
}

std::string tensor(const std::string& offset, const std::string& nbytes, const std::string& shape = "[4]") {
  return R"({"arch": "test", "tensors": {"w": {"dtype": "u8", "shape": )" + shape + R"(, "offset": )" + offset +
         R"(, "nbytes": )" + nbytes + R"(}}})";
}

std::string blob(const std::string& offset, const std::string& nbytes) {
  return R"({"arch": "test", "tensors": {}, "blobs": {"b": {"offset": )" + offset + R"(, "nbytes": )" + nbytes +
         "}}}";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: flyx_test DIR\n");
    return 2;
  }
  const std::string path = std::string(argv[1]) + "/flyx_test.flyx";
  const std::string data = "0123456789abcdef";  // 16 bytes of data
  // Whether the reader takes the file; the error it gives otherwise.
  auto opens = [&](const std::string& bytes, std::string* err) {
    if (!write_file(path, bytes)) {
      std::printf("cannot write %s\n", path.c_str());
      return ++failures, false;
    }
    ModelFile f;
    return f.open(path, err);
  };
  std::string err;

  {  // a file as it should be: the tensor and the blob are the data's bytes
    check(write_file(path, flyx_file(tensor("4", "4"), data)), "writes the file");
    ModelFile f;
    check(f.open(path, &err), "a tensor inside the file: " + err);
    const flyx_tensor_view* w = f.tensor("w");
    check(w && w->nbytes == 4 && std::string(static_cast<const char*>(w->data), 4) == "4567", "the tensor's bytes");
  }  // release the mapping before replacing the fixture
  {
    check(write_file(path, flyx_file(blob("12", "4"), data)), "writes the file");
    ModelFile g;
    check(g.open(path, &err) && g.blob("b") == "cdef", "the blob's bytes: " + err);
  }
  check(opens(flyx_file(tensor("0", "16", "[16]"), data), &err), "a tensor that is all of the data: " + err);
  check(opens(flyx_file(tensor("16", "0", "[0]"), data), &err), "an empty tensor at the data's end: " + err);

  // Outside the file, plainly.
  check(!opens(flyx_file(tensor("0", "17"), data), &err), "a tensor one byte too long");
  check(!opens(flyx_file(tensor("17", "0"), data), &err), "a tensor that starts past the end");
  check(!opens(flyx_file(blob("8", "9"), data), &err), "a blob one byte too long");
  // Outside the file, with numbers whose sum wraps round to a place inside it.
  check(!opens(flyx_file(tensor("18446744073709547520", "8"), data), &err),
        "a tensor whose offset wraps round with the data's start");
  check(err.find("outside the file") != std::string::npos, "and the error says where it lies: " + err);
  check(!opens(flyx_file(tensor("8", "18446744073709551612"), data), &err), "a tensor whose size wraps round");
  check(!opens(flyx_file(tensor("18446744073709551615", "18446744073709551615"), data), &err),
        "a tensor whose offset and size both wrap round");
  check(!opens(flyx_file(blob("18446744073709547520", "8"), data), &err), "a blob whose offset wraps round");
  check(!opens(flyx_file(blob("4", "18446744073709551615"), data), &err), "a blob whose size wraps round");

  // A header that is not a model's description: an error, not an exception.
  check(!opens(flyx_file(tensor("0", "4", "[-4]"), data), &err), "a negative dimension");
  check(!opens(flyx_file(tensor("0", "0", "[4]"), data), &err), "tensor nbytes smaller than shape * dtype");
  check(!opens(flyx_file(tensor("0", "8", "[4]"), data), &err), "tensor nbytes larger than shape * dtype");
  check(!opens(flyx_file(R"({"arch": "test"})", data), &err), "no tensors");
  check(!opens(flyx_file(R"({"arch": "test", "tensors": {"w": {"dtype": "u8"}}})", data), &err),
        "a tensor without a shape");
  check(!opens(flyx_file(R"({"arch": "test", "tensors": {"w": 7}})", data), &err), "a tensor that is a number");
  check(!opens(flyx_file(R"({"arch": "test", "tensors": {}, "blobs": {"b": {}}})", data), &err),
        "a blob without an offset");
  check(!opens(flyx_file("not json", data), &err), "a header that is not JSON");
  {  // keys the reader itself looks at, of another type than it expects: read as absent, by open() and by add()
    const std::string odd = R"({"arch": "test", "kind": 1, "companions": 7, "tensors": {"w": {"dtype": "u8", )"
                            R"("shape": [4], "offset": 0, "nbytes": 4}}})";
    check(write_file(path, flyx_file(odd, data)), "writes the file");
    bool threw = false, opened = false, added = false;
    try {
      ModelFile f;
      opened = f.open(path, &err);
      added = opened && f.add(path, &err);
    } catch (const std::exception& e) {
      threw = true;
      err = e.what();
    }
    check(!threw && opened && added, "a kind and companions of the wrong type: " + err);
  }
  // A header length past the end of the file, and one that wraps round with the 24 bytes before the header.
  const std::string header = tensor("0", "4");
  check(!opens(flyx_file(header, data, 1 << 20), &err), "a header longer than the file");
  check(!opens(flyx_file(header, data, 18446744073709551600ull), &err), "a header length that wraps round");

  // Deeply nested JSON headers are rejected before nlohmann::json::parse() so later copies cannot overflow the stack.
  {
    const std::string deep = R"({"arch": "test", "tensors": {}, "x": )" + std::string(200, '[') +
                             std::string(200, ']') + "}";
    err.clear();
    check(!opens(flyx_file(deep, data), &err) && err.find("nest") != std::string::npos,
          "deeply nested .flyx header is rejected cleanly: " + err);
  }

  // Format 2: a header naming the parts a reader must not ignore ("requires"), which a build of format 1 only refuses.
  auto requiring = [&](const std::string& parts) {
    return R"({"arch": "test", "requires": )" + parts + R"(, "tensors": {}})";
  };
  auto format = [&](const std::string& h, uint32_t version) { return flyx_file(h, data, h.size(), version); };
  {
    write_file(path, format(requiring(R"(["lora"])"), 2));
    ModelFile f;
    check(f.open(path, &err) && f.header().at("requires") == nlohmann::json::array({"lora"}),
          "format 2 with \"requires\": opens, the parts in the header: " + err);
  }
  const std::string none = R"({"arch": "test", "tensors": {}})";
  check(!opens(format(none, 2), &err) && err.find("without \"requires\"") != std::string::npos,
        "format 2 without \"requires\": refused (" + err + ")");
  const std::string lora = requiring(R"(["lora"])");
  check(!opens(format(lora, 1), &err) && err.find("format 1 with \"requires\"") != std::string::npos,
        "format 1 with \"requires\", which a build of format 1 only would ignore: refused (" + err + ")");
  for (const char* bad : {"[]", R"("lora")", "[3]", R"([""])", "null"})
    check(!opens(format(requiring(bad), 2), &err) && err.find("not a list of the parts") != std::string::npos,
          std::string("format 2 with \"requires\": ") + bad + ": refused (" + err + ")");
  check(!opens(format(requiring(R"(["lora"])"), 3), &err) && err.find("reads 1 and 2") != std::string::npos,
        "format 3: refused (" + err + ")");

  std::printf(failures ? "%d checks failed\n" : "the .flyx reader: ok\n", failures);
  return failures ? 1 : 0;
}
