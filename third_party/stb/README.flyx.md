# stb_image

`stb_image.h` is a single-header public-domain / MIT image decoder from Sean
Barrett's `stb` library. `flyx` uses it to decode in-memory PNG, JPEG, BMP,
GIF, and TGA images supplied via `--image`, OpenAI `image_url` / `input_image`
base64 data URIs, and Anthropic base64 `image` blocks.

    https://github.com/nothings/stb
    stb_image.h  version 2.34
    sha256 dfc2aadda6295f67f677587773f25e5bf8c8ad427ed0db7c518525c91e72e020

Dual-licensed under the MIT License and Public Domain (`LICENSE`).
The header is vendored verbatim with no local modifications.

`flyx` compiles its implementation once in `core/vision/image.cpp` with
`STB_IMAGE_IMPLEMENTATION`, `STBI_NO_STDIO`, `STBI_NO_HDR`, `STBI_NO_LINEAR`,
and `STBI_FAILURE_USERMSG` (`STBI_NO_STDIO` restricts decoding to memory
buffers already size-bounded by the caller).

To update: copy `stb_image.h` from upstream `nothings/stb`, record the version
and SHA-256 here and in `docs/prior-art.md` / `NOTICE`, and run
`ctest -R vision`.
