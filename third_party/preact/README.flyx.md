# Preact

Preact is a small library for user interfaces written as components, with the
same model as React: a component is a function of its inputs that returns the
markup it shows, and Preact changes the page where the markup changed. The
dashboard (`apps/flyx-server/dashboard/`) is written with it (decision 0081).

    https://www.npmjs.com/package/preact  version 10.26.4
    dist/preact.module.js        sha256 2ce1b7b810fc14cda3f4242e636311b5a5e6d5a6cb1f3274fe948fe3bba3d32e
    hooks/dist/hooks.module.js   sha256 0e62f95470546ddf657f9cf5f84aaf9de01bb869ee742c2095429cd2b4c9ea0f

under the MIT license (`LICENSE`, "Copyright (c) 2015-present Jason Miller").
Both files are byte-for-byte the package's own builds, taken from its npm
tarball. `hooks.module.js` imports `"preact"` by that name; the dashboard's
import map (`index.html`) says where it is, so neither file is edited.

The build embeds them in the executable beside the dashboard's own files,
served as `/dashboard/vendor/preact.module.js` and `/dashboard/vendor/hooks.module.js`
(`CMakeLists.txt`, `cmake/embed_files.cmake`).

To update: take both files from a newer version's tarball
(`https://registry.npmjs.org/preact/-/preact-VERSION.tgz`), note the version
and hashes here, and run `tools/dashboard_sweep.py` against a server with the
new build.
