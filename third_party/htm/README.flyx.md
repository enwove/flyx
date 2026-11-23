# htm

htm reads JavaScript's tagged template strings as markup, so components can be
written in a form close to JSX without a compiler: `` html`<p class="lead">${text}</p>` ``.
The dashboard (`apps/flyx-server/dashboard/`) binds it to Preact's `h` in
`lib/html.js` (decision 0081).

    https://www.npmjs.com/package/htm  version 3.1.1
    dist/htm.module.js   sha256 ab33dd3f38059b9be4d5f5350128eefb2356639c4e0bbe9d9e8b3ba75847e9e4

under the Apache License 2.0 (`LICENSE`, "Copyright 2018 Google Inc."). The file is
byte-for-byte the package's own build, taken from its npm tarball.

The build embeds it in the executable, served as `/dashboard/vendor/htm.module.js`
(`CMakeLists.txt`, `cmake/embed_files.cmake`).

To update: take the file from a newer version's tarball
(`https://registry.npmjs.org/htm/-/htm-VERSION.tgz`), note the version and
hash here, and run `tools/dashboard_sweep.py` against a server with the new
build.
