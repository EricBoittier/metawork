#!/usr/bin/env bash
# Build the mlipcpp metatomic plugin and the examples using it:
#
#   build/metatomic          upstream metatomic-core (pinned, METATOMIC_COMMIT)
#   build/mlip.cpp           EricBoittier/mlip.cpp, branch feat/metatomic-plugin
#   build/venv               metatensor-core + metatomic-core Python packages
#   build/mlipcpp-metatomic.so, build/metatomic_plugin, build/nve-md
#
# Override the checkouts with METATOMIC_DIR / MLIPCPP_DIR, and pass extra CMake
# options to mlip.cpp with MLIPCPP_CMAKE_ARGS (e.g. "-DMLIPCPP_USE_CUDA=ON").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"
METATOMIC_COMMIT=${METATOMIC_COMMIT:-3656f550}
METATOMIC="${METATOMIC_DIR:-$BUILD/metatomic}"
MLIPCPP="${MLIPCPP_DIR:-$BUILD/mlip.cpp}"
JOBS=${JOBS:-$(nproc)}
mkdir -p "$BUILD"

if [ ! -d "$METATOMIC" ]; then
  git clone -q https://github.com/metatensor/metatomic.git "$METATOMIC"
  git -C "$METATOMIC" checkout -q "$METATOMIC_COMMIT"
fi
if [ ! -d "$MLIPCPP" ]; then
  git clone -q -b feat/metatomic-plugin https://github.com/EricBoittier/mlip.cpp.git "$MLIPCPP"
fi

# metatomic-core needs metatensor-core < 0.3, newer than some metawork venvs
VENV="$BUILD/venv"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" -q install numpy ase "metatensor-core>=0.2.4,<0.3"
  "$VENV/bin/pip" -q install "$METATOMIC/python/metatomic_core"
fi
MTS="$("$VENV/bin/python" -c 'import metatensor, os; print(os.path.dirname(metatensor.__file__))')"

cmake -S "$METATOMIC/metatomic-core" -B "$BUILD/metatomic-build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$BUILD/prefix" -DCMAKE_PREFIX_PATH="$MTS/lib/cmake" > "$BUILD/metatomic.log"
cmake --build "$BUILD/metatomic-build" -j "$JOBS" --target install >> "$BUILD/metatomic.log"

# shellcheck disable=SC2086
cmake -S "$MLIPCPP" -B "$BUILD/mlipcpp-build" -DCMAKE_BUILD_TYPE=Release \
  -DMLIPCPP_BUILD_METATOMIC=ON -DMLIPCPP_BUILD_TRAINING=OFF \
  -DCMAKE_PREFIX_PATH="$BUILD/prefix;$MTS/lib/cmake" ${MLIPCPP_CMAKE_ARGS:-} > "$BUILD/mlipcpp.log"
cmake --build "$BUILD/mlipcpp-build" -j "$JOBS" --target mlipcpp-metatomic metatomic_plugin >> "$BUILD/mlipcpp.log"
cp "$BUILD/mlipcpp-build/lib/mlipcpp-metatomic.so" "$BUILD/mlipcpp-build/bin/metatomic_plugin" "$BUILD/"

cc -O2 -std=c11 -Wall -Wextra -I"$BUILD/prefix/include" -I"$MTS/include" "$HERE/nve-md.c" -o "$BUILD/nve-md" \
  -L"$BUILD/prefix/lib" -L"$MTS/lib" -lmetatomic -lmetatensor -lm \
  -Wl,--disable-new-dtags -Wl,-rpath,"$BUILD/prefix/lib" -Wl,-rpath,"$MTS/lib"

ls -la "$BUILD" | grep -v "^d"
