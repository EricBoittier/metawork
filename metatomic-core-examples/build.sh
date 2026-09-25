#!/usr/bin/env bash
# Build metatomic-core (Rust -> libmetatomic + the libmetatensor it links) and
# the C / C++ / Rust examples against it. Outputs go to ./build.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METATOMIC="${METATOMIC_DIR:-$HERE/../metatomic}"
BUILD="$HERE/build"
mkdir -p "$BUILD"

cargo build --release --manifest-path "$METATOMIC/metatomic-core/Cargo.toml"
LIB="$METATOMIC/target/release"
# metatensor-sys builds metatensor-core next to it; take the newest one
MTS=$(ls -dt "$LIB"/build/metatensor-sys-*/out | head -1)
HDF5_INC=/usr/include/hdf5/serial
HDF5_LIB=/usr/lib/x86_64-linux-gnu/hdf5/serial

# metatomic.h includes metatomic/version.h, which CMake normally generates
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' "$METATOMIC/metatomic-core/Cargo.toml" | head -1)
mkdir -p "$BUILD/include/metatomic"
IFS=. read -r MAJOR MINOR PATCH <<< "${VERSION%%-*}"
cat > "$BUILD/include/metatomic/version.h" <<VERSION_H
#pragma once
#define METATOMIC_VERSION "$VERSION"
#define METATOMIC_VERSION_MAJOR $MAJOR
#define METATOMIC_VERSION_MINOR $MINOR
#define METATOMIC_VERSION_PATCH $PATCH
VERSION_H

# the metatomic C++ headers use nlohmann/json; fetch the same release metatomic's CMake does
if [ ! -f /usr/include/nlohmann/json.hpp ] && [ ! -f "$BUILD/include/nlohmann/json.hpp" ]; then
  mkdir -p "$BUILD/include/nlohmann"
  curl -fsSL -o "$BUILD/include/nlohmann/json.hpp" \
    https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
fi

INCLUDES=(-I"$METATOMIC/metatomic-core/include" -I"$BUILD/include" -I"$MTS/include")
LINK=(-L"$LIB" -L"$MTS/lib" -lmetatomic -lmetatensor -Wl,--disable-new-dtags -Wl,-rpath,"$LIB" -Wl,-rpath,"$MTS/lib")

cc -O2 -std=c11 -Wall -Wextra "${INCLUDES[@]}" "$HERE/c/extxyz-gz-to-mta.c" -o "$BUILD/extxyz-gz-to-mta" "${LINK[@]}" -lz
# The HDF5 *devel* headers (not just the runtime .so, which may already be
# present) are only needed for this one C++ example -- the Rust converter/
# scanner below don't touch HDF5 at all. Skip it gracefully rather than
# aborting the whole build (set -e) when a machine has the runtime lib but
# not -devel installed and the user has no root to fix that (e.g. this
# cluster, RHEL9: hdf5-1.12.1 installed, hdf5-devel is not).
if [ -f "$HERE/cpp/h5-features-to-mts.cpp" ] && [ -f "$HDF5_INC/hdf5.h" ]; then
  c++ -O2 -std=c++17 -Wall -Wextra "${INCLUDES[@]}" -I"$HDF5_INC" "$HERE/cpp/h5-features-to-mts.cpp" \
    -o "$BUILD/h5-features-to-mts" "${LINK[@]}" -L"$HDF5_LIB" -lhdf5 -Wl,-rpath,"$HDF5_LIB"
elif [ -f "$HERE/cpp/h5-features-to-mts.cpp" ]; then
  echo "skipping h5-features-to-mts: no HDF5 headers at $HDF5_INC/hdf5.h (only the C++ example needs them)"
fi
if [ -f "$HERE/rust/Cargo.toml" ]; then
  METATOMIC_LIB_DIR="$LIB" METATENSOR_LIB_DIR="$MTS/lib" \
    cargo build --release --manifest-path "$HERE/rust/Cargo.toml" --target-dir "$BUILD/rust"
  cp "$BUILD/rust/release/madcore-scan" "$BUILD/rust/release/madcore-to-diskdataset" "$BUILD/"
fi
ls -la "$BUILD" | grep -v "^d"
