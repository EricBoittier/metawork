---
tags: [cpp, cmake, lang]
---
# C++ and CMake

## CMake workflow
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix $PWD/install
cmake -S . -B build -LH          # list cache vars with help
```
Common `-D` options:
| Var | Use |
|---|---|
| `CMAKE_BUILD_TYPE` | `Debug` / `Release` / `RelWithDebInfo` |
| `CMAKE_PREFIX_PATH` | where to find deps, e.g. `$(python -c "import torch; print(torch.utils.cmake_prefix_path)")` |
| `CMAKE_CUDA_ARCHITECTURES` | `89;90` kuma (L40S/H100), `100;120` lyra (B200/RTX6000) |
| `CMAKE_EXPORT_COMPILE_COMMANDS=ON` | `compile_commands.json` for clangd |
| `BUILD_SHARED_LIBS=ON` | shared instead of static |

Minimal modern `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.22)
project(foo LANGUAGES CXX)
find_package(Torch REQUIRED)
add_library(foo SHARED src/foo.cpp)
target_compile_features(foo PUBLIC cxx_std_17)
target_include_directories(foo PUBLIC include)
target_link_libraries(foo PUBLIC torch)
```
Use `target_*` commands, never global `include_directories`/`add_definitions`.

## Debug a build
```bash
cmake --build build -v                 # show actual compiler commands
rm -rf build                           # stale cache is the #1 cause of weirdness
cmake --find-package -DNAME=Torch -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST
```

## Language reminders
- RAII: resources owned by objects; `std::unique_ptr` default, `std::shared_ptr` when shared.
- Rule of 0: don't write destructors/copy ctors unless you own a raw resource.
- Pass `const T&` for big read-only args, by value when you'll move it in.
- `auto&&` in range-for when you might modify; `const auto&` otherwise.
- ABI: `torch` wheels use `_GLIBCXX_USE_CXX11_ABI` — mismatches give undefined `std::__cxx11` symbols. Check `torch._C._GLIBCXX_USE_CXX11_ABI`.

## Tools
```bash
clang-format -i src/*.cpp
clang-tidy -p build src/foo.cpp
c++filt _ZN3foo3barEv                  # demangle
gdb --args ./build/test_foo            # catch throw → break on exceptions
```
Sanitizers: `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"`.

Related: [[C]], [[CUDA]], [[PyTorch extensions and TorchScript]]
