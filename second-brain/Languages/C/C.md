---
tags: [c, lang]
---
# C

Mostly relevant as the **ABI layer**: metatensor/metatomic expose a C API (`metatensor.h`) that Rust implements and C++/Python/Julia bind to.

## Compile / link
```bash
gcc -std=c11 -O2 -g -Wall -Wextra -o prog main.c
gcc -c -fPIC lib.c -o lib.o && gcc -shared -o libfoo.so lib.o   # shared lib
gcc main.c -I include -L build -lmetatensor -Wl,-rpath,$PWD/build -o prog
```
| Flag | Meaning |
|---|---|
| `-O0 -g` | debug build |
| `-O3 -march=native` | fast, not portable |
| `-fsanitize=address,undefined` | ASan/UBSan (also link with it) |
| `-fPIC` | needed for code in a `.so` |
| `-Wl,-rpath,DIR` | bake library search path into binary |

## Inspect binaries
```bash
nm -D --defined-only libmetatensor.so | grep mts_   # exported symbols
ldd ./prog                                          # which .so get loaded
readelf -d libfoo.so | grep -E 'RPATH|RUNPATH|NEEDED'
objdump -d -M intel fn.o | less
LD_DEBUG=libs ./prog                                # trace library resolution
```

## Debug
```bash
gdb --args ./prog arg1      # run, bt, frame N, p var, break file.c:42
valgrind --leak-check=full ./prog
```

## C API conventions (metatensor style)
- Opaque handles: `typedef struct mts_tensormap_t mts_tensormap_t;` — users only hold pointers.
- Every function returns a status code; error message via a `*_last_error()` call.
- Ownership is explicit: whoever `_create`s must call `_free`.
- Callbacks pass a `void* user_data` for state.
- Keep the header `extern "C"`-safe:
  ```c
  #ifdef __cplusplus
  extern "C" {
  #endif
  ...
  #ifdef __cplusplus
  }
  #endif
  ```

Related: [[C++ and CMake]], [[Rust]]
