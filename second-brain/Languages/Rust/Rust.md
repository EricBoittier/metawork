---
tags: [rust, lang]
---
# Rust

metatensor-core is Rust exposing a C API (`cbindgen` generates `metatensor.h`).

## Toolchain
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update
rustup toolchain list
rustup component add clippy rustfmt rust-src
rustup override set 1.75     # pin toolchain for this directory
```

## cargo
```bash
cargo build [--release]
cargo test                        # all; `cargo test name_filter -- --nocapture` to see prints
cargo test -p metatensor-core     # one workspace crate
cargo run --example foo
cargo clippy --all-targets -- -D warnings
cargo fmt
cargo doc --open
cargo tree -i some_crate          # who depends on it
cargo update -p some_crate        # bump one dep in Cargo.lock
cargo bench                       # criterion benches
```

## FFI / C API pattern
```rust
#[no_mangle]
pub unsafe extern "C" fn mts_thing_free(ptr: *mut Thing) -> mts_status_t {
    catch_unwind(|| {
        if !ptr.is_null() { drop(Box::from_raw(ptr)); }
        Ok(())
    })
}
```
- Never let a panic cross the FFI boundary → wrap in `catch_unwind`.
- `Box::into_raw` to hand ownership to C, `Box::from_raw` to take it back.
- `crate-type = ["cdylib", "staticlib"]` in `Cargo.toml` for C-loadable libs.
- Header: `cbindgen --config cbindgen.toml --output include/foo.h`.

## Ownership cheat-sheet
| Want | Use |
|---|---|
| read-only borrow | `&T` |
| mutate in place | `&mut T` |
| heap, single owner | `Box<T>` |
| shared, single thread | `Rc<T>` (+`RefCell` to mutate) |
| shared, threads | `Arc<T>` (+`Mutex`/`RwLock`) |
| errors | `Result<T, E>` + `?`; `thiserror` for libs, `anyhow` for apps |

## Python bindings
`maturin develop` (pyo3) for quick editable builds; metatensor itself instead loads the cdylib from Python via `ctypes`.

Related: [[C]], [[C++ and CMake]]
