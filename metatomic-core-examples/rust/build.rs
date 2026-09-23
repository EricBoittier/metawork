// Link against libmetatomic (the metatomic-core C API). The library is found
// in METATOMIC_LIB_DIR, by default the cargo target directory of the
// metatomic checkout next to this repository.
fn main() {
    let dir = std::env::var("METATOMIC_LIB_DIR").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../metatomic/target/release").into()
    });
    println!("cargo:rerun-if-env-changed=METATOMIC_LIB_DIR");
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=metatomic");
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,{dir}");
    if let Ok(extra) = std::env::var("METATENSOR_LIB_DIR") {
        // libmetatomic itself needs libmetatensor at run time
        println!("cargo:rustc-link-arg=-Wl,-rpath,{extra}");
    }
}
