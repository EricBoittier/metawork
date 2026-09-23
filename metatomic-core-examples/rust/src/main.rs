//! Summarise gzip-compressed extxyz shards into metatensor TensorMaps.
//!
//! Streams every frame of the given `.extxyz[.gz]` files (one thread per
//! file) and writes a per-structure summary as a `.mts` TensorMap: one block
//! with samples `system` (the `fps_order` header key when present, else the
//! running frame index) and properties `quantity`, whose names are stored in
//! the `quantities` info entry. Categorical `dataset_group` values are stored
//! as integer codes, decoded by the `dataset_group` info entry.
//!
//! With `--composition`, the composition is written as a second, sparse
//! TensorMap: samples `[system, atomic_type]`, property `count`.
//!
//! With `--systems`, every frame is also turned into an `mta_system_t` through
//! the metatomic C API, which validates the cell / pbc of each structure.
//!
//! ```text
//! madcore-scan -o summary.mts [--composition composition.mts] [--systems] shard.extxyz.gz...
//! ```
//!
//! Reading the result from Python:
//!
//! ```python
//! import metatensor
//! summary = metatensor.load("summary.mts")
//! block = summary.block()
//! names = summary.info()["quantities"].split(",")  # column names
//! ```

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use dlpk::DLPackTensor;
use flate2::read::MultiGzDecoder;
use metatensor::{Labels, TensorBlock, TensorMap};
use ndarray::{Array1, Array2};

const QUANTITIES: [&str; 11] = [
    "n_atoms", "energy", "energy_per_atom", "max_force", "rms_force", "epsilon_sq", "k_set",
    "voronoi_population", "periodic", "volume", "dataset_group",
];
const ENERGY_KEYS: [&str; 3] = ["ecumetric_energy", "energy", "REF_energy"];
const FORCE_KEYS: [&str; 3] = ["ecumetric_nc_forces", "forces", "REF_forces"];

const ELEMENTS: [&str; 119] = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
    "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

// ---------------------------------------------------------------------------
// the small part of the metatomic C API used here (see metatomic.h)

#[repr(C)]
struct MtaSystem {
    _private: [u8; 0],
}

extern "C" {
    fn mta_system_create(
        length_unit: *const c_char,
        types: *mut dlpk::sys::DLManagedTensorVersioned,
        positions: *mut dlpk::sys::DLManagedTensorVersioned,
        cell: *mut dlpk::sys::DLManagedTensorVersioned,
        pbc: *mut dlpk::sys::DLManagedTensorVersioned,
        system: *mut *mut MtaSystem,
    ) -> i32;
    fn mta_system_free(system: *mut MtaSystem) -> i32;
    fn mta_last_error(message: *mut *const c_char, origin: *mut *const c_char, data: *mut *mut c_void) -> i32;
}

fn raw(tensor: DLPackTensor) -> *mut dlpk::sys::DLManagedTensorVersioned {
    tensor.into_raw().as_ptr()
}

/// Create (and immediately free) an `mta_system_t`, returning metatomic's
/// error message if the structure is not valid.
fn check_system(types: &[i32], positions: &[f64], cell: [f64; 9], pbc: [bool; 3]) -> Result<(), String> {
    let n = types.len();
    let tensors: [DLPackTensor; 4] = [
        Array1::from_vec(types.to_vec()).try_into().map_err(|e| format!("{e:?}"))?,
        Array2::from_shape_vec((n, 3), positions.to_vec()).unwrap().try_into().map_err(|e| format!("{e:?}"))?,
        Array2::from_shape_vec((3, 3), cell.to_vec()).unwrap().try_into().map_err(|e| format!("{e:?}"))?,
        Array1::from_vec(pbc.to_vec()).try_into().map_err(|e| format!("{e:?}"))?,
    ];
    let [types, positions, cell, pbc] = tensors;
    let mut system = std::ptr::null_mut();
    // SAFETY: mta_system_create takes ownership of the four tensors
    let status = unsafe {
        mta_system_create(c"angstrom".as_ptr(), raw(types), raw(positions), raw(cell), raw(pbc), &mut system)
    };
    if status != 0 {
        let mut message = std::ptr::null();
        unsafe { mta_last_error(&mut message, std::ptr::null_mut(), std::ptr::null_mut()) };
        return Err(unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned());
    }
    unsafe { mta_system_free(system) };
    Ok(())
}

// ---------------------------------------------------------------------------
// extxyz parsing

/// Value of `key=` in an extxyz comment line, without surrounding quotes.
fn header_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let mut rest = line;
    while let Some(start) = rest.find(key) {
        let at_token = start == 0 || rest.as_bytes()[start - 1] == b' ';
        let after = &rest[start + key.len()..];
        if at_token && after.starts_with('=') {
            let value = &after[1..];
            return Some(match value.strip_prefix('"') {
                Some(quoted) => &quoted[..quoted.find('"').unwrap_or(quoted.len())],
                None => value.split_ascii_whitespace().next().unwrap_or(""),
            });
        }
        rest = &rest[start + key.len()..];
    }
    None
}

/// First column of `name` in the per-atom lines, from `Properties=`.
fn column(properties: &str, names: &[&str]) -> Option<usize> {
    let fields: Vec<&str> = properties.split(':').collect();
    let mut start = 0;
    for field in fields.chunks(3) {
        if names.contains(&field[0]) {
            return Some(start);
        }
        start += field.get(2)?.parse::<usize>().ok()?;
    }
    None
}

fn open(path: &Path) -> std::io::Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    let reader: Box<dyn Read> = if path.extension().is_some_and(|e| e == "gz") {
        Box::new(MultiGzDecoder::new(BufReader::with_capacity(1 << 20, file)))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::with_capacity(1 << 20, reader)))
}

#[derive(Default)]
struct Scan {
    systems: Vec<i32>,
    values: Vec<[f64; QUANTITIES.len()]>,
    composition: Vec<[i32; 2]>,
    counts: Vec<f64>,
    groups: Vec<String>,
    atoms: usize,
    invalid: usize,
}

fn scan(path: &Path, first_index: usize, with_systems: bool) -> Result<Scan, String> {
    let error = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut input = open(path).map_err(error)?;
    let mut scan = Scan::default();
    let mut group_codes = HashMap::<String, usize>::new();
    let (mut line, mut header) = (String::new(), String::new());
    let (mut types, mut positions) = (Vec::new(), Vec::new());

    loop {
        line.clear();
        if input.read_line(&mut line).map_err(error)? == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let n: usize = line.trim().parse().map_err(|_| format!("{}: bad atom count {line:?}", path.display()))?;
        header.clear();
        input.read_line(&mut header).map_err(error)?;

        let properties = header_value(&header, "Properties").unwrap_or("species:S:1:pos:R:3");
        let pos = column(properties, &["pos", "positions"]).ok_or("no pos column")?;
        let forces = column(properties, &FORCE_KEYS);
        let pbc_value = header_value(&header, "pbc").unwrap_or("F F F");
        let mut pbc = [false; 3];
        for (p, token) in pbc.iter_mut().zip(pbc_value.split_ascii_whitespace()) {
            *p = token == "T" || token == "True";
        }
        let mut cell = [0.0; 9];
        if let Some(lattice) = header_value(&header, "Lattice") {
            for (c, token) in cell.iter_mut().zip(lattice.split_ascii_whitespace()) {
                *c = token.parse().unwrap_or(0.0);
            }
        }
        for axis in 0..3 {
            if !pbc[axis] {
                cell[3 * axis..3 * axis + 3].fill(0.0); // metatomic wants zero vectors along open axes
            }
        }

        types.clear();
        positions.clear();
        let (mut max_force, mut sum_force2) = (0.0f64, 0.0f64);
        let mut counts = [0i32; 119];
        for _ in 0..n {
            line.clear();
            input.read_line(&mut line).map_err(error)?;
            let tokens: Vec<&str> = line.split_ascii_whitespace().collect();
            let z = ELEMENTS.iter().position(|&e| e == tokens[0]).unwrap_or(0);
            counts[z] += 1;
            types.push(z as i32);
            positions.extend(tokens[pos..pos + 3].iter().map(|t| t.parse::<f64>().unwrap_or(f64::NAN)));
            if let Some(f) = forces {
                let norm2: f64 = tokens[f..f + 3].iter().map(|t| t.parse::<f64>().unwrap_or(0.0).powi(2)).sum();
                max_force = max_force.max(norm2.sqrt());
                sum_force2 += norm2;
            }
        }

        if with_systems {
            if let Err(message) = check_system(&types, &positions, cell, pbc) {
                scan.invalid += 1;
                if scan.invalid <= 5 {
                    eprintln!("{}: frame {}: {message}", path.display(), scan.systems.len());
                }
            }
        }

        let number = |key: &str| header_value(&header, key).and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::NAN);
        let energy = ENERGY_KEYS.iter().map(|k| number(k)).find(|e| !e.is_nan()).unwrap_or(f64::NAN);
        let periodic = pbc.iter().all(|&p| p);
        let volume = if periodic {
            let c = &cell;
            (c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6])).abs()
        } else {
            f64::NAN
        };
        let group = header_value(&header, "dataset_group").unwrap_or("");
        let next_code = group_codes.len();
        let code = *group_codes.entry(group.to_string()).or_insert_with(|| {
            scan.groups.push(group.to_string());
            next_code
        });

        let fps_order = number("fps_order");
        let system = if fps_order.is_nan() { first_index + scan.systems.len() } else { fps_order as usize };
        let system = i32::try_from(system).map_err(|_| "more than 2^31 systems")?;
        for (z, &count) in counts.iter().enumerate().filter(|(_, &c)| c > 0) {
            scan.composition.push([system, z as i32]);
            scan.counts.push(f64::from(count));
        }
        scan.systems.push(system);
        scan.values.push([
            n as f64,
            energy,
            energy / n as f64,
            if forces.is_some() { max_force } else { f64::NAN },
            if forces.is_some() { (sum_force2 / n as f64).sqrt() } else { f64::NAN },
            number("epsilon_sq"),
            number("k_set"),
            number("voronoi_population"),
            f64::from(u8::from(periodic)),
            volume,
            code as f64, // remapped to global codes in main
        ]);
        scan.atoms += n;
    }
    Ok(scan)
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut output, mut composition, mut with_systems, mut inputs) = (None, None, false, Vec::<PathBuf>::new());
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" | "--output" => output = args.next().map(PathBuf::from),
            "--composition" => composition = args.next().map(PathBuf::from),
            "--systems" => with_systems = true,
            _ => inputs.push(arg.into()),
        }
    }
    let output = output.ok_or("usage: madcore-scan -o summary.mts [--composition c.mts] [--systems] files...")?;
    if inputs.is_empty() {
        return Err("no input files".into());
    }

    let start = Instant::now();
    let scans: Vec<Result<Scan, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = inputs.iter().map(|path| scope.spawn(move || scan(path, 0, with_systems))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let scans = scans.into_iter().collect::<Result<Vec<_>, _>>()?;
    let seconds = start.elapsed().as_secs_f64();

    // merge, turning per-file dataset_group codes into global ones
    let mut groups = Vec::<String>::new();
    let mut total = Scan::default();
    for scan in scans {
        let remap: Vec<f64> = scan
            .groups
            .iter()
            .map(|g| {
                let code = groups.iter().position(|x| x == g).unwrap_or_else(|| {
                    groups.push(g.clone());
                    groups.len() - 1
                });
                code as f64
            })
            .collect();
        total.values.extend(scan.values.into_iter().map(|mut v| {
            v[QUANTITIES.len() - 1] = remap[v[QUANTITIES.len() - 1] as usize];
            v
        }));
        total.systems.extend(scan.systems);
        total.composition.extend(scan.composition);
        total.counts.extend(scan.counts);
        total.atoms += scan.atoms;
        total.invalid += scan.invalid;
    }
    let n = total.systems.len();
    println!(
        "{n} frames, {} atoms from {} files in {seconds:.1} s ({:.0} frames/s){}",
        total.atoms,
        inputs.len(),
        n as f64 / seconds,
        if with_systems { format!(", {} invalid systems", total.invalid) } else { String::new() }
    );

    let single_block = |block: TensorBlock| TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).map_err(|e| e.to_string());

    let values = Array2::from_shape_vec((n, QUANTITIES.len()), total.values.concat()).unwrap();
    let samples = Labels::new(["system"], total.systems.iter().map(|&s| [s]).collect::<Vec<_>>());
    let properties = Labels::new(["quantity"], (0..QUANTITIES.len() as i32).map(|i| [i]).collect::<Vec<_>>());
    let block = TensorBlock::new(values.into_dyn(), &samples, &[], &properties).map_err(|e| e.to_string())?;
    let mut summary = single_block(block)?;
    summary.set_info("quantities", &QUANTITIES.join(","));
    summary.set_info("dataset_group", &groups.join(","));
    summary.save(&output).map_err(|e| e.to_string())?;
    println!("wrote {}", output.display());

    if let Some(path) = composition {
        let values = Array2::from_shape_vec((total.counts.len(), 1), total.counts).unwrap();
        let samples = Labels::new(["system", "atomic_type"], total.composition);
        let block = TensorBlock::new(values.into_dyn(), &samples, &[], &Labels::new(["count"], [[0]]))
            .map_err(|e| e.to_string())?;
        single_block(block)?.save(&path).map_err(|e| e.to_string())?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
