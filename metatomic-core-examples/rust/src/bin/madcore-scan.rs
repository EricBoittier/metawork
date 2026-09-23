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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use madcore_scan::{FrameReader, MtaSystem};
use metatensor::{Labels, TensorBlock, TensorMap};
use ndarray::Array2;

const QUANTITIES: [&str; 11] = [
    "n_atoms", "energy", "energy_per_atom", "max_force", "rms_force", "epsilon_sq", "k_set",
    "voronoi_population", "periodic", "volume", "dataset_group",
];
const GROUP: usize = QUANTITIES.len() - 1;

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

fn determinant(c: &[f64; 9]) -> f64 {
    c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6])
}

fn scan(path: &Path, with_systems: bool) -> Result<Scan, String> {
    let mut scan = Scan::default();
    let mut group_codes = HashMap::<String, usize>::new();
    for frame in FrameReader::open(path)? {
        let frame = frame?;
        let n = frame.len();
        if with_systems {
            if let Err(message) = MtaSystem::new(&frame) {
                scan.invalid += 1;
                if scan.invalid <= 5 {
                    eprintln!("{}: frame {}: {message}", path.display(), scan.systems.len());
                }
            }
        }

        let (max_force, rms_force) = match &frame.forces {
            Some(forces) => {
                let norms2: Vec<f64> = forces.chunks_exact(3).map(|f| f.iter().map(|x| x * x).sum()).collect();
                (norms2.iter().copied().fold(0.0, f64::max).sqrt(), (norms2.iter().sum::<f64>() / n as f64).sqrt())
            }
            None => (f64::NAN, f64::NAN),
        };
        let group = frame.value("dataset_group").unwrap_or("").to_string();
        let next = group_codes.len();
        let code = *group_codes.entry(group.clone()).or_insert_with(|| {
            scan.groups.push(group);
            next
        });

        let fps_order = frame.number("fps_order");
        let system = if fps_order.is_nan() { scan.systems.len() } else { fps_order as usize };
        let system = i32::try_from(system).map_err(|_| "more than 2^31 systems")?;
        let mut counts = [0i32; 119];
        for &z in &frame.types {
            counts[z as usize] += 1;
        }
        for (z, &count) in counts.iter().enumerate().filter(|(_, &c)| c > 0) {
            scan.composition.push([system, z as i32]);
            scan.counts.push(f64::from(count));
        }

        let energy = frame.energy();
        scan.systems.push(system);
        scan.values.push([
            n as f64,
            energy,
            energy / n as f64,
            max_force,
            rms_force,
            frame.number("epsilon_sq"),
            frame.number("k_set"),
            frame.number("voronoi_population"),
            f64::from(u8::from(frame.periodic())),
            if frame.periodic() { determinant(&frame.cell).abs() } else { f64::NAN },
            code as f64, // per-file code, remapped to a global one in main
        ]);
        scan.atoms += n;
    }
    Ok(scan)
}

fn single_block(block: TensorBlock) -> Result<TensorMap, String> {
    TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).map_err(|e| e.to_string())
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
        let handles: Vec<_> = inputs.iter().map(|path| scope.spawn(move || scan(path, with_systems))).collect();
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
            v[GROUP] = remap[v[GROUP] as usize];
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
