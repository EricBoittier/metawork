//! Convert extxyz shards into metatrain datasets, one per split.
//!
//! Two output formats are supported, both read by `metatrain` directly
//! (`systems: read_from: <path>`):
//!
//! **`--format memmap`** (default): a `MemmapDataset` directory per split,
//! `train/`, `val/`, `test/`, holding flat arrays: `ns.npy`, `na.npy`
//! (cumulative atom counts), `x.bin` (positions), `a.bin` (types), `c.bin`
//! (cells, zero along open directions, from which metatrain infers pbc),
//! `energy.bin` and `non_conservative_force.bin`, all float32 / int32. This is
//! what metatrain recommends for large datasets: ~7 GB for all of MAD-CORE,
//! and no per-structure deserialization while training. Each structure still
//! goes through `mta_system_create`, so invalid cells / pbc are rejected.
//!
//! **`--format zip`**: a `DiskDataset` zip per split. Each structure becomes
//! the three members metatrain's `DiskDataset` reads
//! (the same layout as `metatrain.utils.data.writers.DiskDatasetWriter`):
//!
//! * `<i>/system.mta`: the system, serialized with the metatomic C API
//!   (`mta_system_create` + `mta_save_buffer`, float64, Angstrom);
//! * `<i>/energy.mts`: energy TensorMap, samples `system`, property `energy`;
//! * `<i>/non_conservative_force.mts`: per-atom forces, samples
//!   `[system, atom]`, component `xyz`, property `non_conservative_force`;
//!
//! where the `.mts` members hold the metatensor buffer as a `uint8` `.npy`
//! array, and the `system` sample of both targets is the entry number `i`.
//! Every zip also gets `metadata/atom_counts.npy`, which metatrain needs for
//! `max_atoms_per_batch`.
//!
//! The split comes from `madcore/make_split.py` (`val` / `test` row indices,
//! everything else is `train`). Rows are identified by the `fps_order` header
//! key, and entry `i` of a split is its `i`-th smallest row, so the mapping
//! back to MAD-CORE rows is `load_split()[name][i]` (restricted to rows below
//! `--max-rows`).
//!
//! ```text
//! madcore-to-diskdataset --split madcore-split-seed0.npz -o out/ [--format memmap|zip] [--max-rows N] shard.extxyz.gz...
//! ```

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::time::Instant;

use madcore_scan::{fix_one_byte_descr, npy_1d, npy_scalar, read_npy_ints, Frame, FrameReader, MtaSystem};
use metatensor::{Labels, TensorBlock, TensorMap};
use ndarray::{Array2, Array3};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

const SPLITS: [&str; 3] = ["train", "val", "test"];

/// For every row below `max_rows`: which split it belongs to, and its entry
/// number in that split.
struct Assignment {
    split: Vec<u8>,
    entry: Vec<u32>,
    counts: [usize; 3],
}

fn read_split(path: &Path, max_rows: Option<usize>) -> Result<Assignment, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut array = |name: &str| -> Result<Vec<i64>, String> {
        let mut bytes = Vec::new();
        archive.by_name(name).map_err(|e| format!("{name}: {e}"))?.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        read_npy_ints(&bytes)
    };
    let n_rows = array("n_rows.npy")?[0] as usize;
    let n_rows = max_rows.map_or(n_rows, |m| m.min(n_rows));
    let mut split = vec![0u8; n_rows];
    for (code, name) in [(1u8, "val.npy"), (2u8, "test.npy")] {
        for row in array(name)? {
            if (row as usize) < n_rows {
                split[row as usize] = code;
            }
        }
    }
    let mut counts = [0usize; 3];
    let entry = split
        .iter()
        .map(|&s| {
            counts[s as usize] += 1;
            (counts[s as usize] - 1) as u32
        })
        .collect();
    Ok(Assignment { split, entry, counts })
}

/// A TensorMap as stored in a DiskDataset member: its buffer, as a uint8 `.npy`.
fn member(tensor: &TensorMap) -> Result<Vec<u8>, String> {
    let mut buffer = Vec::new();
    tensor.save_buffer(&mut buffer).map_err(|e| e.to_string())?;
    Ok(npy_1d("|u1", buffer.len(), &buffer))
}

fn single_block(block: TensorBlock) -> Result<TensorMap, String> {
    TensorMap::new(Labels::new(["_"], [[0]]), vec![block]).map_err(|e| e.to_string())
}

struct Entry {
    index: u32,
    system: Vec<u8>,
    energy: Vec<u8>,
    force: Vec<u8>,
    n_atoms: usize,
}

fn to_entry(frame: &Frame, index: u32) -> Result<Entry, String> {
    let n = frame.len();
    let system = fix_one_byte_descr(&MtaSystem::new(frame)?.save_buffer()?)?;
    let entry = index as i32;

    let energy = TensorBlock::new(
        Array2::from_elem((1, 1), frame.energy()).into_dyn(),
        &Labels::new(["system"], [[entry]]),
        &[],
        &Labels::new(["energy"], [[0]]),
    )
    .map_err(|e| e.to_string())?;

    let forces = frame.forces.clone().ok_or("frame has no forces")?;
    let force = TensorBlock::new(
        Array3::from_shape_vec((n, 3, 1), forces).map_err(|e| e.to_string())?.into_dyn(),
        &Labels::new(["system", "atom"], (0..n as i32).map(|a| [entry, a]).collect::<Vec<_>>()),
        &[Labels::new(["xyz"], [[0], [1], [2]])],
        &Labels::new(["non_conservative_force"], [[0]]),
    )
    .map_err(|e| e.to_string())?;

    Ok(Entry {
        index,
        system,
        energy: member(&single_block(energy)?)?,
        force: member(&single_block(force)?)?,
        n_atoms: n,
    })
}

fn produce(path: &Path, assignment: &Assignment, senders: &[SyncSender<Entry>; 3]) -> Result<usize, String> {
    let mut written = 0;
    for frame in FrameReader::open(path)? {
        let frame = frame?;
        let row = frame.number("fps_order");
        if row.is_nan() {
            return Err(format!("{}: frame without fps_order", path.display()));
        }
        let row = row as usize;
        if row >= assignment.split.len() {
            break; // rows are increasing within a shard
        }
        let split = assignment.split[row] as usize;
        senders[split].send(to_entry(&frame, assignment.entry[row])?).map_err(|e| e.to_string())?;
        written += 1;
    }
    Ok(written)
}

fn write_zip(path: &Path, entries: Receiver<Entry>, count: usize) -> Result<(), String> {
    let error = |e: zip::result::ZipError| format!("{}: {e}", path.display());
    let io_error = |e: std::io::Error| format!("{}: {e}", path.display());
    let file = File::create(path).map_err(io_error)?;
    let mut zip = ZipWriter::new(BufWriter::with_capacity(8 << 20, file));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    let mut atom_counts = vec![-1i64; count];
    for entry in entries {
        for (name, data) in [("system.mta", &entry.system), ("energy.mts", &entry.energy), ("non_conservative_force.mts", &entry.force)] {
            zip.start_file(format!("{}/{name}", entry.index), options).map_err(error)?;
            zip.write_all(data).map_err(io_error)?;
        }
        atom_counts[entry.index as usize] = entry.n_atoms as i64;
    }
    if let Some(missing) = atom_counts.iter().position(|&c| c < 0) {
        return Err(format!("{}: entry {missing} was never written (missing input shard?)", path.display()));
    }
    let bytes: Vec<u8> = atom_counts.iter().flat_map(|c| c.to_le_bytes()).collect();
    zip.start_file("metadata/atom_counts.npy", options).map_err(error)?;
    zip.write_all(&npy_1d("<i8", count, &bytes)).map_err(io_error)?;
    zip.finish().map_err(error)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// MemmapDataset output

const MEMMAP_ARRAYS: [&str; 6] = ["x", "a", "c", "energy", "non_conservative_force", "n"];

fn part_path(dir: &Path, part: usize, name: &str) -> PathBuf {
    dir.join(format!(".part-{part}-{name}.bin"))
}

/// Append one shard's structures to per-split part files, returning the first
/// row seen (to order the parts) and the number of structures written.
fn produce_memmap(path: &Path, part: usize, assignment: &Assignment, output: &Path) -> Result<(usize, usize), String> {
    let io_error = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut files: Vec<Vec<BufWriter<File>>> = SPLITS
        .iter()
        .map(|split| {
            MEMMAP_ARRAYS
                .iter()
                .map(|name| File::create(part_path(&output.join(split), part, name)).map(|f| BufWriter::with_capacity(1 << 20, f)))
                .collect::<std::io::Result<Vec<_>>>()
        })
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(io_error)?;

    let (mut first_row, mut written) = (usize::MAX, 0);
    for frame in FrameReader::open(path)? {
        let frame = frame?;
        let row = frame.number("fps_order");
        if row.is_nan() {
            return Err(format!("{}: frame without fps_order", path.display()));
        }
        let row = row as usize;
        if row >= assignment.split.len() {
            break; // rows are increasing within a shard
        }
        first_row = first_row.min(row);
        MtaSystem::new(&frame)?; // validates types / cell / pbc

        let forces = frame.forces.as_ref().ok_or("frame has no forces")?;
        let out = &mut files[assignment.split[row] as usize];
        let f32s = |values: &[f64]| values.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect::<Vec<u8>>();
        out[0].write_all(&f32s(&frame.positions)).map_err(io_error)?;
        out[1].write_all(&frame.types.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>()).map_err(io_error)?;
        out[2].write_all(&f32s(&frame.cell)).map_err(io_error)?;
        out[3].write_all(&f32s(&[frame.energy()])).map_err(io_error)?;
        out[4].write_all(&f32s(forces)).map_err(io_error)?;
        out[5].write_all(&(frame.len() as i64).to_le_bytes()).map_err(io_error)?;
        written += 1;
    }
    for writer in files.iter_mut().flatten() {
        writer.flush().map_err(io_error)?;
    }
    Ok((first_row, written))
}

/// Concatenate the part files of one split, in row order, into the final
/// MemmapDataset arrays.
fn finish_memmap(dir: &Path, parts: &[usize], expected: usize) -> Result<(), String> {
    let io_error = |e: std::io::Error| format!("{}: {e}", dir.display());
    for name in &MEMMAP_ARRAYS[..5] {
        let mut out = BufWriter::with_capacity(8 << 20, File::create(dir.join(format!("{name}.bin"))).map_err(io_error)?);
        for &part in parts {
            let mut input = File::open(part_path(dir, part, name)).map_err(io_error)?;
            std::io::copy(&mut input, &mut out).map_err(io_error)?;
        }
        out.flush().map_err(io_error)?;
    }

    let mut na = vec![0i64];
    for &part in parts {
        let bytes = std::fs::read(part_path(dir, part, "n")).map_err(io_error)?;
        for count in bytes.chunks_exact(8) {
            na.push(na.last().unwrap() + i64::from_le_bytes(count.try_into().unwrap()));
        }
    }
    let ns = na.len() - 1;
    if ns != expected {
        return Err(format!("{}: wrote {ns} structures, expected {expected} (missing input shard?)", dir.display()));
    }
    let bytes: Vec<u8> = na.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join("na.npy"), npy_1d("<i8", na.len(), &bytes)).map_err(io_error)?;
    // a 0-d array: MemmapDataset uses it directly as a shape
    std::fs::write(dir.join("ns.npy"), npy_scalar("<i8", &(ns as i64).to_le_bytes())).map_err(io_error)?;
    for &part in parts {
        for name in MEMMAP_ARRAYS {
            std::fs::remove_file(part_path(dir, part, name)).map_err(io_error)?;
        }
    }
    Ok(())
}

fn convert_memmap(inputs: &[PathBuf], assignment: &Assignment, output: &Path) -> Result<usize, String> {
    for split in SPLITS {
        std::fs::create_dir_all(output.join(split)).map_err(|e| e.to_string())?;
    }
    let results: Vec<Result<(usize, usize), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(part, path)| scope.spawn(move || produce_memmap(path, part, assignment, output)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let results = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    let mut order: Vec<usize> = (0..inputs.len()).collect();
    order.sort_by_key(|&part| results[part].0);
    for (split, count) in SPLITS.iter().zip(assignment.counts) {
        finish_memmap(&output.join(split), &order, count)?;
    }
    Ok(results.iter().map(|r| r.1).sum())
}

fn convert_zip(inputs: &[PathBuf], assignment: &Assignment, output: &Path) -> Result<usize, String> {
    std::thread::scope(|scope| -> Result<usize, String> {
        let channels = SPLITS.map(|_| sync_channel::<Entry>(4096));
        let (senders, receivers): (Vec<_>, Vec<_>) = channels.into_iter().unzip();
        let writers: Vec<_> = receivers
            .into_iter()
            .zip(SPLITS)
            .zip(assignment.counts)
            .map(|((rx, name), count)| {
                let path = output.join(format!("{name}.zip"));
                scope.spawn(move || write_zip(&path, rx, count))
            })
            .collect();
        let senders: [SyncSender<Entry>; 3] = senders.try_into().unwrap();
        let producers: Vec<_> = inputs
            .iter()
            .map(|path| {
                let senders = senders.clone();
                scope.spawn(move || produce(path, assignment, &senders))
            })
            .collect();
        drop(senders);
        let written = producers.into_iter().map(|p| p.join().unwrap()).sum::<Result<usize, String>>()?;
        for writer in writers {
            writer.join().unwrap()?;
        }
        Ok(written)
    })
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut split, mut output, mut max_rows, mut format, mut inputs) = (None, None, None, "memmap".to_string(), Vec::<PathBuf>::new());
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--split" => split = args.next().map(PathBuf::from),
            "-o" | "--output" => output = args.next().map(PathBuf::from),
            "--max-rows" => max_rows = args.next().and_then(|v| v.parse().ok()),
            "--format" => format = args.next().unwrap_or_default(),
            _ => inputs.push(arg.into()),
        }
    }
    let usage = "usage: madcore-to-diskdataset --split split.npz -o DIR [--format memmap|zip] [--max-rows N] files...";
    let (split, output) = (split.ok_or(usage)?, output.ok_or(usage)?);
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;

    let assignment = read_split(&split, max_rows)?;
    println!(
        "{} rows: {}",
        assignment.split.len(),
        SPLITS.iter().zip(assignment.counts).map(|(s, c)| format!("{s} {c}")).collect::<Vec<_>>().join(", ")
    );

    let start = Instant::now();
    let written = match format.as_str() {
        "memmap" => convert_memmap(&inputs, &assignment, &output)?,
        "zip" => convert_zip(&inputs, &assignment, &output)?,
        other => return Err(format!("unknown --format '{other}', expected memmap or zip")),
    };
    println!("{written} structures in {:.1} s", start.elapsed().as_secs_f64());
    for name in SPLITS {
        let path = if format == "zip" { output.join(format!("{name}.zip")) } else { output.join(name) };
        let size: u64 = if path.is_dir() {
            std::fs::read_dir(&path).map_err(|e| e.to_string())?.filter_map(|e| e.ok()?.metadata().ok()).map(|m| m.len()).sum()
        } else {
            std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
        };
        println!("  {} ({:.2} GB)", path.display(), size as f64 / 1e9);
    }
    Ok(())
}
