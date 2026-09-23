//! Shared pieces of the Rust examples: a streaming extxyz reader, a thin
//! wrapper around the metatomic-core C API (`mta_system_t`), and `.npy`
//! helpers.

use std::ffi::{c_char, c_void, CStr};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use dlpk::DLPackTensor;
use flate2::read::MultiGzDecoder;
use ndarray::{Array1, Array2};

pub const ENERGY_KEYS: [&str; 3] = ["ecumetric_energy", "energy", "REF_energy"];
pub const FORCE_KEYS: [&str; 3] = ["ecumetric_nc_forces", "forces", "REF_forces"];

pub const ELEMENTS: [&str; 119] = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
    "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

// ---------------------------------------------------------------------------
// extxyz

/// Value of `key=` in an extxyz comment line, without surrounding quotes.
pub fn header_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
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

/// First column of any of `names` in the per-atom lines, from `Properties=`.
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

/// One structure. The cell is zero along non-periodic directions, as
/// metatomic requires.
pub struct Frame {
    pub types: Vec<i32>,
    pub positions: Vec<f64>,
    pub forces: Option<Vec<f64>>,
    pub cell: [f64; 9],
    pub pbc: [bool; 3],
    pub header: String,
}

impl Frame {
    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    pub fn value(&self, key: &str) -> Option<&str> {
        header_value(&self.header, key)
    }

    /// A numeric header value, NaN when missing.
    pub fn number(&self, key: &str) -> f64 {
        self.value(key).and_then(|v| v.parse().ok()).unwrap_or(f64::NAN)
    }

    pub fn energy(&self) -> f64 {
        ENERGY_KEYS.iter().map(|k| self.number(k)).find(|e| !e.is_nan()).unwrap_or(f64::NAN)
    }

    pub fn periodic(&self) -> bool {
        self.pbc.iter().all(|&p| p)
    }
}

/// Streaming reader over the frames of an `.extxyz` or `.extxyz.gz` file.
pub struct FrameReader {
    input: Box<dyn BufRead + Send>,
    path: PathBuf,
    line: String,
}

impl FrameReader {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let reader: Box<dyn Read + Send> = if path.extension().is_some_and(|e| e == "gz") {
            Box::new(MultiGzDecoder::new(BufReader::with_capacity(1 << 20, file)))
        } else {
            Box::new(file)
        };
        Ok(FrameReader { input: Box::new(BufReader::with_capacity(1 << 20, reader)), path: path.into(), line: String::new() })
    }

    fn read_line(&mut self) -> Result<usize, String> {
        self.line.clear();
        self.input.read_line(&mut self.line).map_err(|e| format!("{}: {e}", self.path.display()))
    }

    fn read_frame(&mut self) -> Result<Option<Frame>, String> {
        loop {
            if self.read_line()? == 0 {
                return Ok(None);
            }
            if !self.line.trim().is_empty() {
                break;
            }
        }
        let n: usize = self.line.trim().parse().map_err(|_| format!("{}: bad atom count {:?}", self.path.display(), self.line))?;
        self.read_line()?;
        let header = std::mem::take(&mut self.line);

        let properties = header_value(&header, "Properties").unwrap_or("species:S:1:pos:R:3");
        let pos = column(properties, &["pos", "positions"]).ok_or("no pos column in Properties")?;
        let force = column(properties, &FORCE_KEYS);

        let mut pbc = [false; 3];
        for (p, token) in pbc.iter_mut().zip(header_value(&header, "pbc").unwrap_or("F F F").split_ascii_whitespace()) {
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
                cell[3 * axis..3 * axis + 3].fill(0.0);
            }
        }

        let mut frame = Frame {
            types: Vec::with_capacity(n),
            positions: Vec::with_capacity(3 * n),
            forces: force.map(|_| Vec::with_capacity(3 * n)),
            cell,
            pbc,
            header,
        };
        for _ in 0..n {
            self.read_line()?;
            let tokens: Vec<&str> = self.line.split_ascii_whitespace().collect();
            if tokens.len() < pos + 3 {
                return Err(format!("{}: truncated atom line {:?}", self.path.display(), self.line));
            }
            frame.types.push(ELEMENTS.iter().position(|&e| e == tokens[0]).unwrap_or(0) as i32);
            frame.positions.extend(tokens[pos..pos + 3].iter().map(|t| t.parse::<f64>().unwrap_or(f64::NAN)));
            if let (Some(f), Some(forces)) = (force, frame.forces.as_mut()) {
                forces.extend(tokens[f..f + 3].iter().map(|t| t.parse::<f64>().unwrap_or(f64::NAN)));
            }
        }
        Ok(Some(frame))
    }
}

impl Iterator for FrameReader {
    type Item = Result<Frame, String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read_frame().transpose()
    }
}

// ---------------------------------------------------------------------------
// the small part of the metatomic C API used here (see metatomic.h)

mod ffi {
    use std::ffi::{c_char, c_void};

    #[repr(C)]
    pub struct MtaSystem {
        _private: [u8; 0],
    }

    pub type Realloc = unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> *mut u8;

    extern "C" {
        pub fn mta_system_create(
            length_unit: *const c_char,
            types: *mut dlpk::sys::DLManagedTensorVersioned,
            positions: *mut dlpk::sys::DLManagedTensorVersioned,
            cell: *mut dlpk::sys::DLManagedTensorVersioned,
            pbc: *mut dlpk::sys::DLManagedTensorVersioned,
            system: *mut *mut MtaSystem,
        ) -> i32;
        pub fn mta_system_free(system: *mut MtaSystem) -> i32;
        pub fn mta_save_buffer(
            buffer: *mut *mut u8,
            buffer_count: *mut usize,
            realloc_user_data: *mut c_void,
            realloc: Realloc,
            system: *const MtaSystem,
        ) -> i32;
        pub fn mta_last_error(message: *mut *const c_char, origin: *mut *const c_char, data: *mut *mut c_void) -> i32;
    }
}

fn last_error() -> String {
    let mut message: *const c_char = std::ptr::null();
    unsafe { ffi::mta_last_error(&mut message, std::ptr::null_mut(), std::ptr::null_mut()) };
    if message.is_null() {
        return "unknown metatomic error".into();
    }
    unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned()
}

fn raw<A>(array: A) -> Result<*mut dlpk::sys::DLManagedTensorVersioned, String>
where
    A: TryInto<DLPackTensor>,
    A::Error: std::fmt::Debug,
{
    let tensor: DLPackTensor = array.try_into().map_err(|e| format!("{e:?}"))?;
    Ok(tensor.into_raw().as_ptr())
}

/// An owned `mta_system_t` (float64, Angstrom).
pub struct MtaSystem(*mut ffi::MtaSystem);

impl MtaSystem {
    /// Create a system through `mta_system_create`, which validates the
    /// types, positions, cell and pbc.
    pub fn new(frame: &Frame) -> Result<Self, String> {
        let n = frame.len();
        let types = raw(Array1::from_vec(frame.types.clone()))?;
        let positions = raw(Array2::from_shape_vec((n, 3), frame.positions.clone()).map_err(|e| e.to_string())?)?;
        let cell = raw(Array2::from_shape_vec((3, 3), frame.cell.to_vec()).map_err(|e| e.to_string())?)?;
        let pbc = raw(Array1::from_vec(frame.pbc.to_vec()))?;
        let mut system = std::ptr::null_mut();
        // SAFETY: mta_system_create takes ownership of the four tensors
        let status = unsafe { ffi::mta_system_create(c"angstrom".as_ptr(), types, positions, cell, pbc, &mut system) };
        if status != 0 {
            return Err(last_error());
        }
        Ok(MtaSystem(system))
    }

    /// Serialize with `mta_save_buffer`, in the `.mta` format that
    /// `metatomic.torch.load_system` reads.
    pub fn save_buffer(&self) -> Result<Vec<u8>, String> {
        unsafe extern "C" fn grow(user_data: *mut c_void, _ptr: *mut u8, new_size: usize) -> *mut u8 {
            let buffer = unsafe { &mut *user_data.cast::<Vec<u8>>() };
            buffer.resize(new_size, 0);
            buffer.as_mut_ptr()
        }
        let mut buffer = Vec::<u8>::new();
        let mut ptr = buffer.as_mut_ptr();
        let mut count = 0usize;
        let user_data = (&mut buffer as *mut Vec<u8>).cast::<c_void>();
        let status = unsafe { ffi::mta_save_buffer(&mut ptr, &mut count, user_data, grow, self.0) };
        if status != 0 {
            return Err(last_error());
        }
        buffer.truncate(count);
        Ok(buffer)
    }
}

impl Drop for MtaSystem {
    fn drop(&mut self) {
        unsafe { ffi::mta_system_free(self.0) };
    }
}

/// Rewrite a `.mta` buffer from `mta_save_buffer` so that metatomic-torch can
/// read it.
///
/// metatomic-core writes one-byte dtypes with an endianness prefix (`'<b1'`
/// for `pbc`), while numpy (and metatomic-torch's reader) use `'|b1'`. Each
/// `.npy` member of the zip gets its header fixed; the zip is rebuilt so the
/// CRCs match. Remove once metatomic-core writes `'|'` for 1-byte types.
pub fn fix_one_byte_descr(buffer: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Write;
    let err = |e: zip::result::ZipError| e.to_string();
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(buffer)).map_err(err)?;
    let mut out = zip::ZipWriter::new(std::io::Cursor::new(Vec::with_capacity(buffer.len())));
    let options = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for i in 0..archive.len() {
        let mut member = archive.by_index(i).map_err(err)?;
        let name = member.name().to_string();
        let mut data = Vec::with_capacity(member.size() as usize);
        member.read_to_end(&mut data).map_err(|e| e.to_string())?;
        for (from, to) in [(&b"'<b1'"[..], &b"'|b1'"[..]), (b"'<u1'", b"'|u1'"), (b"'<i1'", b"'|i1'")] {
            if let Some(at) = data.windows(from.len()).take(256).position(|w| w == from) {
                data[at..at + from.len()].copy_from_slice(to);
            }
        }
        out.start_file(name, options).map_err(err)?;
        out.write_all(&data).map_err(|e| e.to_string())?;
    }
    Ok(out.finish().map_err(err)?.into_inner())
}

// ---------------------------------------------------------------------------
// .npy

/// A 1-D `.npy` file with the given numpy `descr` (e.g. `|u1`, `<i8`).
pub fn npy_1d(descr: &str, count: usize, data: &[u8]) -> Vec<u8> {
    npy(descr, &format!("({count},)"), data)
}

/// A 0-d `.npy` file, what `np.save(path, 42)` writes.
pub fn npy_scalar(descr: &str, data: &[u8]) -> Vec<u8> {
    npy(descr, "()", data)
}

fn npy(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    // magic (6) + version (2) + header length (2) + header, padded to 64 bytes, ending in '\n'
    let padding = 64 - (10 + header.len() + 1) % 64;
    header.push_str(&" ".repeat(padding % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

/// Parse a little-endian integer `.npy` array (`<i4` or `<i8`, any shape).
pub fn read_npy_ints(bytes: &[u8]) -> Result<Vec<i64>, String> {
    if !bytes.starts_with(b"\x93NUMPY") {
        return Err("not a .npy file".into());
    }
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len]).map_err(|e| e.to_string())?;
    let data = &bytes[start + header_len..];
    if header.contains("'<i4'") {
        Ok(data.chunks_exact(4).map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap()))).collect())
    } else if header.contains("'<i8'") {
        Ok(data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
    } else {
        Err(format!("unsupported .npy dtype in header {header}"))
    }
}
