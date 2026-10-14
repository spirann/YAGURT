//! Access to the raw card image: cluster geometry and one scanning pass.
//!
//! The engine never touches the OS directly: it reads through [`ReadAt`], so
//! the same code can run on a memory-mapped file, a block device or (later)
//! a browser-provided byte source compiled to WebAssembly.

use std::collections::HashMap;

/// Random-access byte source (an image file, a device, a buffer).
pub trait ReadAt: Sync {
    fn size(&self) -> u64;
    /// Fill `buf` from `pos`; returns how many bytes were available.
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> usize;
}

impl ReadAt for [u8] {
    fn size(&self) -> u64 {
        self.len() as u64
    }
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> usize {
        if pos >= self.len() as u64 {
            return 0;
        }
        let p = pos as usize;
        let n = buf.len().min(self.len() - p);
        buf[..n].copy_from_slice(&self[p..p + n]);
        n
    }
}

impl ReadAt for Vec<u8> {
    fn size(&self) -> u64 {
        self.as_slice().size()
    }
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> usize {
        self.as_slice().read_at(pos, buf)
    }
}

/// A read-only, memory-mapped image file.
#[cfg(feature = "mmap")]
pub struct MappedFile(memmap2::Mmap);

#[cfg(feature = "mmap")]
impl MappedFile {
    pub fn open(path: &std::path::Path) -> std::io::Result<MappedFile> {
        let f = std::fs::File::open(path)?;
        // SAFETY: the image is opened read-only and is not expected to change
        // while we read it.
        Ok(MappedFile(unsafe { memmap2::Mmap::map(&f)? }))
    }
}

#[cfg(feature = "mmap")]
impl ReadAt for MappedFile {
    fn size(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> usize {
        self.0[..].read_at(pos, buf)
    }
}

pub const FLAG_ZERO: u8 = 1; // all 0x00 or all 0xFF: never part of a video
pub const FLAG_SIGNATURE: u8 = 2; // starts another file (ftyp / JPEG)
pub const FLAG_HEADER: u8 = 4; // belongs to a parsed MP4 header

#[derive(Clone, Debug)]
pub struct Geometry {
    pub cluster_size: u64,
    /// byte offset of cluster 0 modulo the cluster size
    pub data_offset: u64,
    pub source: String,
}

pub struct Image<'a> {
    pub src: &'a dyn ReadAt,
    pub size: u64,
    pub cluster_size: u64,
    pub data_offset: u64,
}

impl<'a> Image<'a> {
    pub fn new(src: &'a dyn ReadAt, g: &Geometry) -> Self {
        Image { src, size: src.size(), cluster_size: g.cluster_size, data_offset: g.data_offset % g.cluster_size }
    }

    pub fn n_clusters(&self) -> usize {
        ((self.size.saturating_sub(self.data_offset)) / self.cluster_size) as usize
    }

    pub fn cluster_pos(&self, k: usize) -> u64 {
        self.data_offset + k as u64 * self.cluster_size
    }

    pub fn read(&self, pos: u64, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        let got = self.src.read_at(pos, &mut v);
        v.truncate(got);
        v
    }

    /// Fill `buf` exactly; false if the image ends first.
    pub fn read_into(&self, pos: u64, buf: &mut [u8]) -> bool {
        self.src.read_at(pos, buf) == buf.len()
    }
}

fn read(src: &dyn ReadAt, pos: u64, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    let got = src.read_at(pos, &mut v);
    v.truncate(got);
    v
}

fn le16(b: &[u8], p: usize) -> u64 {
    u16::from_le_bytes([b[p], b[p + 1]]) as u64
}

fn le32(b: &[u8], p: usize) -> u64 {
    u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]) as u64
}

/// Read cluster size and data-area start from the (new) file system.
///
/// An in-camera format writes a fresh FAT32/exFAT with the same geometry, so
/// the boot sector still tells us where clusters start.
pub fn detect_geometry(src: &dyn ReadAt) -> Option<Geometry> {
    let mut starts = vec![0u64];
    let mbr = read(src, 0, 512);
    if mbr.len() == 512 && mbr[510..512] == [0x55, 0xAA] {
        for i in 0..4 {
            let e = &mbr[446 + 16 * i..462 + 16 * i];
            let lba = le32(e, 8);
            if e[4] != 0 && lba != 0 && lba * 512 < src.size() {
                starts.push(lba * 512);
            }
        }
    }
    starts.into_iter().find_map(|base| boot_sector(src, base))
}

fn boot_sector(src: &dyn ReadAt, base: u64) -> Option<Geometry> {
    let bs = read(src, base, 512);
    if bs.len() < 512 || bs[510..512] != [0x55, 0xAA] {
        return None;
    }
    if &bs[3..11] == b"EXFAT   " {
        let bps = 1u64 << bs[108];
        let spc = 1u64 << bs[109];
        let heap = le32(&bs, 88);
        return Some(Geometry {
            cluster_size: bps * spc,
            data_offset: base + heap * bps,
            source: format!("exFAT boot sector at 0x{base:x}"),
        });
    }
    let bps = le16(&bs, 11);
    let spc = bs[13] as u64;
    if ![512, 1024, 2048, 4096].contains(&bps) || spc == 0 || spc & (spc - 1) != 0 {
        return None;
    }
    let reserved = le16(&bs, 14);
    let nfats = bs[16] as u64;
    let root_entries = le16(&bs, 17);
    let fatsz = match le16(&bs, 22) {
        0 => le32(&bs, 36),
        v => v,
    };
    if reserved == 0 || nfats == 0 || fatsz == 0 {
        return None;
    }
    let root_sectors = (root_entries * 32).div_ceil(bps);
    let data = base + (reserved + nfats * fatsz + root_sectors) * bps;
    let kind = if root_entries == 0 { "FAT32" } else { "FAT12/16" };
    Some(Geometry { cluster_size: bps * spc, data_offset: data, source: format!("{kind} boot sector at 0x{base:x}") })
}

/// Infer cluster alignment from where `ftyp` boxes start (sector aligned).
pub fn guess_alignment(src: &dyn ReadAt, cluster_size: u64) -> u64 {
    let mut hits: HashMap<u64, usize> = HashMap::new();
    let chunk = 8 << 20;
    let mut buf = vec![0u8; chunk + 8];
    let mut pos = 0u64;
    let size = src.size();
    while pos < size && hits.values().sum::<usize>() < 200_000 {
        let n = src.read_at(pos, &mut buf);
        if n < 8 {
            break;
        }
        let mut s = 0;
        while s + 8 <= n {
            if &buf[s + 4..s + 8] == b"ftyp" {
                *hits.entry((pos + s as u64) % cluster_size).or_default() += 1;
            }
            s += 512;
        }
        pos += chunk as u64;
    }
    hits.into_iter().max_by_key(|&(_, c)| c).map(|(o, _)| o).unwrap_or(0)
}

/// One pass over every cluster: per-cluster flags and the clusters that start with `ftyp`.
pub fn scan(img: &Image, progress: &mut dyn FnMut(f32)) -> (Vec<u8>, Vec<usize>) {
    let n = img.n_clusters();
    let c = img.cluster_size as usize;
    let mut flags = vec![0u8; n];
    let mut starts = Vec::new();
    let per_read = ((16 << 20) / c).max(1);
    let mut buf = vec![0u8; per_read * c];
    let mut k = 0;
    while k < n {
        let cnt = per_read.min(n - k);
        let got = img.src.read_at(img.cluster_pos(k), &mut buf[..cnt * c]);
        for i in 0..cnt {
            if (i + 1) * c > got {
                break;
            }
            let cl = &buf[i * c..(i + 1) * c];
            if &cl[4..8] == b"ftyp" {
                flags[k + i] |= FLAG_SIGNATURE;
                starts.push(k + i);
            } else if cl[..3] == [0xFF, 0xD8, 0xFF] {
                flags[k + i] |= FLAG_SIGNATURE;
            } else if cl.iter().all(|&b| b == 0) || cl.iter().all(|&b| b == 0xFF) {
                flags[k + i] |= FLAG_ZERO;
            }
        }
        k += cnt;
        progress(k as f32 / n as f32);
    }
    (flags, starts)
}
