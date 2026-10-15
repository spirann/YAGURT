//! Minimal ISO-BMFF (MP4/MOV) parser: just enough to rebuild a sample map.
//!
//! A GoPro HERO3 file looks like `ftyp | moov | free | mdat`: the index (moov)
//! sits at the front, so when the header clusters survive we know the exact
//! offset and size of every audio and video sample in the file.

use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Video,
    Audio,
    Other,
}

#[derive(Clone, Debug, Default)]
pub struct Track {
    pub handler: [u8; 4],
    pub codec: [u8; 4],
    pub width: u32,
    pub height: u32,
    pub timescale: u32,
    pub duration: u64,
    pub nal_length_size: usize,
    pub sizes: Vec<u32>,
    pub chunk_offsets: Vec<u64>,
    pub stsc: Vec<(u32, u32, u32)>,
    /// 0-based indexes of key frames (None: every sample is one)
    pub sync: Option<HashSet<u32>>,
    pub sps: Vec<Vec<u8>>,
    pub pps: Vec<Vec<u8>>,
}

impl Track {
    pub fn kind(&self) -> Kind {
        if &self.handler == b"vide" && (&self.codec == b"avc1" || &self.codec == b"avc3") {
            Kind::Video
        } else if &self.handler == b"soun" {
            Kind::Audio
        } else {
            Kind::Other
        }
    }

    /// Absolute file offset of every sample (stsc + stco + stsz).
    pub fn sample_offsets(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.sizes.len());
        let n_chunks = self.chunk_offsets.len() as u32;
        let mut sample = 0usize;
        for (i, &(first, per_chunk, _)) in self.stsc.iter().enumerate() {
            let last = if i + 1 < self.stsc.len() { self.stsc[i + 1].0.saturating_sub(1) } else { n_chunks };
            let mut chunk = first;
            while chunk <= last {
                if chunk == 0 || chunk > n_chunks {
                    break;
                }
                let mut pos = self.chunk_offsets[(chunk - 1) as usize];
                for _ in 0..per_chunk {
                    if sample >= self.sizes.len() {
                        return out;
                    }
                    out.push(pos);
                    pos += self.sizes[sample] as u64;
                    sample += 1;
                }
                chunk += 1;
            }
        }
        out
    }
}

#[derive(Clone, Debug, Default)]
pub struct Movie {
    /// seconds since 1904-01-01
    pub creation_time: u64,
    pub timescale: u32,
    pub duration: u64,
    pub tracks: Vec<Track>,
    /// file offset where the moov box ends
    pub moov_end: u64,
    /// file offset of the first mdat payload byte
    pub mdat_start: u64,
    /// file offset just past the mdat payload (0 if unknown)
    pub mdat_end: u64,
}

impl Movie {
    pub fn video(&self) -> Option<&Track> {
        self.tracks.iter().find(|t| t.kind() == Kind::Video)
    }

    /// All samples sorted by file offset: (offset, size, kind).
    pub fn samples(&self) -> Vec<(u64, u32, Kind)> {
        let mut out = Vec::new();
        for t in &self.tracks {
            let k = t.kind();
            for (o, &s) in t.sample_offsets().iter().zip(&t.sizes) {
                if s > 0 {
                    out.push((*o, s, k));
                }
            }
        }
        out.sort_by_key(|x| x.0);
        out
    }

    pub fn file_size(&self) -> u64 {
        let end = self.samples().iter().map(|(o, s, _)| o + *s as u64).max().unwrap_or(self.mdat_start);
        if self.mdat_end >= end && self.mdat_end > 0 && self.mdat_end - end < 64 * 1024 * 1024 {
            return self.mdat_end;
        }
        end
    }

    pub fn duration_s(&self) -> f64 {
        if self.timescale == 0 {
            0.0
        } else {
            self.duration as f64 / self.timescale as f64
        }
    }
}

type R<T> = Result<T, String>;

fn u8_at(d: &[u8], p: usize) -> R<u8> {
    d.get(p).copied().ok_or_else(|| "truncated box".to_string())
}

fn u16_at(d: &[u8], p: usize) -> R<u16> {
    d.get(p..p + 2).map(|b| u16::from_be_bytes([b[0], b[1]])).ok_or_else(|| "truncated box".into())
}

fn u32_at(d: &[u8], p: usize) -> R<u32> {
    d.get(p..p + 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| "truncated box".into())
}

fn u64_at(d: &[u8], p: usize) -> R<u64> {
    Ok(((u32_at(d, p)? as u64) << 32) | u32_at(d, p + 4)? as u64)
}

fn fourcc(d: &[u8], p: usize) -> R<[u8; 4]> {
    let b = d.get(p..p + 4).ok_or("truncated box")?;
    Ok([b[0], b[1], b[2], b[3]])
}

/// Child boxes of d[start..end]: (type, payload start, box end).
fn boxes(d: &[u8], start: usize, end: usize) -> R<Vec<([u8; 4], usize, usize)>> {
    let mut out = Vec::new();
    let mut pos = start;
    while pos + 8 <= end {
        let mut size = u32_at(d, pos)? as u64;
        let typ = fourcc(d, pos + 4)?;
        let mut hdr = 8;
        if size == 1 {
            size = u64_at(d, pos + 8)?;
            hdr = 16;
        } else if size == 0 {
            size = (end - pos) as u64;
        }
        if size < hdr as u64 || pos as u64 + size > end as u64 {
            return Err(format!("bad box {} at {}", String::from_utf8_lossy(&typ), pos));
        }
        out.push((typ, pos + hdr, pos + size as usize));
        pos += size as usize;
    }
    Ok(out)
}

fn parse_avcc(b: &[u8], t: &mut Track) {
    // b starts at the avcC payload: version, profile, compat, level, length size, #sps
    let mut p = 5;
    for which in 0..2 {
        let Some(&cnt) = b.get(p) else { return };
        let n = if which == 0 { cnt & 0x1F } else { cnt };
        p += 1;
        for _ in 0..n {
            let Ok(len) = u16_at(b, p) else { return };
            let Some(x) = b.get(p + 2..p + 2 + len as usize) else { return };
            if which == 0 { t.sps.push(x.to_vec()) } else { t.pps.push(x.to_vec()) }
            p += 2 + len as usize;
        }
    }
}

const CONTAINERS: [&[u8; 4]; 8] = [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta"];

fn parse_trak(d: &[u8], start: usize, end: usize) -> R<Track> {
    let mut t = Track { nal_length_size: 4, ..Default::default() };
    let mut stack = vec![(start, end)];
    while let Some((s, e)) = stack.pop() {
        for (typ, ps, pe) in boxes(d, s, e)? {
            match &typ {
                x if CONTAINERS.contains(&x) => stack.push((ps, pe)),
                b"tkhd" => {
                    t.width = u32_at(d, pe - 8)? >> 16;
                    t.height = u32_at(d, pe - 4)? >> 16;
                }
                b"mdhd" => {
                    if u8_at(d, ps)? == 1 {
                        t.timescale = u32_at(d, ps + 20)?;
                        t.duration = u64_at(d, ps + 24)?;
                    } else {
                        t.timescale = u32_at(d, ps + 12)?;
                        t.duration = u32_at(d, ps + 16)? as u64;
                    }
                }
                b"hdlr" => t.handler = fourcc(d, ps + 8)?,
                b"stsd" => {
                    if u32_at(d, ps + 4)? > 0 {
                        let esize = u32_at(d, ps + 8)? as usize;
                        t.codec = fourcc(d, ps + 12)?;
                        let entry = d.get(ps + 8..(ps + 8 + esize).min(pe)).unwrap_or(&[]);
                        if let Some(i) = entry.windows(4).position(|w| w == b"avcC") {
                            if let Some(&b) = entry.get(i + 8) {
                                t.nal_length_size = (b & 3) as usize + 1;
                                parse_avcc(&entry[i + 4..], &mut t);
                            }
                        }
                    }
                }
                b"stsz" => {
                    let fixed = u32_at(d, ps + 4)?;
                    let count = u32_at(d, ps + 8)? as usize;
                    if fixed != 0 {
                        t.sizes = vec![fixed; count];
                    } else {
                        t.sizes = (0..count).map(|i| u32_at(d, ps + 12 + 4 * i)).collect::<R<_>>()?;
                    }
                }
                b"stsc" => {
                    let count = u32_at(d, ps + 4)? as usize;
                    t.stsc = (0..count)
                        .map(|i| {
                            let p = ps + 8 + 12 * i;
                            Ok((u32_at(d, p)?, u32_at(d, p + 4)?, u32_at(d, p + 8)?))
                        })
                        .collect::<R<_>>()?;
                }
                b"stss" => {
                    let count = u32_at(d, ps + 4)? as usize;
                    let mut set = HashSet::with_capacity(count);
                    for i in 0..count {
                        set.insert(u32_at(d, ps + 8 + 4 * i)?.saturating_sub(1));
                    }
                    t.sync = Some(set);
                }
                b"stco" => {
                    let count = u32_at(d, ps + 4)? as usize;
                    t.chunk_offsets = (0..count).map(|i| u32_at(d, ps + 8 + 4 * i).map(|v| v as u64)).collect::<R<_>>()?;
                }
                b"co64" => {
                    let count = u32_at(d, ps + 4)? as usize;
                    t.chunk_offsets = (0..count).map(|i| u64_at(d, ps + 8 + 8 * i)).collect::<R<_>>()?;
                }
                _ => {}
            }
        }
    }
    Ok(t)
}

/// Parse the front of an MP4 whose bytes are returned by `read(pos, n)`.
///
/// Walks the top-level boxes until the first `mdat` and parses `moov`.
pub fn parse_header(read: &dyn Fn(u64, usize) -> Vec<u8>) -> R<Movie> {
    const MAX_HEADER: u64 = 64 * 1024 * 1024;
    let head = read(0, 16);
    if head.len() < 16 || &head[4..8] != b"ftyp" {
        return Err("no ftyp".into());
    }
    let mut m = Movie::default();
    let mut pos = 0u64;
    let mut moov: Option<Vec<u8>> = None;
    while pos < MAX_HEADER {
        let h = read(pos, 16);
        if h.len() < 16 {
            return Err("truncated header".into());
        }
        let mut size = u32_at(&h, 0)? as u64;
        let typ = fourcc(&h, 4)?;
        let mut hdr = 8;
        if size == 1 {
            size = u64_at(&h, 8)?;
            hdr = 16;
        }
        if !typ.iter().all(|&c| (32..127).contains(&c) || c == 0xA9) {
            return Err(format!("garbage box type at {pos}"));
        }
        if &typ == b"mdat" {
            m.mdat_start = pos + hdr;
            m.mdat_end = if size >= hdr { pos + size } else { 0 };
            break;
        }
        if size < hdr {
            return Err(format!("bad top-level box size at {pos}"));
        }
        if &typ == b"moov" {
            if size > MAX_HEADER {
                return Err("moov too large".into());
            }
            moov = Some(read(pos, size as usize));
            m.moov_end = pos + size;
        }
        pos += size;
    }
    let moov = moov.ok_or("moov not found before mdat (moov-at-end files are not supported)")?;
    if m.mdat_start == 0 {
        return Err("mdat not found".into());
    }
    for (typ, ps, pe) in boxes(&moov, 8, moov.len())? {
        match &typ {
            b"mvhd" => {
                if u8_at(&moov, ps)? == 1 {
                    m.creation_time = u64_at(&moov, ps + 4)?;
                    m.timescale = u32_at(&moov, ps + 20)?;
                    m.duration = u64_at(&moov, ps + 24)?;
                } else {
                    m.creation_time = u32_at(&moov, ps + 4)? as u64;
                    m.timescale = u32_at(&moov, ps + 12)?;
                    m.duration = u32_at(&moov, ps + 16)? as u64;
                }
            }
            b"trak" => m.tracks.push(parse_trak(&moov, ps, pe)?),
            _ => {}
        }
    }
    if !m.tracks.iter().any(|t| !t.sizes.is_empty()) {
        return Err("no samples in moov".into());
    }
    for t in &m.tracks {
        if t.sample_offsets().len() != t.sizes.len() {
            return Err("inconsistent sample tables".into());
        }
    }
    Ok(m)
}
