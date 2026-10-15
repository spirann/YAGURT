//! Recover JPEG photos (and THM thumbnails) from a raw card image.
//!
//! A photo is written in one go, so its clusters follow each other on the
//! card except where the allocator skipped clusters still used by other
//! files. We follow a photo from its first cluster (the one starting with
//! FF D8 FF) and check every next cluster against the JPEG syntax: marker
//! segments must have sane lengths, and in the compressed data every FF byte
//! must be followed by 00, a restart marker in sequence (RST0..RST7) or the
//! end of image (FF D9). Video data, random data and empty clusters fail that
//! check within a few hundred bytes.
//!
//! Pass 1 takes each photo's clusters strictly in a row until the check fails
//! or the photo ends. Pass 2 continues the photos that did not end, skipping
//! clusters that videos or other photos claimed in pass 1, and looks a little
//! further ahead for the next cluster that passes the check. With a checker
//! (ffmpeg), a photo only ends where it decodes without errors: FF D9 also
//! turns up by chance in the wrong cluster.

use std::collections::HashSet;

use crate::decode::JpegChecker;
use crate::image::{Image, FLAG_HEADER, FLAG_SIGNATURE, FLAG_ZERO};
use crate::rebuild::{par_map, Candidate};

/// Free clusters to try before giving up on the next piece.
pub const WINDOW: usize = 2048;
/// No camera photo is bigger than this.
pub const MAX_SIZE: u64 = 64 << 20;
/// Never inside compressed JPEG data.
const ZERO_RUN: usize = 64;

/// Markers followed by a length field.
fn is_segment(m: u8) -> bool {
    (0xC0..=0xCF).contains(&m) || (0xDA..=0xDF).contains(&m) || (0xE0..=0xFE).contains(&m)
}

/// What may come between scans (progressive JPEG).
fn between_scans(m: u8) -> bool {
    matches!(m, 0xC4 | 0xDA | 0xDB | 0xDC | 0xDD)
}

/// The longest valid length of a segment.
fn max_len(m: u8) -> usize {
    match m {
        0xC4 => 4 * (17 + 256) + 2,
        0xDA => 2 + 1 + 4 * 2 + 3,
        0xDB => 4 * 129 + 2,
        0xDC | 0xDD => 4,
        0xC0..=0xCF => 2 + 6 + 3 * 4, // SOFn: at most 4 components
        _ => 0xFFFF,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Marker,
    Code,
    Len,
    Skip,
    Entropy,
    EntropyFf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feed {
    Ok,
    /// the image ends after this many bytes of the buffer
    End(usize),
    /// first byte that cannot be part of this JPEG
    Bad(usize),
}

/// Resumable JPEG syntax checker, fed one cluster at a time.
#[derive(Clone, Debug)]
pub struct Parser {
    mode: Mode,
    marker: u8,
    lenbuf: Vec<u8>,
    remaining: usize,
    payload: Vec<u8>,
    restart: u16,
    next_rst: u8,
    scans: u32,
    /// FF pairs checked in the last fed buffer
    pub checks: usize,
}

impl Default for Parser {
    fn default() -> Self {
        Parser {
            mode: Mode::Marker,
            marker: 0,
            lenbuf: Vec::new(),
            remaining: 0,
            payload: Vec::new(),
            restart: 0,
            next_rst: 0,
            scans: 0,
            checks: 0,
        }
    }
}

impl Parser {
    /// Check `buf`. The parser state is only meaningful after [`Feed::Ok`].
    pub fn feed(&mut self, buf: &[u8]) -> Feed {
        let n = buf.len();
        let mut i = 0;
        self.checks = 0;
        while i < n {
            match self.mode {
                Mode::Entropy => {
                    let j = buf[i..].iter().position(|&b| b == 0xFF).map(|j| i + j);
                    let seg = &buf[i..j.unwrap_or(n)];
                    if has_zero_run(seg) {
                        return Feed::Bad(i);
                    }
                    match j {
                        None => return Feed::Ok,
                        Some(j) => {
                            i = j + 1;
                            self.mode = Mode::EntropyFf;
                        }
                    }
                }
                Mode::EntropyFf => {
                    let b = buf[i];
                    i += 1;
                    self.checks += 1;
                    match b {
                        0x00 => self.mode = Mode::Entropy,
                        0xFF => {} // fill byte
                        0xD0..=0xD7 => {
                            if self.restart == 0 || b - 0xD0 != self.next_rst {
                                return Feed::Bad(i - 1);
                            }
                            self.next_rst = (self.next_rst + 1) & 7;
                            self.mode = Mode::Entropy;
                        }
                        0xD9 => return Feed::End(i),
                        m if between_scans(m) => {
                            self.marker = m;
                            self.mode = Mode::Len;
                            self.lenbuf.clear();
                        }
                        _ => return Feed::Bad(i - 1),
                    }
                }
                Mode::Marker => {
                    if buf[i] != 0xFF {
                        return Feed::Bad(i);
                    }
                    i += 1;
                    self.mode = Mode::Code;
                }
                Mode::Code => {
                    let b = buf[i];
                    i += 1;
                    if b == 0xFF {
                        continue;
                    }
                    let allowed = if self.scans > 0 { between_scans(b) } else { is_segment(b) };
                    if b == 0xD8 && self.marker == 0 {
                        self.marker = b;
                        self.mode = Mode::Marker;
                    } else if b == 0xD9 && self.scans > 0 {
                        return Feed::End(i);
                    } else if self.marker != 0 && allowed {
                        self.marker = b;
                        self.mode = Mode::Len;
                        self.lenbuf.clear();
                    } else {
                        return Feed::Bad(i - 1);
                    }
                }
                Mode::Len => {
                    let take = (2 - self.lenbuf.len()).min(n - i);
                    self.lenbuf.extend_from_slice(&buf[i..i + take]);
                    i += take;
                    if self.lenbuf.len() == 2 {
                        let ln = u16::from_be_bytes([self.lenbuf[0], self.lenbuf[1]]) as usize;
                        if ln < 2
                            || ln > max_len(self.marker)
                            || (self.marker == 0xDD && ln != 4)
                            || (self.marker == 0xDA && ln < 8)
                        {
                            return Feed::Bad(i - 1);
                        }
                        self.remaining = ln - 2;
                        self.payload.clear();
                        self.mode = Mode::Skip;
                        if self.remaining == 0 {
                            self.segment_done();
                        }
                    }
                }
                Mode::Skip => {
                    let take = self.remaining.min(n - i);
                    if self.marker == 0xDD {
                        self.payload.extend_from_slice(&buf[i..i + take]);
                    }
                    i += take;
                    self.remaining -= take;
                    if self.remaining == 0 {
                        self.segment_done();
                    }
                }
            }
        }
        Feed::Ok
    }

    fn segment_done(&mut self) {
        if self.marker == 0xDD {
            self.restart = u16::from_be_bytes([self.payload[0], self.payload[1]]);
        }
        if self.marker == 0xDA {
            self.scans += 1;
            self.next_rst = 0;
            self.mode = Mode::Entropy;
        } else {
            self.mode = Mode::Marker;
        }
    }
}

fn has_zero_run(b: &[u8]) -> bool {
    let mut run = 0;
    for &x in b {
        run = if x == 0 { run + 1 } else { 0 };
        if run >= ZERO_RUN {
            return true;
        }
    }
    false
}

/// Does this cluster look like the middle of compressed JPEG data?
/// (Every FF followed by 00, a restart marker, FF or the end of image.)
pub fn looks_like_jpeg_data(cl: &[u8]) -> bool {
    let mut seen_ff = false;
    let mut run = 0;
    for (i, &b) in cl.iter().enumerate() {
        if b == 0xFF {
            seen_ff = true;
            if let Some(&nx) = cl.get(i + 1) {
                if !(nx == 0x00 || (0xD0..=0xD9).contains(&nx) || nx == 0xFF) {
                    return false;
                }
            }
        }
        run = if b == 0 { run + 1 } else { 0 };
        if run >= ZERO_RUN {
            return false;
        }
    }
    seen_ff
}

/// Width, height, date (`YYYYmmdd_HHMMSS`) and camera make from a JPEG's headers.
pub fn jpeg_info(data: &[u8]) -> (u32, u32, Option<String>, Option<String>) {
    let (mut w, mut h) = (0, 0);
    let (mut date, mut make) = (None, None);
    let mut p = 2;
    while p + 4 <= data.len() && data[p] == 0xFF {
        let mk = data[p + 1];
        if mk == 0xFF {
            p += 1;
            continue;
        }
        let ln = u16::from_be_bytes([data[p + 2], data[p + 3]]) as usize;
        let seg = &data[(p + 4).min(data.len())..(p + 2 + ln).min(data.len())];
        if mk == 0xE1 && seg.starts_with(b"Exif\0\0") && date.is_none() {
            (date, make) = exif(&seg[6..]);
        } else if matches!(mk, 0xC0..=0xC2) && seg.len() >= 5 {
            h = u16::from_be_bytes([seg[1], seg[2]]) as u32;
            w = u16::from_be_bytes([seg[3], seg[4]]) as u32;
        }
        if mk == 0xDA {
            break;
        }
        p += 2 + ln;
    }
    if w == 0 {
        // headers we could not walk: look for a plausible SOF0
        for i in 0..data.len().saturating_sub(12) {
            let s = &data[i + 2..i + 12];
            if data[i] == 0xFF
                && data[i + 1] == 0xC0
                && s[2] == 8
                && matches!(s[7], 1 | 3)
                && u16::from_be_bytes([s[0], s[1]]) as usize == 8 + 3 * s[7] as usize
            {
                h = u16::from_be_bytes([s[3], s[4]]) as u32;
                w = u16::from_be_bytes([s[5], s[6]]) as u32;
                break;
            }
        }
    }
    (w, h, date, make)
}

/// DateTimeOriginal (or DateTime) and Make from a TIFF/EXIF block.
fn exif(t: &[u8]) -> (Option<String>, Option<String>) {
    if t.len() < 8 || !(t.starts_with(b"II") || t.starts_with(b"MM")) {
        return (None, None);
    }
    let le = t[0] == b'I';
    let u16_at = |p: usize| t.get(p..p + 2).map(|b| if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) });
    let u32_at = |p: usize| {
        t.get(p..p + 4).map(|b| {
            let a = [b[0], b[1], b[2], b[3]];
            if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) }
        })
    };
    // (tag -> ascii value, tag -> long value)
    let ifd = |off: usize| {
        let mut text = Vec::new();
        let mut long = Vec::new();
        let Some(cnt) = u16_at(off) else { return (text, long) };
        for k in 0..cnt as usize {
            let q = off + 2 + 12 * k;
            let (Some(tag), Some(typ), Some(num)) = (u16_at(q), u16_at(q + 2), u32_at(q + 4)) else { break };
            if typ == 2 {
                let vo = if num <= 4 { Some(q + 8) } else { u32_at(q + 8).map(|v| v as usize) };
                if let Some(s) = vo.and_then(|vo| t.get(vo..vo + num as usize)) {
                    let s = s.split(|&b| b == 0).next().unwrap_or(&[]);
                    text.push((tag, String::from_utf8_lossy(s).to_string()));
                }
            } else if typ == 4 {
                if let Some(v) = u32_at(q + 8) {
                    long.push((tag, v));
                }
            }
        }
        (text, long)
    };
    let Some(first) = u32_at(4) else { return (None, None) };
    let (text0, long0) = ifd(first as usize);
    let sub = long0.iter().find(|(tag, _)| *tag == 0x8769).map(|&(_, o)| ifd(o as usize).0).unwrap_or_default();
    let find = |v: &[(u16, String)], tag: u16| v.iter().find(|(t, _)| *t == tag).map(|(_, s)| s.clone());
    let raw = find(&sub, 0x9003).or_else(|| find(&text0, 0x0132));
    let date = raw.and_then(|r| {
        let b = r.as_bytes();
        (b.len() >= 19 && b[..4].iter().all(u8::is_ascii_digit) && b[4] == b':')
            .then(|| format!("{}{}{}_{}{}{}", &r[0..4], &r[5..7], &r[8..10], &r[11..13], &r[14..16], &r[17..19]))
    });
    let make = find(&text0, 0x010F).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    (date, make)
}

/// Cluster state while walking photos.
const FREE: u8 = 0;
const TAKEN: u8 = 1;

pub struct Photo {
    pub start: usize,
    pub offset: u64,
    pub clusters: Vec<usize>,
    /// bytes used in the last cluster
    pub tail: usize,
    parser: Parser,
    pub complete: bool,
    pub skips: usize,
    /// clusters whose end of image did not decode
    rejected: HashSet<usize>,
    before_last: Option<(Parser, usize, usize)>,
    /// decoder errors in the final file, when checked
    pub damage: Option<u64>,
    pub width: u32,
    pub height: u32,
    pub date: Option<String>,
    pub make: Option<String>,
}

struct Choice {
    q: usize,
    trial: Parser,
    res: Feed,
    tried: usize,
}

impl Photo {
    pub fn new(img: &Image, k: usize) -> Photo {
        Photo {
            start: k,
            offset: img.cluster_pos(k),
            clusters: Vec::new(),
            tail: 0,
            parser: Parser::default(),
            complete: false,
            skips: 0,
            rejected: HashSet::new(),
            before_last: None,
            damage: None,
            width: 0,
            height: 0,
            date: None,
            make: None,
        }
    }

    fn cluster(img: &Image, q: usize) -> Vec<u8> {
        img.read(img.cluster_pos(q), img.cluster_size as usize)
    }

    /// Follow the photo from where it stopped, through clusters that are FREE
    /// in `state` (claiming them as TAKEN). `stop` ends the walk at clusters
    /// flagged as the start of another file. With `skip`, free clusters that
    /// fail the check are stepped over, up to WINDOW of them; there an end of
    /// image found in the first bytes of a cluster is weak evidence, so a
    /// cluster that passes in full is preferred.
    fn walk(&mut self, img: &Image, state: &mut [u8], skip: bool, stop: Option<&[u8]>, checker: Option<&dyn JpegChecker>) {
        let c = img.cluster_size;
        let n_cl = img.n_clusters();
        if self.clusters.is_empty() {
            let res = self.parser.feed(&Self::cluster(img, self.start));
            if let Feed::Bad(_) = res {
                return;
            }
            self.take(self.start, res, state);
        }
        while !self.complete && (self.clusters.len() as u64) * c < MAX_SIZE {
            let mut q = *self.clusters.last().unwrap();
            let mut tried = 0;
            let mut choice: Option<Choice> = None;
            let mut fallback: Option<Choice> = None;
            for _ in 0..n_cl.saturating_sub(1) {
                q = (q + 1) % n_cl;
                if stop.is_some_and(|f| f[q] & FLAG_SIGNATURE != 0) {
                    break;
                }
                if state[q] != FREE || self.rejected.contains(&q) {
                    continue;
                }
                let mut trial = self.parser.clone();
                let res = trial.feed(&Self::cluster(img, q));
                match res {
                    Feed::Ok => choice = Some(Choice { q, trial, res, tried }),
                    Feed::End(n) if checker.is_some() => {
                        if self.verified(img, checker, Some((q, n))) {
                            choice = Some(Choice { q, trial, res, tried });
                        } else {
                            self.rejected.insert(q);
                            if !skip || self.rejected.len() > 16 {
                                return;
                            }
                        }
                    }
                    Feed::End(_) => {
                        if skip && trial.checks < 4 {
                            if fallback.is_none() {
                                fallback = Some(Choice { q, trial, res, tried });
                            }
                        } else {
                            choice = Some(Choice { q, trial, res, tried });
                        }
                    }
                    Feed::Bad(_) => {}
                }
                if choice.is_some() {
                    break;
                }
                tried += 1;
                if !skip || tried >= WINDOW {
                    break;
                }
            }
            let Some(ch) = choice.or(fallback) else { return };
            self.skips += (ch.tried > 0) as usize;
            self.before_last = Some((std::mem::replace(&mut self.parser, ch.trial), self.tail, self.skips - (ch.tried > 0) as usize));
            self.take(ch.q, ch.res, state);
        }
    }

    fn take(&mut self, q: usize, res: Feed, state: &mut [u8]) {
        self.clusters.push(q);
        state[q] = TAKEN;
        self.tail = usize::MAX;
        if let Feed::End(n) = res {
            self.complete = true;
            self.tail = n;
        }
    }

    /// Forget the last cluster (its end of image did not decode).
    fn undo_last(&mut self) -> Option<usize> {
        let (parser, tail, skips) = self.before_last.take()?;
        let q = self.clusters.pop()?;
        self.rejected.insert(q);
        self.parser = parser;
        self.tail = tail;
        self.skips = skips;
        self.complete = false;
        Some(q)
    }

    /// Does the photo, ended in cluster q after n bytes, decode cleanly?
    fn verified(&mut self, img: &Image, checker: Option<&dyn JpegChecker>, end: Option<(usize, usize)>) -> bool {
        let Some(ck) = checker else { return true };
        let mut cl = self.clusters.clone();
        let mut tail = self.tail;
        if let Some((q, n)) = end {
            cl.push(q);
            tail = n;
        }
        let d = ck.jpeg_damage(&assemble(img, &cl, tail));
        self.damage = Some(d);
        d == 0
    }

    /// The recovered file (an end of image is added to a photo cut short, so
    /// viewers show the part we have).
    pub fn data(&self, img: &Image) -> Vec<u8> {
        let mut out = assemble(img, &self.clusters, self.tail);
        if !self.complete {
            out.extend_from_slice(&[0xFF, 0xD9]);
        }
        out
    }

    pub fn size(&self, img: &Image) -> u64 {
        let c = img.cluster_size as usize;
        match self.clusters.len() {
            0 => 0,
            n => ((n - 1) * c + self.tail.min(c)) as u64,
        }
    }

    fn describe(&mut self, img: &Image) {
        // camera headers can hold large previews, so parse the whole file
        (self.width, self.height, self.date, self.make) = jpeg_info(&self.data(img));
    }

    pub fn is_thumbnail(&self) -> bool {
        if self.width > 0 && self.height > 0 {
            return self.width * self.height <= 320 * 240;
        }
        self.clusters.len() <= 1
    }

    pub fn name(&self) -> String {
        let prefix = if self.make.as_deref().unwrap_or("").to_lowercase().starts_with("gopro") { "GOPRO" } else { "JPEG" };
        let stamp = self.date.as_deref().unwrap_or("unknown_date");
        format!("{prefix}_{stamp}_{}x{}_at_0x{:x}.JPG", self.width, self.height, self.offset)
    }

    /// Runs of [logical_start, physical_start, length].
    pub fn cluster_map(&self) -> Vec<(usize, usize, usize)> {
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        for (i, &q) in self.clusters.iter().enumerate() {
            match runs.last_mut() {
                Some(r) if r.1 + r.2 == q => r.2 += 1,
                _ => runs.push((i, q, 1)),
            }
        }
        runs
    }
}

fn assemble(img: &Image, clusters: &[usize], tail: usize) -> Vec<u8> {
    let c = img.cluster_size as usize;
    let mut out = Vec::with_capacity(clusters.len() * c);
    for (i, &q) in clusters.iter().enumerate() {
        let take = if i + 1 == clusters.len() { tail.min(c) } else { c };
        out.extend_from_slice(&img.read(img.cluster_pos(q), take));
    }
    out
}

/// Find and rebuild every JPEG on the card. The clusters of `videos` are
/// never used for photos. Returns the photos that have at least their first
/// cluster.
pub fn recover_photos(
    img: &Image,
    flags: &[u8],
    videos: &[Candidate],
    checker: Option<&dyn JpegChecker>,
    log: &mut dyn FnMut(&str),
) -> Vec<Photo> {
    let n_cl = img.n_clusters();
    let mut state: Vec<u8> = (0..n_cl)
        .map(|k| (flags[k] & (FLAG_ZERO | FLAG_HEADER | FLAG_SIGNATURE) != 0 || k < img.first_cluster) as u8)
        .collect();
    for v in videos.iter().filter(|v| v.ok()) {
        for &q in v.mapping.iter().flatten() {
            state[q] = TAKEN;
        }
    }
    let mut photos: Vec<Photo> = (0..n_cl)
        .filter(|&k| flags[k] & FLAG_SIGNATURE != 0 && flags[k] & FLAG_HEADER == 0)
        .filter(|&k| img.read(img.cluster_pos(k), 3) == [0xFF, 0xD8, 0xFF])
        .map(|k| Photo::new(img, k))
        .collect();

    // pass 1: strictly consecutive free clusters, up to the next file start
    for ph in photos.iter_mut() {
        ph.walk(img, &mut state, false, Some(flags), None);
    }
    photos.retain(|ph| !ph.clusters.is_empty());
    if let Some(ck) = checker {
        // an end of image that does not decode was found by chance
        let ended: Vec<usize> = (0..photos.len()).filter(|&i| photos[i].complete).collect();
        let damage = par_map(ended.iter().map(|&i| &photos[i]).collect(), |ph: &Photo| ck.jpeg_damage(&ph.data(img)));
        for (&i, d) in ended.iter().zip(damage) {
            let ph = &mut photos[i];
            ph.damage = Some(d);
            if d > 0 && ph.clusters.len() > 1 {
                if let Some(q) = ph.undo_last() {
                    state[q] = FREE;
                }
            }
        }
    }
    // pass 2: continue unfinished photos past clusters claimed by others
    for ph in photos.iter_mut().filter(|ph| !ph.complete) {
        ph.walk(img, &mut state, true, None, checker);
    }
    for ph in photos.iter_mut() {
        ph.describe(img);
    }
    let done = photos.iter().filter(|p| p.complete).count();
    log(&format!(
        "  {} JPEG starts, {} complete photos, {} cut short ({} needed skipping)",
        photos.len(),
        done,
        photos.len() - done,
        photos.iter().filter(|p| p.skips > 0).count()
    ));
    photos
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_jpeg() -> Vec<u8> {
        // SOI, DRI(2), SOF0 8x8 1 comp, SOS 1 comp, data with stuffing and RST0, EOI
        let mut d = vec![0xFF, 0xD8, 0xFF, 0xDD, 0, 4, 0, 2];
        d.extend_from_slice(&[0xFF, 0xC0, 0, 11, 8, 0, 8, 0, 8, 1, 1, 0x11, 0]);
        d.extend_from_slice(&[0xFF, 0xDA, 0, 8, 1, 1, 0, 0, 63, 0]);
        d.extend_from_slice(&[0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD0, 0x56, 0xFF, 0xD9]);
        d
    }

    #[test]
    fn parser_accepts_and_ends() {
        let d = tiny_jpeg();
        let mut p = Parser::default();
        assert_eq!(p.feed(&d), Feed::End(d.len()));
        // split anywhere: same result
        for cut in 1..d.len() {
            let mut p = Parser::default();
            assert_eq!(p.feed(&d[..cut]), Feed::Ok, "cut {cut}");
            assert_eq!(p.feed(&d[cut..]), Feed::End(d.len() - cut), "cut {cut}");
        }
    }

    #[test]
    fn parser_rejects() {
        let mut d = tiny_jpeg();
        let n = d.len();
        d[n - 4] = 0xD3; // restart marker out of sequence
        assert!(matches!(Parser::default().feed(&d), Feed::Bad(_)));
        let mut d = tiny_jpeg();
        d[n - 7] = 0x47; // FF 47 is not allowed in entropy data
        assert!(matches!(Parser::default().feed(&d), Feed::Bad(_)));
        assert!(!looks_like_jpeg_data(&[1, 2, 0xFF, 0x47, 3]));
        assert!(looks_like_jpeg_data(&[1, 2, 0xFF, 0x00, 3, 0xFF, 0xD1]));
        assert!(!looks_like_jpeg_data(&[0u8; 100]));
    }

    #[test]
    fn size_past_large_app_segments() {
        // a camera preview in APP2 segments pushes SOF0 past 128 KiB
        let mut d = vec![0xFF, 0xD8];
        for _ in 0..3 {
            d.extend_from_slice(&[0xFF, 0xE2, 0xEA, 0x62]);
            d.extend(std::iter::repeat(7u8).take(60000));
        }
        d.extend_from_slice(&[0xFF, 0xC0, 0, 17, 8, 0x0B, 0xB8, 0x0F, 0xA0, 3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
        assert_eq!((jpeg_info(&d).0, jpeg_info(&d).1), (4000, 3000));
        // and when the segment walk breaks, the SOF0 is still found
        d[4] = 0x10;
        assert_eq!((jpeg_info(&d).0, jpeg_info(&d).1), (4000, 3000));
    }

    #[test]
    fn exif_date() {
        // big-endian TIFF: IFD0 {Make, ExifIFD}, ExifIFD {DateTimeOriginal}
        let mk = b"GoPro\0";
        let dt = b"2014:12:31 16:13:56\0";
        let (ifd0, sub) = (8u32, 8u32 + 2 + 24 + 4);
        let data = sub + 2 + 12 + 4;
        let mut t = b"MM\0\x2a".to_vec();
        t.extend_from_slice(&ifd0.to_be_bytes());
        t.extend_from_slice(&2u16.to_be_bytes());
        for (tag, typ, num, val) in [(0x010Fu16, 2u16, mk.len() as u32, data), (0x8769, 4, 1, sub)] {
            t.extend_from_slice(&tag.to_be_bytes());
            t.extend_from_slice(&typ.to_be_bytes());
            t.extend_from_slice(&num.to_be_bytes());
            t.extend_from_slice(&val.to_be_bytes());
        }
        t.extend_from_slice(&[0; 4]);
        t.extend_from_slice(&1u16.to_be_bytes());
        t.extend_from_slice(&0x9003u16.to_be_bytes());
        t.extend_from_slice(&2u16.to_be_bytes());
        t.extend_from_slice(&(dt.len() as u32).to_be_bytes());
        t.extend_from_slice(&(data + mk.len() as u32).to_be_bytes());
        t.extend_from_slice(&[0; 4]);
        t.extend_from_slice(mk);
        t.extend_from_slice(dt);
        assert_eq!(exif(&t), (Some("20141231_161356".into()), Some("GoPro".into())));
    }
}
