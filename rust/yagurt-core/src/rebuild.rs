//! Rebuild fragmented GoPro MP4/LRV files from a raw card image.
//!
//! A GoPro HERO3 records the high-resolution MP4 and the low-resolution LRV
//! at the same time, so on the card their clusters are interleaved, and any
//! cluster still used by an older file is skipped. Copying the data as one
//! contiguous block (what carvers and AMRT do) mixes the streams.
//!
//! The header (ftyp + moov) of each file survives, and its sample tables give
//! the exact offset and size of every frame. H.264 frames are chains of
//! length-prefixed NAL units, so for most clusters we know where NAL headers
//! must sit and what they must look like. We walk the file cluster by cluster
//! in allocation order, skip clusters that fail that test, and search the whole
//! image for the next frame when the trail is lost. A second round excludes
//! clusters other files claimed with certainty.
//!
//! Clusters in the middle of large frames carry no checkable structure. When
//! the free clusters between two checked ones are more than needed, a decoder
//! ([`FrameChecker`]) decodes the frame for candidate placements: the right one
//! decodes with no concealed macroblocks.

use crate::decode::{to_annexb, FrameChecker, START};
use crate::image::{Image, FLAG_HEADER, FLAG_SIGNATURE, FLAG_ZERO};
use crate::mp4::{self, Movie};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const ST_MISSING: u8 = 0;
pub const ST_HEADER: u8 = 1;
/// cluster content checked against expected NAL headers
pub const ST_VALID: u8 = 2;
/// no checkable structure; placed by allocation order (and the decoder)
pub const ST_INFERRED: u8 = 3;

/// how far ahead to look for the next piece before a full search
const WINDOW: usize = 4096;
/// frame starts that must line up to accept a full-image match
const MIN_STARTS: usize = 2;

pub fn nal_plausible(b: &[u8], nls: usize, remaining: u64) -> bool {
    if b.len() < nls + 1 {
        return false;
    }
    let length = b[..nls].iter().fold(0u64, |a, &x| (a << 8) | x as u64);
    if length == 0 || nls as u64 + length > remaining {
        return false;
    }
    let t = b[nls];
    if t & 0x80 != 0 {
        return false;
    }
    let typ = t & 0x1F;
    let refidc = (t >> 5) & 3;
    if !matches!(typ, 1 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12) {
        return false;
    }
    if matches!(typ, 5 | 7 | 8) && refidc == 0 {
        return false;
    }
    if matches!(typ, 6 | 9 | 10 | 11 | 12) && refidc != 0 {
        return false;
    }
    if typ == 9 && length != 2 {
        return false;
    }
    if matches!(typ, 7 | 8) && length > 1024 {
        return false;
    }
    true
}

/// (next video sample index, next NAL position inside the current one, its end)
type State = (usize, Option<u64>, u64);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    Match,
    Nothing,
    Mismatch,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub clusters: usize,
    pub header: usize,
    pub verified: usize,
    pub inferred: usize,
    pub missing: usize,
    pub skips: usize,
    pub full_searches: usize,
    pub ambiguous_runs: usize,
    pub note: Option<String>,
}

impl Stats {
    pub fn recovered_pct(&self) -> f64 {
        if self.clusters == 0 {
            0.0
        } else {
            100.0 * (self.clusters - self.missing) as f64 / self.clusters as f64
        }
    }
}

/// One MP4/LRV whose header was found at physical cluster `hcl`.
pub struct Candidate {
    pub hcl: usize,
    pub offset: u64,
    pub error: Option<String>,
    pub movie: Movie,
    pub file_size: u64,
    /// clusters in the file
    pub n: usize,
    /// header clusters (assumed contiguous)
    pub h: usize,
    nls: usize,
    hlen: usize,
    vid: Vec<(u64, u64)>,
    vid_starts: Vec<u64>,
    pub mapping: Vec<Option<usize>>,
    pub status: Vec<u8>,
    pub stats: Stats,
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + (m <= 2) as i64, m, d)
}

impl Candidate {
    pub fn new(img: &Image, hcl: usize) -> Candidate {
        let offset = img.cluster_pos(hcl);
        let read = |pos: u64, n: usize| img.read(offset + pos, n);
        let mut c = Candidate {
            hcl,
            offset,
            error: None,
            movie: Movie::default(),
            file_size: 0,
            n: 0,
            h: 0,
            nls: 4,
            hlen: 5,
            vid: vec![],
            vid_starts: vec![],
            mapping: vec![],
            status: vec![],
            stats: Stats::default(),
        };
        match mp4::parse_header(&read) {
            Ok(m) => c.movie = m,
            Err(e) => {
                c.error = Some(e);
                return c;
            }
        }
        let cs = img.cluster_size;
        c.file_size = c.movie.file_size();
        c.n = c.file_size.div_ceil(cs) as usize;
        c.h = (c.movie.moov_end.div_ceil(cs) as usize).clamp(1, c.n.max(1));
        if hcl + c.h > img.n_clusters() {
            c.error = Some("header runs past the end of the image".into());
            return c;
        }
        if let Some(v) = c.movie.video() {
            c.nls = v.nal_length_size;
            c.hlen = c.nls + 1;
            let mut vid: Vec<(u64, u64)> = v
                .sample_offsets()
                .iter()
                .zip(&v.sizes)
                .filter(|(_, &s)| s > 0)
                .map(|(&o, &s)| (o, o + s as u64))
                .collect();
            vid.sort_by_key(|x| x.0);
            c.vid_starts = vid.iter().map(|x| x.0).collect();
            c.vid = vid;
        }
        c
    }

    pub fn ok(&self) -> bool {
        self.error.is_none()
    }

    pub fn resolution(&self) -> (u32, u32) {
        self.movie.video().map_or((0, 0), |v| (v.width, v.height))
    }

    /// Recording time as "YYYYMMDD_HHMMSS" (UTC as stored by the camera).
    pub fn created(&self) -> Option<String> {
        let ct = self.movie.creation_time as i64;
        if ct == 0 {
            return None;
        }
        let unix = ct - 2_082_844_800;
        let (y, m, d) = civil_from_days(unix.div_euclid(86400));
        let s = unix.rem_euclid(86400);
        Some(format!("{y:04}{m:02}{d:02}_{:02}{:02}{:02}", s / 3600, s % 3600 / 60, s % 60))
    }

    pub fn name(&self) -> String {
        let (w, h) = self.resolution();
        let stamp = self.created().unwrap_or_else(|| "unknown_date".into());
        let ext = if w > 0 && w <= 432 { "LRV.mp4" } else { "MP4" };
        format!("GOPRO_{stamp}_{w}x{h}_at_0x{:x}.{ext}", self.offset)
    }

    fn state_at_frame(si: usize) -> State {
        (si, None, 0)
    }

    // --- the checker -------------------------------------------------------

    /// Check physical cluster `q` as logical cluster `j`. `phys(c)` gives the
    /// physical cluster already chosen for logical cluster c < j.
    fn validate(
        &self,
        img: &Image,
        j: usize,
        q: usize,
        state: State,
        phys: &dyn Fn(usize) -> Option<usize>,
    ) -> (Verdict, usize, usize, State) {
        let cs = img.cluster_size;
        let lo = j as u64 * cs;
        let hi = lo + cs;
        let (mut si, mut nal, mut end) = state;
        let (mut checked, mut starts) = (0, 0);
        let mut b = [0u8; 9];
        let hlen = self.hlen;
        loop {
            let mut fresh = false;
            if nal.is_none_or(|x| x >= end) {
                if si >= self.vid.len() {
                    break;
                }
                let (o, e) = self.vid[si];
                end = e;
                if o + hlen as u64 > hi {
                    nal = None;
                    break;
                }
                nal = Some(o);
                si += 1;
                fresh = true;
            }
            let ev = nal.unwrap();
            if ev + hlen as u64 > hi {
                break;
            }
            if ev >= lo {
                if !img.read_into(img.cluster_pos(q) + (ev - lo), &mut b[..hlen]) {
                    return (Verdict::Mismatch, checked, starts, state);
                }
            } else if !self.read_split(img, ev, j, q, phys, &mut b[..hlen]) {
                nal = None; // earlier piece unknown: give up on this frame
                continue;
            }
            if !nal_plausible(&b[..hlen], self.nls, end - ev) {
                return (Verdict::Mismatch, checked, starts, state);
            }
            checked += 1;
            // Frame starts come from the moov, so each one matching is
            // independent evidence; NAL positions inside a frame are derived
            // from the data itself.
            starts += fresh as usize;
            let len = b[..self.nls].iter().fold(0u64, |a, &x| (a << 8) | x as u64);
            nal = Some(ev + self.nls as u64 + len);
        }
        let v = if checked > 0 { Verdict::Match } else { Verdict::Nothing };
        (v, checked, starts, (si, nal, end))
    }

    fn read_split(
        &self,
        img: &Image,
        mut pos: u64,
        j: usize,
        q: usize,
        phys: &dyn Fn(usize) -> Option<usize>,
        out: &mut [u8],
    ) -> bool {
        let cs = img.cluster_size;
        let mut filled = 0;
        while filled < out.len() {
            let c = (pos / cs) as usize;
            let pc = if c == j { Some(q) } else { phys(c) };
            let Some(pc) = pc else { return false };
            let within = pos - c as u64 * cs;
            let take = (out.len() - filled).min((cs - within) as usize);
            if !img.read_into(img.cluster_pos(pc) + within, &mut out[filled..filled + take]) {
                return false;
            }
            filled += take;
            pos += take as u64;
        }
        true
    }

    // --- allocation order helpers ------------------------------------------

    fn next(n_cl: usize, p: usize, blocked: &[u8], used: &[u8], extra: &dyn Fn(usize) -> bool) -> Option<usize> {
        let mut k = p;
        for _ in 0..n_cl {
            k += 1;
            if k >= n_cl {
                k = 0;
            }
            if blocked[k] == 0 && used[k] == 0 && !extra(k) {
                return Some(k);
            }
        }
        None
    }

    fn prev(p: usize, blocked: &[u8], used: &[u8]) -> Option<usize> {
        (0..p).rev().find(|&k| blocked[k] == 0 && used[k] == 0)
    }

    // --- full-image search -------------------------------------------------

    /// First logical cluster >= j that holds the whole header of a frame start.
    fn frame_cluster(&self, cs: u64, j: usize, mut si: usize) -> Option<(usize, usize)> {
        while si < self.vid.len() {
            let off = self.vid[si].0;
            let k = (off / cs) as usize;
            if k >= j && off + self.hlen as u64 <= (k as u64 + 1) * cs {
                return Some((k, si));
            }
            si += 1;
        }
        None
    }

    /// Follow the file from candidate q for a while; count frame starts that line up.
    #[allow(clippy::too_many_arguments)]
    fn score(
        &self,
        img: &Image,
        k: usize,
        q: usize,
        si: usize,
        blocked: &[u8],
        used: &[u8],
        mapping: &[Option<usize>],
    ) -> (usize, usize, bool) {
        const MAX_CLUSTERS: usize = 256;
        const WANT_STARTS: usize = 3;
        let n_cl = img.n_clusters();
        let mut overlay: HashMap<usize, usize> = HashMap::new();
        overlay.insert(k, q);
        let mut state = Self::state_at_frame(si);
        let (mut starts, mut checked) = (0, 0);
        let mut cur = q;
        for t in 0..MAX_CLUSTERS {
            let c = k + t;
            if c >= self.n {
                break;
            }
            let res;
            {
                let ov = &overlay;
                let phys = |x: usize| ov.get(&x).copied().or_else(|| mapping.get(x).copied().flatten());
                let taken: HashSet<usize> = ov.values().copied().collect();
                let extra = |x: usize| taken.contains(&x);
                if t == 0 {
                    let r = self.validate(img, c, cur, state, &phys);
                    if r.0 == Verdict::Mismatch {
                        break;
                    }
                    res = (cur, r);
                } else {
                    let Some(mut nxt) = Self::next(n_cl, cur, blocked, used, &extra) else { break };
                    let mut r = self.validate(img, c, nxt, state, &phys);
                    if r.0 == Verdict::Mismatch {
                        // skip a few foreign clusters, like the main walk
                        let mut found = false;
                        for _ in 0..64 {
                            let Some(x) = Self::next(n_cl, nxt, blocked, used, &extra) else { break };
                            nxt = x;
                            r = self.validate(img, c, nxt, state, &phys);
                            if r.0 == Verdict::Match {
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            break;
                        }
                    }
                    res = (nxt, r);
                }
            }
            let (pc, (_v, chk, st, newstate)) = res;
            cur = pc;
            overlay.insert(c, cur);
            state = newstate;
            checked += chk;
            starts += st;
            if starts >= WANT_STARTS || state.0 >= self.vid.len() {
                break;
            }
        }
        let complete = state.0 >= self.vid.len() && starts >= 1;
        (starts, checked, complete)
    }

    /// Find, anywhere on the card, the cluster holding frame si (or a later one).
    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        img: &Image,
        j: usize,
        si: usize,
        blocked: &[u8],
        used: &[u8],
        mapping: &[Option<usize>],
        prefer: &dyn Fn(usize) -> usize,
    ) -> Option<(usize, usize, usize)> {
        let cs = img.cluster_size;
        let (k, si) = self.frame_cluster(cs, j, si)?;
        let (off, end) = self.vid[si];
        let within = off - k as u64 * cs;
        let mut best: Vec<(usize, usize)> = Vec::new();
        let mut b = [0u8; 9];
        for q in 0..img.n_clusters() {
            if blocked[q] != 0 || used[q] != 0 {
                continue;
            }
            if !img.read_into(img.cluster_pos(q) + within, &mut b[..self.hlen])
                || !nal_plausible(&b[..self.hlen], self.nls, end - off)
            {
                continue;
            }
            let (starts, checked, complete) = self.score(img, k, q, si, blocked, used, mapping);
            if starts >= MIN_STARTS || complete {
                best.push((starts * 1000 + checked, q));
            }
        }
        let top = best.iter().map(|x| x.0).max()?;
        let q = best.iter().filter(|x| x.0 == top).map(|x| x.1).min_by_key(|&q| prefer(q))?;
        Some((k, si, q))
    }

    // --- the main walk -----------------------------------------------------

    pub fn rebuild(&mut self, img: &Image, blocked: &[u8], used: &mut [u8]) {
        let n_cl = img.n_clusters();
        let (n, h) = (self.n, self.h);
        let mut mapping: Vec<Option<usize>> = vec![None; n];
        let mut status = vec![ST_MISSING; n];
        let (mut searches, mut jumps) = (0, 0);
        for i in 0..h {
            mapping[i] = Some(self.hcl + i);
            status[i] = ST_HEADER;
            used[self.hcl + i] = 1;
        }
        // frames that start inside the header clusters (mdat not cluster aligned)
        let mut state = Self::state_at_frame(0);
        for i in 0..h {
            let m = &mapping;
            let (v, _, _, st) = self.validate(img, i, m[i].unwrap(), state, &|x| m.get(x).copied().flatten());
            if v != Verdict::Mismatch {
                state = st;
            }
        }
        let mut j = h;
        let mut p = self.hcl + h - 1;
        let mut note = None;
        if j < n {
            // The data rarely follows the header on the card (the camera writes
            // the header last), so locate the first frame by a full search.
            let hcl = self.hcl;
            let res = self.search(img, j, state.0, blocked, used, &mapping, &|q| q.abs_diff(hcl));
            searches += 1;
            match res {
                None => {
                    note = Some("first frame not found".to_string());
                    j = n;
                }
                Some((k, si, q)) => {
                    let m = &mapping;
                    let (_, _, _, st) =
                        self.validate(img, k, q, Self::state_at_frame(si), &|x| m.get(x).copied().flatten());
                    state = st;
                    mapping[k] = Some(q);
                    status[k] = ST_VALID;
                    used[q] = 1;
                    // whatever precedes the first frame (audio, padding) sits just before it
                    let mut pb = q;
                    for c in (j..k).rev() {
                        let Some(x) = Self::prev(pb, blocked, used) else { break };
                        pb = x;
                        mapping[c] = Some(pb);
                        status[c] = ST_INFERRED;
                        used[pb] = 1;
                    }
                    j = k + 1;
                    p = q;
                }
            }
        }
        let none = |_: usize| false;
        while j < n {
            let Some(q0) = Self::next(n_cl, p, blocked, used, &none) else { break };
            let r = {
                let m = &mapping;
                self.validate(img, j, q0, state, &|x| m.get(x).copied().flatten())
            };
            if r.0 != Verdict::Mismatch {
                mapping[j] = Some(q0);
                status[j] = if r.0 == Verdict::Match { ST_VALID } else { ST_INFERRED };
                used[q0] = 1;
                state = r.3;
                p = q0;
                j += 1;
                continue;
            }
            // skip over clusters of the other stream / older files
            let mut q = q0;
            let mut found = None;
            for _ in 0..WINDOW {
                let Some(x) = Self::next(n_cl, q, blocked, used, &none) else { break };
                if x == q0 {
                    break;
                }
                q = x;
                let m = &mapping;
                let r = self.validate(img, j, q, state, &|x| m.get(x).copied().flatten());
                if r.0 == Verdict::Match {
                    found = Some((q, r.3));
                    break;
                }
            }
            if let Some((q, st)) = found {
                mapping[j] = Some(q);
                status[j] = ST_VALID;
                used[q] = 1;
                state = st;
                p = q;
                j += 1;
                jumps += 1;
                continue;
            }
            // lost the trail: look for the next frame anywhere on the card
            let pp = p;
            let res = self.search(img, j, state.0, blocked, used, &mapping, &|qq| (qq + n_cl - pp) % n_cl);
            searches += 1;
            let Some((k, si, q)) = res else { break };
            let m = &mapping;
            let (_, _, _, st) = self.validate(img, k, q, Self::state_at_frame(si), &|x| m.get(x).copied().flatten());
            state = st;
            mapping[k] = Some(q);
            status[k] = ST_VALID;
            used[q] = 1;
            j = k + 1;
            p = q;
        }
        self.mapping = mapping;
        self.status = status;
        self.stats = self.make_stats(searches, jumps, note);
    }

    // --- second pass: place the clusters we could not check -----------------

    /// Re-place every run of unchecked clusters between two checked ones.
    ///
    /// If exactly as many free clusters as needed sit between the run's two
    /// anchors, the placement is certain. Otherwise clusters of some other file
    /// are mixed in; with a decoder we search for the placement under which the
    /// frame decodes cleanly, else we keep the most likely layout.
    pub fn refine(&mut self, img: &Image, blocked: &[u8], soft: &[u8], checker: Option<&dyn FrameChecker>) {
        let n_cl = img.n_clusters();
        let anchors: Vec<usize> =
            (0..self.n).filter(|&i| self.status[i] == ST_HEADER || self.status[i] == ST_VALID).collect();
        let mut used = vec![0u8; n_cl];
        for &i in &anchors {
            used[self.mapping[i].unwrap()] = 1;
        }
        let mut ambiguous = 0;
        for w in anchors.windows(2) {
            let (a, b) = (w[0], w[1]);
            let n = b - a - 1;
            if n == 0 {
                continue;
            }
            let (pa, pb) = (self.mapping[a].unwrap(), self.mapping[b].unwrap());
            let hyps = Self::hypotheses(n_cl, pa, pb, n, blocked, &used, &[], soft);
            if hyps.is_empty() {
                for i in a + 1..b {
                    match self.mapping[i] {
                        Some(q) if used[q] == 0 => used[q] = 1,
                        _ => {
                            self.mapping[i] = None;
                            self.status[i] = ST_MISSING;
                        }
                    }
                }
                continue;
            }
            let mut choice = hyps[0].clone();
            if hyps.len() > 1 {
                ambiguous += 1;
                if let Some(ch) = checker {
                    if let Some(h) = self.resolve_by_decoding(img, a, b, blocked, &used, soft, ch) {
                        choice = h;
                    }
                }
            }
            for (i, &q) in choice.iter().enumerate() {
                self.mapping[a + 1 + i] = Some(q);
                self.status[a + 1 + i] = ST_INFERRED;
                used[q] = 1;
            }
        }
        let (s, j, note) = (self.stats.full_searches, self.stats.skips, self.stats.note.clone());
        self.stats = self.make_stats(s, j, note);
        self.stats.ambiguous_runs = ambiguous;
    }

    /// Plausible placements of n clusters between physical clusters pa and pb,
    /// most likely first. `extra` lists clusters to treat as used.
    #[allow(clippy::too_many_arguments)]
    fn hypotheses(
        n_cl: usize,
        pa: usize,
        pb: usize,
        n: usize,
        blocked: &[u8],
        used: &[u8],
        extra: &[usize],
        soft: &[u8],
    ) -> Vec<Vec<usize>> {
        let free = |k: usize| blocked[k] == 0 && used[k] == 0 && !extra.contains(&k);
        let dist = (pb + n_cl - pa) % n_cl;
        let mut hyps: Vec<Vec<usize>> = Vec::new();
        if dist > 0 && dist <= n + WINDOW {
            let m: Vec<usize> = (1..dist).map(|d| (pa + d) % n_cl).filter(|&k| free(k)).collect();
            if m.len() == n {
                return vec![m];
            }
            if m.len() > n {
                let extra_n = m.len() - n;
                let m2: Vec<usize> = m.iter().copied().filter(|&k| soft[k] == 0).collect();
                if m2.len() == n {
                    hyps.push(m2);
                }
                if n > 64 {
                    if hyps.is_empty() {
                        hyps.push(m[..n].to_vec());
                    }
                    return hyps;
                }
                for s in (0..=n).rev() {
                    // skip one foreign block starting after s
                    let mut h = m[..s].to_vec();
                    h.extend_from_slice(&m[s + extra_n..]);
                    hyps.push(h);
                }
                return dedupe(hyps);
            }
        }
        // the file jumps to another fragment somewhere inside the run
        let mut f = Vec::new();
        let mut k = pa;
        while f.len() < n {
            k = (k + 1) % n_cl;
            if k == pb || k == pa {
                break;
            }
            if free(k) {
                f.push(k);
            }
        }
        let mut bk = Vec::new();
        k = pb;
        while bk.len() < n {
            k = (k + n_cl - 1) % n_cl;
            if k == pa || k == pb {
                break;
            }
            if free(k) {
                bk.push(k);
            }
        }
        bk.reverse();
        for x in (0..=n).rev() {
            if x <= f.len() && n - x <= bk.len() {
                let mut h = f[..x].to_vec();
                h.extend_from_slice(&bk[bk.len() - (n - x)..]);
                let uniq: HashSet<usize> = h.iter().copied().collect();
                if uniq.len() == n {
                    hyps.push(h);
                }
                if n > 64 {
                    break;
                }
            }
        }
        dedupe(hyps)
    }

    fn read_logical(&self, img: &Image, mut pos: u64, size: u64, phys: &dyn Fn(usize) -> Option<usize>) -> Vec<u8> {
        let cs = img.cluster_size;
        let mut out = vec![0u8; size as usize];
        let mut filled = 0usize;
        while filled < out.len() {
            let c = (pos / cs) as usize;
            let within = pos - c as u64 * cs;
            let take = ((out.len() - filled) as u64).min(cs - within) as usize;
            if let Some(q) = phys(c) {
                img.src.read_at(img.cluster_pos(q) + within, &mut out[filled..filled + take]);
            }
            filled += take;
            pos += take as u64;
        }
        out
    }

    /// Index of the video frame holding most of logical cluster c.
    fn frame_for_cluster(&self, cs: u64, c: usize) -> Option<usize> {
        let (lo, hi) = (c as u64 * cs, (c as u64 + 1) * cs);
        let mut i = self.vid_starts.partition_point(|&o| o < hi) as isize - 1;
        let (mut best, mut best_len) = (None, 0u64);
        while i >= 0 {
            let (o, e) = self.vid[i as usize];
            if e <= lo {
                break;
            }
            let ov = e.min(hi) - o.max(lo);
            if ov > best_len {
                best = Some(i as usize);
                best_len = ov;
            }
            i -= 1;
        }
        if best_len > cs / 4 {
            best
        } else {
            None
        }
    }

    /// Concealed macroblocks when frame fi is decoded (up to logical byte `upto`).
    fn frame_damage(
        &self,
        img: &Image,
        fi: usize,
        upto: Option<u64>,
        phys: &dyn Fn(usize) -> Option<usize>,
        checker: &dyn FrameChecker,
    ) -> u64 {
        let (o, mut e) = self.vid[fi];
        if let Some(u) = upto {
            e = e.min(u);
        }
        let mut stream = Vec::new();
        if let Some(v) = self.movie.video() {
            for x in v.sps.iter().chain(&v.pps) {
                stream.extend_from_slice(&START);
                stream.extend_from_slice(x);
            }
        }
        let (data, _) = to_annexb(&self.read_logical(img, o, e.saturating_sub(o), phys), self.nls, true);
        stream.extend_from_slice(&data);
        checker.damage(&stream)
    }

    /// Placements of run a+1..b-1 that keep `prefix` and vary the rest.
    #[allow(clippy::too_many_arguments)]
    fn run_family(
        &self,
        n_cl: usize,
        a: usize,
        b: usize,
        prefix: &[usize],
        blocked: &[u8],
        used: &[u8],
        soft: &[u8],
    ) -> Vec<Vec<usize>> {
        let n = b - a - 1;
        let left = n - prefix.len();
        if left == 0 {
            return vec![prefix.to_vec()];
        }
        let start = prefix.last().copied().unwrap_or(self.mapping[a].unwrap());
        Self::hypotheses(n_cl, start, self.mapping[b].unwrap(), left, blocked, used, prefix, soft)
            .into_iter()
            .map(|h| {
                let mut v = prefix.to_vec();
                v.extend(h);
                v
            })
            .collect()
    }

    /// Free clusters that could hold the next of `count` clusters after `start`.
    #[allow(clippy::too_many_arguments)]
    fn next_options(
        n_cl: usize,
        start: usize,
        pb: usize,
        count: usize,
        blocked: &[u8],
        used: &[u8],
        extra: &[usize],
        k: usize,
    ) -> Vec<usize> {
        let free = |q: usize| blocked[q] == 0 && used[q] == 0 && !extra.contains(&q);
        let dist = (pb + n_cl - start) % n_cl;
        if dist > 0 && dist <= count + WINDOW {
            let m: Vec<usize> = (1..dist).map(|d| (start + d) % n_cl).filter(|&q| free(q)).collect();
            if m.len() >= count {
                return m[..m.len() - count + 1].iter().take(k).copied().collect();
            }
        }
        let mut out = Vec::new();
        let mut q = start;
        while out.len() < k {
            q = (q + 1) % n_cl;
            if q == pb || q == start {
                break;
            }
            if free(q) {
                out.push(q);
            }
        }
        let mut back = Vec::new();
        q = pb;
        while back.len() < count {
            // or jump to the fragment that ends at pb
            q = (q + n_cl - 1) % n_cl;
            if q == start {
                break;
            }
            if free(q) {
                back.push(q);
            }
        }
        if back.len() == count && !out.contains(back.last().unwrap()) {
            out.push(*back.last().unwrap());
        }
        out
    }

    /// Find the placement of run a+1..b-1 under which its frame decodes cleanly.
    ///
    /// A correct placement decodes with zero concealed macroblocks, a wrong one
    /// practically never does. Decoding the frame truncated after each cluster
    /// tells where the decoder first derails, so we grow a prefix of clusters
    /// known to be right one position at a time, keeping at each step the
    /// candidate that lets the decoder get furthest, and back up when no
    /// candidate gets past the current position.
    #[allow(clippy::too_many_arguments)]
    fn resolve_by_decoding(
        &self,
        img: &Image,
        a: usize,
        b: usize,
        blocked: &[u8],
        used: &[u8],
        soft: &[u8],
        checker: &dyn FrameChecker,
    ) -> Option<Vec<usize>> {
        const BUDGET: usize = 600;
        const K: usize = 12;
        let cs = img.cluster_size;
        let n_cl = img.n_clusters();
        let n = b - a - 1;
        let pb = self.mapping[b].unwrap();
        let mut frames: Vec<usize> = (a + 1..b).filter_map(|c| self.frame_for_cluster(cs, c)).collect();
        frames.sort();
        frames.dedup();
        if frames.is_empty() {
            return None;
        }
        let calls = AtomicUsize::new(0);
        let score = |h: &[usize], upto: Option<u64>| -> u64 {
            calls.fetch_add(1, Ordering::Relaxed);
            let overlay: HashMap<usize, usize> = h.iter().enumerate().map(|(i, &q)| (a + 1 + i, q)).collect();
            let phys = |x: usize| overlay.get(&x).copied().or_else(|| self.mapping.get(x).copied().flatten());
            let fs: &[usize] = if upto.is_none() { &frames } else { &frames[frames.len() - 1..] };
            fs.iter().map(|&fi| self.frame_damage(img, fi, upto, &phys, checker)).sum()
        };
        let derail = |h: &[usize], sc: u64, mut lo: usize| -> usize {
            let mut hi = n - 1;
            while lo < hi && calls.load(Ordering::Relaxed) < BUDGET {
                let mid = (lo + hi) / 2;
                if score(h, Some((a + 2 + mid) as u64 * cs)) <= sc {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            lo
        };

        let mut best: Option<(Vec<usize>, u64)> = None;
        let mut first = self.run_family(n_cl, a, b, &[], blocked, used, soft);
        first.truncate(BUDGET / 4); // cheap first guesses
        for (h, sc) in par_map(first, |h| {
            let s = score(&h, None);
            (h, s)
        }) {
            if sc == 0 {
                return Some(h);
            }
            if best.as_ref().is_none_or(|x| sc < x.1) {
                best = Some((h, sc));
            }
        }
        if best.is_none() || frames.len() != 1 {
            return best.map(|x| x.0);
        }

        let mut prefix: Vec<usize> = Vec::new();
        let mut tried: HashMap<usize, HashSet<usize>> = HashMap::new();
        while calls.load(Ordering::Relaxed) < BUDGET {
            let l = prefix.len();
            if l >= n {
                break;
            }
            let start = prefix.last().copied().unwrap_or(self.mapping[a].unwrap());
            let done = tried.get(&l).cloned().unwrap_or_default();
            let cands: Vec<usize> = Self::next_options(n_cl, start, pb, n - l, blocked, used, &prefix, K)
                .into_iter()
                .filter(|q| !done.contains(q))
                .collect();
            if cands.is_empty() {
                if prefix.pop().is_none() {
                    break;
                }
                continue;
            }
            // Each candidate costs a few decodes; run them side by side.
            let results: Vec<(usize, Vec<usize>, u64, usize)> =
                par_map(cands.iter().take(K).copied().collect(), |q| {
                    let mut pre = prefix.clone();
                    pre.push(q);
                    let h = self.run_family(n_cl, a, b, &pre, blocked, used, soft).into_iter().next()?;
                    let s = score(&h, None);
                    let d = if s == 0 { n } else { derail(&h, s, l) };
                    Some((q, h, s, d))
                })
                .into_iter()
                .flatten()
                .collect();
            let mut top: Option<(usize, u64, usize, Vec<usize>)> = None;
            for (q, h, sc, d) in results {
                if sc == 0 {
                    return Some(h);
                }
                if best.as_ref().is_none_or(|x| sc < x.1) {
                    best = Some((h.clone(), sc));
                }
                if top.as_ref().is_none_or(|t| d > t.0 || (d == t.0 && sc < t.1)) {
                    top = Some((d, sc, q, h));
                }
            }
            let Some((d, _sc, q, h)) = top else { break };
            tried.entry(l).or_default().insert(q);
            if d <= l {
                // nothing gets past this position: an earlier pick was wrong
                if prefix.pop().is_none() {
                    break;
                }
                continue;
            }
            prefix = h[..(l + 1).max(d - 1)].to_vec();
            for i in l + 1..prefix.len() {
                tried.remove(&i);
            }
        }
        best.map(|x| x.0)
    }

    fn make_stats(&self, searches: usize, jumps: usize, note: Option<String>) -> Stats {
        let count = |s: u8| self.status.iter().filter(|&&x| x == s).count();
        Stats {
            clusters: self.n,
            header: count(ST_HEADER),
            verified: count(ST_VALID),
            inferred: count(ST_INFERRED),
            missing: count(ST_MISSING),
            skips: jumps,
            full_searches: searches,
            ambiguous_runs: 0,
            note,
        }
    }

    pub fn strong_clusters(&self) -> Vec<usize> {
        self.mapping
            .iter()
            .zip(&self.status)
            .filter(|(_, &s)| s == ST_VALID || s == ST_HEADER)
            .filter_map(|(q, _)| *q)
            .collect()
    }

    /// Write the rebuilt file; missing clusters become zeros.
    pub fn write_to(&self, img: &Image, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let cs = img.cluster_size as usize;
        let mut buf = vec![0u8; cs];
        let mut remaining = self.file_size as usize;
        for q in &self.mapping {
            if remaining == 0 {
                break;
            }
            let take = cs.min(remaining);
            buf[..take].fill(0);
            if let Some(q) = q {
                img.src.read_at(img.cluster_pos(*q), &mut buf[..take]);
            }
            w.write_all(&buf[..take])?;
            remaining -= take;
        }
        Ok(())
    }

    /// Compact list of (logical_start, physical_start, length, status) runs.
    pub fn cluster_map(&self) -> Vec<(usize, Option<usize>, usize, u8)> {
        let mut runs: Vec<(usize, Option<usize>, usize, u8)> = Vec::new();
        for (c, (&q, &st)) in self.mapping.iter().zip(&self.status).enumerate() {
            if let Some(r) = runs.last_mut() {
                let contiguous = match (r.1, q) {
                    (None, None) => true,
                    (Some(p), Some(q)) => p + r.2 == q,
                    _ => false,
                };
                if r.3 == st && contiguous {
                    r.2 += 1;
                    continue;
                }
            }
            runs.push((c, q, 1, st));
        }
        runs
    }
}

/// Map `f` over `items` on all CPU cores, keeping the order.
fn par_map<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).max(1);
    let mut out = Vec::with_capacity(items.len());
    let mut items = items.into_iter().peekable();
    while items.peek().is_some() {
        let batch: Vec<T> = items.by_ref().take(threads).collect();
        let f = &f;
        out.extend(std::thread::scope(|sc| {
            let hs: Vec<_> = batch.into_iter().map(|x| sc.spawn(move || f(x))).collect();
            hs.into_iter().map(|h| h.join().expect("worker panicked")).collect::<Vec<R>>()
        }));
    }
    out
}

fn dedupe(hyps: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut seen = HashSet::new();
    hyps.into_iter().filter(|h| seen.insert(h.clone())).collect()
}

/// Rebuild every parsable candidate, `rounds` times (see module doc).
pub fn recover_all(
    img: &Image,
    flags: &mut [u8],
    cands: &mut [Candidate],
    rounds: usize,
    checker: Option<&dyn FrameChecker>,
    log: &mut dyn FnMut(&str),
) {
    let n_cl = img.n_clusters();
    let mut base: Vec<u8> = flags.iter().map(|&f| (f & (FLAG_ZERO | FLAG_SIGNATURE) != 0) as u8).collect();
    let good: Vec<usize> = (0..cands.len()).filter(|&i| cands[i].ok()).collect();
    for &i in &good {
        let c = &cands[i];
        for k in 0..c.h {
            base[c.hcl + k] = 1;
            flags[c.hcl + k] |= FLAG_HEADER;
        }
    }
    let mut claims: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut soft_claims: HashMap<usize, Vec<usize>> = HashMap::new();
    for r in 0..rounds {
        for (idx, &i) in good.iter().enumerate() {
            let mut blocked = base.clone();
            for (&other, qs) in &claims {
                if other != i {
                    for &q in qs {
                        blocked[q] = 1;
                    }
                }
            }
            let (hcl, h) = (cands[i].hcl, cands[i].h);
            for k in 0..h {
                blocked[hcl + k] = 0; // own header is excluded via `used`
            }
            let mut used = vec![0u8; n_cl];
            cands[i].rebuild(img, &blocked, &mut used);
            let mut soft = vec![0u8; n_cl];
            for (&other, qs) in &soft_claims {
                if other != i {
                    for &q in qs {
                        soft[q] = 1;
                    }
                }
            }
            let ch = if r + 1 == rounds { checker } else { None };
            cands[i].refine(img, &blocked, &soft, ch);
            log(&format!(
                "  round {} [{}/{}] {}  {:.2}% recovered",
                r + 1,
                idx + 1,
                good.len(),
                cands[i].name(),
                cands[i].stats.recovered_pct()
            ));
        }
        claims = good.iter().map(|&i| (i, cands[i].strong_clusters())).collect();
        soft_claims = good.iter().map(|&i| (i, cands[i].mapping.iter().flatten().copied().collect())).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plausible() {
        assert!(nal_plausible(&[0, 0, 0, 2, 9], 4, 100));
        assert!(nal_plausible(&[0, 0, 0x10, 0, 0x65], 4, 5000));
        assert!(!nal_plausible(&[0, 0, 0x10, 0, 0x65], 4, 1000));
        assert!(!nal_plausible(&[0, 0, 0, 5, 9], 4, 100));
        assert!(!nal_plausible(&[0, 0, 0, 0x10, 0x85], 4, 100));
        assert!(!nal_plausible(&[0, 0, 0, 0, 0x41], 4, 100));
    }

    #[test]
    fn dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(16294), (2014, 8, 12));
    }
}
