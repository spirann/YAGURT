//! YAGURT engine: rebuild GoPro videos from a raw image of a formatted SD card.
//!
//! The engine reads the card only through [`image::ReadAt`] and decodes only
//! through [`decode::FrameChecker`], so it has no OS-specific I/O and can be
//! wrapped by a command line tool, a desktop UI or a WebAssembly build.
//!
//! ```no_run
//! # use yagurt_core::*;
//! # fn demo(card: &Vec<u8>) {
//! let mut rec = Recovery::open(card, &Options::default(), &mut |_| {});
//! rec.rebuild(2, None, &mut |line| println!("{line}"));
//! for c in rec.videos() {
//!     let mut out = Vec::new();
//!     c.write_to(&rec.img, &mut out).unwrap();
//! }
//! # }
//! ```

pub mod decode;
pub mod image;
pub mod mp4;
pub mod photos;
pub mod rebuild;

pub use decode::{FrameChecker, JpegChecker};
pub use image::{Geometry, Image, ReadAt};
pub use photos::Photo;
pub use rebuild::{Candidate, Stats};

#[derive(Clone, Debug, Default)]
pub struct Options {
    pub cluster_size: Option<u64>,
    pub data_offset: Option<u64>,
    /// only rebuild headers at these byte offsets (empty: all)
    pub only: Vec<u64>,
}

pub struct Recovery<'a> {
    pub img: Image<'a>,
    pub geometry: Geometry,
    pub flags: Vec<u8>,
    pub candidates: Vec<Candidate>,
    pub photos: Vec<Photo>,
}

/// Work out cluster size and alignment: options, then boot sector, then a guess.
pub fn resolve_geometry(src: &dyn ReadAt, opts: &Options) -> Geometry {
    let detected = image::detect_geometry(src);
    if let Some(cs) = opts.cluster_size {
        let (off, source) = match opts.data_offset {
            Some(o) => (o, "command line".to_string()),
            None => match &detected {
                Some(g) if g.cluster_size == cs => (g.data_offset, "command line + boot sector".into()),
                _ => (image::guess_alignment(src, cs), "command line + auto alignment".into()),
            },
        };
        return Geometry { cluster_size: cs, data_offset: off, source };
    }
    if let Some(mut g) = detected {
        if let Some(o) = opts.data_offset {
            g.data_offset = o;
        }
        return g;
    }
    let cs = 0x8000;
    let off = opts.data_offset.unwrap_or_else(|| image::guess_alignment(src, cs));
    Geometry { cluster_size: cs, data_offset: off, source: "default 32 KiB (no boot sector found)".into() }
}

impl<'a> Recovery<'a> {
    /// Detect geometry, scan the whole image once and parse every MP4 header.
    pub fn open(src: &'a dyn ReadAt, opts: &Options, progress: &mut dyn FnMut(f32)) -> Recovery<'a> {
        let geometry = resolve_geometry(src, opts);
        let img = Image::new(src, &geometry);
        let (flags, starts) = image::scan(&img, progress);
        let candidates = starts
            .into_iter()
            .map(|k| Candidate::new(&img, k))
            .filter(|c| opts.only.is_empty() || opts.only.contains(&c.offset))
            .collect();
        Recovery { img, geometry, flags, candidates, photos: Vec::new() }
    }

    /// Rebuild every parsable video (see [`rebuild::recover_all`]).
    pub fn rebuild(&mut self, rounds: usize, checker: Option<&dyn FrameChecker>, log: &mut dyn FnMut(&str)) {
        rebuild::recover_all(&self.img, &mut self.flags, &mut self.candidates, rounds.max(1), checker, log);
    }

    /// Find and rebuild the JPEG photos, around the clusters the videos use
    /// (so call it after [`Recovery::rebuild`]).
    pub fn recover_photos(&mut self, checker: Option<&dyn JpegChecker>, log: &mut dyn FnMut(&str)) {
        self.photos = photos::recover_photos(&self.img, &self.flags, &self.candidates, checker, log);
    }

    pub fn videos(&self) -> impl Iterator<Item = &Candidate> {
        self.candidates.iter().filter(|c| c.ok())
    }

    /// Machine-readable account of where every part of every file came from.
    pub fn report_json(&self, image_name: &str) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "{{\n \"image\": {},\n \"cluster_size\": {},\n \"data_offset\": {},\n \"files\": [",
            json_str(image_name),
            self.img.cluster_size,
            self.img.data_offset
        ));
        for (i, c) in self.videos().enumerate() {
            let st = &c.stats;
            let map: Vec<String> = c
                .cluster_map()
                .iter()
                .map(|(l, p, n, s)| format!("[{l},{},{n},{s}]", p.map_or("null".into(), |p| p.to_string())))
                .collect();
            s.push_str(&format!(
                "{}\n  {{\"file\": {}, \"header_offset\": {}, \"clusters\": {}, \"header\": {}, \"verified\": {}, \
                 \"inferred\": {}, \"missing\": {}, \"skips\": {}, \"full_searches\": {}, \"ambiguous_runs\": {}, \
                 \"recovered_pct\": {:.2}, \"note\": {}, \"cluster_map\": [{}]}}",
                if i > 0 { "," } else { "" },
                json_str(&c.name()),
                c.offset,
                st.clusters,
                st.header,
                st.verified,
                st.inferred,
                st.missing,
                st.skips,
                st.full_searches,
                st.ambiguous_runs,
                st.recovered_pct(),
                st.note.as_deref().map_or("null".into(), json_str),
                map.join(",")
            ));
        }
        s.push_str("\n ],\n \"photos\": [");
        for (i, p) in self.photos.iter().enumerate() {
            let map: Vec<String> = p.cluster_map().iter().map(|(l, q, n)| format!("[{l},{q},{n}]")).collect();
            s.push_str(&format!(
                "{}\n  {{\"file\": {}, \"offset\": {}, \"clusters\": {}, \"size\": {}, \"complete\": {}, \
                 \"skips\": {}, \"decode_errors\": {}, \"cluster_map\": [{}]}}",
                if i > 0 { "," } else { "" },
                json_str(&p.name()),
                p.offset,
                p.clusters.len(),
                p.size(&self.img),
                p.complete,
                p.skips,
                p.damage.map_or("null".into(), |d| d.to_string()),
                map.join(",")
            ));
        }
        s.push_str("\n ]\n}\n");
        s
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}
