//! Using a decoder as an oracle: does this candidate byte sequence decode cleanly?
//!
//! Inside a large H.264 frame there is no structure we can check by hand, so
//! when several cluster orders are possible we rebuild the frame for each one
//! and let a decoder tell us which order produces no concealed macroblocks.

pub const START: [u8; 4] = [0, 0, 0, 1];

/// Something that decodes an Annex-B H.264 stream and says how damaged it is
/// (0 = decoded cleanly).
pub trait FrameChecker: Sync {
    fn damage(&self, annexb: &[u8]) -> u64;
}

/// Something that decodes one JPEG and says how many errors it hit (0 = clean).
pub trait JpegChecker: Sync {
    fn jpeg_damage(&self, jpeg: &[u8]) -> u64;
}

/// Convert one length-prefixed sample to Annex-B. Returns (bytes, broken).
/// With `partial`, a NAL cut short by the end of `sample` is kept.
pub fn to_annexb(sample: &[u8], nls: usize, partial: bool) -> (Vec<u8>, bool) {
    let mut out = Vec::with_capacity(sample.len() + 16);
    let mut p = 0;
    let n = sample.len();
    while p + nls <= n {
        let ln = sample[p..p + nls].iter().fold(0usize, |a, &b| (a << 8) | b as usize);
        p += nls;
        if ln == 0 || p + ln > n {
            if partial && ln != 0 && p < n {
                out.extend_from_slice(&START);
                out.extend_from_slice(&sample[p..]);
            }
            return (out, true);
        }
        out.extend_from_slice(&START);
        out.extend_from_slice(&sample[p..p + ln]);
        p += ln;
    }
    (out, p != n)
}

/// Score ffmpeg's log: the number of concealed macroblocks.
///
/// Garbage in the middle of a CABAC slice often ends the slice early; ffmpeg
/// only reports that at info level ("concealing N DC, N AC, N MV errors").
/// The later the decoder derails, the fewer macroblocks it conceals, so this
/// also ranks wrong placements. Other decoder errors only add 1, so that they
/// never outweigh that ranking but still keep the score above 0.
pub fn score_ffmpeg_log(log: &str) -> u64 {
    let mut score = 0;
    for line in log.lines() {
        if let Some(i) = line.find("concealing ") {
            let rest = &line[i + 11..];
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if rest[num.len()..].starts_with(" DC") {
                score += num.parse::<u64>().unwrap_or(0);
                continue;
            }
        }
        if line.contains("[h264") && line.to_ascii_lowercase().contains("error") {
            score += 1;
        }
    }
    score
}

#[cfg(not(target_arch = "wasm32"))]
pub mod ffmpeg {
    use super::{score_ffmpeg_log, FrameChecker, JpegChecker};
    use std::io::{Read, Write};
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    pub struct Ffmpeg {
        pub path: PathBuf,
    }

    impl Ffmpeg {
        /// Find ffmpeg on the PATH (or use `path`), checking that it runs.
        pub fn find(path: Option<PathBuf>) -> Option<Ffmpeg> {
            let path = path.unwrap_or_else(|| PathBuf::from("ffmpeg"));
            let ok = Command::new(&path)
                .arg("-version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            ok.then_some(Ffmpeg { path })
        }

        /// Error lines printed while decoding a whole file (for --verify).
        pub fn file_errors(&self, file: &std::path::Path) -> usize {
            let out = Command::new(&self.path)
                .args(["-v", "error", "-i"])
                .arg(file)
                .args(["-f", "null", "-"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .output();
            match out {
                Ok(o) => String::from_utf8_lossy(&o.stderr).lines().filter(|l| !l.trim().is_empty()).count(),
                Err(_) => usize::MAX,
            }
        }
    }

    impl Ffmpeg {
        /// Run ffmpeg with `input` on stdin; returns its stderr.
        fn decode(&self, args: &[&str], input: &[u8]) -> Option<String> {
            let mut child = Command::new(&self.path)
                .args(args)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .ok()?;
            let mut stdin = child.stdin.take().unwrap();
            let mut stderr = child.stderr.take().unwrap();
            let log = std::thread::scope(|s| {
                let reader = s.spawn(move || {
                    let mut v = Vec::new();
                    let _ = stderr.read_to_end(&mut v);
                    v
                });
                let _ = stdin.write_all(input);
                drop(stdin);
                reader.join().unwrap_or_default()
            });
            let _ = child.wait();
            Some(String::from_utf8_lossy(&log).into_owned())
        }
    }

    impl FrameChecker for Ffmpeg {
        fn damage(&self, annexb: &[u8]) -> u64 {
            let args = ["-hide_banner", "-nostdin", "-v", "info", "-threads", "1", "-f", "h264", "-i", "pipe:0", "-f", "null", "-"];
            self.decode(&args, annexb).map_or(u64::MAX / 4, |log| score_ffmpeg_log(&log))
        }
    }

    impl JpegChecker for Ffmpeg {
        fn jpeg_damage(&self, jpeg: &[u8]) -> u64 {
            let args = ["-hide_banner", "-nostdin", "-v", "error", "-threads", "1", "-f", "jpeg_pipe", "-i", "pipe:0", "-f", "null", "-"];
            self.decode(&args, jpeg).map_or(u64::MAX / 4, |log| log.lines().filter(|l| !l.trim().is_empty()).count() as u64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annexb() {
        let s = [0, 0, 0, 2, 9, 0x10, 0, 0, 0, 3, 0x65, 1, 2];
        let (b, broken) = to_annexb(&s, 4, false);
        assert!(!broken);
        assert_eq!(b, [0, 0, 0, 1, 9, 0x10, 0, 0, 0, 1, 0x65, 1, 2]);
        let (b, broken) = to_annexb(&s[..12], 4, true);
        assert!(broken);
        assert_eq!(b, [0, 0, 0, 1, 9, 0x10, 0, 0, 0, 1, 0x65, 1]);
    }

    #[test]
    fn log_score() {
        let log = "[h264 @ 0x1] concealing 1330 DC, 1330 AC, 1330 MV errors in I frame\n\
                   [h264 @ 0x1] error while decoding MB 3 4\nInput #0";
        assert_eq!(score_ffmpeg_log(log), 1331);
    }
}
