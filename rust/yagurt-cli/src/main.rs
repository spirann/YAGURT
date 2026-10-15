//! `yagurt IMAGE OUTDIR`: rebuild GoPro videos from a formatted SD card image.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use yagurt_core::decode::ffmpeg::Ffmpeg;
use yagurt_core::image::MappedFile;
use yagurt_core::{FrameChecker, Options, ReadAt, Recovery};

const USAGE: &str = "\
Rebuild GoPro MP4/LRV videos from a raw image of a formatted SD card.

usage: yagurt IMAGE [OUTDIR] [options]

  IMAGE                   raw image of the card (ddrescue output, dd, ...)
  OUTDIR                  directory for recovered videos

options:
  --list                  only list the videos found
  --only OFFSET           only rebuild the header at this byte offset (repeatable)
  --cluster-size N        cluster size, e.g. 0x8000 (default: from the boot sector)
  --data-offset N         byte offset of the first cluster (default: auto)
  --rounds N              refinement rounds (default 2)
  --ffmpeg PATH           path to ffmpeg (default: found in PATH)
  --no-decode-check       do not use ffmpeg to choose between cluster orders
  --verify                decode every recovered file with ffmpeg and report errors
  -h, --help              show this help
  -V, --version           show the version";

struct Args {
    image: PathBuf,
    outdir: Option<PathBuf>,
    list: bool,
    verify: bool,
    no_decode_check: bool,
    ffmpeg: Option<PathBuf>,
    rounds: usize,
    opts: Options,
}

fn parse_num(s: &str) -> Result<u64, String> {
    let r = if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(h, 16)
    } else {
        s.parse()
    };
    r.map_err(|_| format!("not a number: {s}"))
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        image: PathBuf::new(),
        outdir: None,
        list: false,
        verify: false,
        no_decode_check: false,
        ffmpeg: None,
        rounds: 2,
        opts: Options::default(),
    };
    let mut pos = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("yagurt {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--list" => a.list = true,
            "--verify" => a.verify = true,
            "--no-decode-check" => a.no_decode_check = true,
            "--ffmpeg" => a.ffmpeg = Some(PathBuf::from(val("--ffmpeg")?)),
            "--rounds" => a.rounds = parse_num(&val("--rounds")?)? as usize,
            "--only" => a.opts.only.push(parse_num(&val("--only")?)?),
            "--cluster-size" => a.opts.cluster_size = Some(parse_num(&val("--cluster-size")?)?),
            "--data-offset" => a.opts.data_offset = Some(parse_num(&val("--data-offset")?)?),
            s if s.starts_with('-') => return Err(format!("unknown option {s}")),
            _ => pos.push(PathBuf::from(arg)),
        }
    }
    let mut pos = pos.into_iter();
    a.image = pos.next().ok_or("IMAGE is required")?;
    a.outdir = pos.next();
    if pos.next().is_some() {
        return Err("too many arguments".into());
    }
    if !a.list && a.outdir.is_none() {
        return Err("OUTDIR is required unless --list is given".into());
    }
    Ok(a)
}

fn main() -> ExitCode {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&a) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(a: &Args) -> Result<(), String> {
    let src = MappedFile::open(&a.image).map_err(|e| format!("{}: {e}", a.image.display()))?;
    println!("image: {} ({:.2} GB)", a.image.display(), src.size() as f64 / 1e9);

    let mut last = -1i32;
    let mut rec = Recovery::open(&src, &a.opts, &mut |f| {
        let pct = (f * 100.0) as i32;
        if pct != last {
            last = pct;
            eprint!("\rscanning image: {pct:3}%");
        }
    });
    eprintln!();
    println!(
        "clusters: {} bytes, first cluster at byte 0x{:x} ({})",
        rec.img.cluster_size, rec.img.data_offset, rec.geometry.source
    );
    let good = rec.videos().count();
    println!("\n{} MP4 headers found, {} parsable:", rec.candidates.len(), good);
    for c in &rec.candidates {
        match &c.error {
            None => {
                let (w, h) = c.resolution();
                println!(
                    "  0x{:010x}  {:<48} {:>5}x{:<5} {:7.1}s {:8.1} MB",
                    c.offset,
                    c.name(),
                    w,
                    h,
                    c.movie.duration_s(),
                    c.file_size as f64 / 1e6
                );
            }
            Some(e) => println!("  0x{:010x}  skipped: {e}", c.offset),
        }
    }
    if a.list || good == 0 {
        return Ok(());
    }
    let outdir = a.outdir.as_ref().unwrap();
    std::fs::create_dir_all(outdir).map_err(|e| format!("{}: {e}", outdir.display()))?;

    let ffmpeg = Ffmpeg::find(a.ffmpeg.clone());
    let checker: Option<&dyn FrameChecker> = if a.no_decode_check {
        None
    } else if let Some(f) = &ffmpeg {
        Some(f)
    } else {
        println!(
            "note: ffmpeg not found, so ambiguous cluster orders are guessed \
             (install ffmpeg or pass --ffmpeg for better results)"
        );
        None
    };
    println!("\nrebuilding:");
    rec.rebuild(a.rounds, checker, &mut |line| println!("{line}"));

    println!("\nresults:");
    for c in rec.videos() {
        let path = outdir.join(c.name());
        let f = File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut w = BufWriter::new(f);
        c.write_to(&rec.img, &mut w).and_then(|_| w.flush()).map_err(|e| format!("{}: {e}", path.display()))?;
        let st = &c.stats;
        let mut line = format!(
            "  {:<48} {:6.2}%  (verified {}, inferred {}, missing {})",
            c.name(),
            st.recovered_pct(),
            st.verified,
            st.inferred,
            st.missing
        );
        if a.verify {
            match &ffmpeg {
                Some(f) => line += &format!("  ffmpeg errors: {}", f.file_errors(&path)),
                None => line += "  (no ffmpeg to verify)",
            }
        }
        println!("{line}");
    }
    let report = outdir.join("yagurt_report.json");
    std::fs::write(&report, rec.report_json(&a.image.display().to_string()))
        .map_err(|e| format!("{}: {e}", report.display()))?;
    println!("\nwrote {} files and yagurt_report.json to {}", good, Path::new(outdir).display());
    Ok(())
}
