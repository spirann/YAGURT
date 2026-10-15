//! Desktop app: pick a card image, scan it, rebuild the GoPro videos.
//!
//! All the work happens on one worker thread that owns the mapped image and
//! the scan results, so a scan is done once and reused by "Recover".

#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc::{channel, Receiver};

use slint::{ModelRc, SharedString, StandardListViewItem, VecModel};
use yagurt_core::decode::ffmpeg::Ffmpeg;
use yagurt_core::image::MappedFile;
use yagurt_core::{FrameChecker, JpegChecker, Options, Recovery};

slint::include_modules!();

enum Job {
    Scan(PathBuf),
    Recover { outdir: PathBuf, use_ffmpeg: bool, photos: bool },
}

/// Thread-safe handle for posting UI updates from the worker.
#[derive(Clone)]
struct Ui(slint::Weak<AppWindow>);

impl Ui {
    fn with(&self, f: impl FnOnce(&AppWindow) + Send + 'static) {
        let w = self.0.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(ui) = w.upgrade() {
                f(&ui);
            }
        });
    }
    fn status(&self, s: String) {
        self.with(move |ui| ui.set_status(s.into()));
    }
    fn log(&self, s: String) {
        self.with(move |ui| {
            let mut l = ui.get_log().to_string();
            l.push_str(&s);
            l.push('\n');
            ui.set_log(l.into());
        });
    }
    fn progress(&self, p: f32) {
        self.with(move |ui| ui.set_progress(p));
    }
    fn busy(&self, b: bool) {
        self.with(move |ui| ui.set_busy(b));
    }
}

type Row = Vec<String>;

fn rows(rec: &Recovery, done: bool) -> Vec<Row> {
    let photos = rec.photos.iter().filter(|p| !p.is_thumbnail()).map(|p| {
        vec![
            p.name(),
            format!("{}x{}", p.width, p.height),
            "photo".into(),
            format!("{:.1} MB", p.size(&rec.img) as f64 / 1e6),
            if p.complete { "complete".into() } else { "cut short".into() },
        ]
    });
    rec.videos()
        .map(|c| {
            let (w, h) = c.resolution();
            vec![
                c.name(),
                format!("{w}x{h}"),
                format!("{:.0} s", c.movie.duration_s()),
                format!("{:.1} MB", c.file_size as f64 / 1e6),
                if done { format!("{:.1}%", c.stats.recovered_pct()) } else { String::new() },
            ]
        })
        .chain(photos)
        .collect()
}

fn set_rows(ui: &Ui, r: Vec<Row>) {
    ui.with(move |w| {
        let model: Vec<ModelRc<StandardListViewItem>> = r
            .into_iter()
            .map(|row| {
                let items: Vec<StandardListViewItem> =
                    row.into_iter().map(|s| StandardListViewItem::from(SharedString::from(s))).collect();
                ModelRc::from(Rc::new(VecModel::from(items)))
            })
            .collect();
        w.set_videos(ModelRc::from(Rc::new(VecModel::from(model))));
    });
}

fn worker(rx: Receiver<Job>, ui: Ui) {
    let mut next = rx.recv().ok();
    while let Some(job) = next.take() {
        let Job::Scan(path) = job else {
            next = rx.recv().ok();
            continue;
        };
        ui.busy(true);
        ui.status(format!("Scanning {} …", path.display()));
        let src = match MappedFile::open(&path) {
            Ok(s) => s,
            Err(e) => {
                ui.status(format!("Cannot open {}: {e}", path.display()));
                ui.busy(false);
                next = rx.recv().ok();
                continue;
            }
        };
        let pu = ui.clone();
        let mut last = -1i32;
        let mut rec = Recovery::open(&src, &Options::default(), &mut |f| {
            let pct = (f * 100.0) as i32;
            if pct != last {
                last = pct;
                pu.progress(f);
            }
        });
        let found = rec.videos().count();
        ui.log(format!(
            "{}: clusters of {} bytes ({}); {} MP4 headers, {} usable",
            path.display(),
            rec.img.cluster_size,
            rec.geometry.source,
            rec.candidates.len(),
            found
        ));
        for c in rec.candidates.iter().filter(|c| !c.ok()) {
            ui.log(format!("  skipped header at 0x{:x}: {}", c.offset, c.error.as_deref().unwrap_or("")));
        }
        set_rows(&ui, rows(&rec, false));
        ui.with(move |w| w.set_scanned(found > 0));
        ui.status(if found > 0 {
            format!("Found {found} videos. Choose an output folder and press Recover.")
        } else {
            "No video headers found in this image.".into()
        });
        ui.progress(0.0);
        ui.busy(false);

        // keep the scan while the user recovers (possibly several times)
        loop {
            match rx.recv() {
                Ok(Job::Recover { outdir, use_ffmpeg, photos }) => {
                    ui.busy(true);
                    recover(&mut rec, &outdir, use_ffmpeg, photos, &path, &ui);
                    ui.busy(false);
                }
                Ok(j @ Job::Scan(_)) => {
                    next = Some(j);
                    break;
                }
                Err(_) => return,
            }
        }
    }
}

fn recover(rec: &mut Recovery, outdir: &PathBuf, use_ffmpeg: bool, photos: bool, image: &std::path::Path, ui: &Ui) {
    if let Err(e) = std::fs::create_dir_all(outdir) {
        ui.status(format!("Cannot create {}: {e}", outdir.display()));
        return;
    }
    let ffmpeg = if use_ffmpeg { Ffmpeg::find(None) } else { None };
    if use_ffmpeg && ffmpeg.is_none() {
        ui.log("ffmpeg not found: ambiguous cluster orders will be guessed".into());
    }
    let checker = ffmpeg.as_ref().map(|f| f as &dyn FrameChecker);
    let total = rec.videos().count() * 2;
    let mut done = 0;
    ui.status("Rebuilding videos …".into());
    let lu = ui.clone();
    rec.rebuild(2, checker, &mut |line| {
        done += 1;
        lu.progress(done as f32 / total.max(1) as f32);
        lu.log(line.trim().to_string());
    });
    let mut written = 0;
    for c in rec.videos() {
        let path = outdir.join(c.name());
        let res = std::fs::File::create(&path).and_then(|f| {
            let mut w = std::io::BufWriter::new(f);
            c.write_to(&rec.img, &mut w)?;
            std::io::Write::flush(&mut w)
        });
        match res {
            Ok(()) => written += 1,
            Err(e) => ui.log(format!("cannot write {}: {e}", path.display())),
        }
    }
    let mut photo_note = String::new();
    if photos {
        ui.status("Recovering photos …".into());
        let lu = ui.clone();
        rec.recover_photos(ffmpeg.as_ref().map(|f| f as &dyn JpegChecker), &mut |line| lu.log(line.trim().to_string()));
        let (mut big, mut thumbs) = (0, 0);
        for p in &rec.photos {
            let mut dir = outdir.join("photos");
            if p.is_thumbnail() {
                dir.push("thumbnails");
                thumbs += 1;
            } else {
                big += 1;
            }
            let path = dir.join(p.name());
            if let Err(e) = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, p.data(&rec.img))) {
                ui.log(format!("cannot write {}: {e}", path.display()));
            }
        }
        photo_note = format!(", {big} photos and {thumbs} thumbnails");
    }
    let _ = std::fs::write(outdir.join("yagurt_report.json"), rec.report_json(&image.display().to_string()));
    set_rows(ui, rows(rec, true));
    ui.progress(1.0);
    ui.status(format!("Wrote {written} videos{photo_note} and yagurt_report.json to {}.", outdir.display()));
}

fn main() -> Result<(), slint::PlatformError> {
    let app = AppWindow::new()?;
    app.set_ffmpeg_status(
        if Ffmpeg::find(None).is_some() { "ffmpeg found" } else { "ffmpeg not found in PATH" }.into(),
    );
    let (tx, rx) = channel::<Job>();
    let ui = Ui(app.as_weak());
    std::thread::spawn(move || worker(rx, ui));

    // `yagurt-gui [IMAGE [OUTDIR]]`: prefill, and scan right away
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    if let Some(image) = args.next() {
        app.set_image_path(image.display().to_string().into());
        if let Some(out) = args.next() {
            app.set_output_dir(out.display().to_string().into());
        }
        let _ = tx.send(Job::Scan(image));
    }

    let w = app.as_weak();
    app.on_choose_image(move || {
        if let Some(p) = rfd::FileDialog::new().set_title("Choose the card image").pick_file() {
            if let Some(ui) = w.upgrade() {
                ui.set_image_path(p.display().to_string().into());
            }
        }
    });
    let w = app.as_weak();
    app.on_choose_output(move || {
        if let Some(p) = rfd::FileDialog::new().set_title("Folder for recovered videos").pick_folder() {
            if let Some(ui) = w.upgrade() {
                ui.set_output_dir(p.display().to_string().into());
            }
        }
    });
    let (w, t) = (app.as_weak(), tx.clone());
    app.on_scan(move || {
        if let Some(ui) = w.upgrade() {
            ui.set_scanned(false);
            let _ = t.send(Job::Scan(PathBuf::from(ui.get_image_path().as_str())));
        }
    });
    let (w, t) = (app.as_weak(), tx);
    app.on_recover(move || {
        if let Some(ui) = w.upgrade() {
            let _ = t.send(Job::Recover {
                outdir: PathBuf::from(ui.get_output_dir().as_str()),
                use_ffmpeg: ui.get_use_ffmpeg(),
                photos: ui.get_use_photos(),
            });
        }
    });
    let w = app.as_weak();
    app.on_open_output(move || {
        if let Some(ui) = w.upgrade() {
            let dir = ui.get_output_dir().to_string();
            let opener = if cfg!(target_os = "macos") {
                "open"
            } else if cfg!(target_os = "windows") {
                "explorer"
            } else {
                "xdg-open"
            };
            let _ = std::process::Command::new(opener).arg(dir).spawn();
        }
    });
    app.run()
}
