# YAGURT

Rebuild GoPro videos from a raw image of an SD card that was formatted in
the camera. Written for a GoPro HERO3 card; it should work for any camera that
writes H.264 MP4 files with the index (`moov`) at the start of the file.

The repository holds two versions of the same method:

| Folder | What | Use it for |
| --- | --- | --- |
| [`rust/`](rust) | The app: a recovery engine, a command line tool (`yagurt`) and a desktop app (`yagurt-gui`, built with [Slint](https://slint.dev)) | Recovering your videos |
| [`python/`](python) | The proof of concept, standard library only | Experimenting with the method |

Both read the image only; it is never modified. Linux (Ubuntu) is the first
target; macOS should work with the same steps (Homebrew instead of apt).

## Quick start on Ubuntu

Install the build tools, ffmpeg and the libraries the desktop app needs, then
Rust:

```sh
sudo apt install build-essential pkg-config git ffmpeg libfontconfig-dev libxkbcommon-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # then open a new terminal
```

ffmpeg is optional but strongly recommended: it is used to choose between
possible cluster orders inside large frames (see below).

Get the code and build it:

```sh
git clone https://github.com/spirann/YAGURT.git
cd YAGURT/rust
cargo build --release --workspace
```

### Desktop app

```sh
./target/release/yagurt-gui
```

Choose the card image, press **Scan**, choose an output folder and press
**Recover videos**. You can also pass the paths to start scanning right away:
`./target/release/yagurt-gui ~/backup/gopro_card.img ~/recovered`.

### Command line

```sh
./target/release/yagurt ~/backup/gopro_card.img --list          # see what is on the card
./target/release/yagurt ~/backup/gopro_card.img ~/recovered     # rebuild every video
```

Each video gets a file named after its recording date, resolution and where
its header sits on the card, for example
`GOPRO_20140812_153012_1920x1080_at_0x1a2b0000.MP4`. Low-resolution preview
copies (LRV) end in `.LRV.mp4`. A `yagurt_report.json` next to them says, for
every file, how many of its clusters were verified, inferred or missing.

| Option | What it does |
| --- | --- |
| `--list` | Only list the videos found. |
| `--only 0x1a2b0000` | Only rebuild the video whose header is at this offset (repeatable). |
| `--cluster-size 0x8000` | Force the cluster size if the boot sector is missing or wrong. SDXC cards formatted as exFAT usually use `0x20000`. |
| `--data-offset N` | Force the byte offset of the first cluster. |
| `--rounds N` | Refinement rounds (default 2). |
| `--ffmpeg PATH` | Path to `ffmpeg` if it is not on the PATH. |
| `--no-decode-check` | Skip the ffmpeg step (faster, less accurate). |
| `--verify` | Decode every recovered file with ffmpeg and print the number of errors. |

### Python proof of concept

Needs Python 3.9 or newer, nothing to install. Same options as the Rust tool:

```sh
cd YAGURT/python
python3 -m yagurt ~/backup/gopro_card.img ~/recovered
```

## Why other tools give unplayable videos

A formatted card still holds all the data, but the file system no longer says
which clusters belong to which file. Carving tools (and AMRT, which this
project started from) find the header of each video and assume its data is one
contiguous block next to it.

On a GoPro that assumption is wrong. While recording, the camera writes two
files at once: the full-resolution MP4 and a low-resolution LRV preview. The
file system hands out clusters to both in turn, so on the card they are
interleaved (`MP4 MP4 MP4 LRV MP4 MP4 LRV ...`), and clusters still used by
older files are skipped too. Copying a contiguous block mixes the two streams
and the result does not decode.

## How YAGURT rebuilds a file

1. It reads the boot sector of the (new) file system to learn the cluster size
   and where clusters start, then scans every cluster once, noting MP4 headers,
   JPEG starts and empty clusters.
2. For each header it parses the `moov` index: the exact position and size of
   every video frame and audio packet in the original file.
3. H.264 frames are chains of length-prefixed NAL units, so for most clusters
   we know where a NAL header must sit and what it must look like. YAGURT walks
   the file cluster by cluster, tries the next free cluster on the card first
   (that is what the camera did), skips clusters that fail the check (they
   belong to the other stream or to an older file), and if it loses the trail
   it searches the whole card for the cluster holding the next frame.
4. Every file is rebuilt twice; the second time, clusters that other files
   claimed with certainty are excluded.
5. Clusters in the middle of large key frames have nothing to check. When more
   free clusters than needed sit between two verified ones, YAGURT asks ffmpeg
   to decode the frame for candidate placements and keeps the one that decodes
   with no concealed macroblocks. Without ffmpeg it guesses the most likely
   layout, which is usually but not always right.

The Rust engine (`rust/yagurt-core`) reads the image only through a small
`ReadAt` trait and decodes only through a `FrameChecker` trait, so it has no
OS-specific I/O and can later be built for WebAssembly.

## Tests

The tests build simulated formatted cards (MP4 and LRV interleaved, up to half
of the clusters held by older recordings or other videos) and check that both
files come back byte for byte. They need ffmpeg with libx264 (Ubuntu's ffmpeg
has it).

```sh
cd rust && cargo test --workspace
cd python && python3 -m unittest discover -s tests -v
```

## Limits

- The video's header clusters must have survived: YAGURT needs the `moov` index.
  Videos whose header was overwritten are not recovered (yet).
- Clusters overwritten after the format are lost; those parts of the video are
  filled with zeros so the rest still plays.
- A large card with many videos can take a while: the scan reads the whole
  image once, and the ffmpeg step runs many short decodes.

## If the result is still not playable

Send the output of `--list`, the `yagurt_report.json` and, if possible, one
broken file. The report says exactly where each part of each file came from.
