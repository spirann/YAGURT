# YAGURT

Rebuild GoPro videos from a raw image of an SD card that was formatted in
the camera. Written for a GoPro HERO3 card; it should work for any camera that
writes H.264 MP4 files with the index (`moov`) at the start of the file.

Pure Python 3 standard library, no install needed. Runs on Windows, macOS and
Linux. [ffmpeg](https://ffmpeg.org/download.html) is optional but strongly
recommended (see below).

## Quick start

1. Install **64-bit** Python 3.9 or newer ([python.org](https://www.python.org/downloads/)).
2. Optional but recommended: install ffmpeg and make sure `ffmpeg -version` works
   in a terminal, or pass its location with `--ffmpeg`.
3. Download this repository (green *Code* button, *Download ZIP*) and unzip it.
4. Open a terminal in the unzipped folder and run:

```
python -m yagurt "D:\backup\gopro_card.img" "D:\recovered"
```

(On macOS/Linux use `python3` and normal paths.) The image is only read, never
modified. To see what is on the card first, without writing anything:

```
python -m yagurt "D:\backup\gopro_card.img" --list
```

Each video found gets a file named after its recording date, resolution and
where its header sits on the card, for example
`GOPRO_20140812_153012_1920x1080_at_0x1a2b0000.MP4`. Low-resolution preview
copies (LRV) end in `.LRV.mp4`. A `yagurt_report.json` next to them says, for
every file, how many of its clusters were verified, inferred or missing.

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

On simulated formatted cards (MP4 and LRV interleaved, up to 60% of clusters
held by older recordings) it rebuilds both files byte for byte. Run
`python -m unittest discover -s tests -v` to repeat those tests (needs ffmpeg
with libx264).

## Options

| Option | What it does |
| --- | --- |
| `--list` | Only list the videos found. |
| `--only 0x1a2b0000` | Only rebuild the video whose header is at this offset (repeatable). |
| `--cluster-size 0x8000` | Force the cluster size if the boot sector is missing or wrong. SDXC cards formatted as exFAT usually use `0x20000`. |
| `--data-offset N` | Force the byte offset of the first cluster. |
| `--ffmpeg PATH` | Path to `ffmpeg` if it is not on the PATH. |
| `--no-decode-check` | Skip the ffmpeg step (faster, less accurate). |
| `--verify` | Decode every recovered file with ffmpeg and print the number of errors. |

## Limits

- The video's header clusters must have survived: YAGURT needs the `moov` index.
  Videos whose header was overwritten are not recovered (yet).
- Clusters overwritten after the format are lost; those parts of the video are
  filled with zeros so the rest still plays.
- The image must be readable through 64-bit Python (any modern install is).
- A large card with many videos can take a while: the scan reads the whole
  image once, and the ffmpeg step runs many short decodes.

## If the result is still not playable

Send the output of `--list`, the `yagurt_report.json` and, if possible, one
broken file. The report says exactly where each part of each file came from.
