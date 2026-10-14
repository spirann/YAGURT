"""Command line entry point: ``python -m yagurt IMAGE OUTDIR``."""

import argparse
import json
import os
import shutil
import subprocess
import sys

from . import __version__
from .decodecheck import DecodeChecker
from .image import Image
from .rebuild import find_candidates, recover_all


def parse_int(s):
    return int(s, 0)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="yagurt",
        description="Recover GoPro MP4/LRV videos from a raw image of a formatted SD card.")
    ap.add_argument("image", help="raw image of the card (ddrescue output, dd, ...)")
    ap.add_argument("outdir", nargs="?", help="directory for recovered videos")
    ap.add_argument("--cluster-size", type=parse_int,
                    help="cluster size in bytes, e.g. 0x8000 (default: read from the boot sector)")
    ap.add_argument("--data-offset", type=parse_int,
                    help="byte offset of the first cluster of the data area (default: auto)")
    ap.add_argument("--list", action="store_true", help="only list the videos found")
    ap.add_argument("--only", type=parse_int, action="append", metavar="OFFSET",
                    help="only rebuild the header at this byte offset (repeatable)")
    ap.add_argument("--rounds", type=int, default=2, help="refinement rounds (default 2)")
    ap.add_argument("--no-decode-check", action="store_true",
                    help="do not use ffmpeg to choose between possible cluster orders")
    ap.add_argument("--ffmpeg", help="path to ffmpeg (default: found in PATH)")
    ap.add_argument("--verify", action="store_true",
                    help="decode every recovered file with ffmpeg and report errors")
    ap.add_argument("--version", action="version", version="yagurt " + __version__)
    a = ap.parse_args(argv)

    if not a.list and not a.outdir:
        ap.error("outdir is required unless --list is given")

    img = Image(a.image)
    try:
        return run(a, img)
    finally:
        img.close()


def run(a, img):
    print("image: %s (%.2f GB)" % (a.image, img.size / 1e9))

    geo = img.detect_geometry()
    if a.cluster_size:
        off = a.data_offset
        src = "command line"
        if off is None:
            off = geo[1] if geo and geo[0] == a.cluster_size else img.guess_alignment(a.cluster_size)[0]
            src += " + auto alignment"
        img.set_geometry(a.cluster_size, off, src)
    elif geo:
        img.set_geometry(geo[0], a.data_offset if a.data_offset is not None else geo[1], geo[2])
    else:
        off, _n = img.guess_alignment(0x8000)
        img.set_geometry(0x8000, a.data_offset if a.data_offset is not None else off,
                         "default 32 KiB (no boot sector found)")
    print("clusters: %d bytes, first cluster at byte 0x%x (%s)" % (
        img.cluster_size, img.data_offset, img.geometry_source))

    flags, starts = img.scan()
    cands = find_candidates(img, starts)
    if a.only:
        cands = [c for c in cands if c.offset in a.only]
    good = [c for c in cands if c.ok]
    print("\n%d MP4 headers found, %d parsable:" % (len(cands), len(good)))
    for c in cands:
        if c.ok:
            w, h = c.resolution()
            print("  0x%010x  %-40s %5dx%-5d %7.1fs %8.1f MB" % (
                c.offset, c.name(), w, h, c.movie.duration_s, c.file_size / 1e6))
        else:
            print("  0x%010x  skipped: %s" % (c.offset, c.error))
    if a.list or not good:
        return 0

    os.makedirs(a.outdir, exist_ok=True)
    checker = None
    if not a.no_decode_check:
        checker = DecodeChecker(a.ffmpeg)
        if not checker.available:
            print("note: ffmpeg not found, so ambiguous cluster orders are guessed "
                  "(install ffmpeg or pass --ffmpeg for better results)")
            checker = None
    print("\nrebuilding:")
    recover_all(img, flags, good, rounds=a.rounds, checker=checker)

    report = []
    ffmpeg = (a.ffmpeg or shutil.which("ffmpeg")) if a.verify else None
    if a.verify and not ffmpeg:
        print("warning: --verify needs ffmpeg in PATH; skipping verification")
    print("\nresults:")
    for c in good:
        out = os.path.join(a.outdir, c.name())
        c.write(out)
        entry = {"file": c.name(), "header_offset": c.offset, **c.stats,
                 "cluster_map": c.cluster_map()}
        line = "  %-48s %6.2f%%  (verified %d, inferred %d, missing %d)" % (
            c.name(), c.stats["recovered_pct"], c.stats["verified"],
            c.stats["inferred"], c.stats["missing"])
        if ffmpeg:
            errs = decode_errors(ffmpeg, out)
            entry["ffmpeg_errors"] = errs
            line += "  ffmpeg errors: %d" % errs
        print(line)
        report.append(entry)
    with open(os.path.join(a.outdir, "yagurt_report.json"), "w") as f:
        json.dump({"image": a.image, "cluster_size": img.cluster_size,
                   "data_offset": img.data_offset, "files": report}, f, indent=1)
    print("\nwrote %d files and yagurt_report.json to %s" % (len(good), a.outdir))
    return 0


def decode_errors(ffmpeg, path):
    p = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-f", "null", "-"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return len([l for l in p.stderr.decode("utf-8", "replace").splitlines() if l.strip()])


if __name__ == "__main__":
    sys.exit(main())
