"""End-to-end tests on simulated formatted GoPro cards (needs ffmpeg + libx264).

Run with:  python -m unittest discover -s tests -v
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from yagurt import cli  # noqa: E402
from yagurt.photos import Parser, jpeg_info  # noqa: E402
from yagurt.rebuild import nal_plausible  # noqa: E402


def read(path):
    with open(path, "rb") as f:
        return f.read()


def have_x264():
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True).stdout
    return b"libx264" in out


class NalTest(unittest.TestCase):
    def test_plausible(self):
        self.assertTrue(nal_plausible(b"\x00\x00\x00\x02\x09", 4, 100))  # access unit delimiter
        self.assertTrue(nal_plausible(b"\x00\x00\x10\x00\x65", 4, 5000))  # IDR slice
        self.assertFalse(nal_plausible(b"\x00\x00\x10\x00\x65", 4, 1000))  # longer than the frame
        self.assertFalse(nal_plausible(b"\x00\x00\x00\x05\x09", 4, 100))  # AUD must be 2 bytes
        self.assertFalse(nal_plausible(b"\x00\x00\x00\x10\x85", 4, 100))  # forbidden bit set
        self.assertFalse(nal_plausible(b"\x00\x00\x00\x00\x41", 4, 100))  # zero length


class JpegTest(unittest.TestCase):
    # SOI, DRI(2), SOF0 8x8, SOS, data with stuffing and RST0, EOI
    JPEG = (b"\xff\xd8\xff\xdd\x00\x04\x00\x02"
            b"\xff\xc0\x00\x0b\x08\x00\x08\x00\x08\x01\x01\x11\x00"
            b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00"
            b"\x12\xff\x00\x34\xff\xd0\x56\xff\xd9")

    def test_parser_any_split(self):
        d = self.JPEG
        self.assertEqual(Parser().feed(d), ("end", len(d)))
        for cut in range(1, len(d)):
            p = Parser()
            self.assertEqual(p.feed(d[:cut]), ("ok", None))
            self.assertEqual(p.feed(d[cut:]), ("end", len(d) - cut))
        self.assertEqual(jpeg_info(d)[:2], (8, 8))

    def test_parser_rejects(self):
        d = self.JPEG
        self.assertEqual(Parser().feed(d[:-4] + b"\xd3" + d[-3:])[0], "bad")  # RST out of order
        self.assertEqual(Parser().feed(d[:-7] + b"\x47" + d[-6:])[0], "bad")  # FF 47 in data


@unittest.skipUnless(have_x264(), "ffmpeg with libx264 is needed to build test cards")
class CardTest(unittest.TestCase):
    def run_card(self, **kw):
        import fake_card
        with tempfile.TemporaryDirectory() as d:
            img, mp4, lrv, _truth = fake_card.build(d, **kw)
            out = os.path.join(d, "out")
            self.assertEqual(cli.main([img, out]), 0)
            got = {os.path.basename(p): read(p) for p in glob.glob(os.path.join(out, "GOPRO_*"))}
            self.assertEqual(len(got), 2, got.keys())
            want = {read(mp4), read(lrv)}
            self.assertEqual(set(got.values()), want)

    def test_interleaved_with_old_random_data(self):
        self.run_card(seed=2, junk_ratio=0.3)

    def test_interleaved_with_old_video_data(self):
        self.run_card(seed=4, junk_ratio=0.3, video_junk=True)

    def test_photos_around_videos_and_old_files(self):
        import fake_card
        with tempfile.TemporaryDirectory() as d:
            img, mp4, lrv, truth = fake_card.build(d, seed=6, junk_ratio=0.4, video_junk=True, photos=True)
            out = os.path.join(d, "out")
            self.assertEqual(cli.main([img, out]), 0)
            got = {read(p) for p in glob.glob(os.path.join(out, "photos", "*.JPG"))}
            self.assertEqual(got, set(truth["photos"]))
            vids = {read(p) for p in glob.glob(os.path.join(out, "GOPRO_*"))}
            self.assertEqual(vids, {read(mp4), read(lrv)})


if __name__ == "__main__":
    unittest.main()
