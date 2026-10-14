"""Use ffmpeg as an oracle: does this candidate byte sequence decode cleanly?

Inside a large H.264 frame there is no structure we can check by hand, so
when several cluster orders are possible we rebuild the affected frames for
each one and let the decoder tell us which order produces no errors.
"""

import re
import shutil
import subprocess

START = b"\x00\x00\x00\x01"
CONCEAL = re.compile(rb"concealing (\d+) DC")


class DecodeChecker:
    def __init__(self, ffmpeg=None):
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg")
        self.calls = 0

    @property
    def available(self):
        return bool(self.ffmpeg)

    def errors(self, annexb):
        """How badly an Annex-B stream decodes: the number of concealed macroblocks.

        Garbage in the middle of a CABAC slice often ends the slice early;
        ffmpeg only reports that at info level ("concealing N DC ... errors").
        The later the decoder derails, the fewer macroblocks it conceals, so
        this also ranks wrong placements. Other decoder errors only add 1, so
        that they never outweigh that ranking but still keep the score above 0.
        """
        self.calls += 1
        p = subprocess.run(
            [self.ffmpeg, "-hide_banner", "-nostdin", "-v", "info", "-threads", "1",
             "-f", "h264", "-i", "pipe:0", "-f", "null", "-"],
            input=annexb, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        score = 0
        for line in p.stderr.splitlines():
            m = CONCEAL.search(line)
            if m:
                score += int(m.group(1))
            elif b"[h264" in line and b"error" in line.lower():
                score += 1
        return score


def to_annexb(sample, nls, partial=False):
    """Convert one length-prefixed sample to Annex-B. Returns (bytes, broken).

    With partial=True a NAL cut short by the end of ``sample`` is kept.
    """
    out = bytearray()
    p = 0
    n = len(sample)
    while p + nls <= n:
        ln = int.from_bytes(sample[p:p + nls], "big")
        p += nls
        if ln == 0 or p + ln > n:
            if partial and ln and p < n:
                out += START + sample[p:]
            return bytes(out), True
        out += START + sample[p:p + ln]
        p += ln
    return bytes(out), p != n
