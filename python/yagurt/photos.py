"""Recover JPEG photos (and THM thumbnails) from a raw card image.

A photo is written in one go, so its clusters follow each other on the card
except where the allocator skipped clusters still used by other files. We
follow a photo from its first cluster (the one starting with FF D8 FF) and
check every next cluster against the JPEG syntax: marker segments must have
sane lengths, and in the compressed data every FF byte must be followed by
00, a restart marker in sequence (RST0..RST7) or the end of image (FF D9).
Video data, random data and empty clusters fail that check within a few
hundred bytes.

Pass 1 takes each photo's clusters strictly in a row until the check fails
or the photo ends. Pass 2 continues the photos that did not end, skipping
clusters that videos or other photos claimed in pass 1, and looks a little
further ahead for the next cluster that passes the check.
"""

import copy
import os
import struct
from concurrent.futures import ThreadPoolExecutor

from .image import FLAG_HEADER, FLAG_SIGNATURE, FLAG_ZERO

WINDOW = 2048  # free clusters to try before giving up on the next piece
MAX_SIZE = 64 << 20  # no camera photo is bigger than this
ZERO_RUN = bytes(64)  # never inside compressed JPEG data

# markers followed by a length field
_SEGMENT = set(range(0xC0, 0xD0)) | {0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF} | set(range(0xE0, 0xFF))
# what may come between scans (progressive JPEG), with their longest valid length
_BETWEEN_SCANS = {0xC4: 4 * (17 + 256) + 2, 0xDA: 2 + 1 + 4 * 2 + 3, 0xDB: 4 * 129 + 2,
                  0xDC: 4, 0xDD: 4}
_MAX_LEN = {m: 2 + 6 + 3 * 4 for m in range(0xC0, 0xD0)}  # SOFn: at most 4 components
_MAX_LEN.update(_BETWEEN_SCANS)


class Parser:
    """Resumable JPEG syntax checker, fed one cluster at a time."""

    def __init__(self):
        self.mode = "marker"  # marker | code | len | skip | entropy | entropy_ff
        self.marker = 0
        self.lenbuf = b""
        self.remaining = 0
        self.payload = b""  # DRI payload being read
        self.restart = 0  # restart interval, 0 = none
        self.next_rst = 0
        self.scans = 0
        self.checks = 0  # FF pairs checked in the last fed buffer

    def feed(self, buf):
        """Check ``buf``. Returns ("ok", None), ("end", n) when the image ends
        after n bytes of buf, or ("bad", i) at the first byte that cannot be
        part of this JPEG. The parser state is only meaningful after "ok"."""
        i, n = 0, len(buf)
        self.checks = 0
        while i < n:
            m = self.mode
            if m == "entropy":
                j = buf.find(b"\xff", i)
                seg = buf[i:] if j < 0 else buf[i:j]
                if ZERO_RUN in seg:
                    return "bad", i
                if j < 0:
                    return "ok", None
                i = j + 1
                self.mode = "entropy_ff"
            elif m == "entropy_ff":
                b = buf[i]
                i += 1
                self.checks += 1
                if b == 0x00:
                    self.mode = "entropy"
                elif b == 0xFF:
                    pass  # fill byte
                elif 0xD0 <= b <= 0xD7:
                    if not self.restart or b - 0xD0 != self.next_rst:
                        return "bad", i - 1
                    self.next_rst = (self.next_rst + 1) & 7
                    self.mode = "entropy"
                elif b == 0xD9:
                    return "end", i
                elif b in _BETWEEN_SCANS:  # next scan or tables (progressive JPEG)
                    self.marker = b
                    self.mode = "len"
                    self.lenbuf = b""
                else:
                    return "bad", i - 1
            elif m == "marker":
                if buf[i] != 0xFF:
                    return "bad", i
                i += 1
                self.mode = "code"
            elif m == "code":
                b = buf[i]
                i += 1
                if b == 0xFF:
                    continue
                if b == 0xD8 and self.marker == 0:
                    self.marker = b
                    self.mode = "marker"
                elif b == 0xD9 and self.scans:
                    return "end", i
                elif self.marker and (b in _BETWEEN_SCANS if self.scans else b in _SEGMENT):
                    self.marker = b
                    self.mode = "len"
                    self.lenbuf = b""
                else:
                    return "bad", i - 1
            elif m == "len":
                take = buf[i:i + 2 - len(self.lenbuf)]
                self.lenbuf += take
                i += len(take)
                if len(self.lenbuf) == 2:
                    ln = struct.unpack(">H", self.lenbuf)[0]
                    if ln < 2 or ln > _MAX_LEN.get(self.marker, 0xFFFF) or \
                            (self.marker == 0xDD and ln != 4) or (self.marker == 0xDA and ln < 8):
                        return "bad", i - 1
                    self.remaining = ln - 2
                    self.payload = b""
                    self.mode = "skip"
                    if not self.remaining:
                        self._segment_done()
            elif m == "skip":
                take = min(self.remaining, n - i)
                if self.marker == 0xDD:
                    self.payload += buf[i:i + take]
                i += take
                self.remaining -= take
                if not self.remaining:
                    self._segment_done()
        return "ok", None

    def _segment_done(self):
        if self.marker == 0xDD:
            self.restart = struct.unpack(">H", self.payload[:2])[0]
        if self.marker == 0xDA:
            self.scans += 1
            self.next_rst = 0
            self.mode = "entropy"
        else:
            self.mode = "marker"


def jpeg_info(data):
    """(width, height, date 'YYYYmmdd_HHMMSS' or None, make) from a JPEG's headers."""
    w = h = 0
    date = make = None
    p = 2
    while p + 4 <= len(data) and data[p] == 0xFF:
        mk = data[p + 1]
        if mk == 0xFF:
            p += 1
            continue
        ln = struct.unpack_from(">H", data, p + 2)[0]
        seg = data[p + 4:p + 2 + ln]
        if mk == 0xE1 and seg[:6] == b"Exif\x00\x00" and date is None:
            date, make = _exif(seg[6:])
        elif mk in (0xC0, 0xC1, 0xC2) and len(seg) >= 5:
            h, w = struct.unpack_from(">HH", seg, 1)
        if mk == 0xDA:
            break
        p += 2 + ln
    if not w:  # headers we could not walk: look for a plausible SOF0/SOF2
        i = data.find(b"\xff\xc0")
        while i >= 0 and not w:
            seg = data[i + 2:i + 12]
            if len(seg) == 10 and seg[2] == 8 and seg[7] in (1, 3) and \
                    struct.unpack_from(">H", seg)[0] == 8 + 3 * seg[7]:
                h, w = struct.unpack_from(">HH", seg, 3)
            i = data.find(b"\xff\xc0", i + 2)
    return w, h, date, make


def _exif(t):
    """DateTimeOriginal (or DateTime) and Make from a TIFF/EXIF block."""
    if len(t) < 8 or t[:2] not in (b"II", b"MM"):
        return None, None
    e = "<" if t[:2] == b"II" else ">"

    def ifd(off):
        out = {}
        if off + 2 > len(t):
            return out
        cnt = struct.unpack_from(e + "H", t, off)[0]
        for k in range(cnt):
            q = off + 2 + 12 * k
            if q + 12 > len(t):
                break
            tag, typ, num = struct.unpack_from(e + "HHI", t, q)
            if typ == 2:  # ASCII
                vo = q + 8 if num <= 4 else struct.unpack_from(e + "I", t, q + 8)[0]
                out[tag] = t[vo:vo + num].split(b"\x00")[0].decode("ascii", "replace")
            elif typ == 4:
                out[tag] = struct.unpack_from(e + "I", t, q + 8)[0]
        return out

    try:
        ifd0 = ifd(struct.unpack_from(e + "I", t, 4)[0])
        sub = ifd(ifd0[0x8769]) if isinstance(ifd0.get(0x8769), int) else {}
    except struct.error:
        return None, None
    raw = sub.get(0x9003) or ifd0.get(0x0132)
    date = None
    if raw and len(raw) >= 19 and raw[:4].isdigit() and raw[4] == ":":
        date = raw[0:4] + raw[5:7] + raw[8:10] + "_" + raw[11:13] + raw[14:16] + raw[17:19]
    return date, (ifd0.get(0x010F) or "").strip() or None


class Photo:
    def __init__(self, img, k):
        self.img = img
        self.start = k
        self.offset = img.cluster_pos(k)
        self.clusters = []
        self.tail = 0  # bytes used in the last cluster
        self.parser = Parser()
        self.complete = False
        self.skips = 0
        self.rejected = set()  # clusters whose JPEG end did not decode
        self.before_last = None
        self.damage = None  # decoder errors in the final file, when checked
        self.width = self.height = 0
        self.date = self.make = None

    def walk(self, free, skip, claim, stop=lambda q: False, checker=None):
        """Follow the photo from where it stopped.

        ``free(q)`` says whether q may hold photo data and ``stop(q)`` ends
        the walk at q. With ``skip``, free clusters that fail the check are
        stepped over, up to WINDOW of them. An end of image found in the
        first bytes of a cluster is weak evidence (FF D9 turns up by chance),
        so while skipping a cluster that passes in full is preferred. With a
        ``checker``, a photo is only ended where the result decodes cleanly.
        """
        img = self.img
        C, N = img.cluster_size, img.n_clusters
        if not self.clusters:
            res, n = self.parser.feed(img.cluster(self.start))
            if res == "bad":
                return
            self._take(self.start, res, n, claim)
        while not self.complete and len(self.clusters) * C < MAX_SIZE:
            q = self.clusters[-1]
            tried = 0
            choice = fallback = None
            for _ in range(N - 1):
                q = (q + 1) % N
                if stop(q):
                    break
                if not free(q) or q in self.rejected:
                    continue
                trial = copy.copy(self.parser)
                res, n = trial.feed(img.cluster(q))
                if res == "ok":
                    choice = (q, trial, res, n, tried)
                elif res == "end" and checker:
                    if self._verified(checker, q, n):
                        choice = (q, trial, res, n, tried)
                    else:
                        self.rejected.add(q)
                        if not skip or len(self.rejected) > 16:
                            return
                elif res == "end":
                    if skip and trial.checks < 4:
                        fallback = fallback or (q, trial, res, n, tried)
                    else:
                        choice = (q, trial, res, n, tried)
                if choice:
                    break
                tried += 1
                if not skip or tried >= WINDOW:
                    break
            choice = choice or fallback
            if choice is None:
                return
            q, trial, res, n, tried = choice
            self.skips += tried > 0
            self.before_last = (self.parser, self.tail, self.skips - (tried > 0))
            self.parser = trial
            self._take(q, res, n, claim)

    def undo_last(self):
        """Forget the last cluster (its end of image did not decode)."""
        q = self.clusters.pop()
        self.rejected.add(q)
        self.parser, self.tail, self.skips = self.before_last
        self.complete = False
        return q

    def _verified(self, checker, q=None, n=None):
        """Does the photo, ended in cluster q after n bytes, decode cleanly?"""
        if not checker:
            return True
        data = self._assemble(self.clusters + ([q] if q is not None else []),
                              n if q is not None else self.tail)
        self.damage = checker.jpeg_errors(data)
        return self.damage == 0

    def _take(self, q, res, n, claim):
        self.clusters.append(q)
        claim(q)
        self.tail = self.img.cluster_size
        if res == "end":
            self.complete = True
            self.tail = n

    def _assemble(self, clusters, tail):
        out = bytearray()
        for i, q in enumerate(clusters):
            c = self.img.cluster(q)
            out += c[:tail] if i == len(clusters) - 1 else c
        return out

    def data(self):
        out = self._assemble(self.clusters, self.tail)
        if not self.complete:
            out += b"\xff\xd9"  # let viewers show the part we have
        return bytes(out)

    def describe(self):
        # camera headers can hold large previews, so parse the whole file
        self.width, self.height, self.date, self.make = jpeg_info(self.data())

    @property
    def is_thumbnail(self):
        if self.width and self.height:
            return self.width * self.height <= 320 * 240
        return len(self.clusters) <= 1

    def name(self):
        prefix = "GOPRO" if (self.make or "").lower().startswith("gopro") else "JPEG"
        stamp = self.date or "unknown_date"
        return "%s_%s_%dx%d_at_0x%x.JPG" % (prefix, stamp, self.width, self.height, self.offset)

    def cluster_map(self):
        runs = []
        for i, q in enumerate(self.clusters):
            if runs and runs[-1][1] + runs[-1][2] == q:
                runs[-1][2] += 1
            else:
                runs.append([i, q, 1])
        return runs

    def report(self):
        return {"file": self.name(), "offset": self.offset, "clusters": len(self.clusters),
                "size": len(self.data()) - (0 if self.complete else 2),
                "complete": self.complete, "skips": self.skips, "decode_errors": self.damage,
                "cluster_map": self.cluster_map()}


def recover_photos(img, flags, videos=(), checker=None, log=print):
    """Find and rebuild every JPEG on the card.

    ``videos`` are rebuilt video candidates: their clusters are never used
    for photos. With a ``checker`` (ffmpeg), a photo only ends where it
    decodes without errors. Returns the list of Photo objects that have at least their
    first cluster."""
    N = img.n_clusters
    taken = bytearray(N)
    for k in range(N):
        if flags[k] & (FLAG_ZERO | FLAG_HEADER | FLAG_SIGNATURE) or k < img.first_cluster:
            taken[k] = 1
    for v in videos:
        for q in v.mapping:
            if q is not None:
                taken[q] = 1
    starts = [k for k in range(N) if flags[k] & FLAG_SIGNATURE
              and not flags[k] & FLAG_HEADER and img.cluster(k)[:3] == b"\xff\xd8\xff"]
    photos = [Photo(img, k) for k in starts]

    # pass 1: strictly consecutive free clusters, up to the next file start
    owner = set()
    for ph in photos:
        ph.walk(lambda q: not taken[q] and q not in owner, False, owner.add,
                stop=lambda q: flags[q] & FLAG_SIGNATURE)
    photos = [ph for ph in photos if ph.clusters]
    if checker:  # an end of image that does not decode was found by chance
        ended = [ph for ph in photos if ph.complete]
        with ThreadPoolExecutor(os.cpu_count() or 2) as pool:
            damage = list(pool.map(lambda ph: checker.jpeg_errors(ph.data()), ended))
        for ph, d in zip(ended, damage):
            ph.damage = d
            if d and len(ph.clusters) > 1:
                owner.discard(ph.undo_last())
    for q in owner:
        taken[q] = 1
    # pass 2: continue unfinished photos past clusters claimed by others
    unfinished = [ph for ph in photos if not ph.complete]

    def claim(q):
        taken[q] = 1
    for ph in unfinished:
        ph.walk(lambda q: not taken[q], True, claim, checker=checker)
    for ph in photos:
        ph.describe()
    done = sum(ph.complete for ph in photos)
    log("  %d JPEG starts, %d complete photos, %d cut short (%d needed skipping)" % (
        len(photos), done, len(photos) - done, sum(1 for ph in photos if ph.skips)))
    return photos
