"""Minimal ISO-BMFF (MP4/MOV) parser: just enough to rebuild a sample map.

A GoPro HERO3 file looks like ``ftyp | moov | free | mdat``: the index (moov)
sits at the front, so when the header clusters survive we know the exact
offset and size of every audio and video sample in the file.
"""

import struct
from dataclasses import dataclass, field

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta"}

KIND_VIDEO = "video"
KIND_AUDIO = "audio"
KIND_OTHER = "other"


class MP4Error(Exception):
    pass


@dataclass
class Track:
    handler: bytes = b""
    codec: bytes = b""
    width: int = 0
    height: int = 0
    timescale: int = 0
    duration: int = 0
    nal_length_size: int = 4
    sizes: list = field(default_factory=list)
    chunk_offsets: list = field(default_factory=list)
    stsc: list = field(default_factory=list)
    sync: set = None  # 0-based indexes of key frames (None: every sample)
    sps: list = field(default_factory=list)
    pps: list = field(default_factory=list)

    @property
    def kind(self):
        if self.handler == b"vide" and self.codec in (b"avc1", b"avc3"):
            return KIND_VIDEO
        if self.handler == b"soun":
            return KIND_AUDIO
        return KIND_OTHER

    def sample_offsets(self):
        """Absolute file offset of every sample (stsc + stco + stsz)."""
        out = []
        n_chunks = len(self.chunk_offsets)
        if not self.stsc:
            return out
        sample = 0
        for i, (first_chunk, per_chunk, _desc) in enumerate(self.stsc):
            last_chunk = self.stsc[i + 1][0] - 1 if i + 1 < len(self.stsc) else n_chunks
            for chunk in range(first_chunk, last_chunk + 1):
                if chunk - 1 >= n_chunks:
                    break
                pos = self.chunk_offsets[chunk - 1]
                for _ in range(per_chunk):
                    if sample >= len(self.sizes):
                        return out
                    out.append(pos)
                    pos += self.sizes[sample]
                    sample += 1
        return out


@dataclass
class Movie:
    creation_time: int = 0  # seconds since 1904-01-01
    timescale: int = 0
    duration: int = 0
    tracks: list = field(default_factory=list)
    moov_end: int = 0  # file offset where the moov box ends
    mdat_start: int = 0  # file offset of the first mdat payload byte
    mdat_end: int = 0  # file offset just past the mdat payload (0 if unknown)
    brand: bytes = b""

    @property
    def video(self):
        for t in self.tracks:
            if t.kind == KIND_VIDEO:
                return t
        return None

    def samples(self):
        """All samples sorted by file offset: list of (offset, size, kind, track)."""
        out = []
        for ti, t in enumerate(self.tracks):
            kind = t.kind
            for off, size in zip(t.sample_offsets(), t.sizes):
                if size:
                    out.append((off, size, kind, ti))
        out.sort()
        return out

    def file_size(self):
        end = max((o + s for o, s, _k, _t in self.samples()), default=self.mdat_start)
        if self.mdat_end and self.mdat_end >= end:
            # trust the declared mdat size unless it is absurdly larger than the data
            if self.mdat_end - end < 64 * 1024 * 1024:
                return self.mdat_end
        return end

    @property
    def duration_s(self):
        return self.duration / self.timescale if self.timescale else 0.0


def _boxes(data, start, end):
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack_from(">I4s", data, pos)
        hdr = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            raise MP4Error("bad box %r at %d" % (typ, pos))
        yield typ, pos + hdr, pos + size
        pos += size


def _full(data, pos):
    return data[pos], pos + 4


def _parse_avcc(b, t):
    try:
        p = 5
        for target in (t.sps, t.pps):
            n = b[p] & 0x1F if target is t.sps else b[p]
            p += 1
            for _ in range(n):
                ln = struct.unpack_from(">H", b, p)[0]
                target.append(bytes(b[p + 2:p + 2 + ln]))
                p += 2 + ln
    except (IndexError, struct.error):
        pass


def _parse_trak(data, start, end):
    t = Track()
    stack = [(start, end)]
    while stack:
        s, e = stack.pop()
        for typ, ps, pe in _boxes(data, s, e):
            if typ in CONTAINERS:
                stack.append((ps, pe))
            elif typ == b"tkhd":
                t.width = struct.unpack_from(">I", data, pe - 8)[0] >> 16
                t.height = struct.unpack_from(">I", data, pe - 4)[0] >> 16
            elif typ == b"mdhd":
                ver, p = _full(data, ps)
                if ver == 1:
                    t.timescale, t.duration = struct.unpack_from(">IQ", data, p + 16)
                else:
                    t.timescale, t.duration = struct.unpack_from(">II", data, p + 8)
            elif typ == b"hdlr":
                t.handler = data[ps + 8:ps + 12]
            elif typ == b"stsd":
                n = struct.unpack_from(">I", data, ps + 4)[0]
                if n:
                    esize, t.codec = struct.unpack_from(">I4s", data, ps + 8)
                    entry = data[ps + 8:ps + 8 + esize]
                    i = entry.find(b"avcC")
                    if i >= 0 and i + 9 <= len(entry):
                        t.nal_length_size = (entry[i + 8] & 3) + 1
                        _parse_avcc(entry[i + 4:], t)
            elif typ == b"stsz":
                fixed, count = struct.unpack_from(">II", data, ps + 4)
                if fixed:
                    t.sizes = [fixed] * count
                else:
                    t.sizes = list(struct.unpack_from(">%dI" % count, data, ps + 12))
            elif typ == b"stsc":
                count = struct.unpack_from(">I", data, ps + 4)[0]
                raw = struct.unpack_from(">%dI" % (3 * count), data, ps + 8)
                t.stsc = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
            elif typ == b"stss":
                count = struct.unpack_from(">I", data, ps + 4)[0]
                t.sync = {n - 1 for n in struct.unpack_from(">%dI" % count, data, ps + 8)}
            elif typ == b"stco":
                count = struct.unpack_from(">I", data, ps + 4)[0]
                t.chunk_offsets = list(struct.unpack_from(">%dI" % count, data, ps + 8))
            elif typ == b"co64":
                count = struct.unpack_from(">I", data, ps + 4)[0]
                t.chunk_offsets = list(struct.unpack_from(">%dQ" % count, data, ps + 8))
    return t


def parse_header(read, max_header=64 * 1024 * 1024):
    """Parse the front of an MP4 whose bytes are returned by ``read(pos, n)``.

    Walks the top-level boxes until the first ``mdat`` and parses ``moov``.
    Raises MP4Error if the file does not start with ftyp or moov is not
    before mdat.
    """
    head = read(0, 16)
    if len(head) < 16 or head[4:8] != b"ftyp":
        raise MP4Error("no ftyp")
    m = Movie(brand=head[8:12])
    pos = 0
    moov = None
    while pos < max_header:
        h = read(pos, 16)
        if len(h) < 8:
            raise MP4Error("truncated header")
        size, typ = struct.unpack_from(">I4s", h, 0)
        hdr = 8
        if size == 1:
            size = struct.unpack_from(">Q", h, 8)[0]
            hdr = 16
        if not all(32 <= c < 127 or c == 0xA9 for c in typ):
            raise MP4Error("garbage box type at %d" % pos)
        if typ == b"mdat":
            m.mdat_start = pos + hdr
            m.mdat_end = pos + size if size >= hdr else 0
            break
        if size < hdr:
            raise MP4Error("bad top-level box size at %d" % pos)
        if typ == b"moov":
            if size > max_header:
                raise MP4Error("moov too large")
            moov = read(pos, size)
            m.moov_end = pos + size
        pos += size
    if moov is None:
        raise MP4Error("moov not found before mdat (moov-at-end files are not supported)")
    if not m.mdat_start:
        raise MP4Error("mdat not found")

    for typ, ps, pe in _boxes(moov, 8, len(moov)):
        if typ == b"mvhd":
            ver, p = _full(moov, ps)
            if ver == 1:
                m.creation_time = struct.unpack_from(">Q", moov, p)[0]
                m.timescale, m.duration = struct.unpack_from(">IQ", moov, p + 16)
            else:
                m.creation_time = struct.unpack_from(">I", moov, p)[0]
                m.timescale, m.duration = struct.unpack_from(">II", moov, p + 8)
        elif typ == b"trak":
            m.tracks.append(_parse_trak(moov, ps, pe))
    if not any(t.sizes for t in m.tracks):
        raise MP4Error("no samples in moov")
    for t in m.tracks:
        offs = t.sample_offsets()
        if len(offs) != len(t.sizes):
            raise MP4Error("inconsistent sample tables")
    return m
