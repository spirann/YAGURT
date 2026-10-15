"""Access to the raw card image: cluster geometry and a single scanning pass."""

import mmap
import os
import re
import struct
import sys
from collections import Counter

FLAG_ZERO = 1  # cluster is all 0x00 or all 0xFF: never part of a video
FLAG_SIGNATURE = 2  # cluster starts another file (ftyp / JPEG): never mid-video
FLAG_HEADER = 4  # cluster belongs to a parsed MP4 header
FLAG_JPEG = 8  # cluster looks like compressed JPEG data: never part of a video

# In compressed JPEG data FF is only followed by 00, a restart marker, FF or
# the end of image; any other pair shows a cluster is not JPEG data.
_NOT_JPEG = re.compile(rb"\xff[^\x00\xd0-\xd9\xff]")
_ZERO_RUN = bytes(64)


def looks_like_jpeg_data(buf, start=0, end=None):
    end = len(buf) if end is None else end
    if _NOT_JPEG.search(buf, start, end) is not None:
        return False
    chunk = buf[start:end]
    return _ZERO_RUN not in chunk and b"\xff" in chunk


class Image:
    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        self._f = open(path, "rb")
        self.mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.cluster_size = 0
        self.data_offset = 0
        self.first_cluster = 0
        self.geometry_source = ""

    def close(self):
        self.mm.close()
        self._f.close()

    # --- geometry -----------------------------------------------------------

    def set_geometry(self, cluster_size, data_offset, source):
        self.cluster_size = cluster_size
        self.data_offset = data_offset % cluster_size
        # clusters before the file system's data area (boot sector, FATs) hold no file data
        self.first_cluster = data_offset // cluster_size
        self.geometry_source = source

    @property
    def n_clusters(self):
        return (self.size - self.data_offset) // self.cluster_size

    def cluster_pos(self, k):
        return self.data_offset + k * self.cluster_size

    def cluster(self, k):
        p = self.cluster_pos(k)
        return self.mm[p:p + self.cluster_size]

    def detect_geometry(self):
        """Read cluster size and data-area start from the (new) file system.

        An in-camera format writes a fresh FAT32/exFAT with the same geometry,
        so the boot sector still tells us where clusters start.
        """
        for base in self._partition_starts():
            g = self._boot_sector(base)
            if g:
                return g
        return None

    def _partition_starts(self):
        starts = [0]
        mbr = self.mm[0:512]
        if len(mbr) == 512 and mbr[510:512] == b"\x55\xaa":
            for i in range(4):
                e = mbr[446 + 16 * i:462 + 16 * i]
                ptype = e[4]
                lba = struct.unpack_from("<I", e, 8)[0]
                if ptype and lba and lba * 512 < self.size:
                    starts.append(lba * 512)
        return starts

    def _boot_sector(self, base):
        bs = self.mm[base:base + 512]
        if len(bs) < 512 or bs[510:512] != b"\x55\xaa":
            return None
        if bs[3:11] == b"EXFAT   ":
            bps = 1 << bs[108]
            spc = 1 << bs[109]
            heap = struct.unpack_from("<I", bs, 88)[0]
            return bps * spc, base + heap * bps, "exFAT boot sector at 0x%x" % base
        bps = struct.unpack_from("<H", bs, 11)[0]
        spc = bs[13]
        if bps not in (512, 1024, 2048, 4096) or spc == 0 or spc & (spc - 1):
            return None
        reserved = struct.unpack_from("<H", bs, 14)[0]
        nfats = bs[16]
        root_entries = struct.unpack_from("<H", bs, 17)[0]
        fatsz = struct.unpack_from("<H", bs, 22)[0] or struct.unpack_from("<I", bs, 36)[0]
        if not reserved or not nfats or not fatsz:
            return None
        root_sectors = (root_entries * 32 + bps - 1) // bps
        data = base + (reserved + nfats * fatsz + root_sectors) * bps
        kind = "FAT32" if root_entries == 0 else "FAT12/16"
        return bps * spc, data, "%s boot sector at 0x%x" % (kind, base)

    def guess_alignment(self, cluster_size, limit=200000):
        """Infer cluster alignment from where ``ftyp`` boxes start."""
        hits = Counter()
        pos = 0
        mm = self.mm
        n = 0
        while n < limit:
            i = mm.find(b"ftyp", pos)
            if i < 0:
                break
            start = i - 4
            if start >= 0 and start % 512 == 0:
                hits[start % cluster_size] += 1
                n += 1
            pos = i + 4
        if not hits:
            return 0, 0
        off, count = hits.most_common(1)[0]
        return off, count

    # --- scanning -----------------------------------------------------------

    def scan(self, progress=True):
        """One pass over every cluster.

        Returns (flags, mp4_starts) where flags is a bytearray with one entry
        per cluster and mp4_starts lists cluster indexes beginning with ftyp.
        """
        n = self.n_clusters
        c = self.cluster_size
        flags = bytearray(n)
        starts = []
        zero = bytes(c)
        ones = b"\xff" * c
        mm = self.mm
        step = max(1, n // 100)
        for k in range(n):
            p = self.data_offset + k * c
            head = mm[p:p + 12]
            if head[4:8] == b"ftyp":
                flags[k] |= FLAG_SIGNATURE
                starts.append(k)
            elif head[:3] == b"\xff\xd8\xff":
                flags[k] |= FLAG_SIGNATURE
            elif head[:8] == zero[:8] or head[:8] == ones[:8]:
                buf = mm[p:p + c]
                if buf == zero or buf == ones:
                    flags[k] |= FLAG_ZERO
            elif looks_like_jpeg_data(mm, p, p + c):
                flags[k] |= FLAG_JPEG
            if progress and k % step == 0:
                sys.stderr.write("\rscanning image: %3d%%" % (100 * k // n))
                sys.stderr.flush()
        if progress:
            sys.stderr.write("\rscanning image: 100%\n")
        return flags, starts
