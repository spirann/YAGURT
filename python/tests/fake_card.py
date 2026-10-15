"""Build a fake, formatted GoPro SD card image for testing.

The image mimics what a HERO3 leaves behind: a FAT32 volume whose clusters
hold an MP4 and its LRV recorded simultaneously (interleaved cluster
bursts), skipping clusters used by older files, with the headers written
after the data, plus a THM thumbnail and a photo. Needs ffmpeg with libx264.
"""

import os
import random
import struct
import subprocess


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)

CLUSTER = 0x8000
PART_LBA = 8192
RESERVED = 32
FATSZ = 1000


def make_video(path, size, bitrate, seconds, seed):
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=%s:rate=30" % size,
        "-f", "lavfi", "-i", "sine=frequency=%d:sample_rate=48000" % (300 + seed),
        "-t", str(seconds), "-vf", "noise=alls=30:allf=t:all_seed=%d" % seed,
        "-c:v", "libx264", "-x264-params", "aud=1:keyint=15", "-b:v", bitrate,
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", "-b:a", "128k",
        "-movflags", "+faststart", path], check=True)


def exif_app1(date, make="GoPro"):
    """A minimal EXIF APP1 segment with Make and DateTimeOriginal."""
    mk = make.encode() + b"\x00"
    dt = date.encode() + b"\x00"
    ifd0 = 8
    exif_ifd = ifd0 + 2 + 2 * 12 + 4
    data = exif_ifd + 2 + 12 + 4
    t = b"MM\x00\x2a" + struct.pack(">I", ifd0)
    t += struct.pack(">H", 2)
    t += struct.pack(">HHII", 0x010F, 2, len(mk), data)
    t += struct.pack(">HHII", 0x8769, 4, 1, exif_ifd) + bytes(4)
    t += struct.pack(">H", 1) + struct.pack(">HHII", 0x9003, 2, len(dt), data + len(mk)) + bytes(4)
    t += mk + dt
    body = b"Exif\x00\x00" + t
    return b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body


def make_photo(path, size, seed, date, preview=False):
    """A real JPEG with EXIF, as a camera would write it. With ``preview``,
    large APP2 segments (like a camera's embedded preview) push the frame
    header past the first 128 KiB."""
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=%s" % size,
        "-vf", "noise=alls=40:allf=t:all_seed=%d" % seed, "-frames:v", "1",
        "-q:v", "2", "-pix_fmt", "yuvj420p", path], check=True)
    d = _read(path)
    d = d[:2] + exif_app1(date) + d[2:]
    if preview:
        rnd = random.Random(seed)
        app2 = b"\xff\xe2" + struct.pack(">H", 60002) + bytes(rnd.randrange(255) for _ in range(60000))
        d = d[:2] + app2 * 3 + d[2:]
    _write(path, d)
    return d


def _boxes(d, s, e):
    while s + 8 <= e:
        size, typ = struct.unpack_from(">I4s", d, s)
        yield typ, s, size
        s += size


def pad_header_to_cluster(data):
    """Grow the header with a 'free' box so mdat payload is cluster aligned (like GoPro)."""
    d = bytearray(data)
    mdat = d.find(b"mdat") - 4
    target = ((mdat + 8 + 8 + CLUSTER - 1) // CLUSTER) * CLUSTER
    pad = target - (mdat + 8)
    # shift all chunk offsets
    stack = [(0, len(d))]
    while stack:
        s, e = stack.pop()
        for typ, p, size in _boxes(d, s, e):
            if typ in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
                stack.append((p + 8, p + size))
            elif typ == b"stco":
                n = struct.unpack_from(">I", d, p + 12)[0]
                for i in range(n):
                    o = p + 16 + 4 * i
                    struct.pack_into(">I", d, o, struct.unpack_from(">I", d, o)[0] + pad)
            if typ == b"mdat":
                break
    free = struct.pack(">I4s", pad, b"free") + bytes(pad - 8)
    return bytes(d[:mdat]) + free + bytes(d[mdat:])


def build(workdir, n_clusters=1200, junk_ratio=0.15, wrap=True, seed=1, video_junk=False,
          photos=False):
    """Write card.img in workdir. Returns (image, mp4, lrv, truth) where truth
    maps each file's logical cluster to its cluster index counted from the
    start of the FAT data area.

    junk_ratio: share of clusters held by older files while recording.
    video_junk: fill those clusters with another H.264 recording (whose header
    is gone) instead of random bytes; the hardest case to tell apart.
    photos: also write real JPEGs: an older photo, a newer one allocated
    around it and around old data, and one taken while the video recorded.
    Their bytes are returned in truth["photos"].
    """
    rnd = random.Random(seed)
    os.makedirs(workdir, exist_ok=True)
    mp4p = os.path.join(workdir, "orig.mp4")
    lrvp = os.path.join(workdir, "orig_lrv.mp4")
    make_video(mp4p, "960x540", "8M", 6, seed)
    make_video(lrvp, "432x240", "700k", 6, seed + 100)
    junk_src = None
    if video_junk:
        jp = os.path.join(workdir, "old.mp4")
        make_video(jp, "960x540", "8M", 8, seed + 200)
        jd = _read(jp)
        jd = jd[jd.find(b"mdat") + 4:]
        junk_src = [jd[i:i + CLUSTER] for i in range(0, len(jd) - CLUSTER, CLUSTER)]
    mp4 = pad_header_to_cluster(_read(mp4p))
    _write(mp4p, mp4)
    lrv = _read(lrvp)

    data_start = PART_LBA * 512 + (RESERVED + 2 * FATSZ) * 512
    img = bytearray(data_start + n_clusters * CLUSTER)
    # MBR + FAT32 boot sector of the "new" (post-format) file system
    struct.pack_into("<B3xBxxxII", img, 446, 0x80, 0x0C, PART_LBA, n_clusters * 64 + 4096)
    img[510:512] = b"\x55\xaa"
    bs = PART_LBA * 512
    img[bs:bs + 3] = b"\xeb\x58\x90"
    img[bs + 3:bs + 11] = b"MSDOS5.0"
    struct.pack_into("<HBHBH", img, bs + 11, 512, CLUSTER // 512, RESERVED, 2, 0)
    struct.pack_into("<I", img, bs + 36, FATSZ)
    img[bs + 82:bs + 90] = b"FAT32   "
    img[bs + 510:bs + 512] = b"\x55\xaa"

    def put(k, chunk):
        p = data_start + k * CLUSTER
        img[p:p + len(chunk)] = chunk

    occupied = [False] * n_clusters
    # older files that were on the card while recording: random-looking data
    k = 0
    while k < n_clusters:
        if rnd.random() < junk_ratio / 4:
            run = rnd.randint(1, 8)
            for i in range(k, min(n_clusters, k + run)):
                occupied[i] = True
                put(i, rnd.choice(junk_src) if junk_src else rnd.randbytes(CLUSTER))
            k += run
        k += 1
    def chunks(data):
        return [data[i:i + CLUSTER] for i in range(0, len(data), CLUSTER)]

    truth = {"mp4": {}, "lrv": {}, "photos": []}
    if photos:
        p0, p1, p2 = (make_photo(os.path.join(workdir, "p%d.jpg" % i), sz, seed + i, dt, preview=i == 1)
                      for i, (sz, dt) in enumerate([
                          ("1280x960", "2014:12:31 16:13:56"), ("1600x1200", "2014:12:31 16:20:01"),
                          ("1280x960", "2014:12:31 16:25:30")]))
        truth["photos"] = [p0, p1, p2]
        p2_chunks = chunks(p2)
    else:
        p0 = b"\xff\xd8\xff\xe1\x00\x10Exif\x00\x00MM\x00\x2a" + rnd.randbytes(3 * CLUSTER)
        p2_chunks = []
    for i, c in enumerate(chunks(p0)):
        occupied[700 + i] = True
        put(700 + i, c)

    mc, lc = chunks(mp4), chunks(lrv)
    mh = (mp4.find(b"mdat") + 4 + CLUSTER - 1) // CLUSTER  # header clusters
    lh = 1  # the LRV moov fits in its first cluster; its mdat starts inside it
    cursor = [n_clusters - 150 if wrap else 10]

    def alloc():
        for _ in range(n_clusters):
            k = cursor[0]
            cursor[0] = (k + 1) % n_clusters
            if not occupied[k]:
                occupied[k] = True
                return k
        raise RuntimeError("card full")

    if photos:  # newer photo, allocated over old data and the older photo
        cursor[0] = 690
        for c in chunks(p1):
            put(alloc(), c)
        cursor[0] = n_clusters - 150 if wrap else 10
    mi, li = mh, lh
    while mi < len(mc) or li < len(lc):
        if p2_chunks and mi > len(mc) // 3:  # photo taken while recording
            for _ in range(rnd.randint(1, 3)):
                if p2_chunks:
                    put(alloc(), p2_chunks.pop(0))
        for _ in range(rnd.randint(2, 8)):
            if mi < len(mc):
                k = alloc()
                put(k, mc[mi])
                truth["mp4"][mi] = k
                mi += 1
        for _ in range(rnd.randint(1, 2)):
            if li < len(lc):
                k = alloc()
                put(k, lc[li])
                truth["lrv"][li] = k
                li += 1
    thm = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + rnd.randbytes(5000)
    put(alloc(), thm)
    for i in range(mh):
        k = alloc()
        put(k, mc[i])
        truth["mp4"][i] = k
    for i in range(lh):
        k = alloc()
        put(k, lc[i])
        truth["lrv"][i] = k

    imgp = os.path.join(workdir, "card.img")
    _write(imgp, img)
    return imgp, mp4p, lrvp, truth


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1])[0])
