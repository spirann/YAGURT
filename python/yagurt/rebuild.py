"""Rebuild fragmented GoPro MP4/LRV files from a raw card image.

Why the files recovered by other tools do not play
---------------------------------------------------
A GoPro HERO3 records the high-resolution MP4 and the low-resolution LRV at
the same time. The FAT allocator hands out free clusters in increasing order
to both files, so on the card their clusters are interleaved
(MP4 MP4 LRV MP4 MP4 MP4 LRV ...), and any cluster still used by an older
file is skipped. After a format the cluster chains are gone; copying the
data as one contiguous block (what carvers do) mixes the two streams and the
result does not decode.

How this module puts them back together
---------------------------------------
The header (ftyp + moov) at the front of each file survives, and its sample
tables (stsz/stsc/stco) give the exact offset and size of every video and
audio sample in the file. H.264 samples are a chain of length-prefixed NAL
units, so for every logical cluster of the file we know where some NAL
headers must sit and what they must look like (a plausible length that
stays inside the sample, a valid NAL type byte). That lets us test whether a
given physical cluster is the next piece of the file.

We walk the file cluster by cluster, always trying the next free physical
cluster first (that is what the allocator did), skip clusters that fail the
test (they belong to the other stream or an older file), and when we lose
the trail completely we search the whole image for the cluster that holds
the next video frame. A second round excludes clusters that other files
claimed with certainty, which removes most interleaving ambiguity.

Clusters in the middle of a large frame (key frames span many clusters)
carry no checkable structure. When the free clusters between two checked
ones are more than needed, we let ffmpeg decode the frame for candidate
placements: the right one decodes with no concealed macroblocks.
"""

import bisect
import time
from datetime import datetime, timedelta, timezone

from . import mp4
from .decodecheck import to_annexb
from .image import FLAG_HEADER, FLAG_JPEG, FLAG_SIGNATURE, FLAG_ZERO

ST_MISSING = 0
ST_HEADER = 1
ST_VALID = 2  # cluster content checked against expected NAL headers
ST_INFERRED = 3  # no checkable structure in it; placed by allocation order

WINDOW = 4096  # how far ahead to look for the next piece before a full search
MIN_STARTS = 2  # frame starts that must line up to accept a full-image match

NAL_TYPES = frozenset((1, 5, 6, 7, 8, 9, 10, 11, 12))


def nal_plausible(b, nls, remaining):
    """Is ``b`` (nls length bytes + NAL header byte) a believable NAL start?"""
    length = int.from_bytes(b[:nls], "big")
    if length == 0 or nls + length > remaining:
        return False
    t = b[nls]
    if t & 0x80:
        return False
    typ = t & 0x1F
    ref = (t >> 5) & 3
    if typ not in NAL_TYPES:
        return False
    if typ in (5, 7, 8) and ref == 0:
        return False
    if typ in (6, 9, 10, 11, 12) and ref != 0:
        return False
    if typ == 9 and length != 2:
        return False
    if typ in (7, 8) and length > 1024:
        return False
    return True


class Candidate:
    """One MP4/LRV whose header was found at physical cluster ``hcl``."""

    def __init__(self, img, hcl):
        self.img = img
        self.hcl = hcl
        self.offset = img.cluster_pos(hcl)
        self.error = None
        self.movie = None
        try:
            self.movie = mp4.parse_header(self._read_contig)
        except (mp4.MP4Error, ValueError, IndexError) as e:  # struct errors are ValueError
            self.error = str(e)
            return
        C = img.cluster_size
        self.file_size = self.movie.file_size()
        self.n = (self.file_size + C - 1) // C
        self.h = max(1, min(self.n, (self.movie.moov_end + C - 1) // C))
        if hcl + self.h > img.n_clusters:
            self.error = "header runs past the end of the image"
            return
        v = self.movie.video
        self.nls = v.nal_length_size if v else 4
        self.hlen = self.nls + 1
        self.vid = []
        if v:
            self.vid = sorted((o, o + s) for o, s in zip(v.sample_offsets(), v.sizes) if s)
        self.vid_starts = [o for o, _e in self.vid]
        self.mapping = []
        self.status = bytearray()
        self.stats = {}

    def _read_contig(self, pos, n):
        p = self.offset + pos
        return self.img.mm[p:p + n]

    # --- descriptive --------------------------------------------------------

    @property
    def ok(self):
        return self.error is None

    def created(self):
        ct = self.movie.creation_time
        if not ct:
            return None
        try:
            return datetime(1904, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=ct)
        except OverflowError:
            return None

    def resolution(self):
        v = self.movie.video
        return (v.width, v.height) if v else (0, 0)

    def name(self):
        w, h = self.resolution()
        d = self.created()
        stamp = d.strftime("%Y%m%d_%H%M%S") if d else "unknown_date"
        ext = "LRV.mp4" if w and w <= 432 else "MP4"
        return "GOPRO_%s_%dx%d_at_0x%x.%s" % (stamp, w, h, self.offset, ext)

    # --- the checker --------------------------------------------------------

    def validate(self, j, q, state, phys):
        """Check physical cluster ``q`` as logical cluster ``j``.

        ``state`` = (next video sample index, next NAL position, sample end).
        ``phys(c)`` gives the physical cluster already chosen for logical
        cluster c < j (or None). Returns (verdict, checked, new_state) where
        verdict is True (structure matched), None (nothing to check) or
        False (mismatch).
        """
        C = self.img.cluster_size
        lo = j * C
        hi = lo + C
        vid = self.vid
        nls = self.nls
        hlen = self.hlen
        si, nal, end = state
        checked = starts = 0
        mm = self.img.mm
        base = self.img.data_offset
        while True:
            fresh = False
            if nal is None or nal >= end:
                if si >= len(vid):
                    break
                nal, end = vid[si]
                if nal + hlen > hi:
                    nal = None
                    break
                si += 1
                fresh = True
            ev = nal
            if ev + hlen > hi:
                break
            if ev >= lo:
                p = base + q * C + (ev - lo)
                b = mm[p:p + hlen]
            else:
                b = self._read_split(ev, j, q, phys)
                if b is None:  # earlier piece unknown: give up on this frame
                    nal = None
                    continue
            if not nal_plausible(b, nls, end - ev):
                self.last_starts = starts
                return False, checked, state
            checked += 1
            starts += fresh
            nal = ev + nls + int.from_bytes(b[:nls], "big")
        # Frame starts come from the moov, so each one matching is independent
        # evidence; NAL positions inside a frame are derived from the data itself.
        self.last_starts = starts
        return (True if checked else None), checked, (si, nal, end)

    def _read_split(self, pos, j, q, phys):
        """Read ``hlen`` logical bytes starting before cluster j."""
        C = self.img.cluster_size
        out = b""
        n = self.hlen
        while n:
            c = pos // C
            pc = q if c == j else phys(c)
            if pc is None:
                return None
            within = pos - c * C
            take = min(n, C - within)
            p = self.img.cluster_pos(pc) + within
            out += self.img.mm[p:p + take]
            pos += take
            n -= take
        return out

    def state_at_frame(self, si):
        return (si, None, 0)

    # --- allocation order helpers -------------------------------------------

    def _next(self, p, blocked, used, extra=()):
        N = self.img.n_clusters
        k = p
        for _ in range(N):
            k += 1
            if k >= N:
                k = 0
            if not blocked[k] and not used[k] and k not in extra:
                return k
        return None

    def _prev(self, p, blocked, used):
        k = p - 1
        while k >= 0:
            if not blocked[k] and not used[k]:
                return k
            k -= 1
        return None

    # --- full-image search --------------------------------------------------

    def _frame_cluster(self, j, si):
        """First logical cluster >= j that holds the whole header of a frame start."""
        C = self.img.cluster_size
        while si < len(self.vid):
            off = self.vid[si][0]
            k = off // C
            if k >= j and off + self.hlen <= (k + 1) * C:
                return k, si
            si += 1
        return None, None

    def _score(self, k, q, si, blocked, used, max_clusters=256, want_starts=3):
        """Follow the file from candidate q for a while; count frame starts that line up."""
        overlay = {k: q}
        mapping = self.mapping
        phys = lambda c: overlay[c] if c in overlay else mapping[c]
        state = self.state_at_frame(si)
        starts = checked = 0
        cur = q
        for t in range(max_clusters):
            c = k + t
            if c >= self.n:
                break
            if t:
                nxt = self._next(cur, blocked, used, overlay.values())
                if nxt is None:
                    break
                verdict, chk, st = self.validate(c, nxt, state, phys)
                if verdict is False:  # skip a few foreign clusters, like the main walk
                    for _ in range(64):
                        nxt = self._next(nxt, blocked, used, overlay.values())
                        if nxt is None:
                            break
                        verdict, chk, st = self.validate(c, nxt, state, phys)
                        if verdict:
                            break
                    if not verdict:
                        break
                cur = nxt
                overlay[c] = cur
            else:
                verdict, chk, st = self.validate(c, cur, state, phys)
                if verdict is False:
                    break
            state = st
            checked += chk
            starts += self.last_starts
            if starts >= want_starts or state[0] >= len(self.vid):
                break
        complete = state[0] >= len(self.vid) and starts >= 1
        return starts, checked, complete

    def search(self, j, si, blocked, used, prefer):
        """Find, anywhere on the card, the cluster holding frame si (or later)."""
        k, si = self._frame_cluster(j, si)
        if k is None:
            return None
        C = self.img.cluster_size
        off, end = self.vid[si]
        within = off - k * C
        nls, hlen = self.nls, self.hlen
        mm = self.img.mm
        base = self.img.data_offset + within
        best = []
        N = self.img.n_clusters
        for q in range(N):
            if blocked[q] or used[q]:
                continue
            p = base + q * C
            if not nal_plausible(mm[p:p + hlen], nls, end - off):
                continue
            starts, checked, complete = self._score(k, q, si, blocked, used)
            if starts >= MIN_STARTS or complete:
                best.append((starts * 1000 + checked, q))
        if not best:
            return None
        top = max(s for s, _q in best)
        tied = [q for s, q in best if s == top]
        q = min(tied, key=prefer)
        return k, si, q, len(best)

    # --- the main walk ------------------------------------------------------

    def rebuild(self, blocked, used):
        img = self.img
        C = img.cluster_size
        n, h = self.n, self.h
        self.mapping = mapping = [None] * n
        self.status = status = bytearray(n)
        phys = mapping.__getitem__
        searches = jumps = 0
        for i in range(h):
            mapping[i] = self.hcl + i
            status[i] = ST_HEADER
            used[self.hcl + i] = 1

        # frames that start inside the header clusters (mdat not cluster aligned)
        state = self.state_at_frame(0)
        for i in range(h):
            v, _c, st = self.validate(i, mapping[i], state, phys)
            state = st if v is not False else state

        def claim(c, q, st):
            mapping[c] = q
            status[c] = st
            used[q] = 1

        j = h
        p = self.hcl + h - 1
        if j < n:
            # The data rarely follows the header on the card (the camera writes
            # the header last), so locate the first frame by a full search.
            res = self.search(j, state[0], blocked, used, prefer=lambda q: abs(q - self.hcl))
            searches += 1
            if res is None:
                self.stats = self._stats(searches, jumps, "first frame not found")
                return
            k, si, q, _nc = res
            v, _c, state = self.validate(k, q, self.state_at_frame(si), phys)
            claim(k, q, ST_VALID)
            # whatever precedes the first frame (audio, padding) sits just before it
            pb = q
            for c in range(k - 1, j - 1, -1):
                pb = self._prev(pb, blocked, used)
                if pb is None:
                    break
                claim(c, pb, ST_INFERRED)
            j, p = k + 1, q

        while j < n:
            q0 = self._next(p, blocked, used)
            if q0 is None:
                break
            v, chk, st = self.validate(j, q0, state, phys)
            if v is not False:
                claim(j, q0, ST_VALID if v else ST_INFERRED)
                state, p, j = st, q0, j + 1
                continue
            # skip over clusters of the other stream / older files
            q = q0
            found = None
            for _ in range(WINDOW):
                q = self._next(q, blocked, used)
                if q is None or q == q0:
                    break
                v, chk, st = self.validate(j, q, state, phys)
                if v:
                    found = (q, st)
                    break
            if found:
                claim(j, found[0], ST_VALID)
                state, p, j = found[1], found[0], j + 1
                jumps += 1
                continue
            # lost the trail: look for the next frame anywhere on the card
            N = img.n_clusters
            res = self.search(j, state[0], blocked, used,
                              prefer=lambda qq: (qq - p) % N)
            searches += 1
            if res is None:
                break
            k, si, q, _nc = res
            v, _c, state = self.validate(k, q, self.state_at_frame(si), phys)
            claim(k, q, ST_VALID)
            j, p = k + 1, q
        self.stats = self._stats(searches, jumps, None)

    # --- second pass: place the clusters we could not check ------------------

    def refine(self, blocked, soft, checker):
        """Re-place every run of unchecked clusters between two checked ones.

        If exactly as many free clusters as needed sit between the run's two
        anchors, the placement is certain. Otherwise clusters of some other
        file are mixed in; with ffmpeg available we search for the placement
        under which the frame decodes cleanly, else we keep the most likely
        layout (contiguous after the first anchor).
        """
        N = self.img.n_clusters
        mapping, status = self.mapping, self.status
        anchors = [i for i in range(self.n) if status[i] in (ST_HEADER, ST_VALID)]
        used = bytearray(N)
        for i in anchors:
            used[mapping[i]] = 1
        ambiguous = 0
        for a, b in zip(anchors, anchors[1:]):
            n = b - a - 1
            if n <= 0:
                continue
            hyps = self._hypotheses(mapping[a], mapping[b], n, blocked, used, soft)
            if not hyps:
                for i in range(a + 1, b):  # keep the walk's guess if still free
                    q = mapping[i]
                    if q is not None and not used[q]:
                        used[q] = 1
                    else:
                        mapping[i] = None
                        status[i] = ST_MISSING
                continue
            choice = hyps[0]
            if len(hyps) > 1:
                ambiguous += 1
                if checker is not None and checker.available:
                    choice = self._resolve_by_decoding(a, b, blocked, used, soft, checker) or choice
            for i, q in enumerate(choice):
                mapping[a + 1 + i] = q
                status[a + 1 + i] = ST_INFERRED
                used[q] = 1
        self.stats = self._stats(self.stats.get("full_searches", 0),
                                 self.stats.get("skips", 0), self.stats.get("note"))
        self.stats["ambiguous_runs"] = ambiguous

    def _hypotheses(self, Pa, Pb, n, blocked, used, soft):
        """Plausible placements of n clusters between physical clusters Pa and Pb,
        most likely first."""
        N = self.img.n_clusters
        free = lambda k: not blocked[k] and not used[k]
        dist = (Pb - Pa) % N
        hyps = []
        if 0 < dist <= n + WINDOW:
            M = [k for k in ((Pa + d) % N for d in range(1, dist)) if free(k)]
            if len(M) == n:
                return [M]
            if len(M) > n:
                extra = len(M) - n
                M2 = [k for k in M if not soft[k]]
                if len(M2) == n:
                    hyps.append(M2)
                if n > 64:
                    return hyps or [M[:n]]
                for s in range(n, -1, -1):  # skip one foreign block starting after s
                    hyps.append(M[:s] + M[s + extra:])
                return _dedupe(hyps)
        # the file jumps to another fragment somewhere inside the run
        F, k = [], Pa
        while len(F) < n:
            k = (k + 1) % N
            if k == Pb:
                break
            if free(k):
                F.append(k)
        B, k = [], Pb
        while len(B) < n:
            k = (k - 1) % N
            if k == Pa:
                break
            if free(k):
                B.append(k)
        B.reverse()
        for x in range(n, -1, -1):
            if x <= len(F) and n - x <= len(B):
                h = F[:x] + B[len(B) - (n - x):]
                if len(set(h)) == n:
                    hyps.append(h)
                if n > 64:
                    break
        return _dedupe(hyps)

    def _read_logical(self, pos, size, phys):
        C = self.img.cluster_size
        out = bytearray()
        while size > 0:
            c = pos // C
            within = pos - c * C
            take = min(size, C - within)
            q = phys(c)
            if q is None:
                out += bytes(take)
            else:
                p = self.img.cluster_pos(q) + within
                out += self.img.mm[p:p + take]
            pos += take
            size -= take
        return bytes(out)

    def _frame_for_cluster(self, c):
        """Index of the video frame holding most of logical cluster c, or None."""
        C = self.img.cluster_size
        lo, hi = c * C, (c + 1) * C
        i = bisect.bisect_right(self.vid_starts, hi - 1) - 1
        best, best_len = None, 0
        while i >= 0:
            o, e = self.vid[i]
            if e <= lo:
                break
            ov = min(e, hi) - max(o, lo)
            if ov > best_len:
                best, best_len = i, ov
            i -= 1
        return best if best_len > C // 4 else None

    def _frame_damage(self, fi, upto, phys, checker):
        """Concealed macroblocks when frame fi is decoded up to logical byte ``upto``."""
        o, e = self.vid[fi]
        if upto is not None:
            e = min(e, upto)
        v = self.movie.video
        head = b"".join(b"\x00\x00\x00\x01" + x for x in v.sps + v.pps)
        data, _broken = to_annexb(self._read_logical(o, e - o, phys), self.nls, partial=True)
        return checker.errors(head + data)

    def _run_family(self, a, b, prefix, blocked, used, soft):
        """Hypotheses for run a+1..b-1 that keep ``prefix`` and vary the rest."""
        n = b - a - 1
        left = n - len(prefix)
        start = prefix[-1] if prefix else self.mapping[a]
        if left == 0:
            return [list(prefix)]
        u = self._with(used, prefix)
        return [list(prefix) + h for h in self._hypotheses(start, self.mapping[b], left, blocked, u, soft)]

    def _run_score(self, a, b, hyp, frames, checker, upto=None):
        mapping = self.mapping
        overlay = {a + 1 + i: q for i, q in enumerate(hyp)}
        ph = lambda x: overlay[x] if x in overlay else mapping[x]
        return sum(self._frame_damage(fi, upto, ph, checker) for fi in frames)

    def _resolve_by_decoding(self, a, b, blocked, used, soft, checker, budget=600, k=12):
        """Find the placement of run a+1..b-1 under which its frame decodes cleanly.

        A correct placement decodes with zero concealed macroblocks, a wrong
        one practically never does. Decoding the frame truncated after each
        cluster tells where the decoder first derails, so we grow a prefix of
        clusters known to be right, one position at a time, keeping at each
        step the candidate that lets the decoder get furthest, and back up when
        no candidate gets past the current position.
        """
        C = self.img.cluster_size
        n = b - a - 1
        Pb = self.mapping[b]
        frames = sorted({fi for fi in (self._frame_for_cluster(c) for c in range(a + 1, b))
                         if fi is not None})
        if not frames:
            return None
        calls = [0]

        def score(h, upto=None):
            calls[0] += 1
            return self._run_score(a, b, h, frames if upto is None else frames[-1:], checker, upto)

        def derail(h, sc, lo):
            hi = n - 1
            while lo < hi and calls[0] < budget:
                mid = (lo + hi) // 2
                if score(h, upto=(a + 2 + mid) * C) <= sc:
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        def complete(prefix):
            fam = self._run_family(a, b, prefix, blocked, used, soft)
            return fam[0] if fam else None

        best, best_s = None, None
        for h in self._run_family(a, b, [], blocked, used, soft):  # cheap first guesses
            if calls[0] >= budget // 4:
                break
            sc = score(h)
            if sc == 0:
                return h
            if best_s is None or sc < best_s:
                best, best_s = h, sc
        if best is None or len(frames) != 1:
            return best

        prefix = []
        tried = {}
        while calls[0] < budget:
            L = len(prefix)
            if L >= n:
                break
            start = prefix[-1] if prefix else self.mapping[a]
            u = self._with(used, prefix)
            done = tried.get(L, ())
            cands = [q for q in self._next_options(start, Pb, n - L, blocked, u, k)
                     if q not in done]
            if not cands:
                if not prefix:
                    break
                prefix.pop()
                continue
            top = None
            for q in cands[:k]:
                if calls[0] >= budget:
                    break
                h = complete(prefix + [q])
                if h is None:
                    continue
                sc = score(h)
                if sc == 0:
                    return h
                if sc < best_s:
                    best, best_s = h, sc
                d = derail(h, sc, L)
                if top is None or d > top[0] or (d == top[0] and sc < top[1]):
                    top = (d, sc, q, h)
            if top is None:
                break
            d, sc, q, h = top
            tried.setdefault(L, set()).add(q)
            if d <= L:
                # nothing gets past this position: an earlier pick was wrong
                if not prefix:
                    break
                prefix.pop()
                continue
            prefix = h[:max(L + 1, d - 1)]
            for i in range(L + 1, len(prefix)):
                tried.pop(i, None)
        return best

    def _next_options(self, start, Pb, count, blocked, used, k):
        """Free clusters that could hold the next of ``count`` clusters after ``start``."""
        N = self.img.n_clusters
        free = lambda q: not blocked[q] and not used[q]
        dist = (Pb - start) % N
        if 0 < dist <= count + WINDOW:
            M = [q for q in ((start + d) % N for d in range(1, dist)) if free(q)]
            if len(M) >= count:
                return M[:len(M) - count + 1][:k]
        out, q = [], start
        while len(out) < k:
            q = (q + 1) % N
            if q == Pb or q == start:
                break
            if free(q):
                out.append(q)
        back, q = [], Pb
        while len(back) < count:  # or jump to the fragment that ends at Pb
            q = (q - 1) % N
            if q == start:
                break
            if free(q):
                back.append(q)
        if len(back) == count and back[-1] not in out:
            out.append(back[-1])
        return out

    @staticmethod
    def _with(used, extra):
        u = bytearray(used)
        for q in extra:
            u[q] = 1
        return u

    def _stats(self, searches, jumps, note):
        st = self.status
        s = {
            "clusters": self.n,
            "header": st.count(ST_HEADER),
            "verified": st.count(ST_VALID),
            "inferred": st.count(ST_INFERRED),
            "missing": st.count(ST_MISSING),
            "skips": jumps,
            "full_searches": searches,
        }
        if note:
            s["note"] = note
        s["recovered_pct"] = round(100.0 * (self.n - s["missing"]) / self.n, 2) if self.n else 0
        return s

    def strong_clusters(self):
        return [q for q, st in zip(self.mapping, self.status) if st in (ST_VALID, ST_HEADER)]

    def write(self, path):
        img = self.img
        C = img.cluster_size
        zero = bytes(C)
        remaining = self.file_size
        with open(path, "wb") as f:
            for q in self.mapping:
                take = min(C, remaining)
                if take <= 0:
                    break
                if q is None:
                    f.write(zero[:take])
                else:
                    p = img.cluster_pos(q)
                    f.write(img.mm[p:p + take])
                remaining -= take

    def cluster_map(self):
        """Compact list of [logical_start, physical_start, length, status] runs."""
        runs = []
        for c, (q, st) in enumerate(zip(self.mapping, self.status)):
            if runs:
                r = runs[-1]
                if r[3] == st and ((q is None and r[1] is None) or
                                   (q is not None and r[1] is not None and r[1] + r[2] == q)):
                    r[2] += 1
                    continue
            runs.append([c, q, 1, st])
        return runs


def _dedupe(hyps):
    seen, out = set(), []
    for h in hyps:
        t = tuple(h)
        if t not in seen:
            seen.add(t)
            out.append(h)
    return out


def find_candidates(img, starts):
    return [Candidate(img, k) for k in starts]


def recover_all(img, flags, cands, rounds=2, checker=None, log=print):
    """Rebuild every parsable candidate, ``rounds`` times (see module doc)."""
    N = img.n_clusters
    base = bytearray(N)
    for k in range(N):
        if flags[k] & (FLAG_ZERO | FLAG_SIGNATURE | FLAG_JPEG) or k < img.first_cluster:
            base[k] = 1
    good = [c for c in cands if c.ok]
    for c in good:
        for i in range(c.h):
            base[c.hcl + i] = 1
            flags[c.hcl + i] |= FLAG_HEADER
    claims = {}
    soft_claims = {}
    for r in range(rounds):
        t0 = time.time()
        for idx, c in enumerate(good):
            blocked = bytearray(base)
            for other, qs in claims.items():
                if other is not c:
                    for q in qs:
                        blocked[q] = 1
            for i in range(c.h):  # own header is excluded via `used`
                blocked[c.hcl + i] = 0
            used = bytearray(N)
            c.rebuild(blocked, used)
            soft = bytearray(N)
            for other, qs in soft_claims.items():
                if other is not c:
                    for q in qs:
                        soft[q] = 1
            c.refine(blocked, soft, checker if r == rounds - 1 else None)
            log("  round %d  [%d/%d] %s  %s%% recovered" % (
                r + 1, idx + 1, len(good), c.name(), c.stats.get("recovered_pct")))
        claims = {c: c.strong_clusters() for c in good}
        soft_claims = {c: [q for q in c.mapping if q is not None] for c in good}
        log("  round %d done in %.1fs" % (r + 1, time.time() - t0))
    return good
