# -*- coding: utf-8 -*-
"""解析 fragmented MP4 的 trun 采样表，精确得出各轨样本数与末帧时间。

比 ffprobe -count_frames（要全解码）快几个数量级：只读 box 头与 trun。
"""
import struct
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def walk(f, start, end):
    """Yield (offset, size, type, header_size) of top-level boxes."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        hdr = f.read(16)
        if len(hdr) < 8:
            break
        size = struct.unpack(">I", hdr[:4])[0]
        btype = hdr[4:8].decode("latin1", "replace")
        hsize = 8
        if size == 1:
            if len(hdr) < 16:
                break
            size = struct.unpack(">Q", hdr[8:16])[0]
            hsize = 16
        elif size == 0:
            size = end - pos
        if size < hsize:
            break
        yield pos, size, btype, hsize
        pos += size


def parse_moov(f, off, size):
    """Return {track_id: (handler, timescale, is_video)}."""
    tracks = {}
    for o, s, t, h in walk(f, off + 8, off + size):
        if t != "trak":
            continue
        tid = handler = None
        timescale = None
        for o2, s2, t2, h2 in walk(f, o + h, o + s):
            if t2 == "tkhd":
                f.seek(o2 + h2)
                d = f.read(s2 - h2)
                ver = d[0]
                tid = struct.unpack(">I", d[20:24])[0] if ver == 1 else \
                    struct.unpack(">I", d[12:16])[0]
            elif t2 == "mdia":
                for o3, s3, t3, h3 in walk(f, o2 + h2, o2 + s2):
                    if t3 == "mdhd":
                        f.seek(o3 + h3)
                        d = f.read(s3 - h3)
                        ver = d[0]
                        timescale = struct.unpack(
                            ">I", d[20:24])[0] if ver == 1 else \
                            struct.unpack(">I", d[12:16])[0]
                    elif t3 == "hdlr":
                        f.seek(o3 + h3 + 8)
                        handler = f.read(4).decode("latin1", "replace")
        if tid is not None:
            tracks[tid] = (handler, timescale)
    return tracks


def main(path):
    p = Path(path)
    fsize = p.stat().st_size
    tracks = {}
    samples = {}      # track_id -> total sample count
    last_tfdt = {}    # track_id -> last fragment baseMediaDecodeTime
    last_frag_times = {}  # track_id -> list of (tfdt, sample_count)
    n_moof = 0
    with open(p, "rb") as f:
        moov = None
        for o, s, t, h in walk(f, 0, fsize):
            if t == "moov":
                moov = (o, s)
            elif t == "moof":
                n_moof += 1
                for o2, s2, t2, h2 in walk(f, o + h, o + s):
                    if t2 != "traf":
                        continue
                    tid = None
                    tfdt = None
                    trun_count = 0
                    for o3, s3, t3, h3 in walk(f, o2 + h2, o2 + s2):
                        f.seek(o3 + h3)
                        d = f.read(s3 - h3)
                        if t3 == "tfhd":
                            flags = struct.unpack(">I", d[0:4])[0] & 0xFFFFFF
                            tid = struct.unpack(">I", d[4:8])[0]
                        elif t3 == "tfdt":
                            ver = d[0]
                            tfdt = struct.unpack(">Q", d[4:12])[0] if ver == 1 \
                                else struct.unpack(">I", d[4:8])[0]
                        elif t3 == "trun":
                            trun_count = struct.unpack(">I", d[4:8])[0]
                    if tid is not None:
                        samples[tid] = samples.get(tid, 0) + trun_count
                        if tfdt is not None:
                            last_frag_times.setdefault(tid, []).append(
                                (tfdt, trun_count))
        if moov:
            tracks = parse_moov(f, moov[0], moov[1])

    print(f"文件: {p.name}")
    print(f"大小: {fsize:,} bytes   moof 数: {n_moof}")
    for tid, (handler, timescale) in sorted(tracks.items()):
        n = samples.get(tid, 0)
        frags = last_frag_times.get(tid, [])
        info = f"  track {tid} [{handler}] timescale={timescale} 样本数={n}"
        if frags and timescale:
            tfdt, sc = frags[-1]
            t_end = (tfdt + sc) / timescale
            info += (f"  末片段起始时间={tfdt / timescale:.3f}s"
                     f"  覆盖到={t_end:.3f}s")
            if handler == "vide":
                tick = timescale / (30000 / 1001)
                info += f"\n      → 末帧索引≈{int(round(t_end * 30000 / 1001)) - 1}"
        print(info)


if __name__ == "__main__":
    main(sys.argv[1])
