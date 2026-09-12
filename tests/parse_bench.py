"""Parse bench logs into a per-stage timing table.

    python tests/parse_bench.py <logfile> [<logfile> ...]

Handles the ANSI/UTF-16 noise that onnxruntime's C++ logger injects into the
output. Emits one table per ###BENCH### section plus a summary.
"""
import json
import re
import sys
from pathlib import Path


def clean(line: str) -> str:
    line = line.replace("\x00", "")
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    return line.rstrip()


def parse(path: Path):
    text = "\n".join(clean(l) for l in path.read_text(encoding="utf-8",
                                                       errors="replace").splitlines())
    runs = []
    starts = [m for m in re.finditer(r"###BENCH### ([^\n(]+)", text)]
    for i, m in enumerate(starts):
        label = m.group(1).strip()
        seg_end = (starts[i + 1].start() if i + 1 < len(starts)
                   else text.find("###SUMMARY###") if text.find("###SUMMARY###") != -1
                   else len(text))
        tail = text[m.end():seg_end]
        profs = re.findall(r"\[PROF\] n=\s*(\d+)\s+(.*)", tail)
        enc = re.search(r"\[PROF\]\[sbs\] encode thread: d2h_wait=([\d.]+)ms\s+"
                        r"encode=([\d.]+)ms\s+n=(\d+)\s+(.*)", tail)
        res = re.search(r"###RESULT### (.+)", tail)
        stages = {}
        for n, body in profs:
            for k, v in re.findall(r"([\w+]+)=\s*(\d+\.\d+)ms", body):
                stages.setdefault(k, []).append(float(v))
        entry = {"label": label, "stages_ms": {
            k: round(sum(v) / len(v), 1) for k, v in stages.items()},
            "profs": len(profs)}
        if enc:
            entry["d2h_wait_ms"] = float(enc.group(1))
            entry["encode_ms"] = float(enc.group(2))
            entry["vram"] = enc.group(4).strip()
        if res:
            entry.update(json.loads(res.group(1)))
        runs.append(entry)
    return runs


def main():
    all_runs = []
    for arg in sys.argv[1:]:
        all_runs.extend(parse(Path(arg)))

    print(f"{'变体':<26}{'fps':>7}{'predict':>9}{'stab':>7}{'rendr+pk':>9}"
          f"{'d2h+enc':>9}{'ort_enc':>9}{'D2Hwait':>9}{'encode':>8}{'峰值显存':>9}")
    for r in all_runs:
        s = r.get("stages_ms", {})
        g = lambda k: f"{s[k]:.0f}" if k in s else "-"
        fps = f"{r.get('fps') or 0:.2f}"
        print(f"{r['label']:<26}{fps:>7}{g('predict'):>9}{g('stabilize'):>7}"
              f"{g('render+pack'):>9}{g('d2h+write'):>9}{g('ort_enc'):>9}"
              f"{r.get('d2h_wait_ms', 0):>9.1f}{r.get('encode_ms', 0):>8.1f}"
              f"{r.get('peak_vram_mb', 0):>8}MB")
    out = Path(sys.argv[1]).with_suffix("") .with_name("parsed_bench.json")
    out = Path(sys.argv[1]).parent / "parsed_bench.json"
    out.write_text(json.dumps(all_runs, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
