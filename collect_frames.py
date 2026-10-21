"""
Copy one frame number of every train of every run into one folder (and a zip), to download it in one go.

    python collect_frames.py                      # frame 10, clean frames  -> frames_010/ + frames_010.zip
    python collect_frames.py --frame 20 --annotated   # the copies with scale bar + time stamp
    python collect_frames.py --runs r571 r572

Then on your own computer:
    scp <user>@max-display.desy.de:<repo>/frames_010.zip .
"""
import argparse
import glob
import os
import re
import shutil

_TF_RE = re.compile(r"_tid(\d+)_(\d+)\.[A-Za-z0-9]+$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=int, default=10, help="frame number (default 10)")
    ap.add_argument("--raw", default="raw", help="folder holding the jpg_<run>/ folders (default raw)")
    ap.add_argument("--runs", nargs="*", help="only these runs, e.g. r571 r572 (default: all)")
    ap.add_argument("--annotated", action="store_true", help="use the _annotated copies (scale bar + time stamp)")
    ap.add_argument("--out", help="output folder (default frames_<NNN>[_annotated])")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    out = a.out or f"frames_{a.frame:03d}" + ("_annotated" if a.annotated else "")
    os.makedirs(out, exist_ok=True)
    want = {f"r{int(r.lstrip('rR')):03d}" for r in a.runs} if a.runs else None
    n = 0
    for d in sorted(glob.glob(os.path.join(a.raw, "jpg_*"))):
        if not os.path.isdir(d) or d.endswith("_annotated") != a.annotated:
            continue
        run = re.match(r"jpg_(r\d+)", os.path.basename(d))
        if want is not None and (not run or f"r{int(run.group(1)[1:]):03d}" not in want):
            continue
        for p in sorted(os.listdir(d)):
            m = _TF_RE.search(p)
            if m and int(m.group(2)) == a.frame:
                shutil.copy2(os.path.join(d, p), os.path.join(out, p))
                n += 1
    print(f"{n} frames copied to {out}/")
    if n and not a.no_zip:
        print("zip:", shutil.make_archive(out, "zip", out))


if __name__ == "__main__":
    main()
