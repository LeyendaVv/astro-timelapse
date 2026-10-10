"""Report the time gaps between frames, because the model assumes equal ones.

`measure_rotation.py` fits one theta per *frame index*, not per second, and the
renderers bridge each interval by index fraction.  That only means "per 40 s"
if the frames really are 40 s apart.  A single 46 s gap fed in as 40 s costs
about 0.22 of lag-1 ZNCC and drags the fitted theta with it, and nothing else in
the pipeline will tell you: the interval only matters as a number, and the
scripts otherwise never look at the clock again.

EXIF DateTimeOriginal is the only place that says whether the spacing is
uniform, so read it first and hand the fit the longest evenly spaced run.

Usage:
  python check_intervals.py --src DIR --first 4678 --last 4775
"""

from __future__ import annotations

import argparse
import datetime as dt
import os

from PIL import Image

_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 36867


def stamp(path):
    exif = Image.open(path).getexif().get_ifd(_EXIF_IFD)
    return dt.datetime.strptime(exif[_DATETIME_ORIGINAL], "%Y:%m:%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with the numbered frames")
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    ap.add_argument("--pattern", default="DSC_%04d.JPG")
    args = ap.parse_args()

    stamps = []
    for n in range(args.first, args.last + 1):
        p = os.path.join(args.src, args.pattern % n)
        if not os.path.exists(p):
            raise SystemExit("missing frame %s" % p)
        stamps.append((n, stamp(p)))
    gaps = [int((stamps[i + 1][1] - stamps[i][1]).total_seconds())
            for i in range(len(stamps) - 1)]
    distinct = sorted(set(gaps))
    print("%d frames  %s -> %s  (%.1f min)"
          % (len(stamps), stamps[0][1], stamps[-1][1],
             (stamps[-1][1] - stamps[0][1]).total_seconds() / 60.0))
    print("gaps: min %d  max %d  mean %.1f  median %d"
          % (min(gaps), max(gaps), sum(gaps) / float(len(gaps)),
             sorted(gaps)[len(gaps) // 2]))

    if len(distinct) == 1:
        print("uniform %d s - the index-based model fits exactly; "
              "pass --interval-s %d" % (distinct[0], distinct[0]))
        return

    print("NOT uniform, distinct gaps %s" % distinct)
    print("   counts %s" % {g: gaps.count(g) for g in distinct})

    # Longest run of evenly spaced frames: the part the fit can trust.
    best = (0, 0, 0)
    start = 0
    for i in range(1, len(gaps) + 1):
        if i == len(gaps) or gaps[i] != gaps[start]:
            if i - start > best[0]:
                best = (i - start, start, gaps[start])
            start = i
    length, start, gap = best
    print("longest evenly spaced run: %d gaps of %d s -> frames %d..%d"
          % (length, gap, stamps[start][0], stamps[start + length][0]))
    print("recommended: --first %d --last %d --interval-s %d"
          % (stamps[start][0], stamps[start + length][0], gap))
    print("Each off-spacing frame fed in as if it were %d s widens the trails "
          "and lowers the ZNCC table, so prefer the run above." % distinct[0])


if __name__ == "__main__":
    main()
