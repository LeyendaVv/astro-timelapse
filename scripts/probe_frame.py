"""Try several foreground-compositing rules on ONE output frame.

Rendering a 4K comet frame costs minutes because every sub-sample is warped,
but comparing composite rules needs the *same* accumulator, not a re-render.
This accumulates up to one output frame, caches that accumulator on disk, and
applies each rule to it - so the first run costs the warp pass and every later
run costs seconds.

The cache is keyed on a hash of every argument that changes the accumulated
pixels plus the contents of the model file, so changing grading, matte, timing
or the model rebuilds it instead of silently comparing a stale render.

The rules only differ where the matte is partially open, so always look at a
1:1 crop near the foreground: downscaled thumbnails hide the whole effect.

Usage:
  python probe_frame.py --src FRAMES --first 4944 --last 5033 --lit 4973 \
      --model rotation_model.json --frame 237 --detail 1200,1050,1000,500
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time

import numpy as np
from PIL import Image

import astro_kit as ak
import render_video as rv

# Arguments that describe the output or the probe itself rather than the
# pixels in the accumulator.
_ACC_KEY_SKIP = frozenset((
    "work", "out", "outdir", "acc_cache", "frame", "variants", "detail",
    "preview", "preview_times",
))


def acc_key(args):
    """Short hash of everything that decides the accumulated frame."""
    items = sorted((k, v) for k, v in vars(args).items()
                   if k not in _ACC_KEY_SKIP)
    h = hashlib.md5(repr(items).encode("utf-8"))
    if args.model and os.path.exists(args.model):
        with open(args.model, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:8]


def parse_box(text):
    if not text:
        return None
    x, y, w, h = (int(v) for v in text.split(","))
    return (x, y, x + w, y + h)


def main():
    ap = rv.build_parser()
    ap.description = __doc__
    ap.add_argument("--frame", type=int, default=237,
                    help="output frame index to probe")
    ap.add_argument("--variants", default="alpha,lighten,screen")
    ap.add_argument("--detail", default="", metavar="X,Y,W,H",
                    help="1:1 crop of the output frame to include in the sheet")
    ap.add_argument("--acc-cache", default="",
                    help="where to keep the accumulator (default: work/acc_<frame>.npy)")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    if args.mode not in ("comet", "stream", "growth"):
        args.mode = "comet"
    args = rv.apply_defaults(args)
    outdir = args.outdir or os.path.join(args.work, "probe")
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    p = rv.Project(args)
    key = acc_key(args)
    acc_path = args.acc_cache or os.path.join(
        args.work, "acc_%03d_%s.npy" % (args.frame, key))
    if os.path.exists(acc_path):
        acc = np.load(acc_path)
        zoom = p.zoom_at(args.frame * p.step)
        print("loaded accumulator %s [%.0fs]" % (acc_path, time.time() - t0))
    else:
        print("accumulator key %s (rebuilds when any render parameter or the "
              "model changes)" % key)
        acc = zoom = None
        for n, a, z in rv.comet_raw(p):
            if n == args.frame:
                acc, zoom = a, z
                break
        if acc is None:
            raise SystemExit("frame %d is past the end of the render" % args.frame)
        np.save(acc_path, acc)
        print("accumulated frame %d (zoom %.4f) -> %s [%.0fs]"
              % (args.frame, zoom, acc_path, time.time() - t0))

    detail = parse_box(args.detail)
    tiles, labels = [], []
    for rule in [v.strip() for v in args.variants.split(",") if v.strip()]:
        img = ak.compose(acc, zoom, p.ov, (p.out_w, p.out_h), rule=rule,
                         med_w=p.med_w, med_h=p.med_h)
        path = os.path.join(outdir, "frame%03d_%s.png" % (args.frame, rule))
        Image.fromarray(img).save(path)
        print("wrote %s" % path)
        im = Image.fromarray(img).convert("RGB")
        scale = 720.0 / im.width
        tiles.append(im.resize((720, int(round(im.height * scale))), Image.LANCZOS))
        labels.append(rule)
        if detail:
            d = im.crop(detail)
            tiles.append(d)
            labels.append("%s 1:1" % rule)
    ak.contact_sheet(tiles, os.path.join(outdir, "probe_%03d.png" % args.frame),
                     labels=labels, cols=len([v for v in args.variants.split(",") if v.strip()]))
    print("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
