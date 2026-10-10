"""Render star-trail / comet-tail / time-lapse videos from a fixed-tripod
night-sky photo sequence.

Modes
  cache   build the graded-frame cache only (every other mode reuses it)
  lapse   rotate-interpolate single frames - round stars, smooth motion
  growth  per-frame max stack - trails grow out of the stars; no foreground
          matte is used because the foreground is static and the max stack
          already keeps its one bright (light-painted) appearance
  stream  the same stack with an exponential decay - every star drags a tail
  comet   pole rotation + bridged sub-samples + decay - continuous comet tails

The two things that decide whether it looks right:

* the exposure gap.  With 15 s exposures 48 s apart only ~1/3 of each interval
  is exposed, so a plain stack is a string of beads.  `--subs 4` warps each
  interval's frame to several phases (u - u_k) * ADVANCE * theta so each
  interval stays covered; all samples live in the original image plane.
* the foreground.  A static object that is lit in exactly one frame must be
  *lightened* onto the stack (max), never alpha-composited: the light-painted
  frame's own sky and leaf gaps are darker than the trail stack, so an
  alpha-over punches a trail-free hole and leaves a hard, dark rim.

Usage:
  python render_video.py --src FRAMES --first 4944 --last 5033 --lit 4973 \
      --model rotation_model.json --mode comet --tau 13 --name 白杨银河
  python render_video.py ... --mode cache      # once, then re-render cheaply
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time

import numpy as np
from PIL import Image

import astro_kit as ak


def parse_size(text):
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("size must look like 3840x2160")


# ------------------------------------------------------------------ project

class Project:
    """Everything the render loops need, built once from the CLI arguments."""

    def __init__(self, args):
        self.args = args
        t0 = time.time()
        self.seq = ak.Sequence(args.src, args.first, args.last, args.pattern)
        self.pole, self.theta = ak.load_model(args.model)
        self.out_w, self.out_h = parse_size(args.out_size)
        self.fps = args.fps
        self.n_out = int(round(args.fps * args.duration))
        self.z0, self.z1 = args.zoom
        self.n_src = self.seq.n
        self.step = (self.n_src - 1) / (self.n_out - 1.0)
        self.subs = args.subs
        self.tau = args.tau
        self.rule = args.composite
        self.lit = args.lit
        self.work = args.work

        im = Image.open(self.seq.path(1))
        self.full_w, self.full_h = im.size
        self.med_w = args.med or self.full_w
        self.med_h = int(round(self.full_h * self.med_w / float(self.full_w)))
        self.med_s = self.med_w / float(self.full_w)
        self.cx, self.cy = self.pole * self.med_s
        os.makedirs(self.work, exist_ok=True)

        self.m = self.m_blank = None
        self.box = None
        if args.lit and not args.no_foreground:
            self.m, self.m_blank, self.box = ak.foreground_matte(
                self.seq, args.lit, (self.med_w, self.med_h),
                m_lo=args.matte_lo, m_hi=args.matte_hi, area=args.mask_area,
                blank=args.blank, out_dir=self.work)
        elif args.lit:
            _, self.m_blank, _ = ak.foreground_matte(
                self.seq, args.lit, (self.med_w, self.med_h),
                m_lo=args.matte_lo, m_hi=args.matte_hi, area=args.mask_area,
                blank=args.blank, out_dir=self.work)

        sky_small = self._sky_small()
        self.wbs, self.gains = ak.analyse(
            self.seq, sky_small, wb_strength=args.wb, deflicker=args.deflicker)
        self.tabs = [ak.make_lut(g, w, gamma=args.gamma)
                     for g, w in zip(self.gains, self.wbs)]
        if args.lit:
            # the deflicker gain is meant for the sky; on the light-painted
            # frame it would only flatten the foreground
            self.tabs[args.lit - 1] = ak.make_lut(1.0, self.wbs[args.lit - 1],
                                                  gamma=args.gamma)

        self.cache = ak.GradeCache(
            self.seq, self.tabs, (self.med_w, self.med_h),
            denoise=args.denoise, sat=args.sat,
            cache_dir=None if args.no_cache else self.cache_dir())

        self.ov = None
        if args.lit:
            lit = ak.grade_frame(self.seq, args.lit, self.tabs[args.lit - 1],
                                 (self.med_w, self.med_h),
                                 denoise=args.denoise, sat=args.sat)
            if self.m is not None:
                self.ov = ak.overlay_parts(lit, self.m, self.box,
                                           matte_gain=args.matte_gain)
            self._lit_graded = lit
        self._blank_cache = {}
        print("pole (%.0f, %.0f) full res ; %+.5f deg/interval ; %d frames ; "
              "MED %dx%d ; %d output frames [%.0fs]"
              % (self.pole[0], self.pole[1], self.theta, self.n_src,
                 self.med_w, self.med_h, self.n_out, time.time() - t0))

    # -- setup helpers ------------------------------------------------------

    def _sky_small(self):
        if self.m_blank is not None:
            small = np.asarray(Image.fromarray(
                (self.m_blank * 255).astype(np.uint8)).resize((378, 252), Image.BOX))
            return ~small.astype(bool)
        return np.ones((252, 378), bool)

    def cache_dir(self):
        a = self.args
        key = "|".join(str(x) for x in (
            self.med_w, self.med_h, a.gamma, a.sat, a.denoise, a.wb,
            a.deflicker, self.seq.first, self.seq.last))
        tag = hashlib.md5(key.encode()).hexdigest()[:8]
        return os.path.join(self.work, "graded_%dx%d_%s" % (self.med_w, self.med_h, tag))

    # -- per-frame ----------------------------------------------------------

    def frame_for_stack(self, i):
        g = self.cache[i]
        if self.m_blank is None or i != self.lit:
            return g
        if i not in self._blank_cache:
            x = g.astype(np.float32) * (1.0 - self.m_blank[:, :, None])
            self._blank_cache[i] = np.clip(x, 0, 255).astype(np.uint8)
        return self._blank_cache[i]

    def zoom_at(self, u):
        return self.z0 + (self.z1 - self.z0) * (u / (self.n_src - 1))

    def finish(self, acc, zoom):
        return ak.compose(acc, zoom, self.ov, (self.out_w, self.out_h),
                          rule=self.rule, med_w=self.med_w, med_h=self.med_h)


# ------------------------------------------------------------------- frames

def lapse_frames(p):
    for n in range(p.n_out):
        u = n * p.step
        k = min(int(u) + 1, p.n_src)
        f = u - int(u)
        acc = ak.warp(p.frame_for_stack(k), ak.ADVANCE * f * p.theta,
                      p.cx, p.cy, p.med_w, p.med_h)
        yield p.finish(acc, p.zoom_at(u))


def growth_frames(p, stream=False):
    decay = float(np.exp(-p.step / p.tau)) if stream else 1.0
    acc, next_k = None, 1
    for n in range(p.n_out):
        U = n * p.step
        if acc is not None and decay != 1.0:
            acc *= decay
        while next_k <= min(int(U) + 1, p.n_src):
            g = p.frame_for_stack(next_k).astype(np.float32)
            acc = g if acc is None else np.maximum(acc, g)
            next_k += 1
        yield p.finish(acc, p.zoom_at(U))


def comet_raw(p):
    """Yield (n, accumulator, zoom) for the comet mode, before compositing.

    Kept separate from comet_frames so a probe can grab the accumulator at one
    output frame and try several compositing rules on it without re-rendering.
    """
    du = 1.0 / p.subs
    decay = float(np.exp(-p.step / p.tau))
    acc, next_sub = None, 0.0
    for n in range(p.n_out):
        U = n * p.step
        if acc is not None and decay != 1.0:
            acc *= decay
        while next_sub <= U + 1e-9:
            i = int(np.floor(next_sub + 1e-9))
            k = min(i + 1, p.n_src)
            w = ak.warp(p.frame_for_stack(k),
                        ak.ADVANCE * (next_sub - i) * p.theta,
                        p.cx, p.cy, p.med_w, p.med_h).astype(np.float32)
            acc = w if acc is None else np.maximum(acc, w)
            next_sub += du
            if next_sub > p.n_src - 1 + 1e-9:
                break
        yield n, acc, p.zoom_at(U)


def comet_frames(p):
    for _, acc, zoom in comet_raw(p):
        yield p.finish(acc, zoom)


MODES = {"lapse": lambda p: lapse_frames(p),
         "growth": lambda p: growth_frames(p, stream=False),
         "stream": lambda p: growth_frames(p, stream=True),
         "comet": lambda p: comet_frames(p)}


# ---------------------------------------------------------------- previews

def previews(args, p):
    """Save stills at `--preview-times` plus a contact sheet."""
    times = [float(t) for t in args.preview_times.split(",") if t.strip()]
    want = {int(round(t * p.fps)): t for t in times}
    tiles, labels = [], []
    os.makedirs(args.preview, exist_ok=True)
    stem = os.path.basename(args.out).rsplit(".", 1)[0]
    for n, arr in enumerate(MODES[args.mode](p)):
        if n in want:
            t = want.pop(n)
            path = os.path.join(args.preview, "%s_%ss.png" % (stem, t))
            Image.fromarray(arr).save(path)
            small = Image.open(path).convert("RGB")
            scale = 640.0 / small.width
            tiles.append(small.resize((640, int(round(small.height * scale))),
                                      Image.LANCZOS))
            labels.append("%gs" % t)
        if not want:
            break
    if tiles:
        ak.contact_sheet(tiles, os.path.join(args.preview, "%s_contact.png" % stem),
                         labels=labels, cols=min(3, len(tiles)))


# --------------------------------------------------------------------- cli

def build_parser():
    ap = argparse.ArgumentParser(
        description="Render a night-sky timelapse / star-trail / comet video.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with the numbered frames")
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    ap.add_argument("--pattern", default="DSC_%04d.JPG")
    ap.add_argument("--model", required=True, help="rotation_model.json")
    ap.add_argument("--mode", default="comet",
                    choices=["cache", "lapse", "growth", "stream", "comet"])
    ap.add_argument("--lit", type=int, default=0,
                    help="1-based index of the light-painted frame (static foreground)")
    ap.add_argument("--no-foreground", action="store_true",
                    help="plain max stack: still blanks that frame, no compositing")
    ap.add_argument("--name", default="sequence")
    ap.add_argument("--out", default="")
    ap.add_argument("--out-size", default="3840x2160")
    ap.add_argument("--med", type=int, default=0,
                    help="working width for the stack (default: native)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--tau", type=float, default=0.0,
                    help="tail length; comet: source intervals (13), stream: frames (55)")
    ap.add_argument("--subs", type=int, default=4,
                    help="bridging sub-samples per exposure interval")
    ap.add_argument("--zoom", type=float, nargs=2, default=[1.0, 1.05],
                    metavar=("Z0", "Z1"))
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--fade-in", type=float, default=0.6)
    ap.add_argument("--fade-out", type=float, default=0.8)
    ap.add_argument("--composite", default="lighten",
                    choices=["lighten", "screen", "alpha"],
                    help="how the light-painted frame joins the stack; "
                         "alpha is kept for comparison and is wrong for stacks")
    ap.add_argument("--matte-gain", type=float, default=1.4,
                    help="saturate the matte so leaves keep their real value")
    ap.add_argument("--matte-lo", type=float, default=3.0)
    ap.add_argument("--matte-hi", type=float, default=16.0)
    ap.add_argument("--mask-area", type=int, default=20000,
                    help="smallest blob kept as foreground, in med pixels")
    ap.add_argument("--blank", type=int, default=8,
                    help="how far the light-painted frame is blanked beyond the matte")
    ap.add_argument("--wb", type=float, default=0.8)
    ap.add_argument("--deflicker", type=float, default=0.8)
    ap.add_argument("--gamma", type=float, default=0.84)
    ap.add_argument("--sat", type=float, default=1.16)
    ap.add_argument("--denoise", type=float, default=1.0)
    ap.add_argument("--work", default=".astro_work")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--preview", default="")
    ap.add_argument("--preview-times", default="2,9.5")
    ap.add_argument("--test", action="store_true",
                    help="fast low-resolution probe of the whole pipeline")
    return ap


def apply_defaults(args):
    if args.tau <= 0:
        args.tau = 13.0 if args.mode in ("comet",) else 55.0
    if args.test:
        args.out_size = "1280x720"
        args.med = 2048
        args.blank = max(2, args.blank // 2)
        args.out = args.out or ("test_%s_%s.mp4" % (args.name, args.mode))
    if not args.out:
        h = parse_size(args.out_size)[1]
        args.out = "%s_%s_%gs_%dp.mp4" % (args.name, args.mode, args.duration, h)
    return args


def main():
    ap = build_parser()
    args = apply_defaults(ap.parse_args())
    t0 = time.time()
    p = Project(args)

    if args.mode == "cache":
        p.cache.build()
        print("cache ready -> %s  [%.0fs]" % (p.cache.cache_dir, time.time() - t0))
        return

    if args.preview:
        previews(args, p)
        return

    ak.write_video(MODES[args.mode](p), args.out, (p.out_w, p.out_h), p.fps,
                   crf=args.crf, preset=args.preset, duration=args.duration,
                   fade_in=args.fade_in, fade_out=args.fade_out,
                   log=args.mode, verbose_every=25)
    print("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
