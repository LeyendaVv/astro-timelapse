"""Measure the sky's rotation (pole + degrees per interval) from a sequence.

Two independent stages, because either one alone is easy to get wrong:

1.  tile phase correlation between consecutive frames gives a dense
    displacement field; a robust fit of  v = theta * (-(y-cy), (x-cx))
    recovers the pole and the rate without tracking individual stars;
2.  `--refine` then scores the model by what the renderer actually needs - the
    ZNCC of the sky after warping a later frame back onto an earlier one -
    averaged over pairs spread across the whole sequence, and optimises the
    three parameters directly from several starting points.

Sign convention (the part that is easy to get backwards):

    theta maps a LATER frame back onto an EARLIER one,
    therefore the sky itself advances by -theta per interval.

`sky_sign()` scores both directions on several separations and flips the
emitted theta if the fit came out the other way, so a model written by this
script can always be consumed with `astro_kit.ADVANCE = -1`.

Usage:
  python measure_rotation.py --src DIR --first 4944 --last 5033 \
      --lit 4973 --interval-s 48 --refine --out rotation_model.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize

import astro_kit as ak


# --------------------------------------------------------------- stage 1

class Tiles:
    def __init__(self, seq, scale, tile, step, up, max_shift, highpass=3.0):
        self.seq = seq
        self.scale = scale
        self.tile = tile
        self.step = step
        self.up = up
        self.max_shift = max_shift
        self.highpass = highpass
        self.w, self.h = 0, 0
        self._cache = {}
        self._win = None

    def frame(self, i):
        if i not in self._cache:
            im = Image.open(self.seq.path(i)).convert("L")
            if not self.w:
                self.w, self.h = im.width // self.scale, im.height // self.scale
            a = np.asarray(im.resize((self.w, self.h), Image.BOX)).astype(np.float32)
            self._cache[i] = a - ndimage.gaussian_filter(a, self.highpass)
            while len(self._cache) > 6:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[i]

    def window(self, n):
        if self._win is None or self._win.shape[0] != n:
            self._win = np.hanning(n)[:, None] * np.hanning(n)[None, :]
        return self._win

    def phase_shift(self, a, b):
        """Sub-pixel (dy, dx) that best maps b onto a inside one tile."""
        A = np.fft.rfft2(a * self.window(a.shape[0]))
        B = np.fft.rfft2(b * self.window(b.shape[0]))
        R = A * np.conj(B)
        mag = np.abs(R)
        mag[mag < 1e-9] = 1e-9
        R /= mag
        h, w = a.shape
        up = np.fft.irfft2(R, s=(h * self.up, w * self.up))
        flat = np.unravel_index(np.argmax(up), up.shape)

        def parab(vals):
            lo, mid, hi = vals
            den = lo - 2.0 * mid + hi
            return 0.0 if den == 0 else float(np.clip(0.5 * (lo - hi) / den, -1, 1))

        iy, ix = flat
        ry = parab((up[(iy - 1) % (h * self.up), ix],
                    up[iy, ix],
                    up[(iy + 1) % (h * self.up), ix]))
        rx = parab((up[iy, (ix - 1) % (w * self.up)],
                    up[iy, ix],
                    up[iy, (ix + 1) % (w * self.up)]))
        dy, dx = (iy + ry) / self.up, (ix + rx) / self.up
        if dy > h / 2.0:
            dy -= h
        if dx > w / 2.0:
            dx -= w
        return dy, dx

    def collect(self, max_pairs=0, verbose=True, t0=None):
        t0 = t0 or time.time()
        # Load one frame first: the grid below depends on the working size,
        # which is only discovered when the first image is read.
        self.frame(1)
        ys = range(0, self.h - self.tile, self.step)
        xs = range(0, self.w - self.tile, self.step)
        ys, xs = list(ys), list(xs)
        if verbose:
            print("tiles %dx%d of %d px, %d pairs"
                  % (len(xs), len(ys), self.tile, self.seq.n - 1))
        rows = []
        n_pairs = self.seq.n - 1
        if max_pairs:
            n_pairs = min(n_pairs, max_pairs)
        for k in range(1, n_pairs + 1):
            A, B = self.frame(k), self.frame(k + 1)
            for y0 in ys:
                for x0 in xs:
                    dy, dx = self.phase_shift(A[y0:y0 + self.tile, x0:x0 + self.tile],
                                              B[y0:y0 + self.tile, x0:x0 + self.tile])
                    if abs(dy) > self.max_shift or abs(dx) > self.max_shift:
                        continue
                    rows.append((x0 + self.tile / 2.0, y0 + self.tile / 2.0,
                                 dx, dy, k))
            if verbose and (k % 10 == 0 or k == n_pairs):
                print("  pair %d/%d  %d rows  [%.0fs]"
                      % (k, n_pairs, len(rows), time.time() - t0), flush=True)
        return np.array(rows) if rows else np.zeros((0, 5))


def fit_displacement(data, scale, verbose=True):
    """Robust fit of the displacement field -> (pole_full, theta_deg, resid, kept)."""
    if len(data) < 40:
        raise SystemExit("not enough tile displacements (%d) to fit" % len(data))
    X, Y, VX, VY, _ = data.T
    keep = np.ones(len(X), bool)
    r = np.zeros(len(X))
    sol = np.zeros(3)
    for it in range(12):
        m = int(keep.sum())
        A = np.zeros((2 * m, 3))
        rhs = np.zeros(2 * m)
        A[0::2, 0] = -Y[keep]
        A[0::2, 1] = 1.0
        rhs[0::2] = VX[keep]
        A[1::2, 0] = X[keep]
        A[1::2, 2] = -1.0
        rhs[1::2] = VY[keep]
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        a, b, d = sol
        r = np.hypot(-a * Y + b - VX, a * X - d - VY)
        thr = max(0.8, np.percentile(r[keep], 60))
        keep = r < thr
        if keep.sum() < 40:
            break
        if verbose:
            print("  trim %2d: %5d/%d inliers  thr %.2f px  theta %+.5f deg/int"
                  % (it + 1, int(keep.sum()), len(X), thr, np.degrees(a)))
    a, b, d = sol
    if abs(a) < 1e-9:
        raise SystemExit("degenerate fit (theta ~ 0)")
    pole = np.array([d / a * scale, b / a * scale])
    resid = {"p50": float(np.percentile(r[keep], 50)),
             "p90": float(np.percentile(r[keep], 90))}
    return pole, float(np.degrees(a)), resid, int(keep.sum())


# --------------------------------------------------------------- stage 2

def sky_mask(seq, size, lit=None, border=0.02, verbose=True):
    """Boolean sky mask at `size`: foreground removed, borders dropped."""
    if lit:
        m, m_blank, _ = ak.foreground_matte(seq, lit, size, verbose=verbose)
        sky = m_blank < 0.5
    else:
        sky = np.ones((size[1], size[0]), bool)
    bx = int(round(border * size[0]))
    by = int(round(border * size[1]))
    sky[:by, :] = False
    sky[-by:, :] = False
    sky[:, :bx] = False
    sky[:, -bx:] = False
    if verbose:
        print("sky mask: %.1f%% of the frame" % (100.0 * sky.mean()))
    return sky


class Refiner:
    """Maximise the mean sky ZNCC of `warp(later, +(b-a)*theta)` onto `earlier`."""

    def __init__(self, seq, sky, scale, pairs, verbose=True):
        self.seq = seq
        self.sky = sky
        self.scale = scale
        self.pairs = pairs
        self.verbose = verbose
        self.w = sky.shape[1]
        self.h = sky.shape[0]
        self._f = {}
        self.calls = 0

    def frame(self, i):
        if i not in self._f:
            self._f[i] = self.seq.gray(i, (self.w, self.h), highpass=6.0)
        return self._f[i]

    def zncc(self, a, b, pole, theta):
        ra = self.frame(a)
        w = ak.warp(self.frame(b), (b - a) * theta,
                    pole[0] / self.scale, pole[1] / self.scale,
                    self.w, self.h)
        av, bv = ra[self.sky], w[self.sky]
        av = av - av.mean()
        bv = bv - bv.mean()
        d = np.sqrt((av * av).sum() * (bv * bv).sum())
        return float((av * bv).sum() / d) if d > 0 else -1.0

    def objective(self, p):
        px, py, theta = p
        if abs(px) > 60000 or abs(py) > 60000 or abs(theta) > 1.0:
            return 1e3
        z = [self.zncc(a, min(b, self.seq.n), np.array([px, py]), theta)
             for a, b in self.pairs]
        self.calls += 1
        return -float(np.mean(z)) * 100.0

    def table(self, pole, theta, lags=None, tag=""):
        lags = lags or [d for d in (1, 2, 4, 8, 16, 32, 64, self.seq.n - 1)
                        if d <= self.seq.n - 1]
        rows = []
        for d in lags:
            z = self.zncc(1, min(1 + d, self.seq.n), pole, theta)
            rows.append((d, z))
            if self.verbose:
                print("  %s lag %3d -> ZNCC %+.3f" % (tag, d, z))
        return rows

    def sign_test(self, pole, theta, lags=(1, 2, 4, 8)):
        """Mean ZNCC for +theta (correct) vs -theta (flipped)."""
        lags = [d for d in lags if d <= self.seq.n - 1]
        fwd = np.mean([self.zncc(1, min(1 + d, self.seq.n), pole, theta) for d in lags])
        rev = np.mean([self.zncc(1, min(1 + d, self.seq.n), pole, -theta) for d in lags])
        return float(fwd), float(rev)


# --------------------------------------------------------------- cli

def parse_pairs(n, lags, keep=22):
    """Spread `keep` frame pairs across the sequence, short lags first."""
    pairs, l = [], 0
    while len(pairs) < keep:
        d = lags[l % len(lags)]
        step = max(1, (n - d) // max(1, keep // len(lags)))
        for a in range(1, n - d + 1, step):
            pairs.append((a, a + d))
            if len(pairs) >= keep:
                break
        l += 1
        if l > 4 * len(lags):
            break
    return sorted(set(pairs))[:keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with the numbered frames")
    ap.add_argument("--first", type=int, required=True, help="first frame number, e.g. 4944")
    ap.add_argument("--last", type=int, required=True, help="last frame number")
    ap.add_argument("--pattern", default="DSC_%04d.JPG")
    ap.add_argument("--lit", type=int, default=0,
                    help="1-based index of the light-painted frame (builds a sky mask)")
    ap.add_argument("--interval-s", type=float, default=0.0)
    ap.add_argument("--scale", type=int, default=2, help="work at 1/scale (stage 1)")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--step", type=int, default=192)
    ap.add_argument("--up", type=int, default=8, help="cross-correlation upsample factor")
    ap.add_argument("--max-shift", type=float, default=40.0)
    ap.add_argument("--max-pairs", type=int, default=0, help="limit stage 1 (0 = all)")
    ap.add_argument("--refine", action="store_true",
                    help="run the ZNCC refinement and the full validation table")
    ap.add_argument("--refine-scale", type=int, default=4)
    ap.add_argument("--out", default="rotation_model.json")
    ap.add_argument("--work", default=".astro_work")
    ap.add_argument("--overlay", default="", help="write a first/last frame overlay here")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    t0 = time.time()
    seq = ak.Sequence(args.src, args.first, args.last, args.pattern)
    print("%d frames %s .. %s" % (seq.n, os.path.basename(seq.path(1)),
                                  os.path.basename(seq.path(seq.n))))

    # ---- stage 1: displacement field
    tiles = Tiles(seq, args.scale, args.tile, args.step, args.up, args.max_shift)
    data = tiles.collect(args.max_pairs, t0=t0)
    np.save(os.path.join(args.work, "displacement.npy"), data)
    print("collected %d tile displacements [%.0fs]" % (len(data), time.time() - t0))
    pole, theta, resid, kept = fit_displacement(data, args.scale)
    print("stage 1: pole (%.0f, %.0f) full res  %+.5f deg/interval  "
          "resid p50 %.2f / p90 %.2f px  (%d tiles kept)"
          % (pole[0], pole[1], theta, resid["p50"], resid["p90"], kept))

    zncc_table = []
    if args.refine:
        size = (6048 // args.refine_scale, 4032 // args.refine_scale)
        im = Image.open(seq.path(1))
        size = (im.width // args.refine_scale, im.height // args.refine_scale)
        sky = sky_mask(seq, size, args.lit or None)
        pairs = parse_pairs(seq.n, (1, 2, 3, 4, 7, 11, 17))
        print("refining on %d pairs: %s" % (len(pairs), pairs))
        ref = Refiner(seq, sky, args.refine_scale, pairs)
        seeds = [np.array([pole[0], pole[1], theta]),
                 np.array([pole[0] * 1.006, pole[1], theta * 0.99]),
                 np.array([pole[0] * 0.99, pole[1] + 40.0, theta * 1.01])]
        best = None
        for i, s in enumerate(seeds):
            r = minimize(ref.objective, s, method="Nelder-Mead",
                         options={"maxiter": 260, "xatol": 2.0, "fatol": 0.02})
            print("  seed %d (%.0f, %.0f, %+.5f) -> (%.1f, %.1f, %+.5f)  "
                  "ZNCC %.4f  [%.0fs]"
                  % (i, s[0], s[1], s[2], r.x[0], r.x[1], r.x[2],
                     -r.fun / 100.0, time.time() - t0), flush=True)
            if best is None or r.fun < best.fun:
                best = r
        pole, theta = np.array(best.x[:2], float), float(best.x[2])
        print("stage 2: pole (%.0f, %.0f)  %+.5f deg/interval  ZNCC %.4f"
              % (pole[0], pole[1], theta, -best.fun / 100.0))

        fwd, rev = ref.sign_test(pole, theta)
        if rev > fwd:
            print("  !! the fit came out with the opposite sign "
                  "(+theta %.3f vs -theta %.3f) - flipping it" % (fwd, rev))
            theta = -theta
            fwd, rev = ref.sign_test(pole, theta)
        print("  sign check: warp(later, +theta) ZNCC %+.3f vs %+.3f reversed"
              % (fwd, rev))
        print("validation table (lag -> ZNCC of the later frame warped back):")
        zncc_table = [{"intervals": int(d), "zncc": float(z)}
                      for d, z in ref.table(pole, theta)]
        if args.overlay:
            make_overlay(seq, ref, pole, theta, args.overlay)

    doc = ak.save_model(args.out, pole, theta, first=seq.first, last=seq.last,
                        interval_s=args.interval_s, zncc=zncc_table,
                        residual=resid if not args.refine else None,
                        extra={"refined": bool(args.refine),
                               "scale_stage1": args.scale,
                               "source": os.path.abspath(args.src)})
    print("\nwrote %s  [%.0fs]" % (args.out, time.time() - t0))
    print("consume it with astro_kit.ADVANCE = -1 "
          "(the sky advances by -theta per interval)")


def make_overlay(seq, ref, pole, theta, path):
    """Red = first frame, cyan = last frame warped back with +theta."""
    a = seq.gray(1, (ref.w, ref.h))
    b = ak.warp(seq.gray(seq.n, (ref.w, ref.h)), (seq.n - 1) * theta,
                pole[0] / ref.scale, pole[1] / ref.scale, ref.w, ref.h)
    rgb = np.stack([a, b, b], -1)
    lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99.5)
    rgb = np.clip((rgb - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    print("wrote %s (star points overlapping = the model is right)" % path)


if __name__ == "__main__":
    main()
