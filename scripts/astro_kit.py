"""Shared helpers for the astro-timelapse skill.

Plain numpy / Pillow / scipy plus ffmpeg on PATH, so the calibrator, the
renderer and the single-frame probe can all import the same code.
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None

W_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# The stored theta is the angle that maps a LATER frame back onto an EARLIER
# one, so the sky itself advances by -theta per interval.  measure_rotation.py
# verifies this by scoring both signs; renderers must not flip it.
ADVANCE = -1.0


class Sequence:
    """A numbered photo sequence, e.g. DSC_4944.JPG ... DSC_5033.JPG."""

    def __init__(self, src, first, last, pattern="DSC_%04d.JPG"):
        self.src = src
        self.first = int(first)
        self.last = int(last)
        self.pattern = pattern
        self.n = self.last - self.first + 1

    def path(self, i):
        """Path of frame i, counting from 1."""
        return os.path.join(self.src, self.pattern % (self.first + i - 1))

    def gray(self, i, size, highpass=0.0):
        a = np.asarray(Image.open(self.path(i)).convert("L").resize(
            size, Image.BOX)).astype(np.float32)
        if highpass:
            a = a - ndimage.gaussian_filter(a, highpass)
        return a

    def rgb(self, i, size, resample=Image.LANCZOS):
        return np.asarray(Image.open(self.path(i)).convert("RGB").resize(
            size, resample))


def warp(a, ang_deg, cx, cy, w=None, h=None, resample=Image.BICUBIC):
    """Rotate image content by +ang_deg about (cx, cy).

    Works on float / uint8 grayscale and on uint8 RGB arrays; PIL has no
    float32 RGB mode, so callers must keep colour frames 8-bit.
    """
    h = h or a.shape[0]
    w = w or a.shape[1]
    t = np.radians(ang_deg)
    ct, st = np.cos(t), np.sin(t)
    R = np.array([[ct, st], [-st, ct]])
    c = np.array([cx, cy], dtype=float)
    b = c - R @ c
    return np.asarray(Image.fromarray(a).transform(
        (w, h), Image.AFFINE,
        (R[0, 0], R[0, 1], b[0], R[1, 0], R[1, 1], b[1]), resample=resample))


def load_model(path):
    """Return (pole_full_res_xy, theta_deg_per_interval)."""
    m = json.load(open(path, encoding="utf-8"))
    return np.array(m["pole_full"], dtype=float), float(m["theta_deg_per_interval"])


def save_model(path, pole_full, theta, *, first, last, interval_s,
               zncc=None, residual=None, extra=None):
    doc = {
        "pole_full": [float(pole_full[0]), float(pole_full[1])],
        "theta_deg_per_interval": float(theta),
        "interval_s": float(interval_s),
        "first_frame": int(first),
        "last_frame": int(last),
        "warp_convention":
            "content rotated by +theta per interval maps a later frame back "
            "onto an earlier one; the sky advances by -theta",
        "zncc": zncc or [],
    }
    if residual is not None:
        doc["residual_px"] = {k: float(v) for k, v in residual.items()}
    if extra:
        doc.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    return doc


# ---------------------------------------------------------------- foreground

def foreground_matte(seq, lit, size, *, m_lo=3.0, m_hi=16.0, smooth=1.5,
                     area=20000, blank=8, neighbours=(-6, -3, 3, 6),
                     right_limit=0.85, out_dir=None, verbose=True):
    """Soft coverage of a static foreground that is lit in exactly one frame.

    Driven by how much brighter each pixel is in the light-painted frame than
    in its neighbours, so the matte follows leaves and twigs instead of being a
    dilated silhouette.  Returns (m, m_blank, box) as arrays at `size`:

      m        soft coverage, 0 = sky, 1 = foreground
      m_blank  m grown by `blank` px, used to keep that one frame out of the
               stack so its bright foreground cannot ghost
      box      (y0, y1, x0, x1) of m > 0.02
    """
    def gray(i):
        return seq.gray(i, size)

    lit_g = gray(lit)
    ref = np.median(np.stack([gray(lit + d) for d in neighbours]), axis=0)
    diff = ndimage.gaussian_filter(lit_g - ref, 2.0)
    m = np.clip((diff - m_lo) / (m_hi - m_lo), 0.0, 1.0).astype(np.float32)
    m = ndimage.gaussian_filter(m, smooth)

    # keep the foreground, drop halos of stars that happen to be lit only here
    hard = m > 0.35
    hard = ndimage.binary_closing(hard, np.ones((5, 5), bool))
    lab, n = ndimage.label(hard)
    if n:
        sizes = ndimage.sum(hard, lab, range(1, n + 1))
        coms = ndimage.center_of_mass(hard, lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        for j, (size_j, (cy, cx)) in enumerate(zip(sizes, coms), start=1):
            keep[j] = (size_j > area) and (cx < right_limit * size[0])
        if not keep.any():                      # never drop everything
            keep[1 + int(np.argmax(sizes))] = True
        if verbose:
            dropped = [(int(s), int(c[0]), int(c[1])) for s, c, k in
                       zip(sizes, coms, keep[1:]) if not k and s > 400]
            print("  dropped %d sky blobs (e.g. %s)" % (len(dropped), dropped[:6]))
        m *= keep[lab].astype(np.float32)

    solid = m > 0.02
    reach = ndimage.distance_transform_edt(~solid) <= blank
    m_blank = np.maximum(m, reach.astype(np.float32))
    ys, xs = np.nonzero(solid)
    box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    if verbose:
        print("foreground: matte %.1f%% solid, %.1f%% touched, blanked %.1f%%"
              % (100.0 * (m > 0.5).mean(), 100.0 * solid.mean(),
                 100.0 * (m_blank > 0.5).mean()))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray((m * 255).astype(np.uint8)).save(
            os.path.join(out_dir, "matte.png"))
        Image.fromarray((m_blank * 255).astype(np.uint8)).save(
            os.path.join(out_dir, "matte_blank.png"))
    return m, m_blank, box


def overlay_parts(lit_rgb, m, box, matte_gain=1.4):
    """(lit foreground, weighted coverage, raw matte, box) for the lighten pass.

    `matte_gain` saturates the coverage over the leaves so they keep the value
    they have in the light-painted frame; the outer ramp still fades to zero.
    """
    y0, y1, x0, x1 = box
    lit = lit_rgb.astype(np.float32)[y0:y1, x0:x1]
    mm = m[y0:y1, x0:x1]
    w_lit = np.clip(mm * matte_gain, 0.0, 1.0)[:, :, None]
    return lit, w_lit, mm, box


# -------------------------------------------------------------------- grading

def analyse(seq, sky_small, *, wb_strength=0.8, deflicker=0.8, small=(378, 252)):
    """Per-frame white balance and deflicker gain, measured on sky pixels only.

    Restricting the estimate to sky keeps a dark foreground from dragging the
    background level down.  Returns (wbs[n,3], gains[n]).
    """
    wbs, lv = [], []
    for i in range(1, seq.n + 1):
        a = np.asarray(Image.open(seq.path(i)).convert("RGB").resize(
            small, Image.BOX)).astype(np.float32)
        lum = a @ W_LUMA
        m = (lum < np.percentile(lum, 60)) & sky_small
        bg = a[m].mean(axis=0)
        w = (bg.mean() / bg) ** wb_strength
        w = w / (w @ W_LUMA)
        wbs.append(w)
        lv.append(float((bg * w) @ W_LUMA))
    lv = np.array(lv)
    gains = np.clip((np.median(lv) / lv) ** deflicker, 0.8, 1.5)
    print("deflicker %.3f..%.3f ; sky level p50 %.1f/255"
          % (gains.min(), gains.max(), np.median(lv)))
    return np.array(wbs), gains


def make_lut(gain, wb, *, gamma=0.84):
    x = np.arange(256, dtype=np.float64)
    lut = np.empty((3, 256), dtype=np.uint8)
    for ch in range(3):
        v = np.clip(x * gain * wb[ch], 0, 255.0)
        lut[ch] = np.clip(255.0 * (v / 255.0) ** gamma, 0, 255).astype(np.uint8)
    return lut


def grade_frame(seq, i, lut, size, *, denoise=1.0, sat=1.16):
    a = np.asarray(Image.open(seq.path(i)).convert("RGB").resize(size, Image.LANCZOS))
    if denoise:
        a = ndimage.gaussian_filter(a.astype(np.float32), (denoise, denoise, 0.0))
    out = np.empty(a.shape, dtype=np.uint8)
    for ch in range(3):
        out[:, :, ch] = lut[ch][np.clip(a[:, :, ch], 0, 255).astype(np.uint8)]
    x = out.astype(np.float32)
    lum = x @ W_LUMA
    x = lum[:, :, None] + (x - lum[:, :, None]) * sat
    return np.clip(x, 0, 255).astype(np.uint8)


class GradeCache:
    """Graded frames, memoised on disk so every render mode reuses them."""

    def __init__(self, seq, luts, size, *, denoise=1.0, sat=1.16,
                 cache_dir=None, keep=4):
        self.seq = seq
        self.luts = luts
        self.size = tuple(size)
        self.denoise = denoise
        self.sat = sat
        self.cache_dir = cache_dir
        self.keep = keep
        self._mem = {}

    def path_for(self, i):
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, "%03d.npy" % i)

    def __getitem__(self, i):
        if i in self._mem:
            return self._mem[i]
        p = self.path_for(i)
        if p and os.path.exists(p):
            res = np.load(p)
        else:
            res = grade_frame(self.seq, i, self.luts[i - 1], self.size,
                              denoise=self.denoise, sat=self.sat)
            if p and os.path.isdir(self.cache_dir):
                np.save(p, res)
        self._mem[i] = res
        while len(self._mem) > self.keep:
            self._mem.pop(next(iter(self._mem)))
        return res

    def build(self, verbose=True):
        assert self.cache_dir, "build() needs a cache directory"
        os.makedirs(self.cache_dir, exist_ok=True)
        for i in range(1, self.seq.n + 1):
            self[i]
            if verbose and i % 10 == 0:
                print("  cached %d/%d" % (i, self.seq.n), flush=True)


# -------------------------------------------------------------------- output

def view_box(zoom, med_w, med_h, out_w, out_h):
    """Source rectangle (medium pixels) the output frame shows."""
    w = med_w / zoom
    h = w * out_h / float(out_w)
    ox = (med_w - w) / 2.0
    oy = (med_h - h) / 2.0
    return (ox, oy, ox + w, oy + h)


def compose(acc, zoom, ov, out_size, *, rule="lighten", med_w=None, med_h=None):
    """Add the lit foreground over the stack, then centre-crop/zoom to output.

    `lighten` (max) adds light without ever subtracting any, so star trails keep
    running through the gaps of a static foreground.  `alpha` is kept for
    comparison but must not be used for a stack: the light-painted frame's sky
    is darker than the trails it would replace.
    """
    if ov is not None:
        lit, w_lit, mm, (y0, y1, x0, x1) = ov
        sub = acc[y0:y1, x0:x1]
        if rule == "lighten":
            blend = np.maximum(sub.astype(np.float32), lit * w_lit)
        elif rule == "screen":
            l = lit * w_lit
            s = sub.astype(np.float32)
            blend = s + l - s * l / 255.0
        elif rule == "alpha":
            s = sub.astype(np.float32)
            m3 = mm[:, :, None]
            blend = s * (1.0 - m3) + lit * m3
        else:
            raise ValueError("unknown composite rule %r" % rule)
        if acc.dtype != np.uint8:
            acc = np.clip(acc, 0, 255).astype(np.uint8)
        elif not acc.flags.writeable:
            acc = acc.copy()
        acc[y0:y1, x0:x1] = np.clip(blend, 0, 255).astype(np.uint8)
    elif acc.dtype != np.uint8:
        acc = np.clip(acc, 0, 255).astype(np.uint8)
    box = view_box(zoom, med_w, med_h, out_size[0], out_size[1])
    return np.asarray(Image.fromarray(acc).resize(out_size, Image.BICUBIC, box=box))


def ffmpeg_cmd(dst, size, fps, *, crf=17, preset="slow", profile="high",
               level="5.1", duration=None, fade_in=0.0, fade_out=0.0,
               ffmpeg="ffmpeg"):
    if size[0] % 2 or size[1] % 2:
        raise ValueError("libx264 needs even width/height, got %dx%d"
                         % (size[0], size[1]))
    vf = []
    if fade_in:
        vf.append("fade=t=in:st=0:d=%g" % fade_in)
    if fade_out:
        vf.append("fade=t=out:st=%g:d=%g" % (duration - fade_out, fade_out))
    cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % size, "-r", str(fps), "-i", "-"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-profile:v", profile, "-level", level,
            "-movflags", "+faststart", dst]
    return cmd


def write_video(frames, dst, size, fps, *, verbose_every=25, log=None, **kw):
    """Stream RGB frames through ffmpeg into `dst`."""
    tmp = dst + ".part.mp4"
    proc = subprocess.Popen(ffmpeg_cmd(tmp, size, fps, **kw), stdin=subprocess.PIPE)
    n = 0
    for n, arr in enumerate(frames, start=1):
        proc.stdin.write(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())
        if log and verbose_every and (n - 1) % verbose_every == 0:
            print("    %s frame %d" % (log, n - 1), flush=True)
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed with code %d" % proc.returncode)
    os.replace(tmp, dst)
    print("wrote %s (%d frames)" % (dst, n))


_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/PingFang.ttc",
]


def font(size):
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def contact_sheet(tiles, path, *, labels=None, cols=None, gap=8, label_h=30,
                  font_size=24, label_color=(255, 235, 190)):
    """Tile PIL images into one sheet with optional captions."""
    tiles = [t.convert("RGB") for t in tiles]
    cols = cols or len(tiles)
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (cols * tw + gap * (cols - 1),
                              rows * (th + label_h) + gap * (rows - 1)), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    f = font(font_size)
    for k, t in enumerate(tiles):
        x = (k % cols) * (tw + gap)
        y = (k // cols) * (th + label_h + gap)
        sheet.paste(t, (x, y + label_h))
        if labels and k < len(labels):
            d.text((x + 10, y + 4), labels[k], font=f, fill=label_color)
    sheet.save(path)
    print("wrote %s" % path)
    return sheet
