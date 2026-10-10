# Pipeline detail

Everything here comes from real passes over two fixed-tripod milky-way
sequences (91 and 101 frames, 15 s exposures at 40-48 s intervals). Read the
section you need; the entrypoint already carries the invariants.

## Inputs

`Sequence` assumes a numbered pattern, default `DSC_%04d.JPG`, and `--first` /
`--last` are the frame numbers, not offsets. Frames may be JPEG; RAW has to be
developed first, since the scripts read through Pillow.

`--lit` is a 1-based index into that sequence: frame number `first + lit - 1`.

## Rotation calibration

Before fitting, confirm the frames are evenly spaced (`check_intervals.py`).
theta is degrees per *frame index*, so one frame that arrived 46 s after its
neighbour instead of 40 s is read as if it held 40 s worth of rotation. Measured
on a 99-frame sequence: that single off gap took lag-1 ZNCC from 0.835 down to
0.609 and pulled theta from 0.1053 to 0.1066 deg/interval. Everything after the
first few frames of that sequence was exactly 40 s, so trimming to the evenly
spaced run restored the table - check the spacing before blaming the optics or
the fit.

`measure_rotation.py` runs two independent stages, because either alone is easy
to get wrong.

1. **Displacement field.** Tile phase correlation between consecutive frames at
   `1/--scale` resolution gives a dense field; a robust iterative fit of
   `v = theta * (-(y-cy), (x-cx))` recovers pole and rate without tracking
   individual stars. Report prints residual p50/p90 in tile pixels.
2. **`--refine`.** Scores the model by what the renderer actually needs - the
   ZNCC of the sky after warping a later frame back onto an earlier one - over
   pairs spread across the whole sequence, and optimizes pole x, pole y and
   theta with Nelder-Mead from several seeds. All seeds should converge to the
   same solution; if they do not, distrust the model. The sky mask comes from
   the foreground matte when `--lit` is given, so foreground pixels cannot vote.

`sky_sign()` then scores both signs over several separations and flips the
emitted theta if the fit came out the other way, so a model written by this
script can always be consumed with `ADVANCE = -1`.

Do not read precision into pole and theta beyond the sign and the table. The
objective has a long flat ridge between them: on one sequence two solutions
3.3 deg of sky rotation apart - pole (-465, -1600) at 0.1116 deg/interval, and
pole (-680, -1817) at 0.1053 - scored ZNCC 0.606 and 0.613, which is not a
meaningful difference. Both render correctly. What the fit is reliable for is
the sign, the order of magnitude, and a ZNCC table that decays smoothly; a
hand-tuned start is only needed when the seeds disagree.

Measured example (24 mm, 48 s interval, 89 intervals): pole (7373, 31) in
6048x4032 coordinates - outside the frame, upper right - at 0.13118 deg per
interval. `warp(frame1, -theta)` against frame2 gives sky ZNCC 0.833 versus
-0.006 reversed. Long baselines decorrelate honestly: ZNCC falls to roughly
0.59 at 4 intervals, 0.27 at 16, 0.04 at 89, because per-frame noise,
atmosphere and light level all differ. That is why long-star-trail renders look
slightly furry while comet renders, which bridge only about ±0.75 interval,
stay crisp.

## Exposure gap

Exposure / interval is the fraction of each interval actually recorded. At
15 s / 48 s a plain stack is a chain of dots. The fix is not interpolation
between rendered frames: warp each interval's source frame to sub-phases
`(u - u_k) * ADVANCE * theta` and accumulate those, with `--subs 4` sub-samples
per interval. The rotation is genuine sky rotation inside the original image
plane, so nothing about the composition changes.

## Grading

`analyse()` estimates per-frame white balance and a deflicker gain on sky
pixels only, using the lower 60% of luminance within the sky mask so the
background level is not dragged down by a dark foreground:

```
w = (mean(bg) / bg) ** wb_strength,  renormalised so w @ luma == 1
gain = clip((median(level) / level) ** deflicker, 0.8, 1.5)
```

`make_lut()` bakes gain and white balance into a per-channel LUT and applies
gamma (`--gamma 0.84` by default). A 1 px pre-denoise (`--denoise 1.0`) pays off
on high-ISO sequences; saturation is applied after the LUT (`--sat 1.16`).
The light-painted frame's LUT is rebuilt with gain 1.0.

`GradeCache` writes the graded frames as `.npy` under
`work/graded_<w>x<h>_<hash>/`, keyed on the parameters that change the pixels,
so `--mode cache` is the only slow pass.

## Foreground matte and composite

- Matte: `lit_gray - median(neighbour frames at -6,-3,+3,+6)`, Gaussian smoothed,
  mapped linearly from `--matte-lo 3` to `--matte-hi 16` grey levels, then
  connected components filtered by `--mask-area` and by centroid position
  (`right_limit`, for foregrounds that can only be on one side). If everything
  would be dropped, the largest blob is kept.
- `m_blank` grows the matte by `--blank` px; it only excludes the light-painted
  frame from the stack, it is not the composite mask.
- Composite: `compose()` supports `lighten` (max, the default),
  `screen` and `alpha`, with `--matte-gain 1.4` saturating the coverage on
  leaves. `alpha` exists for comparison and must not be used on a stack.
- In `growth` mode no matte is used at all: a static foreground is already
  correct in a max stack because only one frame lights it.

Numeric anchor for `--mask-area`: the tree body was two blobs of 2.66 M and
0.55 M px at 4096 width, while the largest star halo was about 6 k px, so
20000 is a safe floor. At 400 px, bright star halos were taken for foreground
and the composite replaced patches of sky, leaving a row of round black spots.

## Output and verification

`write_video()` streams RGB frames into `ffmpeg` (libx264, CRF 17, preset slow,
yuv420p, `+faststart`) and writes to `<out>.part.mp4` before replacing the
destination, so a failed render never leaves a half file at the final name.

Verification that is actually worth doing:

- Sign check and lag->ZNCC table from `measure_rotation.py`.
- `--overlay` first/last frame: star points must coincide.
- `--preview` stills plus a `probe_frame.py` sheet at 1:1 near the foreground.
- On a new sequence, one `--test` pass end to end before any 4K render.

Rough 4K costs with the cache warm on a 16 GB machine: growth ~2 min,
stream ~4 min, lapse ~7 min, comet ~8 min, plus about 35 s of setup per run.
Three concurrent renders start swapping and become slower than two.

## Symptom -> cause

| Symptom | Cause | Fix |
|---|---|---|
| Trails are a chain of dots | exposures shorter than the interval | raise `--subs` (4 is usually enough) |
| Motion runs backwards, dots fragment | `ADVANCE` sign or a model written with the opposite convention | re-run `measure_rotation.py --refine` and keep its sign |
| Hard dark rim, opaque patch over the foreground, trails erased under it | alpha/coverage composite | `--composite lighten` |
| Row of round black spots | matte area threshold too low, star halos kept | raise `--mask-area` |
| Bright foreground ghosts elsewhere in the frame | light-painted frame not blanked along the matte | raise `--blank` |
| A band with no trails just outside the foreground | blanked every frame, or a wide hard feather | blank only the lit frame and keep the matte soft |
| Background pulses frame to frame | deflicker missing or estimated including foreground | check `--deflicker` and the sky mask |
| Leaves darker than in the light-painted frame | matte not saturated | raise `--matte-gain` |
| Furry, wide trails after many intervals | honest long-baseline decorrelation | expected; prefer `comet` for crisp trails |
| Lag table worse than the same rig on another night, trails wider than expected | irregular frame spacing fed to an index-based model | `check_intervals.py`, then fit the longest evenly spaced run |

## Known tradeoffs

- Satellite and aircraft lines are not removed; masking them is a separate pass.
- The lit foreground comes from a single frame, so it carries its own noise
  level and is not stacked like the sky.
