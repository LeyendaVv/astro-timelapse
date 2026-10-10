---
name: astro-timelapse
description: Turn a fixed-tripod night-sky photo sequence into star-trail, comet-tail, or time-lapse videos and stacked stills - calibrate the sky's rotation, bridge the exposure gaps, deflicker, and light a static foreground. Use when the input is a numbered folder of frames (DSC_4944.JPG ... DSC_5033.JPG) rather than ordinary video footage.
---

# Astro Timelapse

Turn a fixed-tripod night-sky sequence of numbered frames into video. Four looks
come out of one calibrated rotation model; only the accumulation rule differs.

Needs numpy, Pillow, scipy, and `ffmpeg` on PATH. All scripts live in
`scripts/` and import `astro_kit.py`, so run them from that directory.

## Workflow

0. Check the frame spacing before anything else. The model is one theta per
   frame *index*, so irregular gaps quietly drag both the fit and the
   validation numbers.

   ```
   python check_intervals.py --src FRAMES --first 4944 --last 5033
   ```

   Feed the fit the longest evenly spaced run it reports.

1. Calibrate the sky's rotation once per sequence. `--lit` is the 1-based index
   of the light-painted frame; omit it for a foreground-free sequence.

   ```
   python measure_rotation.py --src FRAMES --first 4944 --last 5033 \
       --lit 30 --interval-s 48 --refine --out rotation_model.json
   ```

   Read the output before trusting it: the sign check must prefer `+theta`, lag 1
   ZNCC must be clearly higher than the reversed sign, and ZNCC should decay
   smoothly with lag. With `--overlay` the star points of the two frames must
   land on top of each other. Pole and theta sit on a flat ridge and can move a
   few percent without changing the score, so judge the sign and the table, not
   the last digit.

2. Build the graded-frame cache once; every mode reuses it.

   ```
   python render_video.py --src FRAMES --first 4944 --last 5033 --lit 30 \
       --model rotation_model.json --mode cache
   ```

3. Render. Check stills with `--preview --preview-times 2,9.5` and the whole
   pipeline fast with `--test` before committing to a 4K pass.

   ```
   python render_video.py --src FRAMES --first 4944 --last 5033 --lit 30 \
       --model rotation_model.json --mode comet --tau 13 --name 白杨银河
   ```

4. To compare foreground compositing rules, do not re-render: `probe_frame.py`
   accumulates one output frame, caches it, and re-applies each rule in seconds.
   The accumulator is keyed on a hash of the render parameters and the model
   file, so it rebuilds instead of silently comparing a stale render.

## Mode selection

| mode | accumulation | looks like |
|---|---|---|
| `lapse` | one frame at a time, warped to the output phase | round stars gliding; a clean time-lapse |
| `growth` | per-frame max stack, no matte | trails grow out of the stars; last frame is a full star-trail still |
| `stream` | max stack with exponential decay, `--tau` in output frames | every star drags a short tail |
| `comet` | pole rotation + bridged sub-samples + decay, `--tau` in source intervals | continuous comet tails with no beading; usually the best |

`--tau` reads as star rain around 8, balanced around 13, and closer to a star
trail around 22. A static foreground needs no matte in `growth`: the max stack
already keeps its one lit appearance.

## Invariants that decide whether it looks right

- **Sign.** The stored `theta_deg_per_interval` maps a *later* frame back onto
  an *earlier* one, so the sky advances by `-theta`. `astro_kit.ADVANCE`
  encodes this and renderers must not flip it. A flipped sign turns trails into
  beads and is the most common failure.
- **Exposure gap.** When the exposure is much shorter than the interval, a plain
  stack is a string of beads. `--subs N` warps each interval's frame to several
  phases `(u - u_k) * ADVANCE * theta` so the interval stays covered. Keep every
  sample in the original image plane; do not add camera-style rotation.
- **Foreground joins by lighten, never alpha.** The light-painted frame's own
  sky and leaf gaps are darker than the trail stack, so alpha punches a
  trail-free hole and leaves a hard dark rim. Build the matte from
  `bright frame - median(neighbour frames)` so it follows twigs and leaves, and
  keep the minimum blob area far above star-halo size or bright stars get
  treated as foreground.
- **Blank only the light-painted frame.** Grow the matte by `--blank` px and
  remove that one frame from the stack there so its bright foreground cannot
  ghost as the sky rotates. The dark-silhouette frames stay in.
- **Deflicker measures on sky pixels only**, or a dark foreground drags the
  background estimate down. The light-painted frame keeps gain 1.0; deflicker
  there would only flatten its foreground.
- **Judge composites on 1:1 crops** near the foreground. Downscaled thumbnails
  hide exactly the artifact that matters.
- **Cache graded frames on disk** - that is the expensive step. On a 16 GB
  machine run at most two renders at once, and give concurrent renders distinct
  output names, since the ffmpeg temp file is derived from the output name.

## Deliverable convention

Unless the user asks otherwise: 3840x2160, 25 fps, 250 frames = exactly 10.00 s,
fade in 0.6 s / out 0.8 s, no audio, named
`<name>_<mode>_<duration>s_<height>p.mp4`, plus a 1920x1080 copy. Keep extracted
stills, contact sheets, the model JSON, and a short production note next to the
renders so a later pass can reproduce them.

## More detail

Read [references/pipeline.md](references/pipeline.md) for the calibration
method and its evidence, the exposure-gap math, the grading formulas, the
per-parameter defaults, and a symptom-to-cause troubleshooting table.
