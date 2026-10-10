# astro-timelapse

A Codex skill that turns a fixed-tripod night-sky photo sequence into
star-trail, comet-tail, or time-lapse videos and stacked stills. It calibrates
the sky's rotation, bridges the exposure gaps, deflickers, and lights a static
foreground that was painted with a torch on exactly one frame.

## Install

Clone it into your skills directory, so Codex discovers it on the next run.

    git clone https://github.com/LeyendaVv/astro-timelapse.git \
        "${CODEX_HOME:-$HOME/.codex}/skills/astro-timelapse"

On Windows:

    git clone https://github.com/LeyendaVv/astro-timelapse.git "$env:USERPROFILE\.codex\skills\astro-timelapse"

Then just ask for the job, e.g. *"use astro-timelapse on the frames in
D:\shoots\2026-10-04"*.

## Requirements

`numpy`, `Pillow`, `scipy`, and `ffmpeg` on PATH.

## What is in here

| Path | Contents |
|---|---|
| `SKILL.md` | What Codex loads: workflow, mode selection, the invariants that decide whether a render looks right |
| `references/pipeline.md` | Calibration method and its evidence, exposure-gap maths, grading formulas, parameter defaults, symptom-to-cause table |
| `scripts/check_intervals.py` | Reads EXIF and reports the frame spacing, because the model assumes it is uniform |
| `scripts/measure_rotation.py` | Two-stage calibration: tile phase correlation, then ZNCC refinement and a sign check |
| `scripts/astro_kit.py` | Shared library: sequence access, warping, foreground matte, grading, compositing, video encoding |
| `scripts/render_video.py` | The four render modes, the graded-frame cache, and preview stills |
| `scripts/probe_frame.py` | Caches one accumulator frame so compositing rules can be compared in seconds |

## Modes

`lapse` (round stars gliding), `growth` (trails grow out of the stars),
`stream` (every star drags a short tail), `comet` (continuous comet tails -
usually the best looking).

## Verified on

Two fixed-tripod milky-way sequences, 90 and 99 frames at 15 s exposures:

| Sequence | Foreground | Result |
|---|---|---|
| 90 frames, 48 s apart | tree lit on frame 30 only | all four modes render; the 720p test frame matches the delivered 4K master (mean 3.93 vs 3.21, p99 15 both) |
| 99 frames, 40 s apart, pole in a different quadrant | none | calibration reproduces the shipped model's sign and magnitude; comet render matches the delivered master |

The skill has been debugged against both: three real bugs in the scripts were
found and fixed this way (a tile grid laid out before the first frame was read,
an odd-height default that libx264 rejects, and a shape error in the alpha
compositing path).

## Limits

Satellite and aircraft lines are not removed. The rotation model is one angle
per frame index, so a sequence with irregular spacing must be trimmed to its
longest evenly spaced run first (`check_intervals.py` tells you which).

## License

MIT - see [LICENSE](LICENSE).
