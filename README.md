# scan_telescope_data

A Python script that walks a tree of astrophotography data, plate solves a few light frames from each observing session, and works out whether the session was shot on an equatorial (EQ) or alt-azimuth (alt-az) mount. It also tallies targets, filters, exposures and integration time. Results go to a CSV file, one row per session, filter and exposure.

## Background

The script started as a way to sort a large archive of Seestar S50 data. The Seestar runs alt-az by default and can also run in EQ mode, and over the years it has been used both ways, including with third-party software that spoofed an equatorial setup before ZWO added official EQ mode. Folder names and FITS headers don't reliably say which mode a session used, so the script measures it from the sky instead. It later grew to handle N.I.N.A. output too, so one run can cover both sources.

## How the mount is detected

A single plate solve tells you how the camera is oriented, which says nothing about the mount. What separates the two mount types is how that orientation changes over a session.

- On an equatorial mount the field stays put. A meridian flip on a German equatorial adds about 180 degrees.
- On an alt-az mount the field rotates by the change in parallactic angle, which depends on the target's position, the observing site and the time.

For each session the script solves a frame near the start and a frame near the end, takes the rotation of the later frame relative to the first, and compares it with both predictions. The equatorial prediction is zero change (or about 180 degrees after a flip). The alt-az prediction is computed from each frame's solved RA/Dec, the site latitude and longitude, and the frame's DATE-OBS time. Solver rotation sign conventions differ, so the alt-az comparison accepts either sign, applied consistently across the session.

A verdict needs the winning hypothesis to fit within its tolerance and the losing one to miss by clearly more. If the two ends can't separate the hypotheses, the script solves up to five frames spread evenly across the session and tries again. Two cases trigger this: an end-to-end change near 180 degrees, which fits both a meridian flip and an alt-az mount, and a session whose alt-az rotation swings through a large angle in the middle while ending near where it started.

Seestar scopes (S50, S30, anything with SEESTAR in the TELESCOP or INSTRUME header) can't meridian flip, since they rotate past the meridian instead. For those the flip case is removed, so equatorial means no change at all.

Each session ends up with one of these verdicts:

| Verdict | Meaning |
|---|---|
| EQUATORIAL | The field held still, or flipped by about 180 degrees |
| ALT-AZ | The field rotated as the parallactic angle predicts |
| INCONCLUSIVE | Not enough rotation, too few solved frames, or both hypotheses fit |
| CONFLICT | The rotation verdict disagrees with the EQMODE header |
| UNKNOWN | No frame in the session could be plate solved |

### Thresholds

These are constants near the top of the script.

| Constant | Value | Role |
|---|---|---|
| `FIT_TOLERANCE_DEG` | 8.0 | A mount type fits if every sampled frame agrees within this many degrees |
| `FLIP_TOLERANCE_DEG` | 12.0 | Wider equatorial tolerance when a meridian flip is involved (one real flip landed 8.8 degrees off 180) |
| `NEAR_ZENITH_DEG` | 10.0 | Closer than this to zenith, the alt-az prediction swings wildly with small errors |
| `NEAR_ZENITH_TOLERANCE_DEG` | 20.0 | Alt-az tolerance used for those near-zenith targets |
| `REJECT_DEG` | 2.0 | The losing hypothesis must miss by at least this much |
| `REJECT_RATIO` | 3.0 | ...and by at least this multiple of the winner's miss |
| `MIN_PREDICTED_ROTATION_DEG` | 2.0 | Alt-az rotation needed over the session to tell the mounts apart |
| `MIN_SWEEP_DEG` | 6.0 | Mid-session swing that justifies solving extra frames |
| `MOUNT_SAMPLE_FRAMES` | 5 | Maximum frames solved per session when the ends are ambiguous |
| `MIN_SPAN_NO_LOCATION_MIN` | 30.0 | Minimum session length to judge without a site location |

When the headers have no site location, the script reads `SITE_LATITUDE` and `SITE_LONGITUDE` from the environment. With neither available it falls back to the observed change alone: a session of at least 30 minutes whose rotation stays within 8 degrees is called equatorial, one that deviates by 24 degrees or more is called alt-az, and anything between is inconclusive.

### EQMODE cross-check

Newer Seestar firmware writes an `EQMODE` FITS header. When it is present, the script compares it with the rotation verdict. Measured rotation wins when decisive, and a disagreement is reported as CONFLICT. When rotation can't decide, the header value is used instead. The earliest frames in my archive that carry the header date from 2025-05-22, while ZWO added official EQ mode around March 2025, so the header does not date the switch from spoofing to native EQ mode.

### Pooling short sessions

A short run or a single mosaic panel barely rotates under alt-az, so it can't settle the question alone. After all sessions are processed, inconclusive ones are re-judged together with other sessions from the same night on the same instrument (sessions within 3 hours of each other). The pooled verdict is applied only when none of the sessions in the group was classified differently.

## How sessions are found and solved

1. Walks each directory you give it and collects folders named `light` or `lights`, in any case. It doesn't descend into those folders, which saves listing large folders twice on a network share, and it reports directories it can't read instead of skipping them silently.
2. Orders the FITS frames by the timestamp in the file name (Seestar `20260101-022917` and NINA `2026-08-16_03-52-56` styles), or by DATE-OBS when names lack one.
3. Splits the frames into sessions wherever the gap exceeds 6 hours (`SESSION_GAP_HOURS`), since a long gap can mean another night or a changed mount.
4. Prefers frames without `failed` in the name (Seestar's rejected frames) when choosing what to solve, but still counts them toward totals.
5. Solves one frame at each end of the session, then more only when the result is ambiguous.

Both ends of a session must come from the same solver, and for Seti Astro Suite Pro from the same mode, because rotations from different solvers aren't comparable.

### Solvers

Which solvers run, and in what order, depends on the options you pass.

| Invocation | Solver order |
|---|---|
| no solver option | ASTAP, then astrometry.net if an API key is set |
| `--sasp` | Seti Astro Suite Pro only (blind, then seeded) |
| `--sasp-fallback` | Seti Astro Suite Pro (blind, then seeded), then ASTAP, then astrometry.net |
| `--astrometry-only` | astrometry.net only |

Seti Astro Suite Pro is off unless you ask for it, so a plain run starts with ASTAP.

ASTAP is the default. Up to 3 frames per end are tried (`MAX_ASTAP_ATTEMPTS`), and the first attempt is "thorough": a hinted search with the header position, then a retry without downsampling, then a whole-sky search that ignores the header position entirely. The last step exists because some Seestar headers carry a position far from the real field (one 2024 frame was 30 degrees off). Folders where ASTAP already failed get only one attempt in later sessions. The Linux ASTAP package is the GUI build and would open a window for every solve, so the script sets Qt's offscreen platform; set `ASTAP_QT_PLATFORM` to an empty string to turn that off. The solution is read from the `.ini` file ASTAP writes, not from its console output, because the Linux build prints a different format than Windows.

astrometry.net is the fallback after ASTAP fails, using one frame. It needs an API key in `ASTROMETRY_NET_API_KEY`. Submissions are private, and each polling stage waits up to 10 minutes. `--astrometry-only` skips ASTAP, which helps with frames that have too few stars for it, such as narrowband filters.

Seti Astro Suite Pro (`--sasp`) uses its headless `solve` command. It solves blind first, trying up to 4 frames per end, and only then seeded with the header position (1 frame per end). The blind pass was the right default after a 1664-solve run of the archive, where 1650 solved on the first try and the other 14 on later ones, while all 18 seeded solves came on the first try. Rotation is taken from the blind solution's roll when at least 8 stars matched, which keeps every frame measured the same way even when the later refinement step fails on sparse narrowband fields. Otherwise it comes from `CROTA2` or the CD matrix in the `.ini` file. If only one end solves blind, the script retries both ends seeded. `--sasp` alone uses nothing else. `--sasp-fallback` also continues with ASTAP and astrometry.net, and if SASPro solves only one end, both ends are redone with ASTAP.

## Filters, exposures and objects

The filter comes from the file name when it matches a NINA or Seestar pattern, and from the FITS header otherwise. Common narrowband names are merged under one short name, so Hydrogen Alpha, H-alpha and Ha all report as Ha, and likewise for Oiii and Sii. Exposure time comes from the file name (for example `60.00s`). Every filter shot during a session shares that session's mount verdict.

Object names are normalized across catalogs (Messier, NGC, IC, Sharpless, Caldwell, Barnard, and common names), and mosaic panels are recognized from suffixes such as `_01`, `_1of4`, `_p1` and `_panel1`. The end-of-run summary groups sessions by normalized object and prints files, integration time (total and per filter), mount types and mosaic panels. When a frame's header has no TELESCOP value, a `TELESCOPE_DEFAULTS` table maps the camera name to a telescope name; it currently maps `ATR585M` to `RedCat 51`.

## Requirements

- Python 3 with `astropy`, `numpy` and `requests`, listed in `requirements.txt` (`requests` is imported at startup even if astrometry.net is never used). Tested with Python 3.14.10, astropy 8.0.2, numpy 2.5.4 and requests 2.33.1. A virtual environment keeps them away from the system Python:

  ```bash
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -r requirements.txt
  ```
- [ASTAP](https://www.hnsky.org/astap.htm) and a star database. The script expects `/usr/bin/astap` and `/usr/share/astap/data`, set by `ASTAP_PATH` and `ASTAP_DATABASE` in the configuration block. It was first written on Windows, so those two paths are what to change when moving between platforms.
- Optional: Seti Astro Suite Pro, found through `SASP_PATH` (default `~/.local/bin/setiastrosuitepro`). It needs the Gaia data installed and its indexes created before it can solve anything ([Seti Astro Suite - Blind Solve Instructions](https://www.youtube.com/watch?v=MLroe8W_Zh4)). `SASP_SOLVER` picks its backend (`gaia` by default, or `astap`, `astrometry`, `all`).
- Optional: an astrometry.net API key from <https://nova.astrometry.net/api_help>

## Usage

```bash
python3 scan_telescope_data.py [-o output.csv] [--sasp | --sasp-fallback] [--astrometry-only] <directory> [<directory> ...]
```

| Option | Effect |
|---|---|
| `-o`, `--output` | CSV file name (default `telescope_analysis.csv`) |
| `--sasp` | Solve with Seti Astro Suite Pro only |
| `--sasp-fallback` | Solve with SASPro first, then ASTAP and astrometry.net if it fails |
| `--astrometry-only` | Skip ASTAP and solve with astrometry.net |

| Environment variable | Purpose |
|---|---|
| `ASTROMETRY_NET_API_KEY` | astrometry.net key |
| `SITE_LATITUDE`, `SITE_LONGITUDE` | Site location in degrees, east longitude positive, used when headers lack one |
| `SASP_PATH`, `SASP_SOLVER` | Location and backend of Seti Astro Suite Pro |
| `ASTAP_QT_PLATFORM` | Qt platform for the ASTAP GUI build (default `offscreen`) |

Example:

```bash
export SITE_LATITUDE=32.9 SITE_LONGITUDE=-96.7
python3 scan_telescope_data.py -o telescope_analysis.csv /path/to/nina /path/to/seestar
```

## Output

`telescope_analysis.csv` has one row per session, filter and exposure combination, with these columns:

`Target`, `Normalized Name`, `Catalog Type`, `Is Mosaic`, `Mosaic Details`, `Mount Type`, `Mount Evidence`, `EQMODE`, `Filter`, `Session`, `Date`, `Exposure`, `Frames`, `Integration (h)`, `Directory`, `Latitude`, `Longitude`, `Total Files`, `Instrument`, `Telescope`, `Solved File`, `Solver Used`, `Header Offset (deg)`, `Attempts`

`Mount Evidence` records the numbers behind each verdict: how many frames were solved, the observed end-to-end rotation, what alt-az predicts, and the worst miss for each hypothesis. `Header Offset (deg)` is the angular distance between the header pointing and where the frame actually solved, so large values flag frames whose header position isn't the real field. That includes the Seestar sessions shot before official EQ mode, which were spoofed to a different position.

A second file, `<output>_solve_stats.csv`, logs every solver pass (method, end of session, tries, outcome, seconds, folder). The same data is summarized on screen at the end of a run, with a histogram of how many frames each solver needed, which is how the per-end attempt limits were chosen.

The console summary also reports how many sessions were classified, how many were inconclusive, conflicting or unsolved, which solvers were used, and how many directories had no FITS files.

## Results so far

All of the NINA data scanned to date (ATR585M and ASCOM ToupTek cameras, February to October 2026) was shot on an EQ mount, and the script classified it that way. The Seestar archive is mostly alt-az, with some sessions in EQ mode.

## Known limitations

- Sessions shot through an external 7nm narrowband filter (the Oct 2024 to Jan 2025 Orion-area nebulae, for example) leave too few stars for ASTAP. Use `--astrometry-only` or SASPro on those.
- Spoofed-EQ Seestar sessions have a fixed polar misalignment of roughly 25 to 30 degrees, which produces steady field rotation that can look like alt-az behavior. They are the hardest case to classify.
- Targets that pass within about 10 degrees of zenith make the alt-az prediction unreliable, so the script allows a wider tolerance there.
- Sidereal time is computed from the Julian date without IERS data. That is accurate to well under a second, which is plenty here.
- Very short sessions rarely rotate enough to separate the mounts and often come back inconclusive unless pooling resolves them.

## History

The commit history was reconstructed from saved revisions of the script, so each commit is one of those revisions.

## License

GPL-3.0. See [LICENSE](LICENSE).
