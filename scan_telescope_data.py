from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u
import numpy as np
import sys
import os
import subprocess
import re
import glob
import csv
import time
import tempfile
import requests
import json
from datetime import datetime, timedelta

# Configuration
ASTAP_PATH = r"/usr/bin/astap"
ASTAP_DATABASE = r"/usr/share/astap/data"
# Set this in your environment instead of hardcoding it:
#   export ASTROMETRY_NET_API_KEY="your key from nova.astrometry.net"
ASTROMETRY_NET_API_KEY = os.environ.get("ASTROMETRY_NET_API_KEY", "")
MAX_ASTAP_ATTEMPTS = 3  # Files to try with ASTAP at each end of the session before using astrometry.net
ASTAP_TIMEOUT_SEC = 60  # A solve that takes longer than this is abandoned
SESSION_GAP_HOURS = 6.0  # A gap this long between frames starts a new observing session

# Mount detection settings
MOUNT_SAMPLE_FRAMES = 5           # Total frames solved when the two ends are ambiguous
FIT_TOLERANCE_DEG = 8.0           # A mount type fits if every sampled frame agrees within this
REJECT_DEG = 2.0                  # The losing mount type must miss by at least this much (solves agree to ~0.3 deg)
REJECT_RATIO = 3.0                # ...and by at least this multiple of the winner's miss
MIN_PREDICTED_ROTATION_DEG = 2.0  # Alt-az rotation needed over the session to tell the mounts apart
MIN_SWEEP_DEG = 6.0               # Mid-session swing in alt-az rotation that justifies solving extra frames
MIN_SPAN_NO_LOCATION_MIN = 30.0   # Minimum session length to judge without a site location

# Used only when the FITS headers have no site location (east longitude positive):
#   export SITE_LATITUDE=32.9 SITE_LONGITUDE=-96.7
ENV_SITE_LATITUDE = os.environ.get("SITE_LATITUDE")
ENV_SITE_LONGITUDE = os.environ.get("SITE_LONGITUDE")

LAT_KEYS = ['SITELAT', 'LATITUDE', 'OBSLAT', 'LAT-OBS', 'LAT']
LON_KEYS = ['SITELONG', 'LONGITUDE', 'OBSLONG', 'LONG-OBS', 'LON', 'LONG']


def read_astap_ini(path):
    """Read the key=value summary file that ASTAP writes next to its output base name."""
    values = {}
    with open(path) as f:
        for line in f:
            if '=' in line:
                key, val = line.split('=', 1)
                values[key.strip().upper()] = val.strip()
    return values


def create_wcs_from_solution(ra_deg, dec_deg, rotation, header):
    """Create an approximate WCS from solved coordinates and rotation (used for astrometry.net results)"""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [header.get('NAXIS1', 1080) / 2, header.get('NAXIS2', 1920) / 2]

    angle_rad = np.radians(rotation)
    scale = 0.0005  # Approximate scale in degrees/pixel

    wcs.wcs.cd = [
        [scale * np.cos(angle_rad), -scale * np.sin(angle_rad)],
        [scale * np.sin(angle_rad), scale * np.cos(angle_rad)]
    ]

    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


# ---------------------------------------------------------------------------
# Mount type detection
#
# A single frame's rotation only says how the camera is oriented. What separates
# the mount types is how that rotation changes during the session:
#   equatorial: the field stays put (a meridian flip adds exactly 180 degrees)
#   alt-az:     the field rotates by the change in parallactic angle
# ---------------------------------------------------------------------------

def wrap180(angle):
    """Wrap an angle in degrees to the range [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def to_float_deg(value):
    """Parse a header latitude/longitude (number or sexagesimal string) to degrees, or None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(Angle(str(value).strip(), unit=u.deg).deg)
    except Exception:
        return None


def get_site_location(header):
    """Return (latitude, longitude) in degrees from the header, falling back to the environment."""
    lat = lon = None
    for key in LAT_KEYS:
        if key in header:
            lat = to_float_deg(header[key])
            if lat is not None:
                break
    for key in LON_KEYS:
        if key in header:
            lon = to_float_deg(header[key])
            if lon is not None:
                break
    if lat is None or lon is None:
        env_lat = to_float_deg(ENV_SITE_LATITUDE) if ENV_SITE_LATITUDE else None
        env_lon = to_float_deg(ENV_SITE_LONGITUDE) if ENV_SITE_LONGITUDE else None
        if env_lat is not None and env_lon is not None:
            return env_lat, env_lon
    if lat is None or lon is None:
        return None, None
    return lat, lon


def get_frame_time(header):
    """Return the UTC exposure start time from the header as an astropy Time, or None."""
    for key in ('DATE-OBS', 'DATE_OBS'):
        if key in header:
            try:
                return Time(str(header[key]).strip().rstrip('Zz'), scale='utc')
            except Exception:
                continue
    return None


def parallactic_angle_deg(ra_deg, dec_deg, lat_deg, lon_deg, obs_time):
    """Parallactic angle in degrees for a sky position, site (east longitude positive) and UTC time."""
    # Mean sidereal time from the Julian date. Accurate to well under a second, which is
    # plenty here and avoids needing IERS data.
    gmst_deg = 280.46061837 + 360.98564736629 * (obs_time.jd - 2451545.0)
    lst_deg = (gmst_deg + lon_deg) % 360.0
    hour_angle = np.radians(lst_deg - ra_deg)
    lat = np.radians(lat_deg)
    dec = np.radians(dec_deg)
    q = np.arctan2(np.sin(hour_angle),
                   np.tan(lat) * np.cos(dec) - np.sin(dec) * np.cos(hour_angle))
    return float(np.degrees(q))


def classify_mount(frames):
    """Decide the mount type from frames solved across a session.

    Frames must be in time order and come from the same solver. Rotation of each frame
    relative to the first is compared with two hypotheses:
      equatorial: no change (or a change of exactly 180 degrees after a meridian flip)
      alt-az:     the change in parallactic angle for that pointing, site and time
    With only the two ends of a session, the alt-az prediction can land close to 180
    degrees and look like a meridian flip; frames in between settle that.

    Returns (mount_type, evidence, ambiguous). mount_type is EQUATORIAL, ALT-AZ or
    INCONCLUSIVE. ambiguous is True only when solving more frames could help.
    """
    if len(frames) < 2:
        return "INCONCLUSIVE", "Fewer than two frames solved, so rotation change is unknown", False
    if len({f['solver'] for f in frames}) > 1:
        return ("INCONCLUSIVE",
                "Frames were solved by different solvers, so their rotations are not comparable", False)

    ref = frames[0]
    times = [f.get('time') for f in frames]
    have_times = all(t is not None for t in times)
    span_min = (times[-1] - times[0]).to(u.min).value if have_times else None
    span_text = f"{span_min / 60.0:.1f} h" if span_min is not None else "unknown span"

    observed = [wrap180(f['rotation'] - ref['rotation']) for f in frames[1:]]
    # Distance of each observed change from 0 or 180 degrees (a meridian flip adds 180)
    equatorial_miss = max(abs((d + 90.0) % 180.0 - 90.0) for d in observed)

    lat, lon = get_site_location(ref['header'])
    if lat is not None and lon is not None and have_times:
        q_ref = parallactic_angle_deg(ref['ra'], ref['dec'], lat, lon, ref['time'])
        predicted = [wrap180(parallactic_angle_deg(f['ra'], f['dec'], lat, lon, f['time']) - q_ref)
                     for f in frames[1:]]
        # The rotation sign convention varies by solver, so allow either sign (used consistently)
        altaz_miss = min(
            max(abs(wrap180(d - sign * p)) for d, p in zip(observed, predicted))
            for sign in (1.0, -1.0)
        )
        evidence = (f"{len(frames)} frames over {span_text}; end-to-end rotation {observed[-1]:+.1f} deg, "
                    f"alt-az predicts {predicted[-1]:+.1f} deg; worst miss: "
                    f"equatorial {equatorial_miss:.1f} deg, alt-az {altaz_miss:.1f} deg")

        # The prediction at the solved frames can be small even when the session sweeps
        # through a larger angle in between (an alt-az session symmetric about the
        # meridian ends at nearly the angle it started at). Extra frames only help when
        # that swing is large, so check the points in between.
        span = times[-1] - times[0]
        sweep = max(
            abs(wrap180(parallactic_angle_deg(ref['ra'], ref['dec'], lat, lon, times[0] + span * (k / 8.0)) - q_ref))
            for k in range(1, 8)
        )
        more_frames_help = sweep >= MIN_SWEEP_DEG and len(frames) < MOUNT_SAMPLE_FRAMES

        if max(abs(p) for p in predicted) < MIN_PREDICTED_ROTATION_DEG:
            if more_frames_help:
                return ("INCONCLUSIVE",
                        evidence + f" (ends show little alt-az rotation but the session passes through "
                                   f"{sweep:.0f} deg in between)", True)
            return ("INCONCLUSIVE",
                    evidence + " (too little expected rotation to tell mounts apart)", False)

        # An end-to-end change near 180 degrees is what a meridian flip produces, and an
        # alt-az mount can land there by coincidence. Only frames in between can tell which.
        if more_frames_help and any(abs(abs(d) - 180.0) <= 15.0 for d in observed):
            return "INCONCLUSIVE", evidence + " (end-to-end change is near 180 degrees)", True

        if equatorial_miss <= altaz_miss:
            winner, winner_miss, loser_miss = "EQUATORIAL", equatorial_miss, altaz_miss
        else:
            winner, winner_miss, loser_miss = "ALT-AZ", altaz_miss, equatorial_miss

        if winner_miss <= FIT_TOLERANCE_DEG and loser_miss >= max(REJECT_DEG, REJECT_RATIO * winner_miss):
            return winner, evidence, False
        # More frames only help when both hypotheses fit the ends (for example an alt-az
        # rotation near 180 degrees that looks like a meridian flip)
        both_fit = equatorial_miss <= FIT_TOLERANCE_DEG and altaz_miss <= FIT_TOLERANCE_DEG
        return "INCONCLUSIVE", evidence + " (cannot separate the two mount types)", both_fit and more_frames_help

    # No site location or times: judge from the observed change alone
    evidence = (f"{len(frames)} frames over {span_text}; worst deviation from a fixed field "
                f"{equatorial_miss:.1f} deg; no site location for an alt-az prediction")
    if span_min is None or span_min < MIN_SPAN_NO_LOCATION_MIN:
        return "INCONCLUSIVE", evidence + " (session too short to judge)", False
    if equatorial_miss <= FIT_TOLERANCE_DEG:
        return "EQUATORIAL", evidence, False
    if equatorial_miss >= 3 * FIT_TOLERANCE_DEG:
        return "ALT-AZ", evidence, False
    return "INCONCLUSIVE", evidence, False


# ---------------------------------------------------------------------------
# Header and filename helpers
# ---------------------------------------------------------------------------

def get_position_from_header(header):
    """Extract RA/Dec from header for initial guess"""
    for ra_key in ['RA', 'CRVAL1']:
        for dec_key in ['DEC', 'CRVAL2']:
            if ra_key in header and dec_key in header:
                try:
                    ra_str = str(header[ra_key]).strip()
                    dec_str = str(header[dec_key]).strip()

                    if ':' in ra_str:
                        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                    else:
                        coord = SkyCoord(float(ra_str), float(dec_str), unit=u.deg)

                    return coord.ra.hour, coord.dec.deg
                except Exception:
                    continue
    return None, None


def extract_exposure_from_filename(filename):
    """Extract exposure time from filename pattern like '10.0s', '20.0s', etc."""
    basename = os.path.basename(filename)

    patterns = [
        r'(\d+\.?\d*)s',   # 10.0s, 20s, 5.5s
        r'_(\d+\.?\d*)_',  # _10.0_, _20_
        r'(\d+\.?\d*)sec', # 10.0sec, 20sec
    ]

    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            exposure = match.group(1)
            try:
                exposure_float = float(exposure)
                return f"{exposure_float:.1f}s"
            except ValueError:
                return f"{exposure}s"

    return "Unknown"


def analyze_exposure_distribution(directory):
    """Analyze exposure times from all FITS filenames in directory"""
    fits_files = glob.glob(os.path.join(directory, '*.fit')) + glob.glob(os.path.join(directory, '*.fits'))

    exposure_counts = {}
    total_files = len(fits_files)

    for fits_file in fits_files:
        exposure = extract_exposure_from_filename(fits_file)
        if exposure not in exposure_counts:
            exposure_counts[exposure] = 0
        exposure_counts[exposure] += 1

    return exposure_counts, total_files


def format_timestamp(timestamp_str):
    """Format timestamp to remove 'T' and decimal seconds"""
    if timestamp_str == 'Unknown' or timestamp_str == 'N/A':
        return timestamp_str

    timestamp_str = timestamp_str.replace('T', ' ')

    if '.' in timestamp_str:
        timestamp_str = timestamp_str.split('.')[0]

    return timestamp_str


def extract_observatory_location(header):
    """Extract observatory latitude and longitude from FITS header (as display strings)"""
    lat = 'Unknown'
    lon = 'Unknown'

    for key in LAT_KEYS:
        if key in header:
            try:
                lat_val = header[key]
                if isinstance(lat_val, (int, float)):
                    lat = f"{lat_val:.2f}"
                else:
                    lat = str(lat_val).strip()
                break
            except Exception:
                continue

    for key in LON_KEYS:
        if key in header:
            try:
                lon_val = header[key]
                if isinstance(lon_val, (int, float)):
                    lon = f"{lon_val:.2f}"
                else:
                    lon = str(lon_val).strip()
                break
            except Exception:
                continue

    return lat, lon


def extract_header_info(header):
    """Extract target name and other useful info from FITS header"""
    info = {
        'target': 'Unknown',
        'filter': 'Unknown',
        'date_obs': 'Unknown',
        'instrument': 'Unknown',
        'latitude': 'Unknown',
        'longitude': 'Unknown'
    }

    target_keys = ['OBJECT', 'TARGNAME', 'TARGET', 'OBJCTRA', 'OBJNAME']
    for key in target_keys:
        if key in header:
            info['target'] = str(header[key]).strip()
            break

    normalized_name, catalog_type, mosaic_info = normalize_object_name(info['target'])
    info['normalized_name'] = normalized_name
    info['catalog_type'] = catalog_type
    info['mosaic_info'] = mosaic_info
    info['is_mosaic'] = mosaic_info['is_mosaic'] if mosaic_info else False

    filter_keys = ['FILTER', 'FILT', 'FILTER1', 'FILTER2']
    for key in filter_keys:
        if key in header:
            info['filter'] = str(header[key]).strip()
            break

    date_keys = ['DATE-OBS', 'DATE_OBS', 'UT', 'MJD-OBS']
    for key in date_keys:
        if key in header:
            raw_date = str(header[key]).strip()
            info['date_obs'] = format_timestamp(raw_date)
            break

    inst_keys = ['INSTRUME', 'TELESCOP', 'OBSERVER']
    for key in inst_keys:
        if key in header:
            info['instrument'] = str(header[key]).strip()
            break

    info['latitude'], info['longitude'] = extract_observatory_location(header)

    return info


# ---------------------------------------------------------------------------
# Plate solvers
# ---------------------------------------------------------------------------

def run_astap_solve(filename, ra_hours=None, dec_degrees=None):
    """Run ASTAP and read the solution from the .ini file it writes.

    The Linux GUI build prints the solution line in a different format than the
    Windows console build, so the key=value .ini file is used instead of stdout.
    Output goes to a temp directory so image folders stay clean.
    """
    print("  Running ASTAP plate solver...")

    if not os.path.exists(ASTAP_PATH):
        print(f"    ERROR: ASTAP not found at {ASTAP_PATH}")
        return None, None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "solve")
            cmd = [ASTAP_PATH, '-f', filename, '-r', '10', '-z', '2', '-fov', '0',
                   '-wcs', '-log', '-o', base]

            if os.path.exists(ASTAP_DATABASE):
                cmd.extend(['-d', ASTAP_DATABASE])

            if ra_hours is not None and dec_degrees is not None:
                cmd.extend(['-ra', f"{ra_hours:.6f}", '-spd', f"{dec_degrees + 90:.6f}"])

            print(f"    Command: {' '.join(cmd)}")
            started = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=ASTAP_TIMEOUT_SEC)
            print(f"    Return code: {result.returncode} ({time.time() - started:.1f} s)")

            ini_path = base + ".ini"
            if not os.path.exists(ini_path):
                print("    ✗ ASTAP wrote no .ini file")
                return None, None

            ini = read_astap_ini(ini_path)
            if ini.get('PLTSOLVD', 'F').upper() != 'T':
                print(f"    ✗ Not solved. {ini.get('ERROR', '')}")
                return None, None

            ra = float(ini['CRVAL1'])
            dec = float(ini['CRVAL2'])
            rotation = float(ini['CROTA2'])
            print(f"    ✓ Solved: RA={ra:.4f}°, DEC={dec:.4f}°, rotation={rotation:.1f}°")
            return {'ra': ra, 'dec': dec, 'rotation': rotation}, rotation

    except subprocess.TimeoutExpired:
        print(f"    ✗ ASTAP timed out after {ASTAP_TIMEOUT_SEC} s")
    except Exception as e:
        print(f"    ✗ ASTAP error: {e}")
    return None, None


def run_astrometry_net_solve(filename, ra_hours=None, dec_degrees=None):
    """Fallback to astrometry.net for plate solving"""
    print("  🔄 ASTAP failed, trying astrometry.net...")

    if not ASTROMETRY_NET_API_KEY:
        print("    ❌ No astrometry.net API key configured (set ASTROMETRY_NET_API_KEY)")
        return None, None

    try:
        # Step 1: Login
        print("    🔐 DEBUG: Logging in to get session...")
        login_url = "https://nova.astrometry.net/api/login"
        login_payload = {'request-json': json.dumps({"apikey": ASTROMETRY_NET_API_KEY})}

        login_response = requests.post(login_url, data=login_payload)
        print(f"    🔐 DEBUG: Login response status: {login_response.status_code}")

        if login_response.status_code != 200:
            print(f"    ❌ DEBUG: Login failed with status {login_response.status_code}")
            return None, None

        login_result = login_response.json()

        if login_result.get('status') != 'success':
            error_msg = login_result.get('errormessage', 'Unknown login error')
            print(f"    ❌ DEBUG: Login failed: {error_msg}")
            return None, None

        session_key = login_result.get('session')
        if not session_key:
            print("    ❌ DEBUG: No session key in login response")
            return None, None

        print("    ✅ DEBUG: Got session")

        # Step 2: Upload with session
        print("    📤 DEBUG: Uploading file with session...")
        upload_url = "https://nova.astrometry.net/api/upload"

        upload_data = {
            "session": session_key,
            "publicly_visible": "n",
            "allow_commercial_use": "n",
            "allow_modifications": "n",
            "scale_units": "degwidth",
            "scale_type": "ul",
            "scale_lower": 0.5,
            "scale_upper": 10.0,
        }

        if ra_hours is not None and dec_degrees is not None:
            upload_data["center_ra"] = float(ra_hours * 15.0)
            upload_data["center_dec"] = float(dec_degrees)
            upload_data["radius"] = 10.0

        with open(filename, 'rb') as f:
            files = {'file': f}
            data = {'request-json': json.dumps(upload_data)}
            response = requests.post(upload_url, files=files, data=data)

        print(f"    📤 DEBUG: Response status: {response.status_code}")

        if response.status_code != 200:
            print(f"    ❌ DEBUG: Upload failed: HTTP {response.status_code}")
            return None, None

        result = response.json()

        if result.get('status') != 'success':
            error_msg = result.get('errormessage', 'Unknown API error')
            print(f"    ❌ DEBUG: API error: {error_msg}")
            return None, None

        subid = result['subid']
        print(f"    ✅ DEBUG: Upload successful, submission ID: {subid}")

        # Wait for processing
        print("    ⏳ DEBUG: Waiting for processing...")
        job_id = None
        for i in range(60):  # Wait up to 10 minutes
            time.sleep(10)

            status_url = f"https://nova.astrometry.net/api/submissions/{subid}"
            status_response = requests.get(status_url)

            if status_response.status_code == 200:
                status_data = status_response.json()
                jobs = status_data.get('jobs', [])

                if jobs and jobs[0] is not None:
                    job_id = jobs[0]
                    print(f"    ✅ DEBUG: Job started: {job_id}")
                    break
            print(f"      ...{i+1}/60")
        else:
            print("    ❌ DEBUG: Timeout waiting for job start")
            return None, None

        # Wait for job completion
        print("    ⏳ DEBUG: Waiting for job completion...")
        for i in range(60):
            time.sleep(10)

            job_url = f"https://nova.astrometry.net/api/jobs/{job_id}"
            job_response = requests.get(job_url)

            if job_response.status_code == 200:
                job_data = job_response.json()
                status = job_data.get('status')

                if status == 'success':
                    print("    ✅ DEBUG: Job completed successfully")
                    break
                elif status == 'failure':
                    print("    ❌ DEBUG: Job failed")
                    return None, None
            print(f"      ...{i+1}/60")
        else:
            print("    ❌ DEBUG: Timeout waiting for job completion")
            return None, None

        # Get results
        cal_url = f"https://nova.astrometry.net/api/jobs/{job_id}/calibration/"
        cal_response = requests.get(cal_url)

        if cal_response.status_code == 200:
            cal_data = cal_response.json()
            ra = cal_data.get('ra')
            dec = cal_data.get('dec')
            orientation = cal_data.get('orientation', 0)

            if ra is not None and dec is not None:
                print(f"    ✅ DEBUG: Solution: RA={ra:.6f}°, DEC={dec:.6f}°")

                with fits.open(filename) as hdul:
                    wcs = create_wcs_from_solution(ra, dec, orientation, hdul[0].header)
                    return wcs, orientation

        print("    ❌ DEBUG: No solution found")
        return None, None

    except Exception as e:
        print(f"    ❌ DEBUG: Exception: {str(e)}")
        import traceback
        print(f"    ❌ DEBUG: Traceback: {traceback.format_exc()}")
        return None, None


def solve_frame(path, use_astrometry_net=False):
    """Plate solve one frame. Returns a dict with solution, rotation, time and header, or None."""
    print(f"\nProcessing: {os.path.basename(path)}")
    print("-" * 50)

    try:
        header = fits.getheader(path)
    except Exception as e:
        print(f"  → Could not read FITS header: {e}")
        return None

    ra, dec = get_position_from_header(header)
    if ra is not None:
        print(f"  Initial position: RA={ra:.3f}h, DEC={dec:.3f}°")

    if use_astrometry_net:
        wcs, rotation = run_astrometry_net_solve(path, ra, dec)
        if wcs is None:
            print("  → astrometry.net could not plate solve image")
            return None
        solved_ra, solved_dec = (float(v) for v in wcs.wcs.crval)
        solver = "ASTROMETRY.NET"
    else:
        solution, rotation = run_astap_solve(path, ra, dec)
        if solution is None:
            print("  → ASTAP could not plate solve image")
            return None
        solved_ra, solved_dec = solution['ra'], solution['dec']
        solver = "ASTAP"

    return {
        'file': path,
        'solver': solver,
        'ra': solved_ra,
        'dec': solved_dec,
        'rotation': float(rotation),
        'time': get_frame_time(header),
        'header': header,
    }


def solve_end(candidates, label):
    """Find one solvable frame from an ordered list of candidates.

    Tries ASTAP on up to MAX_ASTAP_ATTEMPTS candidates, then astrometry.net on one.
    Returns (frame_or_None, attempts_made).
    """
    attempts = 0
    astap_candidates = candidates[:MAX_ASTAP_ATTEMPTS]

    for path in astap_candidates:
        attempts += 1
        print(f"\n🔄 ASTAP attempt {attempts}/{len(astap_candidates)} on {label} frame: {os.path.basename(path)}")
        frame = solve_frame(path)
        if frame:
            print(f"✅ Solved {label} frame with ASTAP")
            return frame, attempts
        print(f"❌ ASTAP failed on {os.path.basename(path)}")

    if candidates and ASTROMETRY_NET_API_KEY:
        path = candidates[MAX_ASTAP_ATTEMPTS] if len(candidates) > MAX_ASTAP_ATTEMPTS else candidates[0]
        attempts += 1
        print(f"\n🔄 astrometry.net attempt on {label} frame: {os.path.basename(path)}")
        frame = solve_frame(path, use_astrometry_net=True)
        if frame:
            print(f"✅ Solved {label} frame with astrometry.net")
            return frame, attempts
        print("❌ astrometry.net also failed")

    return None, attempts


# ---------------------------------------------------------------------------
# Directory handling
# ---------------------------------------------------------------------------

def find_lights_directories(root_path):
    """Find all directories named 'light' or 'lights', in any case, under root_path."""
    lights_dirs = []
    for root, dirs, _files in os.walk(root_path):
        for d in dirs:
            if d.lower() in ("light", "lights"):
                lights_dirs.append(os.path.join(root, d))
    return lights_dirs


def find_all_fits_files(directory):
    """Find ALL FITS files in a directory"""
    fits_files = glob.glob(os.path.join(directory, '*.fit')) + glob.glob(os.path.join(directory, '*.fits'))
    return sorted(fits_files) if fits_files else []


FILENAME_TIME_PATTERNS = [
    re.compile(r'(\d{4})(\d{2})(\d{2})[-_T](\d{2})(\d{2})(\d{2})'),      # 20260101-022917 (Seestar)
    re.compile(r'(\d{4})-(\d{2})-(\d{2})[_T ](\d{2})-(\d{2})-(\d{2})'),  # 2026-08-16_03-52-56 (NINA)
]


def time_from_filename(path):
    """Timestamp embedded in a file name, or None. Only used for ordering and gap detection."""
    name = os.path.basename(path)
    for pattern in FILENAME_TIME_PATTERNS:
        match = pattern.search(name)
        if match:
            try:
                return datetime(*(int(g) for g in match.groups()))
            except ValueError:
                continue
    return None


def order_frames(files):
    """Return [(datetime or None, path)] in time order.

    Uses the timestamp in each file name when every file has one, which avoids opening
    thousands of FITS files. Otherwise reads DATE-OBS from each header.
    """
    stamped = [(time_from_filename(p), p) for p in files]
    if all(t is not None for t, _ in stamped):
        return sorted(stamped, key=lambda item: (item[0], item[1]))

    print("  File names lack timestamps, reading FITS headers to order frames...")
    from_headers = []
    for path in files:
        stamp = None
        try:
            obs_time = get_frame_time(fits.getheader(path))
            if obs_time is not None:
                stamp = obs_time.to_datetime()
        except Exception:
            pass
        from_headers.append((stamp, path))
    return sorted(from_headers, key=lambda item: (item[0] is None, item[0] or datetime.min, item[1]))


def split_sessions(ordered, gap_hours=SESSION_GAP_HOURS):
    """Split time-ordered [(datetime, path)] into sessions wherever the gap exceeds gap_hours."""
    sessions = []
    current = []
    previous = None
    for stamp, path in ordered:
        if current and stamp is not None and previous is not None \
                and (stamp - previous) > timedelta(hours=gap_hours):
            sessions.append(current)
            current = []
        current.append(path)
        if stamp is not None:
            previous = stamp
    if current:
        sessions.append(current)
    return sessions


def prefer_usable(paths):
    """Prefer files without 'failed' in the name (Seestar appears to use it for rejected frames).

    Falls back to every file when fewer than two others remain.
    """
    usable = [p for p in paths if 'failed' not in os.path.basename(p).lower()]
    return usable if len(usable) >= 2 else paths


def pick_sample_indices(count, wanted):
    """Indices of `wanted` positions spread evenly from the first to the last of `count` items."""
    if count <= wanted:
        return list(range(count))
    return sorted({round(i * (count - 1) / (wanted - 1)) for i in range(wanted)})


def frame_sort_key(frame):
    return frame['time'].jd if frame['time'] is not None else float('inf')


FILTER_FILENAME_PATTERNS = [
    # NINA: 2026-08-16_03-52-56_Lum_0.00_60.00s_0000.fits (the filter can contain spaces or be empty)
    re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(?P<filter>.*?)_-?\d+(?:\.\d+)?_\d+(?:\.\d+)?s_\d+',
               re.IGNORECASE),
    # Seestar: Light_M33_20.0s_IRCUT_20251218-193902.fit, with an optional "_failed" before the time stamp
    re.compile(r'_\d+(?:\.\d+)?s_(?P<filter>[^_]+?)(?:_failed)?_\d{8}-\d{6}\.fits?$', re.IGNORECASE),
]
NO_FILTER_LABEL = "(none)"


def filter_from_filename(path):
    """Filter name embedded in a file name, or None when the name has no recognized pattern."""
    name = os.path.basename(path)
    for pattern in FILTER_FILENAME_PATTERNS:
        match = pattern.search(name)
        if match:
            return match.group('filter').strip() or NO_FILTER_LABEL
    return None


def filter_from_header(header):
    """Filter name from the FITS header."""
    for key in ('FILTER', 'FILT', 'FILTER1', 'FILTER2'):
        if key in header:
            value = str(header[key]).strip()
            if value:
                return value
    return NO_FILTER_LABEL


def assign_filters(paths):
    """Map each path to its filter name, from the file name when possible, else from the FITS header."""
    result = {}
    unrecognized = []
    for path in paths:
        name = filter_from_filename(path)
        if name is None:
            unrecognized.append(path)
        else:
            result[path] = name
    if unrecognized:
        print(f"  Reading the filter from the FITS header of {len(unrecognized)} file(s) whose names do not contain it...")
        for path in unrecognized:
            try:
                result[path] = filter_from_header(fits.getheader(path))
            except Exception:
                result[path] = "Unknown"
    return result


def process_session(lights_dir, session, index, total):
    """Solve frames from one observing session and classify the mount.

    Solves one frame at each end of the session. Only when that is ambiguous does it
    solve frames in between. Every filter shot during the session shares the verdict,
    since the filters are interleaved within the same night's mount setup.
    """
    if total > 1:
        print(f"\n📅 Session {index} of {total}: {len(session)} frames")

    filters = assign_filters(session)
    filter_exposure_counts = {}
    for path in session:
        key = (filters[path], extract_exposure_from_filename(path))
        filter_exposure_counts[key] = filter_exposure_counts.get(key, 0) + 1

    print(f"📄 {len(session)} FITS file(s) in this session")
    print("📊 Filter and exposure distribution:")
    for (flt, exp), count in sorted(filter_exposure_counts.items()):
        print(f"      {flt} {exp}: {count} frames")

    ordered = prefer_usable(session)

    mount_type = "UNKNOWN"
    mount_evidence = "No frame could be plate solved"
    header_info = {}
    frames = []
    attempts = 0

    # Solve near the start of the session
    print("\n🔧 Solving a frame from the start of the session")
    frame_a, n = solve_end(ordered, "first")
    attempts += n

    # Solve near the end, skipping the frame already solved
    frame_b = None
    if frame_a is not None:
        frames.append(frame_a)
        later = [p for p in reversed(ordered) if p != frame_a['file']]
        if later:
            print("\n🔧 Solving a frame from the end of the session")
            frame_b, n = solve_end(later, "last")
            attempts += n
        if frame_b is not None:
            frames.append(frame_b)

    frames.sort(key=frame_sort_key)

    if len(frames) >= 2:
        mount_type, mount_evidence, ambiguous = classify_mount(frames)

        # An end-to-end rotation near 180 degrees fits both a meridian flip and an alt-az
        # mount, so solve frames in between to separate them
        if ambiguous and all(f['solver'] == 'ASTAP' for f in frames):
            print("\n  → The two ends cannot separate the mount types, solving frames in between")
            used = {f['file'] for f in frames}
            for idx in pick_sample_indices(len(ordered), MOUNT_SAMPLE_FRAMES)[1:-1]:
                candidates = [p for p in ordered[idx: idx + MAX_ASTAP_ATTEMPTS] if p not in used]
                print(f"\n🔧 Solving a frame from the middle of the session (position {idx + 1}/{len(ordered)})")
                for path in candidates:
                    attempts += 1
                    mid = solve_frame(path)
                    if mid:
                        frames.append(mid)
                        used.add(path)
                        break
            frames.sort(key=frame_sort_key)
            mount_type, mount_evidence, _ = classify_mount(frames)

        print(f"\n  → Determined mount type: {mount_type}")
        print(f"    {mount_evidence}")
    elif len(frames) == 1:
        mount_type = "INCONCLUSIVE"
        mount_evidence = "Only one frame could be solved, so rotation change is unknown"
        print(f"\n  → {mount_evidence}")
    else:
        print("\n❌ Could not plate solve any files in this session")

    # Header info from the earliest solved frame, else from the first file
    if frames:
        header_info = extract_header_info(frames[0]['header'])
    else:
        try:
            header_info = extract_header_info(fits.getheader(ordered[0]))
        except Exception:
            header_info = {}

    solvers = sorted({f['solver'] for f in frames})
    if len(frames) >= 2:
        solved_files = [os.path.basename(frames[0]['file']), os.path.basename(frames[-1]['file'])]
    else:
        solved_files = [os.path.basename(f['file']) for f in frames]

    return {
        'directory': lights_dir,
        'session': f"{index}/{total}",
        'mount_type': mount_type,
        'mount_evidence': mount_evidence,
        'total_files': len(session),
        'filter_exposure_counts': filter_exposure_counts,
        'target': header_info.get('target', 'Unknown'),
        'date': header_info.get('date_obs', 'Unknown'),
        'instrument': header_info.get('instrument', 'Unknown'),
        'latitude': header_info.get('latitude', 'Unknown'),
        'longitude': header_info.get('longitude', 'Unknown'),
        'solved_files': solved_files,
        'solver_used': '+'.join(solvers) if solvers else 'NONE',
        'attempts_made': attempts,
        'total_attempts': len(session)
    }


def process_directory(lights_dir):
    """Split a directory into observing sessions and process each. Returns a list of session results."""
    print(f"\n📁 Processing directory: {lights_dir}")

    fits_files = find_all_fits_files(lights_dir)

    if not fits_files:
        print(f"❌ No FITS files found in {lights_dir} - skipping directory")
        return []

    print(f"📄 Found {len(fits_files)} FITS file(s) total")

    # Frames separated by a long gap are different sessions (different nights, or a
    # mount change in between), so each is classified on its own
    sessions = split_sessions(order_frames(fits_files))
    if len(sessions) > 1:
        print(f"📅 Directory holds {len(sessions)} separate sessions (gap of {SESSION_GAP_HOURS:g} h or more)")

    return [process_session(lights_dir, session, i, len(sessions))
            for i, session in enumerate(sessions, 1)]


# ---------------------------------------------------------------------------
# Object name handling
# ---------------------------------------------------------------------------

def normalize_object_name(object_name):
    """
    Normalize object names to handle different catalog designations.
    Returns a tuple: (normalized_name, catalog_type, mosaic_info)
    """
    if object_name == 'Unknown' or not object_name:
        return 'Unknown', 'Unknown', None

    original_name = str(object_name).strip()
    upper_name = original_name.upper()

    mosaic_info = detect_mosaic_panel(original_name)

    clean_name = original_name
    if mosaic_info and mosaic_info['is_mosaic']:
        clean_name = mosaic_info['base_object']

    catalog_type = 'Unknown'
    normalized = clean_name

    # Messier objects
    if upper_name.startswith('M') and upper_name[1:].isdigit():
        catalog_type = 'Messier'
        normalized = f"M{int(upper_name[1:])}"

    # NGC objects
    elif upper_name.startswith('NGC') and upper_name[3:].replace(' ', '').isdigit():
        catalog_type = 'NGC'
        normalized = f"NGC{int(upper_name[3:].replace(' ', ''))}"

    # IC objects
    elif upper_name.startswith('IC') and upper_name[2:].replace(' ', '').isdigit():
        catalog_type = 'IC'
        normalized = f"IC{int(upper_name[2:].replace(' ', ''))}"

    # Sharpless objects (Sh2-)
    elif 'SH2-' in upper_name or 'SH2-' in upper_name.replace(' ', ''):
        catalog_type = 'Sharpless'
        parts = upper_name.replace(' ', '').split('SH2-')
        if len(parts) > 1 and parts[1].isdigit():
            normalized = f"Sh2-{int(parts[1])}"

    # Caldwell objects
    elif upper_name.startswith('C') and upper_name[1:].isdigit():
        catalog_type = 'Caldwell'
        normalized = f"C{int(upper_name[1:])}"

    # Barnard objects (B-)
    elif upper_name.startswith('B') and '-' in upper_name and upper_name[1:].split('-')[0].isdigit():
        catalog_type = 'Barnard'
        parts = upper_name.split('-')
        if parts[0][1:].isdigit():
            normalized = f"B{int(parts[0][1:])}-{parts[1]}"

    # General names without specific catalog
    else:
        catalog_type = 'Common Name'
        normalized = ' '.join(clean_name.split())

    return normalized, catalog_type, mosaic_info


def detect_mosaic_panel(object_name):
    """
    Detect if object name indicates a mosaic panel.
    Returns dict with mosaic information.
    """
    if not object_name or '_' not in object_name:
        return {
            'is_mosaic': False,
            'base_object': object_name,
            'panel_number': None,
            'total_panels': None
        }

    name_str = str(object_name)

    # Pattern 1: Simple underscore with number (M101_01, M101_1)
    if re.search(r'_(\d+)$', name_str):
        match = re.search(r'^(.+)_(\d+)$', name_str)
        if match:
            return {
                'is_mosaic': True,
                'base_object': match.group(1),
                'panel_number': int(match.group(2)),
                'total_panels': None
            }

    # Pattern 2: Of pattern (M101_1of4, M101_01of12)
    of_match = re.search(r'^(.+)_(\d+)of(\d+)$', name_str, re.IGNORECASE)
    if of_match:
        return {
            'is_mosaic': True,
            'base_object': of_match.group(1),
            'panel_number': int(of_match.group(2)),
            'total_panels': int(of_match.group(3))
        }

    # Pattern 3: P prefix (M101_p1, M101_p01)
    p_match = re.search(r'^(.+)_p(\d+)$', name_str, re.IGNORECASE)
    if p_match:
        return {
            'is_mosaic': True,
            'base_object': p_match.group(1),
            'panel_number': int(p_match.group(2)),
            'total_panels': None
        }

    # Pattern 4: Panel prefix (M101_panel1, M101_panel01)
    panel_match = re.search(r'^(.+)_panel(\d+)$', name_str, re.IGNORECASE)
    if panel_match:
        return {
            'is_mosaic': True,
            'base_object': panel_match.group(1),
            'panel_number': int(panel_match.group(2)),
            'total_panels': None
        }

    return {
        'is_mosaic': False,
        'base_object': object_name,
        'panel_number': None,
        'total_panels': None
    }


def group_similar_objects(results):
    """
    Group results by normalized object names to identify same objects
    with different designations.
    """
    object_groups = {}

    for result in results:
        target = result.get('target', 'Unknown')
        normalized, catalog_type, mosaic_info = normalize_object_name(target)

        result['normalized_name'] = normalized
        result['catalog_type'] = catalog_type
        result['mosaic_info'] = mosaic_info
        result['is_mosaic'] = mosaic_info['is_mosaic'] if mosaic_info else False

        if normalized not in object_groups:
            object_groups[normalized] = {
                'normalized_name': normalized,
                'catalog_type': catalog_type,
                'original_names': set(),
                'directories': [],
                'mount_types': set(),
                'total_files': 0,
                'mosaic_panels': set(),
                'has_mosaic': False,
                'mosaic_catalog_type': catalog_type
            }

        group = object_groups[normalized]
        group['original_names'].add(target)
        if result['directory'] not in group['directories']:
            group['directories'].append(result['directory'])
        group['mount_types'].add(result['mount_type'])
        group['total_files'] += result['total_files']

        if mosaic_info and mosaic_info['is_mosaic']:
            group['has_mosaic'] = True
            panel_info = f"Panel {mosaic_info['panel_number']}"
            if mosaic_info['total_panels']:
                panel_info += f" of {mosaic_info['total_panels']}"
            group['mosaic_panels'].add(panel_info)

            base_normalized, base_catalog_type, _ = normalize_object_name(mosaic_info['base_object'])
            group['mosaic_catalog_type'] = base_catalog_type

    return object_groups


def main():
    if len(sys.argv) != 2:
        print("Usage: python scan_telescope_data.py <directory_path>")
        print("Example: python scan_telescope_data.py /home/brian/nina")
        print("This will walk through all subdirectories and process FITS files")
        print("from each 'light' or 'lights' directory found (any case).")
        print("\nThe mount type is judged separately for each observing session (a gap of")
        print(f"{SESSION_GAP_HOURS:g} hours or more between frames starts a new one) by solving frames")
        print("from it and comparing how the field rotation changed with what an alt-az mount")
        print("would produce at that site and time. CSV rows are split by filter and exposure.")
        print("\nOptional environment variables:")
        print("  ASTROMETRY_NET_API_KEY   fallback solver key from https://nova.astrometry.net/api_help")
        print("  SITE_LATITUDE / SITE_LONGITUDE   site location (east positive) when headers lack it")
        return

    root_path = sys.argv[1]

    if not os.path.exists(root_path):
        print(f"Error: Directory not found: {root_path}")
        return

    print("Searching for 'light'/'lights' directories...")
    lights_dirs = find_lights_directories(root_path)

    if not lights_dirs:
        print("No 'light' or 'lights' directories found!")
        return

    print(f"Found {len(lights_dirs)} 'light'/'lights' directories")
    results = []

    print("\nTelescope Mount Analysis")
    print("=" * 120)
    print(f"Solves one frame at each end of each session; up to {MOUNT_SAMPLE_FRAMES} frames only when that is ambiguous")
    print("Directories with no FITS files will be excluded from output")

    for lights_dir in sorted(lights_dirs):
        results.extend(process_directory(lights_dir))

    if not results:
        print("\n❌ No directories with FITS files found!")
        return

    object_groups = group_similar_objects(results)

    all_rows = []

    for result in results:
        target = result['target']
        normalized_name = result.get('normalized_name', target)
        catalog_type = result.get('catalog_type', 'Unknown')
        mosaic_info = result.get('mosaic_info', {})
        is_mosaic = result.get('is_mosaic', False)
        mount_type = result['mount_type']
        date = result['date']
        directory = result['directory']
        latitude = result['latitude']
        longitude = result['longitude']

        mosaic_display = "No"
        mosaic_detail = ""
        if is_mosaic:
            mosaic_display = "Yes"
            panel_info = f"Panel {mosaic_info['panel_number']}"
            if mosaic_info['total_panels']:
                panel_info += f" of {mosaic_info['total_panels']}"
            mosaic_detail = panel_info

        groups = sorted(result['filter_exposure_counts'].items())

        display_target = target
        if is_mosaic:
            panel_info = f" (Panel {mosaic_info['panel_number']}"
            if mosaic_info['total_panels']:
                panel_info += f" of {mosaic_info['total_panels']}"
            panel_info += ")"
            display_target = f"{mosaic_info['base_object']}{panel_info}"

        base_row = {
            'Target': display_target,
            'Normalized Name': normalized_name,
            'Catalog Type': catalog_type,
            'Is Mosaic': mosaic_display,
            'Mosaic Details': mosaic_detail,
            'Mount Type': mount_type,
            'Mount Evidence': result.get('mount_evidence', ''),
            'Session': result.get('session', '1/1'),
            'Date': date,
            'Directory': directory,
            'Latitude': latitude,
            'Longitude': longitude,
            'Total Files': result['total_files'],
            'Instrument': result.get('instrument', 'Unknown'),
            'Solved File': ' | '.join(result['solved_files']) if result.get('solved_files') else 'None',
            'Solver Used': result.get('solver_used', 'NONE'),
            'Attempts': f"{result.get('attempts_made', 0)}/{result.get('total_attempts', 0)}"
        }

        if groups:
            for (filter_name, exp), count in groups:
                row = dict(base_row)
                row['Filter'] = filter_name
                row['Exposure'] = exp
                row['Frames'] = count
                all_rows.append(row)
        else:
            row = dict(base_row)
            row['Filter'] = 'Unknown'
            row['Exposure'] = 'No data'
            row['Frames'] = result['total_files']
            all_rows.append(row)

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"Showing {len(results)} sessions from {len({r['directory'] for r in results})} directories with FITS files")

    print("\n📊 OBJECT GROUPING SUMMARY:")
    print(f"Found {len(object_groups)} unique objects from {len({r['directory'] for r in results})} directories")

    mosaic_objects = 0
    for norm_name, group in sorted(object_groups.items()):
        if norm_name == 'Unknown':
            continue

        mosaic_indicator = " 🧩 MOSAIC" if group['has_mosaic'] else ""
        print(f"\n  {norm_name} ({group['catalog_type']}){mosaic_indicator}:")
        print(f"    Original names: {', '.join(sorted(group['original_names']))}")
        print(f"    Directories: {len(group['directories'])}")
        print(f"    Total files: {group['total_files']}")
        print(f"    Mount types: {', '.join(sorted(group['mount_types']))}")
        if group['has_mosaic']:
            mosaic_objects += 1
            print(f"    Mosaic panels: {', '.join(sorted(group['mosaic_panels']))}")
            print(f"    Base object catalog: {group['mosaic_catalog_type']}")

    if mosaic_objects > 0:
        print(f"\n🧩 Found {mosaic_objects} objects with mosaic panels")

    header = f"{'Target':<25} {'Normalized':<15} {'Catalog':<10} {'Mosaic':<6} {'Mount':<13} {'Filter':<16} {'Date':<15} {'Solver':<12} {'Directory'}"
    print(f"\n{'=' * 120}")
    print("DETAILED RESULTS")
    print("=" * 120)
    print(header)
    print("-" * len(header))

    for row in all_rows:
        mosaic_indicator = "🧩" if row['Is Mosaic'] == "Yes" else "  "
        print(f"{row['Target']:<25} {row['Normalized Name']:<15} {row['Catalog Type']:<10} {mosaic_indicator:<6} {row['Mount Type']:<13} {row['Filter']:<16} {row['Date']:<15} {row['Solver Used']:<12} {row['Directory']}")

    csv_filename = "telescope_analysis.csv"
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Target', 'Normalized Name', 'Catalog Type', 'Is Mosaic', 'Mosaic Details',
                          'Mount Type', 'Mount Evidence', 'Filter', 'Session', 'Date', 'Exposure', 'Frames',
                          'Directory', 'Latitude', 'Longitude', 'Total Files', 'Instrument',
                          'Solved File', 'Solver Used', 'Attempts']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

        print(f"\n✅ Results saved to: {csv_filename}")
        print(f"📊 Total records: {len(all_rows)}")

        total_dirs_with_files = len(results)
        classified = sum(1 for r in results if r['mount_type'] in ("EQUATORIAL", "ALT-AZ"))
        inconclusive = sum(1 for r in results if r['mount_type'] == "INCONCLUSIVE")
        unsolved = sum(1 for r in results if r['mount_type'] in ("UNKNOWN", "ERROR"))
        astap_dirs = sum(1 for r in results if 'ASTAP' in r.get('solver_used', ''))
        astrometry_dirs = sum(1 for r in results if 'ASTROMETRY.NET' in r.get('solver_used', ''))

        mosaic_dirs = sum(1 for r in results if r.get('is_mosaic', False))

        print(f"📈 Mount classified: {classified}/{total_dirs_with_files} sessions "
              f"({inconclusive} inconclusive, {unsolved} could not be plate solved)")
        print(f"🔧 Solvers used: {astap_dirs} sessions with ASTAP, {astrometry_dirs} with astrometry.net")
        print(f"🧩 Mosaic sessions: {mosaic_dirs}/{total_dirs_with_files}")

        total_lights_dirs_found = len(lights_dirs)
        dirs_with_files = len({r['directory'] for r in results})
        dirs_without_files = total_lights_dirs_found - dirs_with_files
        print(f"📁 Directories: {total_lights_dirs_found} total, {dirs_with_files} with FITS files, {dirs_without_files} without FITS files (excluded)")

    except Exception as e:
        print(f"\n❌ Error writing CSV file: {e}")


if __name__ == "__main__":
    main()
