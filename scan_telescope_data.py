from astropy.io import fits
from astropy.coordinates import SkyCoord
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

# Configuration
ASTAP_PATH = r"/usr/bin/astap"
ASTAP_DATABASE = r"/usr/share/astap/data"
# Set this in your environment instead of hardcoding it:
#   export ASTROMETRY_NET_API_KEY="your key from nova.astrometry.net"
ASTROMETRY_NET_API_KEY = os.environ.get("ASTROMETRY_NET_API_KEY", "")
MAX_ASTAP_ATTEMPTS = 3  # Maximum number of files to try with ASTAP before switching to astrometry.net


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


def analyze_rotation_simple(rotation_angle):
    """Simple analysis based on the rotation angle"""
    print(f"    Field rotation: {rotation_angle:.1f}°")

    # Check if close to 0° or 180° (equatorial) or something else (alt-az)
    normalized_0 = min(rotation_angle % 360, 360 - (rotation_angle % 360))
    normalized_180 = min(abs(rotation_angle - 180), 360 - abs(rotation_angle - 180))

    deviation = min(normalized_0, normalized_180)
    print(f"    Deviation from equatorial: {deviation:.1f}°")

    if deviation < 10:  # Within 10° of 0° or 180°
        return "EQUATORIAL"
    else:
        return "ALT-AZ"


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
    """Extract observatory latitude and longitude from FITS header"""
    lat = 'Unknown'
    lon = 'Unknown'

    lat_keys = ['SITELAT', 'LATITUDE', 'OBSLAT', 'LAT-OBS', 'LAT']
    for key in lat_keys:
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

    lon_keys = ['SITELONG', 'LONGITUDE', 'OBSLONG', 'LONG-OBS', 'LON', 'LONG']
    for key in lon_keys:
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            print(f"    Return code: {result.returncode}")

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
        print("    ✗ ASTAP timed out")
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


def process_file_with_astap(filename, ra_hours=None, dec_degrees=None):
    """Process a file using only ASTAP (for the initial attempts)"""
    print(f"\nProcessing with ASTAP: {os.path.basename(filename)}")
    print("-" * 50)

    try:
        with fits.open(filename) as hdul:
            header = hdul[0].header

            header_info = extract_header_info(header)

            ra, dec = get_position_from_header(header)
            if ra is not None:
                print(f"  Initial position: RA={ra:.3f}h, DEC={dec:.3f}°")

            solution, rotation = run_astap_solve(filename, ra, dec)

            if solution is None:
                print("  → ASTAP could not plate solve image")
                return "UNKNOWN", header_info, "ASTAP"

            print("  Analyzing field rotation...")
            mount_type = analyze_rotation_simple(rotation)
            print(f"  → Determined mount type: {mount_type}")
            return mount_type, header_info, "ASTAP"

    except Exception as e:
        print(f"  → Error processing file with ASTAP: {e}")
        return "ERROR", {}, "ASTAP"


def process_file_with_astrometry(filename, ra_hours=None, dec_degrees=None):
    """Process a file using astrometry.net (fallback)"""
    print(f"\nProcessing with astrometry.net: {os.path.basename(filename)}")
    print("-" * 50)

    try:
        with fits.open(filename) as hdul:
            header = hdul[0].header

            header_info = extract_header_info(header)

            ra, dec = get_position_from_header(header)
            if ra is not None:
                print(f"  Initial position: RA={ra:.3f}h, DEC={dec:.3f}°")

            print("  Calling astrometry.net solver...")
            wcs, rotation = run_astrometry_net_solve(filename, ra, dec)

            if wcs is None:
                print("  → Astrometry.net could not plate solve image")
                return "UNKNOWN", header_info, "ASTROMETRY.NET"

            print("  Analyzing field rotation...")
            mount_type = analyze_rotation_simple(rotation)
            print(f"  → Determined mount type: {mount_type}")
            return mount_type, header_info, "ASTROMETRY.NET"

    except Exception as e:
        print(f"  → Error processing file with astrometry.net: {e}")
        return "ERROR", {}, "ASTROMETRY.NET"


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


def process_directory(lights_dir):
    """Process a directory by trying up to MAX_ASTAP_ATTEMPTS with ASTAP, then ONE file with astrometry.net"""
    print(f"\n📁 Processing directory: {lights_dir}")

    fits_files = find_all_fits_files(lights_dir)

    if not fits_files:
        print(f"❌ No FITS files found in {lights_dir} - skipping directory")
        return None

    exposure_counts, total_files = analyze_exposure_distribution(lights_dir)

    print(f"📄 Found {len(fits_files)} FITS file(s) total")
    print("📊 Exposure distribution:")
    for exp, count in sorted(exposure_counts.items()):
        print(f"      {exp}: {count} frames")

    mount_type = "UNKNOWN"
    header_info = {}
    successful_file = None
    attempts = 0
    solver_used = "NONE"

    # Phase 1: Try ASTAP on first MAX_ASTAP_ATTEMPTS files
    astap_files = fits_files[:MAX_ASTAP_ATTEMPTS]
    print(f"\n🔧 Phase 1: Trying ASTAP on first {len(astap_files)} files")

    for fits_file in astap_files:
        attempts += 1
        print(f"\n🔄 ASTAP Attempt {attempts}/{len(astap_files)}: {os.path.basename(fits_file)}")

        current_mount_type, current_header_info, current_solver = process_file_with_astap(fits_file)

        if current_mount_type not in ["UNKNOWN", "ERROR"]:
            mount_type = current_mount_type
            header_info = current_header_info
            successful_file = fits_file
            solver_used = current_solver
            print(f"✅ Successfully processed {os.path.basename(fits_file)} with ASTAP on attempt {attempts}")
            break
        else:
            print(f"❌ ASTAP failed on {os.path.basename(fits_file)}")

    # Phase 2: If ASTAP failed, try ONLY ONE file with astrometry.net
    if mount_type in ["UNKNOWN", "ERROR"]:
        print("\n🔧 Phase 2: ASTAP failed, trying ONE file with astrometry.net")

        if len(fits_files) > MAX_ASTAP_ATTEMPTS:
            fits_file = fits_files[MAX_ASTAP_ATTEMPTS]
        else:
            fits_file = fits_files[0]

        print(f"\n🔄 Astrometry.net Attempt (1/1): {os.path.basename(fits_file)}")

        current_mount_type, current_header_info, current_solver = process_file_with_astrometry(fits_file)

        if current_mount_type not in ["UNKNOWN", "ERROR"]:
            mount_type = current_mount_type
            header_info = current_header_info
            successful_file = fits_file
            solver_used = current_solver
            print("✅ Successfully processed with astrometry.net")
        else:
            print("❌ Astrometry.net also failed")
            attempts += 1

    if mount_type in ["UNKNOWN", "ERROR"]:
        print("❌ Could not plate solve any files in this directory")
        if fits_files:
            with fits.open(fits_files[0]) as hdul:
                header_info = extract_header_info(hdul[0].header)

    return {
        'directory': lights_dir,
        'mount_type': mount_type,
        'total_files': total_files,
        'exposure_counts': exposure_counts,
        'target': header_info.get('target', 'Unknown'),
        'filter': header_info.get('filter', 'Unknown'),
        'date': header_info.get('date_obs', 'Unknown'),
        'instrument': header_info.get('instrument', 'Unknown'),
        'latitude': header_info.get('latitude', 'Unknown'),
        'longitude': header_info.get('longitude', 'Unknown'),
        'solved_file': successful_file,
        'solver_used': solver_used,
        'attempts_made': attempts,
        'total_attempts': len(fits_files)
    }


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
        print("\nNote: astrometry.net fallback needs an API key in the")
        print("ASTROMETRY_NET_API_KEY environment variable.")
        print("Get one from: https://nova.astrometry.net/api_help")
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
    print(f"Will try up to {MAX_ASTAP_ATTEMPTS} files with ASTAP first, then ONE file with astrometry.net")
    print("Directories with no FITS files will be excluded from output")

    for lights_dir in sorted(lights_dirs):
        result = process_directory(lights_dir)
        if result is not None:
            results.append(result)

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
        filter_val = result['filter']
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

        exposures = sorted(result['exposure_counts'].items())

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
            'Filter': filter_val,
            'Date': date,
            'Directory': directory,
            'Latitude': latitude,
            'Longitude': longitude,
            'Total Files': result['total_files'],
            'Instrument': result.get('instrument', 'Unknown'),
            'Solved File': os.path.basename(result.get('solved_file', '')) if result.get('solved_file') else 'None',
            'Solver Used': result.get('solver_used', 'NONE'),
            'Attempts': f"{result.get('attempts_made', 0)}/{result.get('total_attempts', 0)}"
        }

        if exposures:
            for exp, count in exposures:
                row = dict(base_row)
                row['Exposure'] = exp
                row['Frames'] = count
                all_rows.append(row)
        else:
            row = dict(base_row)
            row['Exposure'] = 'No data'
            row['Frames'] = result['total_files']
            all_rows.append(row)

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"Showing {len(results)} directories with FITS files")

    print("\n📊 OBJECT GROUPING SUMMARY:")
    print(f"Found {len(object_groups)} unique objects from {len(results)} directories")

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

    header = f"{'Target':<25} {'Normalized':<15} {'Catalog':<10} {'Mosaic':<6} {'Mount':<12} {'Filter':<8} {'Date':<15} {'Solver':<12} {'Directory'}"
    print(f"\n{'=' * 120}")
    print("DETAILED RESULTS")
    print("=" * 120)
    print(header)
    print("-" * len(header))

    for row in all_rows:
        mosaic_indicator = "🧩" if row['Is Mosaic'] == "Yes" else "  "
        print(f"{row['Target']:<25} {row['Normalized Name']:<15} {row['Catalog Type']:<10} {mosaic_indicator:<6} {row['Mount Type']:<12} {row['Filter']:<8} {row['Date']:<15} {row['Solver Used']:<12} {row['Directory']}")

    csv_filename = "telescope_analysis.csv"
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Target', 'Normalized Name', 'Catalog Type', 'Is Mosaic', 'Mosaic Details',
                          'Mount Type', 'Filter', 'Date', 'Exposure', 'Frames', 'Directory',
                          'Latitude', 'Longitude', 'Total Files', 'Instrument', 'Solved File',
                          'Solver Used', 'Attempts']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

        print(f"\n✅ Results saved to: {csv_filename}")
        print(f"📊 Total records: {len(all_rows)}")

        successful_solves = sum(1 for r in results if r['mount_type'] not in ["UNKNOWN", "ERROR"])
        total_dirs_with_files = len(results)
        astap_solves = sum(1 for r in results if r.get('solver_used') == 'ASTAP')
        astrometry_solves = sum(1 for r in results if r.get('solver_used') == 'ASTROMETRY.NET')

        mosaic_dirs = sum(1 for r in results if r.get('is_mosaic', False))

        print(f"📈 Success rate: {successful_solves}/{total_dirs_with_files} directories successfully plate solved")
        print(f"🔧 Solvers: {astap_solves} ASTAP, {astrometry_solves} astrometry.net")
        print(f"🧩 Mosaic directories: {mosaic_dirs}/{total_dirs_with_files}")

        total_lights_dirs_found = len(lights_dirs)
        dirs_with_files = len(results)
        dirs_without_files = total_lights_dirs_found - dirs_with_files
        print(f"📁 Directories: {total_lights_dirs_found} total, {dirs_with_files} with FITS files, {dirs_without_files} without FITS files (excluded)")

    except Exception as e:
        print(f"\n❌ Error writing CSV file: {e}")


if __name__ == "__main__":
    main()
