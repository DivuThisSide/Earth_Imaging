from part5 import *

from datetime import datetime, timedelta
import requests
def parse_entity_id(entity_id: str):
    path, row, date = None, None, None

    try:
        if '_' not in entity_id and len(entity_id) >= 16:
            path = int(entity_id[3:6])
            row  = int(entity_id[6:9])

            year = int(entity_id[9:13])
            doy  = int(entity_id[13:16])

            date_obj = datetime(year, 1, 1) + timedelta(days=doy - 1)
            date = date_obj.strftime('%Y-%m-%d')


        else:
            parts = entity_id.split('_')

            if len(parts) >= 3:
                pr = parts[2]
                if len(pr) == 6 and pr.isdigit():
                    path = int(pr[:3])
                    row  = int(pr[3:])

            if len(parts) >= 4:
                raw = parts[3]
                if len(raw) == 8 and raw.isdigit():
                    date = f'{raw[:4]}-{raw[4:6]}-{raw[6:8]}'

    except:
        pass

    return path, row, date


def extract_path_row(entity_id: str):
    p, r, _ = parse_entity_id(entity_id)
    return p, r

WRS_PATH, WRS_ROW, _ = parse_entity_id(scene_id)

if WRS_PATH and WRS_ROW:
    print(f'WRS tile locked → Path {WRS_PATH} / Row {WRS_ROW}')
else:
    print('⚠ Could not parse path/row — using spatial filter only')
    WRS_PATH, WRS_ROW = None, None


def search_best_scene(start, end, cloud_max=20, exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = set()

    scene_filter = {
        'cloudCoverFilter': {'max': cloud_max, 'min': 0},
        'spatialFilter': {
            'filterType': 'mbr',
            'lowerLeft':  {'latitude': AOI['min_lat'], 'longitude': AOI['min_lon']},
            'upperRight': {'latitude': AOI['max_lat'], 'longitude': AOI['max_lon']}
        }
    }

    q = {
        'datasetName': 'landsat_ot_c2_l2',
        'sceneFilter': scene_filter,
        'temporalFilter': {'startDate': start, 'endDate': end},
        'maxResults': 20
    }

    resp = requests.post(url + 'scene-search', json=q, headers=hdr).json()
    scenes = resp['data']['results']

    # Optional tile filtering
    def same_tile(s):
        if WRS_PATH is None:
            return True
        p, r = extract_path_row(s['entityId'])
        return p == WRS_PATH and r == WRS_ROW

    scenes = [
        s for s in scenes
        if s.get('cloudCover', 100) < cloud_max
        and s['entityId'] not in exclude_ids
        and same_tile(s)
    ]

    if not scenes:
        return None

    scenes.sort(key=lambda s: s['cloudCover'])
    return scenes[0]



SEASON_WINDOWS = [
    ('Winter',  '2023-12-01', '2024-02-28'),
    ('Spring',  '2024-03-01', '2024-05-31'),
    ('Summer',  '2024-06-01', '2024-08-31'),
    ('Autumn',  '2024-09-01', '2024-11-30'),
    ('Winter2', '2024-12-01', '2025-02-28'),
]

print('\nSearching for seasonal scenes …\n')
print(f'{"Season":<10} {"Entity ID":<30} {"Date":<12} {"Path":>5} {"Row":>5} Cloud%')
print('─' * 80)

seasonal_scenes = []
used_entity_ids = set()

for season_name, s_start, s_end in SEASON_WINDOWS:
    sc = search_best_scene(s_start, s_end, exclude_ids=used_entity_ids)

    if sc:
        entity_id = sc['entityId']


        p, r, acq_date = parse_entity_id(entity_id)

        if not acq_date:
            print(f"Skipping scene (no date): {entity_id}")
            continue

        used_entity_ids.add(entity_id)

        seasonal_scenes.append({
            'season': season_name,
            'entity_id': entity_id,
            'date': acq_date,
            'cloud': sc.get('cloudCover', 0),
            'path': p,
            'row': r,
            'scene': sc
        })

        print(f"{season_name:<10} {entity_id:<30} {acq_date:<12} "
              f"{str(p):>5} {str(r):>5}  {sc.get('cloudCover',0):.1f}%")

    else:
        print(f"{season_name:<10} — no scene found")

ids = [sm['entity_id'] for sm in seasonal_scenes]

print()
if len(ids) != len(set(ids)):
    print('Duplicate scenes found')
else:
    print(f' All {len(ids)} scenes are unique')

if WRS_PATH is not None and WRS_ROW is not None:
    print(f'Tile: Path {WRS_PATH} / Row {WRS_ROW}')
else:
    print('Tile not locked (multiple tiles possible)')
scene_dates = [sm['date'] for sm in seasonal_scenes]

print("\nDates extracted:")
print(scene_dates)

def safe_parse_date(d):
    try:
        return datetime.strptime(d, '%Y-%m-%d')
    except:
        return None


date_objs = [safe_parse_date(d) for d in scene_dates]

if not any(date_objs):
    print(' ERROR: No valid dates for plotting')
else:
    print(' Ready for NDVI time series plotting')

def download_scene(scene_meta: dict):
    """
    Download a Landsat scene, extract it, and return a band-path dict.
    Returns None on failure. Re-run safe: skips download if already on disk.
    """
    entity_id = scene_meta['entity_id']
    scene_dir = os.path.join(landsat_dir, entity_id)

    # BUG FIX 3 — check for THIS scene's folder, not a generic one.
    # os.path.join(landsat_dir, entity_id) is unique per scene, so the
    # check is correct as written — but we print the path to confirm.
    existing_b4 = []
    if os.path.isdir(scene_dir):
        existing_b4 = [f for f in os.listdir(scene_dir) if f.endswith('_B4.TIF')]

    if existing_b4:
        print(f'  {entity_id}: already on disk at {scene_dir}')
    else:
        # Fetch download options
        opts = requests.post(
            url + 'download-options',
            json={'datasetName': 'landsat_ot_c2_l2', 'entityIds': [entity_id]},
            headers=hdr
        ).json()['data']

        chosen_opt = next(
            (o for o in opts if 'Bundle' in o.get('productName', '') and o.get('available')),
            next((o for o in opts if o.get('available')), None)
        )
        if not chosen_opt:
            print(f'  {entity_id}: no downloadable product — skipping.')
            return None

        dl_resp = requests.post(
            url + 'download-request',
            json={'downloads': [{'entityId': entity_id, 'productId': chosen_opt['id']}],
                  'label': f'ts-{entity_id}'},
            headers=hdr
        ).json()
        ready = dl_resp['data'].get('availableDownloads', [])
        if not ready:
            print(f'  {entity_id}: download not yet ready — re-run later.')
            return None
        dl_url = ready[0]['url']

        tar_path = os.path.join(landsat_dir, f'{entity_id}.tar')
        r = requests.get(dl_url, stream=True)
        with open(tar_path, 'wb') as fh:
            for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                fh.write(chunk)

        os.makedirs(scene_dir, exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(scene_dir)
        print(f'  {entity_id}: downloaded & extracted → {scene_dir}')

    # Locate band TIFs for THIS scene specifically
    files = sorted(os.listdir(scene_dir))
    band_map = {2: None, 3: None, 4: None, 5: None}
    for fname in files:
        for b in band_map:
            if fname.endswith(f'_B{b}.TIF'):
                band_map[b] = os.path.join(scene_dir, fname)

    missing = [b for b, p in band_map.items() if p is None]
    if missing:
        print(f'  {entity_id}: missing bands {missing}.')
        return None

    # Print which files were found so we can visually verify they differ per scene
    print(f'    B4: {os.path.basename(band_map[4])}')
    print(f'    B5: {os.path.basename(band_map[5])}')
    return band_map


print('\nDownloading seasonal scenes …')
for sm in seasonal_scenes:
    print(f'\n[{sm["season"]}]  {sm["entity_id"]}  ({sm["date"]})')
    sm['band_map'] = download_scene(sm)

seasonal_scenes = [sm for sm in seasonal_scenes if sm.get('band_map')]
print(f'\n{len(seasonal_scenes)} scenes ready for NDVI computation.')