from libraries import *

# Required Paths
root        = '../data'
landsat_dir = os.path.join(root, 'landsat')
lulc_dir    = os.path.join(root, 'esri_lulc')
out_dir     = os.path.join(root, 'outputs')
for d in [landsat_dir, lulc_dir, out_dir]:
    os.makedirs(d, exist_ok=True)

# Area of Interest — Nara Valley, Japan
# Coordinates are in (longitude, latitude) format
AOI = {
    'min_lon': 135.800, 'min_lat': 34.635,
    'max_lon': 135.881, 'max_lat': 34.696,
    'area_km2': 50,
    'label': 'Nara Valley'
}

# USGS M2M API
usgs_user  = 'Divyansh_Bansal'
usgs_token = 'bRluH4OPWp6ntn8K2JB3xcSBnq3IJBaPbm75BUn6Bbgf3YVay7qmFxx!rNzFKL!R'
url = 'https://m2m.cr.usgs.gov/api/api/json/stable/'

start_date = '2023-01-01'
end_date = '2025-04-18'
cloudlimit = 20  # percentage

print('Configuration ready.')
print(f'AOI  : {AOI["label"]}')
print(f'Bbox : ({AOI["min_lon"]}, {AOI["min_lat"]}) → ({AOI["max_lon"]}, {AOI["max_lat"]})')

# Authentication

# Sending request using username and token and this gives us a temporary session key to access data
auth_resp = requests.post(
    url + 'login-token',
    json={'username': usgs_user, 'token': usgs_token}
).json()

if auth_resp['errorCode']:
    raise RuntimeError(f"USGS login failed: {auth_resp['errorMessage']}")

sessionkey = auth_resp['data']
hdr = {'X-Auth-Token': sessionkey}
print('Authenticated')

# Scene Search
scene_query = {
    'datasetName': 'landsat_ot_c2_l2',
    "sceneFilter" : {
        "cloudCoverFilter": {
            "max": 20,
            "min": 0
          },
        'spatialFilter': { #spatial region using bounding box (MBR = minimum bounding rectangle)
        'filterType': 'mbr',
        'lowerLeft' : {'latitude': AOI['min_lat'], 'longitude': AOI['min_lon']},
        'upperRight': {'latitude': AOI['max_lat'], 'longitude': AOI['max_lon']}
    }
    },
    'temporalFilter': {'startDate': start_date, 'endDate': end_date},
    'maxResults': 20,
    'startingNumber': 1
}

raw_scenes = requests.post(url + 'scene-search', json=scene_query, headers=hdr).json()
all_scenes = raw_scenes['data']['results']

# Filtering by cloud cover as we keep only the areas where cloudcover < cloudlimit
valid_scenes = [s for s in all_scenes if s.get('cloudCover', 100) < cloudlimit]
valid_scenes.sort(key=lambda s: s['cloudCover'])

print(f'Total scenes found      : {len(all_scenes)}')
print(f'Scenes below {cloudlimit}% clouds : {len(valid_scenes)}')
print()
print(f'{"#":<4} {"Entity ID":<42} {"Cloud%"}')
print('─' * 70)
for idx, sc in enumerate(valid_scenes):
    print(f"{idx:<4} {sc['entityId']:<42} {sc.get('cloudCover',0):.1f}%")

# Best scene the one with lowest cloud cover
chosen        = valid_scenes[0]
scene_id      = chosen['entityId']
scene_date    = chosen.get('acquisitionDate', 'unknown')
scene_cloud   = chosen.get('cloudCover', 0)

print(f'Selected scene : {scene_id}')
print(f'Cloud cover    : {scene_cloud:.1f}%')

# Fetching download options
opts_resp = requests.post(
    url + 'download-options',
    json={'datasetName': 'landsat_ot_c2_l2', 'entityIds': [scene_id]},
    headers=hdr
).json()

available_opts = opts_resp['data']

# Prefer the full Bundle
chosen_opt = next(
    (o for o in available_opts if 'Bundle' in o.get('productName','') and o.get('available')),
    next((o for o in available_opts if o.get('available')), None)
)
print(f'\nDownload product : {chosen_opt["productName"]}')
print(f'Product ID : {chosen_opt["id"]}')
