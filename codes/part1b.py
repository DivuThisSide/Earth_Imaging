from part1a import *

# Request download URL
dl_req = requests.post(
    url + 'download-request',
    json={
        'downloads': [{'entityId': scene_id, 'productId': chosen_opt['id']}],
        'label': 'nara-lulc-project'
    },
    headers=hdr
).json()

ready_urls = dl_req['data'].get('availableDownloads', [])
dl_url = ready_urls[0]['url']
print(f'Download URL obtained: {dl_url}')

# Stream download with progress
tar_path = os.path.join(landsat_dir, f'{scene_id}.tar')

response  = requests.get(dl_url, stream=True)
total  = int(response.headers.get('content-length', 0)) // (1024**2) # total size of image in megabytes(mb)
received  = 0

with open(tar_path, 'wb') as fh:
    for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
        fh.write(chunk)
        received += len(chunk)
        done = received // (1024**2) # size of image (in mb) already downloaded
        if done % 50 == 0 and done > 0:
            pct = done / total * 100 if total else 0
            print(f'{done} / {total} MB  ({pct:.0f}%)')

print(f'\nDownload complete')

# Extract TAR & locate bands
scene_dir = os.path.join(landsat_dir, scene_id)
os.makedirs(scene_dir, exist_ok=True)

with tarfile.open(tar_path) as tf:
    tf.extractall(scene_dir)

extracted = sorted(os.listdir(scene_dir))
print(f'Extracted {len(extracted)} files:')
for f in extracted:
    print(f'{f}')

# Locate band TIFs
band_map = {2: None, 3: None, 4: None, 5: None}
for fname in extracted:
    for b in band_map:
        if fname.endswith(f'_B{b}.TIF'):
            band_map[b] = os.path.join(scene_dir, fname)

b2_blue, b3_green, b4_red, b5_nir = (
    band_map[2], band_map[3], band_map[4], band_map[5]
)

print('\nBand files located:')
for b, p in band_map.items():
    label = {2:'Blue', 3:'Green', 4:'Red', 5:'NIR'}[b]
    print(f'  B{b} ({label}): {os.path.basename(p) if p else "NOT FOUND"}')

# Persist config
config_path = os.path.join(landsat_dir, 'band_paths.json')
cfg = {
    'scene_id'   : scene_id,
    'scene_date' : scene_date,
    'cloud_pct'  : scene_cloud,
    'aoi'        : AOI,
    'bands'      : {'B2': b2_blue, 'B3': b3_green, 'B4': b4_red, 'B5': b5_nir}
}
with open(config_path, 'w') as fh:
    json.dump(cfg, fh, indent=2)
print(f'\nConfig saved')

aoi_geom = mapping(box(AOI['min_lon'], AOI['min_lat'],
                              AOI['max_lon'], AOI['max_lat']))

def load_band(tif_path: str) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        # Step 1: reproject AOI box into scene CRS
        aoi_native_geom = transform_geom('EPSG:4326', src.crs, aoi_geom)
        aoi_shape = shape(aoi_native_geom)   # shapely geometry

        # Step 2: get the actual raster footprint in same CRS
        raster_bounds = box(*src.bounds)     # shapely box from raster extent

        # If the AOI slightly overshoots the scene edge (floating point,or scene doesn't fully cover AOI), we clip to the overlap.
        aoi_clipped = aoi_shape.intersection(raster_bounds)

        if aoi_clipped.is_empty:
            raise ValueError(
                f"AOI does not overlap raster at all: {tif_path}\n"
                f"Raster bounds : {src.bounds}\n"
                f"AOI (native)  : {aoi_shape.bounds}\n"
                "Check that your AOI lon/lat coordinates are correct "
                "and that the downloaded scene covers your area."
            )

        # Step 4: clip and return
        cropped, _ = rio_mask(src, [mapping(aoi_clipped)], crop=True, nodata=0)

    return cropped[0].astype(np.float32)

# Percentile stretch helper
def pct_stretch(arr: np.ndarray, lo=2, hi=98) -> np.ndarray:
    valid = arr[arr > 0]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    p_lo, p_hi = np.percentile(valid, lo), np.percentile(valid, hi)
    out = np.clip(arr, p_lo, p_hi)
    out = (out - p_lo) / (p_hi - p_lo + 1e-9) * 255
    out[arr == 0] = 0
    return out.astype(np.uint8)

# Loading bands
print('Loading and clipping bands to AOI…')
red_raw   = load_band(b4_red)
green_raw = load_band(b3_green)
blue_raw  = load_band(b2_blue)
nir_raw   = load_band(b5_nir)

# Checking if all bands must have same shape
shapes = [red_raw.shape, green_raw.shape, blue_raw.shape, nir_raw.shape]
if len(set(shapes)) > 1:
    min_r = min(s[0] for s in shapes)
    min_c = min(s[1] for s in shapes)
    red_raw   = red_raw  [:min_r, :min_c]
    green_raw = green_raw[:min_r, :min_c]
    blue_raw  = blue_raw [:min_r, :min_c]
    nir_raw   = nir_raw  [:min_r, :min_c]
    print(f'Bands had slightly different shapes — trimmed to {min_r}×{min_c}')

print(f'Array shape (rows × cols): {red_raw.shape}')

RGB = np.stack([pct_stretch(red_raw),
                pct_stretch(green_raw),
                pct_stretch(blue_raw)], axis=-1)

print('RGB composite built.')

# Ploting True Colour
fig, ax = plt.subplots(figsize=(13, 10))
ax.imshow(RGB)
ax.set_title(
    f'True Colour Composite  —  {AOI["label"]}\n'
    f'Scene: {scene_id} | Date: {scene_date} | Cloud: {scene_cloud:.1f}% | Area: ~{AOI["area_km2"]} km²',
    fontsize=11, fontweight='bold'
)
ax.set_xlabel('Column (pixel)')
ax.set_ylabel('Row (pixel)')
ax.tick_params(labelsize=8)

# Colour key
legend_items = [
    mpatches.Patch(label='R = Band 4 (Red)',   facecolor='red'),
    mpatches.Patch(label='G = Band 3 (Green)', facecolor='green'),
    mpatches.Patch(label='B = Band 2 (Blue)',  facecolor='blue'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=9,
          framealpha=0.7, title='Band assignment')

plt.tight_layout()
save_p = os.path.join(out_dir, 'true_colour_composite.png')
plt.savefig(save_p, dpi=150, bbox_inches='tight')
plt.show()
print(f'Image Saved !!!')
