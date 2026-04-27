from part6a import *

def compute_scene_ndvi_direct(b4_path: str, b5_path: str) -> np.ndarray:

    red = load_band(b4_path)
    nir = load_band(b5_path)

    min_r = min(red.shape[0], nir.shape[0])
    min_c = min(red.shape[1], nir.shape[1])
    red, nir = red[:min_r, :min_c], nir[:min_r, :min_c]

    with np.errstate(invalid='ignore', divide='ignore'):
        ndvi_out = np.where(
            (nir + red) > 0,
            (nir - red) / (nir + red),
            np.nan
        ).astype(np.float32)
    return ndvi_out


print('Computing NDVI for each seasonal scene …\n')
for sm in seasonal_scenes:
    # Explicitly use THIS scene's band paths
    sm['ndvi'] = compute_scene_ndvi_direct(sm['band_map'][4], sm['band_map'][5])
    valid = sm['ndvi'][~np.isnan(sm['ndvi'])]
    print(f"  [{sm['season']}]  {sm['date']}  →  "
          f"shape={sm['ndvi'].shape}  mean NDVI={valid.mean():.4f}  "
          f"min={valid.min():.3f}  max={valid.max():.3f}")

print('\nComputing per-class NDVI means …\n')

class_ndvi_ts  = {cls: [] for cls in le.classes_}
class_ndvi_std = {cls: [] for cls in le.classes_}   # standard deviation too
scene_dates    = []

for sm in seasonal_scenes:
    ndvi_scene = sm['ndvi']
    h, w = ndvi_scene.shape
    scene_dates.append(sm['date'])

    sc_r = h / lulc_arr.shape[0]
    sc_c = w / lulc_arr.shape[1]
    lulc_for_scene = zoom(lulc_arr.astype(np.float32), (sc_r, sc_c), order=0).astype(np.int16)
    lulc_for_scene = lulc_for_scene[:h, :w]

    for cls in le.classes_:
        mask = (lulc_for_scene == cls)
        vals = ndvi_scene[mask]
        vals = vals[~np.isnan(vals)]
        if vals.size > 0:
            class_ndvi_ts[cls].append(float(vals.mean()))
            class_ndvi_std[cls].append(float(vals.std()))
        else:
            class_ndvi_ts[cls].append(np.nan)
            class_ndvi_std[cls].append(np.nan)

class_ndvi_ts  = {cls: np.array(v) for cls, v in class_ndvi_ts.items()}
class_ndvi_std = {cls: np.array(v) for cls, v in class_ndvi_std.items()}

# Summary table
header = f"{'Class':<22}" + "".join(f"{d:>13}" for d in scene_dates)
print(header)
print('─' * (22 + 13 * len(scene_dates)))
for cls in le.classes_:
    name = ESRI_CLASSES.get(int(cls), (str(cls),))[0]
    row  = f"{name:<22}"
    for v in class_ndvi_ts[cls]:
        row += f"{v:>13.4f}" if not np.isnan(v) else f"{'N/A':>13}"
    print(row)
