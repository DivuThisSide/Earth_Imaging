from part3 import *

# 8A: Align LULC labels to Landsat grid

# Since, Landsat ~30m and  ESRI LULC ~10m we need to downscale LULC by factor ≈ 3
rows_ls, cols_ls = red_raw.shape
rows_lc, cols_lc = lulc_arr.shape
scale_r = rows_ls / rows_lc
scale_c = cols_ls / cols_lc

# Nearest-neighbour zoom (preserves class integer values)
lulc_resampled = zoom(lulc_arr.astype(np.float32), (scale_r, scale_c), order=0).astype(np.int16)

print(f'Landsat grid   : {rows_ls} × {cols_ls}')
print(f'ESRI LULC grid : {rows_lc} × {cols_lc}')
print(f'Resampled LULC : {lulc_resampled.shape}')

min_r = min(rows_ls, lulc_resampled.shape[0])
min_c = min(cols_ls, lulc_resampled.shape[1])

label_grid  = lulc_resampled[:min_r, :min_c]
red_g       = red_raw        [:min_r, :min_c]
green_g     = green_raw      [:min_r, :min_c]
blue_g      = blue_raw       [:min_r, :min_c]
nir_g       = nir_raw        [:min_r, :min_c]
ndvi_g      = ndvi           [:min_r, :min_c]
ndvi_g      = np.nan_to_num(ndvi_g, nan=0.0)

print(f'Aligned grid   : {label_grid.shape}')

# 8B: Assemble feature stack

# Normalise each band to [0, 1] using its 2nd–98th percentile
def normalise(arr):
    valid = arr[arr != 0]
    lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)

# 5-channel feature cube  (C, H, W)
feat_cube = np.stack([
    normalise(blue_g),
    normalise(green_g),
    normalise(red_g),
    normalise(nir_g),
    ((ndvi_g + 1) / 2).astype(np.float32)   # NDVI scaled to [0,1]
], axis=0)   # shape: (5, H, W)

print(f'Feature cube shape: {feat_cube.shape}  (channels, rows, cols)')

# 8C: Extract patches & labels
PATCH = 11      # 7×7 pixel patch centred on labelled pixel
PAD   = PATCH // 2
MAX_PER_CLASS = 2000   # cap samples per class for balance
MIN_PER_CLASS = 200

# Padded the feature cube so border pixels can have full patches
padded = np.pad(feat_cube, ((0,0),(PAD,PAD),(PAD,PAD)), mode='reflect')

present_classes = [int(v) for v in np.unique(label_grid) if v > 0]
print(f'Classes in scene: {present_classes}')

patches, labels_raw = [], []

rng = np.random.default_rng(seed=42)

for cls in present_classes:
    rows_idx, cols_idx = np.where(label_grid == cls)
    n = len(rows_idx)
      # Oversample minority classes (with replacement) to MIN_PER_CLASS
    target_n = max(min(n, MAX_PER_CLASS), MIN_PER_CLASS)
    use_replace = n < target_n   # allow duplicates for rare classes
    sel = rng.choice(n, size=target_n, replace=use_replace)
    for r, c in zip(rows_idx[sel], cols_idx[sel]):
        patch = padded[:, r:r+PATCH, c:c+PATCH]   # (5, PATCH, PATCH)
        patches.append(patch)
        labels_raw.append(cls)

patches_np = np.stack(patches).astype(np.float32)   # (N, 5, 7, 7)
labels_np  = np.array(labels_raw)

# Re-encode class values to 0-based integers
le = LabelEncoder().fit(labels_np)
labels_enc = le.transform(labels_np)
CLASS_NAMES = [ESRI_CLASSES.get(int(c), (str(c),))[0] for c in le.classes_]

print(f'\nTotal patches  : {len(patches_np)}')
print(f'Patch shape    : {patches_np[0].shape}')
print(f'Num classes    : {len(CLASS_NAMES)}')
print(f'Class names    : {CLASS_NAMES}')

# 8D: Dataset Split into train, test and validation
X_train_val, X_test, y_train_val, y_test = train_test_split(
    patches_np, labels_enc, test_size=0.15, random_state=42, stratify=labels_enc
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
)

print(f'Training   : {len(X_train):>5} patches')
print(f'Validation : {len(X_val):>5} patches')
print(f'Test       : {len(X_test):>5} patches')

# Compute inverse-frequency class weights for balanced training
from collections import Counter
_counts = Counter(y_train)
_total = sum(_counts.values())
_n_cls = len(_counts)
class_weights = torch.tensor(
    [_total / (_n_cls * _counts[i]) for i in range(_n_cls)],
    dtype=torch.float32
)
print(f'\nClass weights: {dict(zip(CLASS_NAMES, class_weights.numpy().round(2)))}')

# Visualising training and test patches on Landsat image
fig, ax = plt.subplots(figsize=(13, 10))
ax.imshow(RGB)

# Find pixel coordinates of training/test samples
all_rows, all_cols = np.where(label_grid > 0)
n_display = min(500, len(all_rows))
disp_idx = rng.choice(len(all_rows), n_display, replace=False)

tr_idx = disp_idx[:int(n_display*0.7)]
te_idx = disp_idx[int(n_display*0.7):]

ax.scatter(all_cols[tr_idx], all_rows[tr_idx],
           s=2, c='lime',   alpha=0.6, label='Train pixels')
ax.scatter(all_cols[te_idx], all_rows[te_idx],
           s=2, c='red',    alpha=0.6, label='Test pixels')

ax.legend(loc='lower right', fontsize=9, markerscale=5)
ax.set_title('Training & Test Pixels overlaid on Landsat Image', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'train_test_overlay.png'), dpi=150)
plt.show()