from part4b import *

# 8I: Full AOI Prediction
best_model = BEST['model'].eval().to(DEVICE)

device = next(best_model.parameters()).device

# --- Test-Time Augmentation helper ---
def predict_tta(model, batch_np, dev):
    """Average softmax over 4 orientations: orig, h-flip, v-flip, hv-flip."""
    t = torch.from_numpy(batch_np).to(dev)
    probs  = torch.softmax(model(t), dim=1)
    probs += torch.softmax(model(t.flip(-1)), dim=1)           # h-flip
    probs += torch.softmax(model(t.flip(-2)), dim=1)           # v-flip
    probs += torch.softmax(model(t.flip(-1).flip(-2)), dim=1)  # hv-flip
    return probs / 4.0

# --- Spatial majority-vote filter (fast, convolution-based) ---
def spatial_majority_filter(pred, kernel=3):
    """Replace each pixel with the most-voted class in its neighbourhood."""
    from scipy.ndimage import uniform_filter
    classes = np.unique(pred[pred > 0])
    if len(classes) == 0:
        return pred
    votes = np.zeros((len(classes), *pred.shape), dtype=np.float32)
    for i, cls in enumerate(classes):
        votes[i] = uniform_filter((pred == cls).astype(np.float32), size=kernel)
    best_idx = votes.argmax(axis=0)
    out = np.zeros_like(pred)
    for i, cls in enumerate(classes):
        out[best_idx == i] = cls
    out[pred == 0] = 0  # keep background
    return out

# Flatten spatial pixels that have valid labels
valid_rows, valid_cols = np.where(label_grid > 0)
n_valid = len(valid_rows)

pred_probs = np.zeros((n_valid, num_classes), dtype=np.float32)
INFER_BATCH = 512

print(f'Running TTA inference on {n_valid:,} valid pixels (4 orientations)…')

for start in range(0, n_valid, INFER_BATCH):
    end = min(start + INFER_BATCH, n_valid)
    batch_patches = np.stack([
        padded[:, r:r+PATCH, c:c+PATCH]
        for r, c in zip(valid_rows[start:end], valid_cols[start:end])
    ]).astype(np.float32)

    with torch.no_grad():
        probs = predict_tta(best_model, batch_patches, device)
        pred_probs[start:end] = probs.cpu().numpy()

pred_labels = pred_probs.argmax(axis=1)

pred_map = np.zeros_like(label_grid, dtype=np.int16)
pred_map[valid_rows, valid_cols] = le.inverse_transform(pred_labels)

# Apply spatial majority filter to smooth noisy predictions
pred_map = spatial_majority_filter(pred_map, kernel=3)

print('TTA inference + spatial smoothing complete.')

# Visualise predicted LULC on Landsat image
pred_rgb = cmap_arr[np.clip(pred_map, 0, max_val)]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, img, title in zip(
    axes,
    [lulc_rgb[:min_r, :min_c], pred_rgb],
    ['ESRI LULC Ground Truth', 'CNN Predicted LULC']
):
    ax.imshow(img)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

legend_patches = [
    mpatches.Patch(facecolor=ESRI_CLASSES[c][1], label=ESRI_CLASSES[c][0])
    for c in le.classes_ if c in ESRI_CLASSES
]
fig.legend(handles=legend_patches, loc='lower center',
           ncol=len(legend_patches), fontsize=9,
           title='ESRI LULC Classes', bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Ground Truth vs CNN Prediction — Dresden AOI', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'cnn_prediction_map.png'), dpi=150, bbox_inches='tight')
plt.show()