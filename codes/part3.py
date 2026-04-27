from part2 import *

# Computing NDVI

nir_f = nir_raw.astype(np.float32)
red_f = red_raw.astype(np.float32)

# Also we need to mask nodata(0) pixels
valid_mask = (nir_f > 0) & (red_f > 0)

ndvi = np.where(
    valid_mask,
    (nir_f - red_f) / (nir_f + red_f + 1e-9),
    np.nan
)

# Summary statistics
valid_ndvi = ndvi[~np.isnan(ndvi)]
print('NDVI Statistics:')
print(f'Min    : {valid_ndvi.min():.4f}')
print(f'Max    : {valid_ndvi.max():.4f}')
print(f'Mean   : {valid_ndvi.mean():.4f}')
print(f'Median : {np.median(valid_ndvi):.4f}')
print(f'Std    : {valid_ndvi.std():.4f}')
print()

# Vegetation fraction (NDVI > 0.3)
veg_frac = (valid_ndvi > 0.3).sum() / valid_ndvi.size * 100
print(f'Pixels with NDVI > 0.3 (vegetation): {veg_frac:.1f}%')

# Displaying NDVI
fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                         gridspec_kw={'width_ratios': [4, 1]})

ax_img, ax_cb = axes

ndvi_plot = ax_img.imshow(ndvi, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
ax_img.set_title(
    f'Normalised Difference Vegetation Index (NDVI)\n'
    f'{AOI["label"]}  —  {scene_date}',
    fontsize=12, fontweight='bold'
)
ax_img.set_xlabel('Column (pixel)')
ax_img.set_ylabel('Row (pixel)')

# Colourbar
cb = fig.colorbar(ndvi_plot, cax=ax_cb, orientation='vertical')
cb.set_label('NDVI value', fontsize=10)

# Annotation ticks with interpretations
tick_vals   = [-0.5, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
tick_labels = [
    '-0.5 Water',
    '-0.2 Bare soil',
     '0.0 Non-veg',
     '0.2 Sparse veg',
     '0.4 Grassland',
     '0.6 Shrub',
     '0.8 Dense veg',
     '1.0 Lush forest'
]
cb.set_ticks(tick_vals)
cb.set_ticklabels(tick_labels, fontsize=7)

plt.tight_layout()
save_p = os.path.join(out_dir, 'ndvi_map.png')
plt.savefig(save_p, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {save_p}')

# NDVI Histogram
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(valid_ndvi, bins=100, color='#3a9e5f', edgecolor='none', alpha=0.85)
ax.axvline(0.3,  color='orange', lw=1.5, ls='--', label='0.3 (veg threshold)')
ax.axvline(valid_ndvi.mean(), color='red', lw=1.5, ls='-', label=f'Mean = {valid_ndvi.mean():.2f}')
ax.set_xlabel('NDVI', fontsize=11)
ax.set_ylabel('Pixel count', fontsize=11)
ax.set_title(f'NDVI Distribution — {AOI["label"]}', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'ndvi_histogram.png'), dpi=150)
plt.show()
