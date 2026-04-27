# from libraries import plt,requests,rioxarray,np,mcolors,rasterio,mpatches,os
# from part1a import AOI,out_dir
from part1b import *

min_lon = AOI['min_lon']
min_lat = AOI['min_lat']
max_lon = AOI['max_lon']
max_lat = AOI['max_lat']
esri_url = "https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer/exportImage"
parameters = {"bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}","bboxSR": 4326,
    "size": "1024,1024", "imageSR": 4326,"format": "tiff",
    "pixelType": "U8","noData": 0,"f": "image"}

response = requests.get(esri_url, params=parameters)

with open("lulc.tif", "wb") as f:
    f.write(response.content)

lulc_data = rioxarray.open_rasterio("lulc.tif", masked=True).squeeze()

print("Data Ingestion Complete")

ESRI_CLASSES = {
    1:  ('Water',              '#419BDF'),
    2:  ('Trees',              '#397D49'),
    4:  ('Flooded Vegetation', '#7A87C6'),
    5:  ('Crops',              '#E49635'),
    7:  ('Built Area',         '#C4281B'),
    8:  ('Bare Ground',        '#A59B8F'),
    9:  ('Snow / Ice',         '#A8EBFF'),
    10: ('Clouds',             '#616161'),
    11: ('Rangeland',          '#E3E2C3'),
}

# Loading & inspecting LULC raster
lulc_tif = "lulc.tif"

with rasterio.open(lulc_tif) as src_lulc:
    lulc_arr = src_lulc.read(1).astype(np.int16)
    lulc_meta = src_lulc.meta.copy()
    lulc_transform = src_lulc.transform

unique_vals, pixel_counts = np.unique(lulc_arr[lulc_arr > 0], return_counts=True)
print('Class distribution in AOI:')
print(f'{"Value":<8} {"Class":<22} {"Pixels":<12} {"Coverage %"}')
print('─' * 55)
total_valid = pixel_counts.sum()
for v, c in zip(unique_vals, pixel_counts):
    name = ESRI_CLASSES.get(int(v), ('Unknown','#000000'))[0]
    print(f'{int(v):<8} {name:<22} {int(c):<12} {c/total_valid*100:.1f}%')

# Visualising LULC

# Build a colour map aligned to class values
max_val  = 12
cmap_arr = np.zeros((max_val + 1, 3), dtype=np.float32)
for val, (name, hex_col) in ESRI_CLASSES.items():
    if val <= max_val:
        cmap_arr[val] = mcolors.to_rgb(hex_col)

lulc_rgb = cmap_arr[np.clip(lulc_arr, 0, max_val)]

fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(lulc_rgb)
ax.set_title(
    f'ESRI Sentinel-2 10 m Land Use Land Cover\n'
    f'{AOI["label"]}',
    fontsize=12, fontweight='bold'
)
ax.set_xlabel('Column (pixel)')
ax.set_ylabel('Row (pixel)')


legend_patches = [
    mpatches.Patch(facecolor=ESRI_CLASSES[int(v)][1],
                   label=f"{int(v):>2}  {ESRI_CLASSES[int(v)][0]}")
    for v in unique_vals if int(v) in ESRI_CLASSES
]
ax.legend(handles=legend_patches, loc='lower right',
          fontsize=9, title='ESRI LULC Classes',
          title_fontsize=9, framealpha=0.8)

plt.tight_layout()
save_p = os.path.join(out_dir, 'esri_lulc_map.png')
plt.savefig(save_p, dpi=150, bbox_inches='tight')
plt.show()
print(f'Image Saved !!!')
