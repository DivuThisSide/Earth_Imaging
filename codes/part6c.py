from part6b import *

def safe_parse_date(d):
    try:
        return datetime.strptime(d, '%Y-%m-%d')
    except ValueError:
        return None

date_objs = [safe_parse_date(d) for d in scene_dates]
valid_date_mask = [d is not None for d in date_objs]

if not any(valid_date_mask):
    print('No parseable dates')
else:
    fig, ax = plt.subplots(figsize=(13, 6))

    for cls in le.classes_:
        name    = ESRI_CLASSES.get(int(cls), (str(cls), '#888888'))[0]
        hex_col = ESRI_CLASSES.get(int(cls), (str(cls), '#888888'))[1]
        vals    = class_ndvi_ts[cls]
        stds    = class_ndvi_std[cls]

        # Keep only entries with valid date AND valid NDVI
        keep = [i for i in range(len(date_objs))
                if valid_date_mask[i] and not np.isnan(vals[i])]
        if len(keep) < 2:
            continue

        x_pts = [date_objs[i] for i in keep]
        y_pts = [vals[i]      for i in keep]
        s_pts = [stds[i]      for i in keep]

        line, = ax.plot(x_pts, y_pts,
                        marker='o', linewidth=2.2, markersize=7,
                        color=hex_col, label=name, zorder=3)

        # ±1 std shaded band
        y_arr = np.array(y_pts)
        s_arr = np.array(s_pts)
        ax.fill_between(x_pts, y_arr - s_arr, y_arr + s_arr,
                        color=hex_col, alpha=0.12, zorder=2)

        ax.annotate(f'{y_pts[-1]:.2f}',
                    xy=(x_pts[-1], y_pts[-1]),
                    xytext=(6, 0), textcoords='offset points',
                    fontsize=8, color=hex_col, va='center', fontweight='bold')

    ax.axhspan(0.6,  1.0, alpha=0.06, color='green',  label='Dense veg (>0.6)')
    ax.axhspan(0.3,  0.6, alpha=0.06, color='lime',   label='Moderate veg (0.3–0.6)')
    ax.axhspan(0.0,  0.3, alpha=0.06, color='yellow', label='Sparse/bare (0–0.3)')
    ax.axhspan(-1.0, 0.0, alpha=0.06, color='blue',   label='Water/non-veg (<0)')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate(rotation=30)
    ax.set_ylim(-0.25, 1.0)
    ax.set_xlabel('Scene Date', fontsize=11)
    ax.set_ylabel('Mean NDVI', fontsize=11)
    ax.set_title(
        f'NDVI Time Series per LULC Class — {AOI["label"]}\n'
        f'({len(seasonal_scenes)} scenes · shading = ±1 std dev)',
        fontsize=12, fontweight='bold'
    )
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, loc='upper left', ncol=2, framealpha=0.85)

    plt.tight_layout()
    
    ts_save = os.path.join(out_dir, 'ndvi_time_series.png')
    plt.savefig(ts_save, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    plt.close('all') 
    
    print(f'Successfully Saved → {os.path.abspath(ts_save)}')    

season_labels = [sm['season'] for sm in seasonal_scenes]
n_classes     = len(le.classes_)
x             = np.arange(len(season_labels))
bar_w         = 0.8 / n_classes

fig, ax = plt.subplots(figsize=(13, 5))

for i, cls in enumerate(le.classes_):
    name    = ESRI_CLASSES.get(int(cls), (str(cls), '#888'))[0]
    hex_col = ESRI_CLASSES.get(int(cls), (str(cls), '#888'))[1]
    vals    = class_ndvi_ts[cls]
    offset  = (i - n_classes / 2) * bar_w + bar_w / 2
    ax.bar(x + offset, vals, width=bar_w,
           label=name, color=hex_col, edgecolor='k', linewidth=0.4, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(season_labels, fontsize=10)
ax.set_xlabel('Season', fontsize=11)
ax.set_ylabel('Mean NDVI', fontsize=11)
ax.set_title(
    f'Seasonal Mean NDVI by LULC Class — {AOI["label"]}  '
    f'(Path {WRS_PATH:03d} / Row {WRS_ROW:03d})',
    fontsize=12, fontweight='bold'
)
ax.set_ylim(-0.1, 1.0)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=8, ncol=3, loc='upper right', framealpha=0.85)

plt.tight_layout()

bar_save = os.path.join(out_dir, 'ndvi_seasonal_bar.png')
plt.savefig(bar_save, dpi=150, bbox_inches='tight')

plt.show()

plt.close('all') 

print(f'Successfully Saved → {os.path.abspath(bar_save)}')

if len(seasonal_scenes) >= 2:
    ndvi_first = seasonal_scenes[0]['ndvi']
    ndvi_last  = seasonal_scenes[-1]['ndvi']
    min_r = min(ndvi_first.shape[0], ndvi_last.shape[0])
    min_c = min(ndvi_first.shape[1], ndvi_last.shape[1])
    delta = ndvi_last[:min_r, :min_c] - ndvi_first[:min_r, :min_c]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_i, (data, title, cmap, vmin, vmax) in zip(axes, [
        (ndvi_first[:min_r, :min_c], f'NDVI — {seasonal_scenes[0]["date"]}',  'RdYlGn', -0.2, 1.0),
        (ndvi_last [:min_r, :min_c], f'NDVI — {seasonal_scenes[-1]["date"]}', 'RdYlGn', -0.2, 1.0),
        (delta,                       'NDVI Change (Last − First)',             'RdBu',   -0.4, 0.4),
    ]):
        im = ax_i.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_i.set_title(title, fontsize=10, fontweight='bold')
        ax_i.axis('off')
        fig.colorbar(im, ax=ax_i, fraction=0.04, pad=0.02)

    fig.suptitle(
        f'NDVI Change Map — {AOI["label"]}  (Path {WRS_PATH:03d} / Row {WRS_ROW:03d})',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    change_save = os.path.join(out_dir, 'ndvi_change_map.png')
    plt.savefig(change_save, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    plt.close('all') 
    
    print(f'Successfully Saved → {os.path.abspath(change_save)}')  


print('FINAL SUMMARY')

print(f'AOI        : {AOI["label"]}')
print(f'WRS-2 Tile : Path {WRS_PATH:03d} / Row {WRS_ROW:03d}')
print(f'Scenes     : {len(seasonal_scenes)}  |  '
      f'Date range : {scene_dates[0]} → {scene_dates[-1]}')
print()
print(f'{"Class":<22}  {"Peak season":<16}  {"Trough season":<16}  {"Swing"}')
print('  ' + '─'*62)
for cls in le.classes_:
    name  = ESRI_CLASSES.get(int(cls), (str(cls),))[0]
    vals  = class_ndvi_ts[cls]
    valid = ~np.isnan(vals)
    if valid.sum() == 0:
        continue
    peak_i   = int(np.nanargmax(vals))
    trough_i = int(np.nanargmin(vals))
    swing    = float(np.nanmax(vals) - np.nanmin(vals))
    print(f'{name:<22}  '
          f'{scene_dates[peak_i]:<16}  '
          f'{scene_dates[trough_i]:<16}  '
          f'{swing:.3f}')
print(f'\nAll outputs saved in: {out_dir}')


print('\n' + '='*30)
print('FILE SYSTEM VERIFICATION')
print('='*30)
expected_files = ['ndvi_map.png', 'ndvi_time_series.png', 'ndvi_seasonal_bar.png', 'ndvi_change_map.png']

for file_name in expected_files:
    full_path = os.path.join(out_dir, file_name)
    if os.path.exists(full_path):
        print(f" FOUND: {file_name}")
    else:
        print(f" MISSING: {file_name} (Looked in: {out_dir})")
