from part4c import *

# 9A: Test-set predictions (with TTA) ─────────────────────────────
te_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long()),
    batch_size=256, shuffle=False
)

all_probs, all_true = [], []
best_model.eval()
with torch.no_grad():
    for xb, yb in te_loader:
        xb_d = xb.to(device)
        # TTA: average softmax over 4 orientations
        probs  = torch.softmax(best_model(xb_d), dim=1)
        probs += torch.softmax(best_model(xb_d.flip(-1)), dim=1)
        probs += torch.softmax(best_model(xb_d.flip(-2)), dim=1)
        probs += torch.softmax(best_model(xb_d.flip(-1).flip(-2)), dim=1)
        probs /= 4.0
        all_probs.append(probs.cpu().numpy())
        all_true.append(yb.numpy())

y_true  = np.concatenate(all_true)
y_proba = np.concatenate(all_probs)
y_pred  = y_proba.argmax(axis=1)

overall_acc = (y_true == y_pred).mean() * 100
print(f'Test-set overall accuracy (with TTA): {overall_acc:.2f}%')
print()
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# 9B: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation='vertical')
ax.set_title('Confusion Matrix — CNN LULC Classifier\n(Test Set)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
plt.show()

# Normalised confusion matrix
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(10, 8))
disp_norm = ConfusionMatrixDisplay(confusion_matrix=np.round(cm_norm, 2), display_labels=CLASS_NAMES)
disp_norm.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation='vertical')
ax.set_title('Normalised Confusion Matrix (Row = True class)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'confusion_matrix_normalised.png'), dpi=150)
plt.show()

# 9C: F1 Scores
f1_per_class = f1_score(y_true, y_pred, average=None)
f1_macro     = f1_score(y_true, y_pred, average='macro')
f1_weighted  = f1_score(y_true, y_pred, average='weighted')

print('F1 Scores per class:')
print(f'{"Class":<22} {"F1 Score"}')
print('─' * 35)
for name, score in zip(CLASS_NAMES, f1_per_class):
    bar = '█' * int(score * 20)
    print(f'{name:<22} {score:.4f}  {bar}')
print('─' * 35)
print(f'{"Macro average":<22} {f1_macro:.4f}')
print(f'{"Weighted average":<22} {f1_weighted:.4f}')

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(CLASS_NAMES, f1_per_class,
              color=[ESRI_CLASSES.get(int(c), ('?','#888'))[1] for c in le.classes_],
              edgecolor='k', linewidth=0.5)
ax.axhline(f1_macro,    color='red',    ls='--', lw=1.5, label=f'Macro avg: {f1_macro:.3f}')
ax.axhline(f1_weighted, color='purple', ls=':',  lw=1.5, label=f'Weighted avg: {f1_weighted:.3f}')
ax.set_ylim(0, 1.05)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('F1 Score per LULC Class', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=30)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

for bar, score in zip(bars, f1_per_class):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{score:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'f1_scores.png'), dpi=150)
plt.show()

# 9D: ROC Curves (one-vs-rest)
from sklearn.preprocessing import label_binarize

y_bin = label_binarize(y_true, classes=list(range(num_classes)))

fig, ax = plt.subplots(figsize=(9, 7))

auc_scores = {}
for i, cls_name in enumerate(CLASS_NAMES):
    if y_bin[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
    roc_auc      = auc(fpr, tpr)
    auc_scores[cls_name] = roc_auc

    hex_col = ESRI_CLASSES.get(int(le.classes_[i]), ('?', '#888888'))[1]
    ax.plot(fpr, tpr, lw=1.8, color=hex_col,
            label=f'{cls_name} (AUC = {roc_auc:.3f})')

ax.plot([0,1],[0,1], 'k--', lw=1, label='Random classifier')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves — One-vs-Rest (CNN LULC Classifier)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'roc_curves.png'), dpi=150)
plt.show()

print('AUC Scores:')
for cls, sc in sorted(auc_scores.items(), key=lambda x: -x[1]):
    print(f'  {cls:<22}: {sc:.4f}')

# 9E: Intersection over Union (IoU)
def compute_iou(y_true, y_pred, num_classes):
    iou_list = []
    for cls_idx in range(num_classes):
        tp = ((y_pred == cls_idx) & (y_true == cls_idx)).sum()
        fp = ((y_pred == cls_idx) & (y_true != cls_idx)).sum()
        fn = ((y_pred != cls_idx) & (y_true == cls_idx)).sum()
        denom = tp + fp + fn
        iou_list.append(tp / denom if denom > 0 else float('nan'))
    return np.array(iou_list)

iou_vals = compute_iou(y_true, y_pred, num_classes)
mean_iou = np.nanmean(iou_vals)

print('IoU (Jaccard Index) per class:')
print(f'{"Class":<22} {"IoU"}')
print('─' * 35)
for name, score in zip(CLASS_NAMES, iou_vals):
    bar = '█' * int(score * 20) if not np.isnan(score) else 'N/A'
    print(f'{name:<22} {score:.4f}  {bar}')
print('─' * 35)
print(f'{"Mean IoU (mIoU)":<22} {mean_iou:.4f}')

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
iou_clean = [v if not np.isnan(v) else 0 for v in iou_vals]
bars = ax.bar(CLASS_NAMES, iou_clean,
              color=[ESRI_CLASSES.get(int(c), ('?','#888'))[1] for c in le.classes_],
              edgecolor='k', linewidth=0.5)
ax.axhline(mean_iou, color='navy', ls='--', lw=1.5, label=f'mIoU = {mean_iou:.3f}')
ax.set_ylim(0, 1.05)
ax.set_ylabel('IoU (Jaccard Index)', fontsize=11)
ax.set_title('IoU Score per LULC Class', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=30)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

for bar, score in zip(bars, iou_clean):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{score:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'iou_scores.png'), dpi=150)
plt.show()

# 9F: Summary
print('FINAL ACCURACY SUMMARY — CNN LULC CLASSIFIER')
print("\n")
print(f'AOI                 : {AOI["label"]}')
print(f'Landsat scene       : {scene_id}')
print(f'Scene date          : {scene_date}')
print(f'CNN best HP         : lr={BEST["hp"]["lr"]}, dropout={BEST["hp"]["dropout"]}')
print(f'Number of classes   : {num_classes}')
print(f'Test-set samples    : {len(y_test)}')
print('─' * 55)
print(f'Overall Accuracy    : {overall_acc:.2f}%')
print(f'Macro F1            : {f1_macro:.4f}')
print(f'Weighted F1         : {f1_weighted:.4f}')
print(f'Mean IoU (mIoU)     : {mean_iou:.4f}')
mean_auc = np.mean(list(auc_scores.values()))
print(f'Mean AUC (OvR)      : {mean_auc:.4f}')

print(f'\nAll output files saved in: {out_dir}')
for fname in sorted(os.listdir(out_dir)):
    print(f'  {fname}')
