from part4a import *

# 8E: CNN architecture
class LandsatCNN(nn.Module):
    """
      Input  : (batch, 5, PATCH, PATCH)
    Output : (batch, num_classes)
    Architecture: 3 conv blocks → global avg pool → 3-layer classifier
    Fixes: removed duplicate AdaptiveAvgPool2d, widened hidden layer,
           added second dropout for regularisation.
    """
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

             nn.AdaptiveAvgPool2d(1),       # → (batch, 128, 1, 1)
        )
        self.classifier = nn.Sequential(
              nn.Flatten(),                  # → (batch, 128)
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),     # lighter second dropout
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

num_classes = len(CLASS_NAMES)
print(f'CNN architecture — {num_classes} output classes')
demo = LandsatCNN(num_classes)
print(demo)
n_params = sum(p.numel() for p in demo.parameters() if p.requires_grad)
print(f'\nTrainable parameters: {n_params:,}')

# 8F: Training utilities (improved)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_loaders(X_tr, y_tr, X_vl, y_vl, batch=128):
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
    vl_ds = TensorDataset(torch.from_numpy(X_vl), torch.from_numpy(y_vl).long())

    loader_kwargs = {
        'num_workers': 2 if DEVICE == 'cuda' else 0,
        'pin_memory': DEVICE == 'cuda'
    }
    return (
        DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=False, **loader_kwargs),
        DataLoader(vl_ds, batch_size=batch, shuffle=False, drop_last=False, **loader_kwargs)
    )


def train_model(model, train_loader, val_loader, lr, class_wts=None, epochs=60, patience=12):
    model.to(DEVICE)
    device = next(model.parameters()).device

    wts = class_wts.to(device) if class_wts is not None else None
    criterion = nn.CrossEntropyLoss(weight=wts, label_smoothing=0.05)
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=4, min_lr=1e-6
    )

    history = {'tr_loss': [], 'vl_loss': [], 'tr_acc': [], 'vl_acc': []}
    best_state = None
    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for xb, yb in train_loader:
            # Data augmentation — random flips (done on CPU for robustness)
            if torch.rand(1).item() > 0.5:
                xb = xb.flip(-1)      # horizontal flip
            if torch.rand(1).item() > 0.5:
                xb = xb.flip(-2)      # vertical flip

            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            tr_loss += loss.item() * len(xb)
            tr_correct += (out.argmax(1) == yb).sum().item()
            tr_n += len(xb)

        # Validation
        model.eval()
        vl_loss, vl_correct, vl_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                vl_loss += loss.item() * len(xb)
                vl_correct += (out.argmax(1) == yb).sum().item()
                vl_n += len(xb)

        epoch_tr_loss = tr_loss / tr_n
        epoch_vl_loss = vl_loss / vl_n
        epoch_tr_acc = tr_correct / tr_n * 100
        epoch_vl_acc = vl_correct / vl_n * 100

        history['tr_loss'].append(epoch_tr_loss)
        history['vl_loss'].append(epoch_vl_loss)
        history['tr_acc'].append(epoch_tr_acc)
        history['vl_acc'].append(epoch_vl_acc)

        scheduler.step(epoch_vl_loss)

        if epoch_vl_loss < best_val_loss:
            best_val_loss = epoch_vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            current_lr = optimiser.param_groups[0]['lr']
            print(f'Epoch {epoch:>3}/{epochs} | '
                  f'Train loss: {epoch_tr_loss:.4f}, acc: {epoch_tr_acc:.1f}% | '
                  f'Val loss: {epoch_vl_loss:.4f}, acc: {epoch_vl_acc:.1f}% | '
                  f'LR: {current_lr:.2e}')

        if no_improve >= patience:
            print(f'Early stop at epoch {epoch} (no val_loss improvement for {patience} epochs)')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

# 8G: Hyperparameter tuning using validation set
HP_GRID = list(ParameterGrid({
    'lr': [1e-3, 5e-4, 3e-4],
    'dropout': [0.3, 0.45],
    'batch': [64, 128]
}))

print(f'Hyperparameter combinations to search: {len(HP_GRID)}')
print('Grid:', HP_GRID)

hp_results = []

for hp in HP_GRID:
    print(f'\n lr={hp["lr"]}, dropout={hp["dropout"]}, batch={hp["batch"]}')
    tr_loader, vl_loader = make_loaders(X_train, y_train, X_val, y_val, batch=hp['batch'])

    net = LandsatCNN(num_classes, dropout=hp['dropout'])
    hist = train_model(net, tr_loader, vl_loader, lr=hp['lr'], class_wts=class_weights, epochs=60, patience=12)
    best_val_acc = max(hist['vl_acc'])

    hp_results.append({'hp': hp, 'val_acc': best_val_acc, 'model': net, 'history': hist})
    print(f'Best val accuracy: {best_val_acc:.1f}%')

hp_results.sort(key=lambda x: x['val_acc'], reverse=True)
BEST = hp_results[0]
print(f'\n★ Best hyperparameters: {BEST["hp"]}  →  val acc {BEST["val_acc"]:.1f}%')

# 8H: Ploting training curves for all HP combinations
fig, axes = plt.subplots(2, len(hp_results), figsize=(5*len(hp_results), 8))
if len(hp_results) == 1:
    axes = axes.reshape(2, 1)

for col, res in enumerate(hp_results):
    hist = res['history']
    label_str = f"lr={res['hp']['lr']}, drop={res['hp']['dropout']}"
    ep = range(1, len(hist['tr_loss'])+1)

    axes[0, col].plot(ep, hist['tr_loss'], label='Train',  color='steelblue')
    axes[0, col].plot(ep, hist['vl_loss'], label='Val',    color='tomato', ls='--')
    axes[0, col].set_title(label_str, fontsize=9)
    axes[0, col].set_ylabel('Loss')
    axes[0, col].legend(fontsize=8)
    axes[0, col].grid(alpha=0.3)

    axes[1, col].plot(ep, hist['tr_acc'], label='Train',  color='steelblue')
    axes[1, col].plot(ep, hist['vl_acc'], label='Val',    color='tomato', ls='--')
    axes[1, col].set_ylabel('Accuracy (%)')
    axes[1, col].set_xlabel('Epoch')
    axes[1, col].legend(fontsize=8)
    axes[1, col].grid(alpha=0.3)

fig.suptitle('CNN Training Curves — Hyperparameter Search', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
plt.show()