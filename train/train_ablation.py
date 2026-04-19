################################################################################
#                                                                              #
#                    Hydrothermal Plume Anomaly Detection                      #
#                Ablation Study: GRU + MLP End-to-End (No Anomaly Loss)        #
#                         Cross-Validation Version                             #
#                                                                              #
# Revised: 2026.04.19                                                          #
#   - Use training set only with 5-fold cross-validation                       #
#   - Fixed GRU hyperparameters; grid search only for MLP hyperparameters      #
#     (hidden size, number of layers, learning rate)                           #
#   - GRUs act purely as feature extractors without anomaly supervision        #
#   - Final training on full training set with best MLP configuration          #
#   - Feature caching mechanism for faster repeated runs                       #
#   - GPU acceleration support (auto-detects CUDA device)                      #
#                                                                              #
################################################################################

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import os
import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

# ====================== Model Definitions ======================
class GRUCellFeatureExtractor(nn.Module):
    """GRU feature extractor (for Strategy B, outputs only hidden states, no anomaly probability)."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.gru_cells.append(nn.GRUCell(in_size, hidden_size))

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        new_h = []
        for i, cell in enumerate(self.gru_cells):
            h_i = h[i]
            out = cell(x if i == 0 else new_h[-1], h_i)
            new_h.append(out)
        return new_h[-1], torch.stack(new_h)  # return last hidden state and stacked states


class MLPClassifier(nn.Module):
    """MLP for plume type classification."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=5):
        super().__init__()
        layers = []
        prev = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, hidden_size))
            layers.append(nn.ReLU())
            prev = hidden_size
        layers.append(nn.Linear(prev, output_size))
        self.mlp = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.mlp(x))


# ====================== Feature Extraction Functions ======================
def extract_features_temp_turb(seq, window_size):
    """Temperature/Turbidity: extract 8 statistical features."""
    if len(seq) < window_size:
        return np.empty((0, 8), dtype=np.float32)
    windows = sliding_window_view(seq, window_size)
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    median = np.median(windows, axis=1)
    q1 = np.percentile(windows, 25, axis=1)
    q3 = np.percentile(windows, 75, axis=1)
    iqr = q3 - q1
    minv = windows.min(axis=1)
    maxv = windows.max(axis=1)
    return np.column_stack([mean, std, median, q1, q3, iqr, minv, maxv]).astype(np.float32)


def extract_features_meth_orp(seq, window_size):
    """Methane/ORP: extract 6 statistical features."""
    if len(seq) < window_size:
        return np.empty((0, 6), dtype=np.float32)
    windows = sliding_window_view(seq, window_size)
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    median = np.median(windows, axis=1)
    q1 = np.percentile(windows, 25, axis=1)
    q3 = np.percentile(windows, 75, axis=1)
    iqr = q3 - q1
    return np.column_stack([mean, std, median, q1, q3, iqr]).astype(np.float32)


# ====================== Dataset Class ======================
class HydrothermalDataset(Dataset):
    """Load raw data, extract sliding window features and cache them."""
    def __init__(self, data_dir, window_size, cache_key, cache_dir):
        self.window_size = window_size
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}_features_w{window_size}.npz")

        if os.path.exists(cache_file):
            try:
                with np.load(cache_file) as data:
                    self.features = [data['feat_temp'], data['feat_turb'], data['feat_meth'], data['feat_orp']]
                    self.plume_labels_aligned = data['plume_labels']
                    print(f"Loaded cached {cache_key} data successfully (window size={window_size})")
                    return
            except:
                print(f"Cache file corrupted, reprocessing {cache_key} data...")

        print(f"Processing {cache_key} data (window size={window_size})...")
        feat_path = os.path.join(data_dir, 'real_time_data.xlsx')
        plume_path = os.path.join(data_dir, 'plume_type_labels.xlsx')
        df_feat = pd.read_excel(feat_path)
        df_plume = pd.read_excel(plume_path)

        # Assume first four columns are temperature, turbidity, methane, ORP
        data = df_feat.iloc[:, :4].values.astype(np.float32)
        plume_labels = df_plume.values.astype(np.float32)

        feat_temp = extract_features_temp_turb(data[:, 0], window_size)
        feat_turb = extract_features_temp_turb(data[:, 1], window_size)
        feat_meth = extract_features_meth_orp(data[:, 2], window_size)
        feat_orp = extract_features_meth_orp(data[:, 3], window_size)

        self.features = [feat_temp, feat_turb, feat_meth, feat_orp]
        T = self.features[0].shape[0]
        self.plume_labels_aligned = plume_labels[window_size-1:, :]

        np.savez(cache_file,
                 feat_temp=feat_temp, feat_turb=feat_turb,
                 feat_meth=feat_meth, feat_orp=feat_orp,
                 plume_labels=self.plume_labels_aligned)
        print(f"{cache_key} data processing complete, samples: {T}")

    def __len__(self):
        return self.features[0].shape[0]

    def __getitem__(self, idx):
        x_list = [torch.from_numpy(self.features[i][idx]) for i in range(4)]
        y_plume = torch.from_numpy(self.plume_labels_aligned[idx])
        return x_list, y_plume


# ====================== Evaluation Function ======================
def evaluate_model(gru_temp, gru_turb, gru_meth, gru_orp, mlp, loader, device):
    """Evaluate classification accuracy of Strategy B model."""
    gru_temp.eval()
    gru_turb.eval()
    gru_meth.eval()
    gru_orp.eval()
    mlp.eval()

    all_preds = []
    all_true = []
    with torch.no_grad():
        for x_list, y_plume in loader:
            x_temp = x_list[0].to(device)
            x_turb = x_list[1].to(device)
            x_meth = x_list[2].to(device)
            x_orp = x_list[3].to(device)
            y = y_plume.to(device)

            feat_temp, _ = gru_temp(x_temp)
            feat_turb, _ = gru_turb(x_turb)
            feat_meth, _ = gru_meth(x_meth)
            feat_orp, _ = gru_orp(x_orp)

            combined = torch.cat([feat_temp, feat_turb, feat_meth, feat_orp], dim=1)
            fluid_probs = mlp(combined)
            pred = torch.argmax(fluid_probs, dim=1).cpu().numpy()
            true = torch.argmax(y, dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_true.extend(true)
    return accuracy_score(all_true, all_preds)


# ====================== Single-Fold Training Function (Fixed GRU Hyperparameters) ======================
def train_one_fold(train_loader, val_loader, device,
                   gru_params,   # fixed GRU hyperparameters dictionary
                   mlp_hidden, mlp_layers, lr, epochs, verbose=False):
    """
    Train Strategy B model on one fold, return best validation accuracy.
    gru_params dictionary contains: temp_hidden, temp_layers, turb_hidden, turb_layers,
                                    meth_hidden, meth_layers, orp_hidden, orp_layers
    """
    # Initialize feature extractors with fixed GRU hyperparameters
    gru_temp = GRUCellFeatureExtractor(8, gru_params['temp_hidden'], gru_params['temp_layers']).to(device)
    gru_turb = GRUCellFeatureExtractor(8, gru_params['turb_hidden'], gru_params['turb_layers']).to(device)
    gru_meth = GRUCellFeatureExtractor(6, gru_params['meth_hidden'], gru_params['meth_layers']).to(device)
    gru_orp  = GRUCellFeatureExtractor(6, gru_params['orp_hidden'], gru_params['orp_layers']).to(device)

    # MLP input dimension = sum of four GRU hidden sizes
    mlp_input_size = (gru_params['temp_hidden'] + gru_params['turb_hidden'] +
                      gru_params['meth_hidden'] + gru_params['orp_hidden'])
    mlp = MLPClassifier(input_size=mlp_input_size, hidden_size=mlp_hidden,
                        num_layers=mlp_layers, output_size=5).to(device)

    # Joint optimization of all parameters
    all_params = (list(gru_temp.parameters()) + list(gru_turb.parameters()) +
                  list(gru_meth.parameters()) + list(gru_orp.parameters()) +
                  list(mlp.parameters()))
    optimizer = optim.Adam(all_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        gru_temp.train()
        gru_turb.train()
        gru_meth.train()
        gru_orp.train()
        mlp.train()

        for x_list, y_plume in train_loader:
            x_temp = x_list[0].to(device)
            x_turb = x_list[1].to(device)
            x_meth = x_list[2].to(device)
            x_orp = x_list[3].to(device)
            y = y_plume.to(device)

            optimizer.zero_grad()
            feat_temp, _ = gru_temp(x_temp)
            feat_turb, _ = gru_turb(x_turb)
            feat_meth, _ = gru_meth(x_meth)
            feat_orp, _ = gru_orp(x_orp)
            combined = torch.cat([feat_temp, feat_turb, feat_meth, feat_orp], dim=1)
            fluid_probs = mlp(combined)
            loss = criterion(fluid_probs, torch.argmax(y, dim=1))
            loss.backward()
            optimizer.step()

        val_acc = evaluate_model(gru_temp, gru_turb, gru_meth, gru_orp, mlp, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {epoch+1:03d} | Val Acc: {val_acc:.4f}")

    return best_val_acc


# ====================== Grid Search (MLP Hyperparameters Only) ======================
def grid_search_mlp_only(dataset, device,
                         gru_params,   # fixed GRU hyperparameters
                         mlp_hidden_list, mlp_layers_list, lr_list,
                         epochs, batch_size, num_workers, n_splits=5):
    """
    Perform grid search over MLP hyperparameters using n_splits-fold cross-validation.
    Returns best MLP hyperparameters and corresponding cross-validation mean accuracy.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_mean_acc = 0.0
    best_params = None
    results = []

    total = len(mlp_hidden_list) * len(mlp_layers_list) * len(lr_list)
    print(f"\nStarting MLP hyperparameter grid search, total {total} combinations.")
    print("Fixed GRU hyperparameters:")
    for k, v in gru_params.items():
        print(f"  {k}: {v}")

    comb_count = 0
    for mlp_h in mlp_hidden_list:
        for mlp_l in mlp_layers_list:
            for lr in lr_list:
                comb_count += 1
                fold_accs = []
                print(f"\n[{comb_count}/{total}] Testing MLP: hidden={mlp_h}, layers={mlp_l}, lr={lr}")

                for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                    train_subset = Subset(dataset, train_idx)
                    val_subset = Subset(dataset, val_idx)
                    train_loader = DataLoader(train_subset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
                    val_loader = DataLoader(val_subset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

                    acc = train_one_fold(
                        train_loader, val_loader, device,
                        gru_params, mlp_h, mlp_l, lr, epochs, verbose=False
                    )
                    fold_accs.append(acc)
                    print(f"    Fold {fold+1}: Acc = {acc:.4f}")

                mean_acc = np.mean(fold_accs)
                std_acc = np.std(fold_accs)
                print(f"  --> Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

                results.append({
                    'mlp_hidden': mlp_h, 'mlp_layers': mlp_l, 'lr': lr,
                    'cv_mean': mean_acc, 'cv_std': std_acc
                })

                if mean_acc > best_mean_acc:
                    best_mean_acc = mean_acc
                    best_params = {'mlp_hidden': mlp_h, 'mlp_layers': mlp_l, 'lr': lr}

    # Save search results
    df_results = pd.DataFrame(results)
    df_results.to_csv("grid_search_mlp_only_results.csv", index=False)
    print(f"\nGrid search completed. Best CV accuracy: {best_mean_acc:.4f}")
    print("Best MLP hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params, best_mean_acc


# ====================== Final Training (Full Dataset) ======================
def train_final_model(dataset, device, gru_params, best_mlp_params,
                      epochs, batch_size, num_workers, save_dir):
    """Train final model on full training set using fixed GRU and best MLP hyperparameters, then save."""
    print("\n===== Training final model on full training set with best hyperparameters =====")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Initialize models
    gru_temp = GRUCellFeatureExtractor(8, gru_params['temp_hidden'], gru_params['temp_layers']).to(device)
    gru_turb = GRUCellFeatureExtractor(8, gru_params['turb_hidden'], gru_params['turb_layers']).to(device)
    gru_meth = GRUCellFeatureExtractor(6, gru_params['meth_hidden'], gru_params['meth_layers']).to(device)
    gru_orp  = GRUCellFeatureExtractor(6, gru_params['orp_hidden'], gru_params['orp_layers']).to(device)

    mlp_input_size = (gru_params['temp_hidden'] + gru_params['turb_hidden'] +
                      gru_params['meth_hidden'] + gru_params['orp_hidden'])
    mlp = MLPClassifier(input_size=mlp_input_size,
                        hidden_size=best_mlp_params['mlp_hidden'],
                        num_layers=best_mlp_params['mlp_layers'],
                        output_size=5).to(device)

    all_params = (list(gru_temp.parameters()) + list(gru_turb.parameters()) +
                  list(gru_meth.parameters()) + list(gru_orp.parameters()) +
                  list(mlp.parameters()))
    optimizer = optim.Adam(all_params, lr=best_mlp_params['lr'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        gru_temp.train()
        gru_turb.train()
        gru_meth.train()
        gru_orp.train()
        mlp.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f'Final Training Epoch {epoch+1:03d}')
        for x_list, y_plume in pbar:
            x_temp = x_list[0].to(device)
            x_turb = x_list[1].to(device)
            x_meth = x_list[2].to(device)
            x_orp = x_list[3].to(device)
            y = y_plume.to(device)

            optimizer.zero_grad()
            feat_temp, _ = gru_temp(x_temp)
            feat_turb, _ = gru_turb(x_turb)
            feat_meth, _ = gru_meth(x_meth)
            feat_orp, _ = gru_orp(x_orp)
            combined = torch.cat([feat_temp, feat_turb, feat_meth, feat_orp], dim=1)
            fluid_probs = mlp(combined)
            loss = criterion(fluid_probs, torch.argmax(y, dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        print(f"Epoch {epoch+1:03d} average loss: {total_loss/len(loader):.4f}")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'gru_temp': gru_temp.state_dict(),
        'gru_turb': gru_turb.state_dict(),
        'gru_meth': gru_meth.state_dict(),
        'gru_orp': gru_orp.state_dict(),
        'mlp': mlp.state_dict(),
    }, os.path.join(save_dir, "best_ablation_B.pth"))

    # Save configuration
    config = {
        'window_size': dataset.window_size,
        'gru_params': gru_params,
        'mlp_params': best_mlp_params,
        'batch_size': batch_size,
        'epochs': epochs
    }
    with open(os.path.join(save_dir, "ablation_B_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Final model and configuration saved to {save_dir}")
    return config


# ====================== Main Program ======================
if __name__ == "__main__":
    # ========== User Configuration ==========
    DATA_DIR = r"./train_data"               # Training data directory
    OUTPUT_MODEL_DIR = r"./Ablation_model"
    CACHE_DIR = "./cache"

    WINDOW_SIZE = 10                          # Sliding window size
    BATCH_SIZE = 512
    NUM_WORKERS = 0
    EPOCHS_PER_FOLD = 20                      # Number of epochs per fold during CV (reduce to speed up)
    FINAL_EPOCHS = 50                         # Number of epochs for final training

    # ---------- Fixed GRU hyperparameters (set based on previous experiments or heuristics) ----------
    GRU_PARAMS = {
        'temp_hidden': 48,
        'temp_layers': 3,
        'turb_hidden': 48,
        'turb_layers': 2,
        'meth_hidden': 32,
        'meth_layers': 3,
        'orp_hidden': 48,
        'orp_layers': 3
    }

    # ---------- MLP hyperparameter search space ----------
    MLP_HIDDEN_LIST = [16, 32, 48, 64]
    MLP_LAYERS_LIST = [1, 2, 3, 4]
    LR_LIST = [0.005, 0.001, 0.0005]

    SEED = 42
    # =====================================

    # Set random seeds for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Load full training dataset
    print("\n===== Loading dataset =====")
    dataset = HydrothermalDataset(DATA_DIR, WINDOW_SIZE, 'train', CACHE_DIR)
    print(f"Dataset sample count: {len(dataset)}")

    # Grid search (MLP hyperparameters only)
    best_mlp_params, best_cv_acc = grid_search_mlp_only(
        dataset, DEVICE,
        GRU_PARAMS,
        MLP_HIDDEN_LIST, MLP_LAYERS_LIST, LR_LIST,
        EPOCHS_PER_FOLD, BATCH_SIZE, NUM_WORKERS,
        n_splits=5
    )

    # Final training and save model
    train_final_model(
        dataset, DEVICE, GRU_PARAMS, best_mlp_params,
        FINAL_EPOCHS, BATCH_SIZE, NUM_WORKERS, OUTPUT_MODEL_DIR
    )

    print("\n===== All done =====")