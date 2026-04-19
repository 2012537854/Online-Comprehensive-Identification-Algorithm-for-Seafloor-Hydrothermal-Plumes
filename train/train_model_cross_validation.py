################################################################################
#                                                                              #
#                    Hydrothermal Plume Anomaly Detection                      #
#                         Cross-Validation Version                             #
#                                                                              #
# Revised: 2026.04.19                                                          #
#   - Use training set only with 5-fold cross-validation                       #
#   - Stage 1: Fixed window size, grid search for best GRU and MLP             #
#              hyperparameters                                                 #
#   - Stage 2: Fixed best hyperparameters, search for best sliding window size #
#   - Final training on full training set with best configuration and save     #
#     the model                                                                #
#   - Add feature caching mechanism to accelerate repeated experiments         #
#   - Support GPU acceleration (auto-detect CUDA device)                       #
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
class GRUCellAnomalyDetector(nn.Module):
    """GRU-based anomaly detector for a single sensor."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.gru_cells.append(nn.GRUCell(in_size, hidden_size))
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        new_h = []
        for i, cell in enumerate(self.gru_cells):
            h_i = h[i]
            out = cell(x if i == 0 else new_h[-1], h_i)
            new_h.append(out)
        last_out = new_h[-1]
        out = self.fc(last_out)
        return self.sigmoid(out), torch.stack(new_h)


class MLPClassifier(nn.Module):
    """MLP for plume type classification."""
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=5):
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


class CombinedModel(nn.Module):
    """Combines four GRU detectors and an MLP classifier."""
    def __init__(self, gru_models, mlp_model):
        super().__init__()
        self.temp_detector = gru_models[0]
        self.turb_detector = gru_models[1]
        self.meth_detector = gru_models[2]
        self.orp_detector  = gru_models[3]
        self.mlp = mlp_model

    def forward(self, x_list, prev_hiddens=None):
        if prev_hiddens is None:
            prev_hiddens = [None, None, None, None]
        temp_prob, h_temp = self.temp_detector(x_list[0], prev_hiddens[0])
        turb_prob, h_turb = self.turb_detector(x_list[1], prev_hiddens[1])
        meth_prob, h_meth = self.meth_detector(x_list[2], prev_hiddens[2])
        orp_prob, h_orp   = self.orp_detector(x_list[3], prev_hiddens[3])
        anomaly_probs = [temp_prob, turb_prob, meth_prob, orp_prob]
        concat_probs = torch.cat(anomaly_probs, dim=1)
        fluid_probs = self.mlp(concat_probs)
        return anomaly_probs, fluid_probs, [h_temp, h_turb, h_meth, h_orp]


# ====================== Feature Extraction ======================
def extract_features_temp_turb(seq, window_size):
    """Extract 8 statistical features for temperature and turbidity."""
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
    """Extract 6 statistical features for methane and ORP."""
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


# ====================== Dataset ======================
class HydrothermalDataset(Dataset):
    """Dataset that loads raw data, extracts sliding window features, and caches them."""
    def __init__(self, data_dir, window_size, cache_key, cache_dir):
        self.window_size = window_size
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}_features_w{window_size}.npz")
        if os.path.exists(cache_file):
            data = np.load(cache_file)
            self.features = [data['feat_temp'], data['feat_turb'], data['feat_meth'], data['feat_orp']]
            self.anomaly_labels_aligned = data['anomaly_labels']
            self.plume_labels_aligned = data['plume_labels']
            return

        print(f"Processing {cache_key} data (window size={window_size})...")
        feat_path = os.path.join(data_dir, 'real_time_data.xlsx')
        anom_path = os.path.join(data_dir, 'anomaly_labels.xlsx')
        plume_path = os.path.join(data_dir, 'plume_type_labels.xlsx')
        df_feat = pd.read_excel(feat_path)
        df_anom = pd.read_excel(anom_path)
        df_plume = pd.read_excel(plume_path)

        if df_feat.shape[1] >= 5:
            data = df_feat.iloc[:, 1:5].values.astype(np.float32)
        else:
            data = df_feat.iloc[:, :4].values.astype(np.float32)

        anomaly_labels = df_anom.values.astype(np.float32)
        plume_labels = df_plume.values.astype(np.float32)

        feat_temp = extract_features_temp_turb(data[:, 0], window_size)
        feat_turb = extract_features_temp_turb(data[:, 1], window_size)
        feat_meth = extract_features_meth_orp(data[:, 2], window_size)
        feat_orp = extract_features_meth_orp(data[:, 3], window_size)

        self.features = [feat_temp, feat_turb, feat_meth, feat_orp]
        T = self.features[0].shape[0]
        self.anomaly_labels_aligned = anomaly_labels[window_size-1:, :]
        self.plume_labels_aligned = plume_labels[window_size-1:, :]

        np.savez(cache_file,
                 feat_temp=feat_temp, feat_turb=feat_turb, feat_meth=feat_meth, feat_orp=feat_orp,
                 anomaly_labels=self.anomaly_labels_aligned,
                 plume_labels=self.plume_labels_aligned)

    def __len__(self):
        return self.features[0].shape[0]

    def __getitem__(self, idx):
        x_list = [torch.from_numpy(self.features[i][idx]) for i in range(4)]
        y_anom = torch.from_numpy(self.anomaly_labels_aligned[idx])
        y_plume = torch.from_numpy(self.plume_labels_aligned[idx])
        return x_list, y_anom, y_plume


# ====================== Evaluation Functions ======================
def evaluate_gru(model, loader, device, sensor_idx):
    """Evaluate GRU anomaly detection accuracy."""
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x_list, y_anom, _ in loader:
            x = x_list[sensor_idx].to(device)
            y = y_anom[:, sensor_idx].to(device).unsqueeze(1)
            prob, _ = model(x)
            pred = (prob > 0.5).float()
            all_preds.extend(pred.cpu().numpy().flatten())
            all_true.extend(y.cpu().numpy().flatten())
    return accuracy_score(all_true, all_preds)


def evaluate_mlp(model, loader, device):
    """Evaluate MLP plume type classification accuracy."""
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x_list, _, y_plume in loader:
            anom_probs = []
            for i, detector in enumerate([model.temp_detector, model.turb_detector,
                                          model.meth_detector, model.orp_detector]):
                x = x_list[i].to(device)
                prob, _ = detector(x)
                anom_probs.append(prob)
            concat_probs = torch.cat(anom_probs, dim=1)
            fluid_probs = model.mlp(concat_probs)
            pred = torch.argmax(fluid_probs, dim=1).cpu().numpy()
            true = torch.argmax(y_plume, dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_true.extend(true)
    return accuracy_score(all_true, all_preds)


# ====================== Training Functions (return best validation accuracy) ======================
def train_gru(model, train_loader, val_loader, optimizer, device, epochs, sensor_name):
    """Train a GRU detector for a fixed number of epochs, tracking best validation accuracy."""
    best_acc = 0.0
    model.to(device)
    sensor_idx = {'Temp':0, 'Turb':1, 'Meth':2, 'ORP':3}[sensor_name]
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        for x_list, y_anom, _ in train_loader:
            x = x_list[sensor_idx].to(device)
            y = y_anom[:, sensor_idx].to(device).unsqueeze(1)
            optimizer.zero_grad()
            prob, _ = model(x)
            loss = criterion(prob, y)
            loss.backward()
            optimizer.step()
        val_acc = evaluate_gru(model, val_loader, device, sensor_idx)
        if val_acc > best_acc:
            best_acc = val_acc
    return best_acc


def train_mlp(combined, train_loader, val_loader, optimizer, device, epochs):
    """Train the MLP part (GRUs frozen), returning best validation accuracy."""
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        combined.train()
        for x_list, _, y_plume in train_loader:
            with torch.no_grad():
                anom_probs = []
                for i, detector in enumerate([combined.temp_detector, combined.turb_detector,
                                              combined.meth_detector, combined.orp_detector]):
                    x = x_list[i].to(device)
                    prob, _ = detector(x)
                    anom_probs.append(prob)
                concat_probs = torch.cat(anom_probs, dim=1)
            y = y_plume.to(device)
            optimizer.zero_grad()
            fluid_probs = combined.mlp(concat_probs)
            loss = criterion(fluid_probs, torch.argmax(y, dim=1))
            loss.backward()
            optimizer.step()
        val_acc = evaluate_mlp(combined, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
    return best_acc


# ====================== Cross-Validation Helpers ======================
def cross_val_gru(sensor_name, input_size, dataset, hidden, layers, lr, epochs,
                  batch_size, num_workers, device, n_splits=5):
    """Perform n_splits-fold cross-validation for a GRU hyperparameter set. Returns mean and std accuracy."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list = []
    sensor_idx = {'Temp':0, 'Turb':1, 'Meth':2, 'ORP':3}[sensor_name]

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = GRUCellAnomalyDetector(input_size, hidden, layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        val_acc = train_gru(model, train_loader, val_loader, optimizer, device, epochs, sensor_name)
        acc_list.append(val_acc)

    return np.mean(acc_list), np.std(acc_list)


def cross_val_mlp(dataset, mlp_hidden, mlp_layers, mlp_lr, epochs,
                  batch_size, num_workers, device,
                  gru_params_dict, n_splits=5):
    """
    Cross-validation for MLP hyperparameters.
    In each fold, GRUs are trained from scratch using the provided best GRU hyperparameters,
    then frozen while training the MLP.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        # Build subset loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Train GRUs
        temp_model = GRUCellAnomalyDetector(
            gru_params_dict['temp_input'], gru_params_dict['temp_hidden'], gru_params_dict['temp_layers']).to(device)
        opt_temp = optim.Adam(temp_model.parameters(), lr=gru_params_dict['temp_lr'])
        train_gru(temp_model, train_loader, val_loader, opt_temp, device, epochs, 'Temp')

        turb_model = GRUCellAnomalyDetector(
            gru_params_dict['turb_input'], gru_params_dict['turb_hidden'], gru_params_dict['turb_layers']).to(device)
        opt_turb = optim.Adam(turb_model.parameters(), lr=gru_params_dict['turb_lr'])
        train_gru(turb_model, train_loader, val_loader, opt_turb, device, epochs, 'Turb')

        meth_model = GRUCellAnomalyDetector(
            gru_params_dict['meth_input'], gru_params_dict['meth_hidden'], gru_params_dict['meth_layers']).to(device)
        opt_meth = optim.Adam(meth_model.parameters(), lr=gru_params_dict['meth_lr'])
        train_gru(meth_model, train_loader, val_loader, opt_meth, device, epochs, 'Meth')

        orp_model = GRUCellAnomalyDetector(
            gru_params_dict['orp_input'], gru_params_dict['orp_hidden'], gru_params_dict['orp_layers']).to(device)
        opt_orp = optim.Adam(orp_model.parameters(), lr=gru_params_dict['orp_lr'])
        train_gru(orp_model, train_loader, val_loader, opt_orp, device, epochs, 'ORP')

        # Freeze GRU parameters
        for m in [temp_model, turb_model, meth_model, orp_model]:
            for p in m.parameters():
                p.requires_grad = False

        # Build and train MLP
        mlp = MLPClassifier(4, mlp_hidden, mlp_layers, 5).to(device)
        combined = CombinedModel([temp_model, turb_model, meth_model, orp_model], mlp).to(device)
        opt_mlp = optim.Adam(combined.mlp.parameters(), lr=mlp_lr)
        mlp_acc = train_mlp(combined, train_loader, val_loader, opt_mlp, device, epochs)
        acc_list.append(mlp_acc)

    return np.mean(acc_list), np.std(acc_list)


# ====================== Dataset Loading ======================
def load_full_dataset(data_dir, cache_dir, window_size):
    """Load the full dataset (no train/val split)."""
    dataset = HydrothermalDataset(data_dir, window_size, 'train', cache_dir)
    return dataset


# ====================== Stage 1: Fixed window, hyperparameter search via CV ======================
def stage1_search(data_dir, grid_result_dir, window_size,
                  temp_hidden_list, temp_layers_list, temp_lr_list,
                  turb_hidden_list, turb_layers_list, turb_lr_list,
                  meth_hidden_list, meth_layers_list, meth_lr_list,
                  orp_hidden_list, orp_layers_list, orp_lr_list,
                  mlp_hidden_list, mlp_layers_list, mlp_lr_list,
                  batch_size, num_workers, gru_epochs, mlp_epochs, device):
    print(f"\n========== Stage 1: Fixed window size = {window_size}, 5-fold CV hyperparameter search ==========")
    dataset = load_full_dataset(data_dir, './cache', window_size)
    os.makedirs(grid_result_dir, exist_ok=True)

    # Helper for GRU search
    def search_gru_for_sensor(sensor_name, input_size, hidden_list, layers_list, lr_list):
        best_mean = 0.0
        best_params = None
        results = []
        total = len(hidden_list) * len(layers_list) * len(lr_list)
        print(f"\nGrid search {sensor_name} GRU ({total} combinations)")
        for hidden in hidden_list:
            for layers in layers_list:
                for lr in lr_list:
                    mean_acc, std_acc = cross_val_gru(
                        sensor_name, input_size, dataset, hidden, layers, lr,
                        gru_epochs, batch_size, num_workers, device
                    )
                    print(f"  {sensor_name}: hidden={hidden}, layers={layers}, lr={lr} -> CV acc = {mean_acc:.4f} ± {std_acc:.4f}")
                    results.append({'hidden': hidden, 'layers': layers, 'lr': lr,
                                    'cv_mean': mean_acc, 'cv_std': std_acc})
                    if mean_acc > best_mean:
                        best_mean = mean_acc
                        best_params = (hidden, layers, lr)
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(grid_result_dir, f"cv_search_{sensor_name.lower()}.csv"), index=False)
        print(f"Best {sensor_name} params: hidden={best_params[0]}, layers={best_params[1]}, lr={best_params[2]}, CV acc={best_mean:.4f}")
        return best_params, best_mean

    temp_params, _ = search_gru_for_sensor('Temp', 8, temp_hidden_list, temp_layers_list, temp_lr_list)
    turb_params, _ = search_gru_for_sensor('Turb', 8, turb_hidden_list, turb_layers_list, turb_lr_list)
    meth_params, _ = search_gru_for_sensor('Meth', 6, meth_hidden_list, meth_layers_list, meth_lr_list)
    orp_params, _  = search_gru_for_sensor('ORP', 6, orp_hidden_list, orp_layers_list, orp_lr_list)

    # Store best GRU hyperparameters
    best_hyper_params = {
        'temp_input': 8, 'temp_hidden': temp_params[0], 'temp_layers': temp_params[1], 'temp_lr': temp_params[2],
        'turb_input': 8, 'turb_hidden': turb_params[0], 'turb_layers': turb_params[1], 'turb_lr': turb_params[2],
        'meth_input': 6, 'meth_hidden': meth_params[0], 'meth_layers': meth_params[1], 'meth_lr': meth_params[2],
        'orp_input': 6, 'orp_hidden': orp_params[0], 'orp_layers': orp_params[1], 'orp_lr': orp_params[2]
    }

    # MLP hyperparameter search
    print(f"\nGrid search MLP ({len(mlp_hidden_list) * len(mlp_layers_list) * len(mlp_lr_list)} combinations)")
    best_mlp_mean = 0.0
    best_mlp_params = None
    mlp_results = []
    for hidden in mlp_hidden_list:
        for layers in mlp_layers_list:
            for lr in mlp_lr_list:
                mean_acc, std_acc = cross_val_mlp(
                    dataset, hidden, layers, lr, mlp_epochs,
                    batch_size, num_workers, device,
                    best_hyper_params
                )
                print(f"  MLP: hidden={hidden}, layers={layers}, lr={lr} -> CV acc = {mean_acc:.4f} ± {std_acc:.4f}")
                mlp_results.append({'hidden': hidden, 'layers': layers, 'lr': lr,
                                    'cv_mean': mean_acc, 'cv_std': std_acc})
                if mean_acc > best_mlp_mean:
                    best_mlp_mean = mean_acc
                    best_mlp_params = (hidden, layers, lr)
    df_mlp = pd.DataFrame(mlp_results)
    df_mlp.to_csv(os.path.join(grid_result_dir, "cv_search_mlp.csv"), index=False)
    print(f"Best MLP params: hidden={best_mlp_params[0]}, layers={best_mlp_params[1]}, lr={best_mlp_params[2]}, CV acc={best_mlp_mean:.4f}")

    best_hyper_params.update({
        'mlp_hidden': best_mlp_params[0], 'mlp_layers': best_mlp_params[1], 'mlp_lr': best_mlp_params[2]
    })
    best_hyper_params['window_size'] = window_size
    best_hyper_params['mlp_cv_accuracy'] = best_mlp_mean
    return best_hyper_params


# ====================== Stage 2: Fixed hyperparameters, window size search via CV ======================
def stage2_search(data_dir, grid_result_dir, best_hyper_params,
                  window_size_list, batch_size, num_workers, gru_epochs, mlp_epochs, device):
    print(f"\n========== Stage 2: Fixed best hyperparameters, 5-fold CV window size search ==========")
    os.makedirs(grid_result_dir, exist_ok=True)
    results = []
    for ws in window_size_list:
        print(f"\nTrying window size: {ws}")
        dataset = load_full_dataset(data_dir, './cache', ws)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            # Train GRUs with best hyperparameters
            temp_h, temp_l, temp_lr = best_hyper_params['temp_hidden'], best_hyper_params['temp_layers'], best_hyper_params['temp_lr']
            temp_model = GRUCellAnomalyDetector(8, temp_h, temp_l).to(device)
            opt_temp = optim.Adam(temp_model.parameters(), lr=temp_lr)
            train_gru(temp_model, train_loader, val_loader, opt_temp, device, gru_epochs, 'Temp')

            turb_h, turb_l, turb_lr = best_hyper_params['turb_hidden'], best_hyper_params['turb_layers'], best_hyper_params['turb_lr']
            turb_model = GRUCellAnomalyDetector(8, turb_h, turb_l).to(device)
            opt_turb = optim.Adam(turb_model.parameters(), lr=turb_lr)
            train_gru(turb_model, train_loader, val_loader, opt_turb, device, gru_epochs, 'Turb')

            meth_h, meth_l, meth_lr = best_hyper_params['meth_hidden'], best_hyper_params['meth_layers'], best_hyper_params['meth_lr']
            meth_model = GRUCellAnomalyDetector(6, meth_h, meth_l).to(device)
            opt_meth = optim.Adam(meth_model.parameters(), lr=meth_lr)
            train_gru(meth_model, train_loader, val_loader, opt_meth, device, gru_epochs, 'Meth')

            orp_h, orp_l, orp_lr = best_hyper_params['orp_hidden'], best_hyper_params['orp_layers'], best_hyper_params['orp_lr']
            orp_model = GRUCellAnomalyDetector(6, orp_h, orp_l).to(device)
            opt_orp = optim.Adam(orp_model.parameters(), lr=orp_lr)
            train_gru(orp_model, train_loader, val_loader, opt_orp, device, gru_epochs, 'ORP')

            for m in [temp_model, turb_model, meth_model, orp_model]:
                for p in m.parameters():
                    p.requires_grad = False

            # Train MLP
            mlp_h, mlp_l, mlp_lr = best_hyper_params['mlp_hidden'], best_hyper_params['mlp_layers'], best_hyper_params['mlp_lr']
            mlp = MLPClassifier(4, mlp_h, mlp_l, 5).to(device)
            combined = CombinedModel([temp_model, turb_model, meth_model, orp_model], mlp).to(device)
            opt_mlp = optim.Adam(combined.mlp.parameters(), lr=mlp_lr)
            mlp_acc = train_mlp(combined, train_loader, val_loader, opt_mlp, device, mlp_epochs)
            fold_accs.append(mlp_acc)

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        print(f"Window size {ws}: CV MLP acc = {mean_acc:.4f} ± {std_acc:.4f}")
        results.append({'window_size': ws, 'cv_mean': mean_acc, 'cv_std': std_acc})

    best_result = max(results, key=lambda x: x['cv_mean'])
    print(f"\nBest window size: {best_result['window_size']}, CV acc = {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(grid_result_dir, "cv_window_search.csv"), index=False)
    return best_result['window_size']


# ====================== Final Training on Full Dataset ======================
def final_train_and_save(data_dir, model_save_dir, best_window, best_hyper_params,
                         batch_size, num_workers, gru_epochs, mlp_epochs, device):
    print(f"\n========== Final training on full dataset with best window {best_window} and hyperparameters ==========")
    dataset = load_full_dataset(data_dir, './cache', best_window)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def train_gru_full(model, train_loader, optimizer, device, epochs, sensor_name):
        """Train GRU on full dataset without validation monitoring."""
        model.to(device)
        sensor_idx = {'Temp':0, 'Turb':1, 'Meth':2, 'ORP':3}[sensor_name]
        criterion = nn.BCELoss()
        for epoch in range(epochs):
            model.train()
            for x_list, y_anom, _ in train_loader:
                x = x_list[sensor_idx].to(device)
                y = y_anom[:, sensor_idx].to(device).unsqueeze(1)
                optimizer.zero_grad()
                prob, _ = model(x)
                loss = criterion(prob, y)
                loss.backward()
                optimizer.step()

    def train_mlp_full(combined, train_loader, optimizer, device, epochs):
        """Train MLP on full dataset without validation monitoring."""
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            combined.train()
            for x_list, _, y_plume in train_loader:
                with torch.no_grad():
                    anom_probs = []
                    for i, detector in enumerate([combined.temp_detector, combined.turb_detector,
                                                  combined.meth_detector, combined.orp_detector]):
                        x = x_list[i].to(device)
                        prob, _ = detector(x)
                        anom_probs.append(prob)
                    concat_probs = torch.cat(anom_probs, dim=1)
                y = y_plume.to(device)
                optimizer.zero_grad()
                fluid_probs = combined.mlp(concat_probs)
                loss = criterion(fluid_probs, torch.argmax(y, dim=1))
                loss.backward()
                optimizer.step()

    # Train GRUs
    temp_h, temp_l, temp_lr = best_hyper_params['temp_hidden'], best_hyper_params['temp_layers'], best_hyper_params['temp_lr']
    final_temp = GRUCellAnomalyDetector(8, temp_h, temp_l).to(device)
    opt_temp = optim.Adam(final_temp.parameters(), lr=temp_lr)
    train_gru_full(final_temp, train_loader, opt_temp, device, gru_epochs, 'Temp')

    turb_h, turb_l, turb_lr = best_hyper_params['turb_hidden'], best_hyper_params['turb_layers'], best_hyper_params['turb_lr']
    final_turb = GRUCellAnomalyDetector(8, turb_h, turb_l).to(device)
    opt_turb = optim.Adam(final_turb.parameters(), lr=turb_lr)
    train_gru_full(final_turb, train_loader, opt_turb, device, gru_epochs, 'Turb')

    meth_h, meth_l, meth_lr = best_hyper_params['meth_hidden'], best_hyper_params['meth_layers'], best_hyper_params['meth_lr']
    final_meth = GRUCellAnomalyDetector(6, meth_h, meth_l).to(device)
    opt_meth = optim.Adam(final_meth.parameters(), lr=meth_lr)
    train_gru_full(final_meth, train_loader, opt_meth, device, gru_epochs, 'Meth')

    orp_h, orp_l, orp_lr = best_hyper_params['orp_hidden'], best_hyper_params['orp_layers'], best_hyper_params['orp_lr']
    final_orp = GRUCellAnomalyDetector(6, orp_h, orp_l).to(device)
    opt_orp = optim.Adam(final_orp.parameters(), lr=orp_lr)
    train_gru_full(final_orp, train_loader, opt_orp, device, gru_epochs, 'ORP')

    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(final_temp.state_dict(), os.path.join(model_save_dir, "best_gru_temp.pth"))
    torch.save(final_turb.state_dict(), os.path.join(model_save_dir, "best_gru_turb.pth"))
    torch.save(final_meth.state_dict(), os.path.join(model_save_dir, "best_gru_meth.pth"))
    torch.save(final_orp.state_dict(), os.path.join(model_save_dir, "best_gru_orp.pth"))

    # Load GRUs and freeze for MLP training
    gru_temp = GRUCellAnomalyDetector(8, temp_h, temp_l).to(device)
    gru_temp.load_state_dict(torch.load(os.path.join(model_save_dir, "best_gru_temp.pth"), map_location=device, weights_only=True))
    gru_turb = GRUCellAnomalyDetector(8, turb_h, turb_l).to(device)
    gru_turb.load_state_dict(torch.load(os.path.join(model_save_dir, "best_gru_turb.pth"), map_location=device, weights_only=True))
    gru_meth = GRUCellAnomalyDetector(6, meth_h, meth_l).to(device)
    gru_meth.load_state_dict(torch.load(os.path.join(model_save_dir, "best_gru_meth.pth"), map_location=device, weights_only=True))
    gru_orp = GRUCellAnomalyDetector(6, orp_h, orp_l).to(device)
    gru_orp.load_state_dict(torch.load(os.path.join(model_save_dir, "best_gru_orp.pth"), map_location=device, weights_only=True))
    for m in [gru_temp, gru_turb, gru_meth, gru_orp]:
        for p in m.parameters():
            p.requires_grad = False

    mlp_h, mlp_l, mlp_lr = best_hyper_params['mlp_hidden'], best_hyper_params['mlp_layers'], best_hyper_params['mlp_lr']
    mlp = MLPClassifier(4, mlp_h, mlp_l, 5).to(device)
    combined = CombinedModel([gru_temp, gru_turb, gru_meth, gru_orp], mlp).to(device)
    opt_mlp = optim.Adam(combined.mlp.parameters(), lr=mlp_lr)
    train_mlp_full(combined, train_loader, opt_mlp, device, mlp_epochs)
    torch.save(combined.state_dict(), os.path.join(model_save_dir, "best_combined.pth"))

    # Save configuration
    config = {
        "window_size": best_window,
        "batch_size": batch_size,
        "temp_turb_input": 8,
        "meth_orp_input": 6,
        "temp_hidden": temp_h, "temp_layers": temp_l,
        "turb_hidden": turb_h, "turb_layers": turb_l,
        "meth_hidden": meth_h, "meth_layers": meth_l,
        "orp_hidden": orp_h, "orp_layers": orp_l,
        "mlp_hidden_size": mlp_h,
        "mlp_num_layers": mlp_l
    }
    with open(os.path.join(model_save_dir, "model_config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\nFinal models and configuration saved to {model_save_dir}")


# ====================== Main Program (5-fold CV version) ======================
if __name__ == "__main__":
    # ---------- User-configurable parameters ----------
    DATA_DIR = "./train_data_10000"                     # Directory containing training data (Excel files)
    GRID_RESULT_DIR = "./grid_search_result"            # Directory to save grid search results
    MODEL_SAVE_DIR = "./2026.4.13.Best_Identification_Model"  # Directory for final best model

    # Data preprocessing
    FIXED_WINDOW = 10                                   # Window size used in stage 1
    WINDOW_SIZE_LIST = [5,10,15,20,30,40,50,60]                             # List of window sizes to try in stage 2
    BATCH_SIZE = 512
    NUM_WORKERS = 0

    # GRU hyperparameter search spaces
    TEMP_HIDDEN_LIST = [16, 32, 48, 64]
    TEMP_LAYERS_LIST = [1, 2, 3, 4]
    TEMP_LR_LIST = [0.005, 0.001, 0.0005]

    TURB_HIDDEN_LIST = [16, 32, 48, 64]
    TURB_LAYERS_LIST = [1, 2, 3, 4]
    TURB_LR_LIST = [0.005, 0.001, 0.0005]

    METH_HIDDEN_LIST = [16, 32, 48, 64]
    METH_LAYERS_LIST = [1, 2, 3, 4]
    METH_LR_LIST = [0.005, 0.001, 0.0005]

    ORP_HIDDEN_LIST = [16, 32, 48, 64]
    ORP_LAYERS_LIST = [1, 2, 3, 4]
    ORP_LR_LIST = [0.005, 0.001, 0.0005]

    # MLP hyperparameter search space
    MLP_HIDDEN_LIST = [16, 32, 48, 64]
    MLP_LAYERS_LIST = [1, 2, 3, 4]
    MLP_LR_LIST = [0.005, 0.001, 0.0005]

    # Number of epochs per hyperparameter combination (reduce for faster search)
    GRU_EPOCHS = 20
    MLP_EPOCHS = 40

    SEED = 42
    # ------------------------------------------------

    # Set random seeds for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Stage 1: Fixed window, hyperparameter search with 5-fold CV
    best_hyper_params = stage1_search(
        DATA_DIR, GRID_RESULT_DIR, FIXED_WINDOW,
        TEMP_HIDDEN_LIST, TEMP_LAYERS_LIST, TEMP_LR_LIST,
        TURB_HIDDEN_LIST, TURB_LAYERS_LIST, TURB_LR_LIST,
        METH_HIDDEN_LIST, METH_LAYERS_LIST, METH_LR_LIST,
        ORP_HIDDEN_LIST, ORP_LAYERS_LIST, ORP_LR_LIST,
        MLP_HIDDEN_LIST, MLP_LAYERS_LIST, MLP_LR_LIST,
        BATCH_SIZE, NUM_WORKERS, GRU_EPOCHS, MLP_EPOCHS, DEVICE)

    # Stage 2: Fixed hyperparameters, window size search with 5-fold CV
    best_window = stage2_search(
        DATA_DIR, GRID_RESULT_DIR, best_hyper_params,
        WINDOW_SIZE_LIST, BATCH_SIZE, NUM_WORKERS, GRU_EPOCHS, MLP_EPOCHS, DEVICE)

    # Final training on full dataset and save model
    final_train_and_save(DATA_DIR, MODEL_SAVE_DIR, best_window, best_hyper_params,
                         BATCH_SIZE, NUM_WORKERS, GRU_EPOCHS, MLP_EPOCHS, DEVICE)