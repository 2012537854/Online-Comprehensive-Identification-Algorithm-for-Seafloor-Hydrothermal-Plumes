# Online-Comprehensive-Identification-Algorithm-for-Seafloor-Hydrothermal-Plumes
Online identification of seafloor hydrothermal plumes from AUV sensor data. Reads real‑time temperature, turbidity, methane, and ORP. Sliding‑window statistical features feed four GRU anomaly detectors. An MLP then fuses anomaly probabilities to classify plume fluid types instantly. Lightweight for onboard deployment.

## 📁 train
### `train_model_cross_validation.py`
Full training pipeline with **5‑fold cross‑validation** and a **two‑stage optimization** strategy:
1. **Stage 1** – Fixes the sliding window size and performs grid search over GRU and MLP hyperparameters.
2. **Stage 2** – Fixes the best hyperparameters from Stage 1 and searches for the optimal window size.
3. **Final training** – Trains the combined model on the entire dataset using the best configuration.
All features are cached to `.npz` files to speed up repeated runs. GPU acceleration is automatically enabled when available.

### `train_ablation.py`
Ablation study (GRU + MLP end‑to‑end training **without** anomaly supervision).
- GRU modules act purely as feature extractors, and their hyperparameters are **fixed**.
- Only the **MLP hyperparameters** (hidden size, number of layers, learning rate) are tuned via 5‑fold cross‑validation.
- Useful for comparing the effect of the anomaly detection branch against the full model.

- ## 📁 Verification
- ### `Verification.ipynb`
Comprehensive evaluation script for a trained model.
- Loads the saved `best_combined.pth` and its configuration.
- Runs inference on validation data and computes:
  - **Anomaly detection metrics** (accuracy, precision, recall, F1‑score) for each sensor.
  - **Fluid type classification** overall accuracy and confusion matrix.
- Generates publication‑quality plots of predicted anomaly probabilities and the confusion matrix.
- Saves all predictions to an Excel file for further analysis.
- Optionally reports **model complexity** (parameter count, size, inference latency).
