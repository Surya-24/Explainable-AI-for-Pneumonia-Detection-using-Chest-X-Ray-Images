# Explainable AI for Pneumonia Detection using Chest X‑Ray Images

This project implements an **end‑to‑end deep learning and explainable AI (XAI) pipeline** for automatic **pneumonia detection from chest X‑ray images**. The pipeline covers **data loading, training, validation, evaluation, and model interpretability** using **Grad‑CAM** and **LIME**.

The solution is built with **PyTorch**, uses a **DenseNet‑121** backbone.
---

## Key Features

-  DenseNet‑121 with ImageNet pretraining
-  Stratified train/validation split
-  Class‑imbalance handling with weighted loss
-  Advanced data augmentation
-  Learning‑rate scheduling & early stopping
-  Comprehensive evaluation (accuracy, report, confusion matrix)
-  Explainable AI using **Grad‑CAM** and **LIME**
-  Visualization of **correct and incorrect predictions**

---

## 📂 Dataset

**Dataset:** Chest X‑Ray Images (Pneumonia)

- Source: Kaggle (`paultimothymooney/chest-xray-pneumonia`)
- Classes:
  - `NORMAL`
  - `PNEUMONIA`

The dataset is automatically downloaded using **kagglehub**.

Expected structure:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
```

---

## ⚙️ Environment & Dependencies

### Required Libraries

- Python ≥ 3.8
- PyTorch
- torchvision
- captum
- lime
- scikit‑image
- scikit‑learn
- matplotlib
- seaborn
- kagglehub

### Installation (Cell 1)
```bash
pip install captum kagglehub lime scikit-image
```
⚠️ **Important:** Restart the runtime after installing dependencies.

---

## 🧠 Model Architecture

- **Backbone:** DenseNet‑121 (ImageNet pretrained)
- **Custom Classifier Head:**
  - Fully connected layers
  - Batch normalization
  - ReLU activations
  - Dropout regularization

```text
DenseNet121 → FC(512) → BN → ReLU → Dropout
             → FC(256) → BN → ReLU → Dropout
             → FC(2)
```

---

## 🏋️ Training Strategy

- Loss: **CrossEntropyLoss** with class weights
- Optimizer: **Adam**
- Learning Rate: `1e‑4`
- Weight Decay: `1e‑4`
- Batch Size: `32`
- Epochs: `15` (with early stopping)
- Scheduler: **ReduceLROnPlateau**

### Data Augmentation

- Random rotation
- Horizontal flipping
- Affine translation
- Brightness & contrast jitter

---

## 📊 Evaluation Metrics

- Overall accuracy
- Precision, recall, F1‑score
- Confusion matrix visualization

All evaluation plots are saved in the `results/` directory.

---

## 🔍 Explainable AI (XAI)

### 1️ Grad‑CAM (Captum)

- Visualizes class‑specific activation maps
- Highlights **regions influencing predictions**
- Applied on DenseNet’s final convolutional block

### 2️ LIME (Local Interpretable Model‑agnostic Explanations)

- Superpixel‑based local explanations
- Shows **positive contributing regions**
- Model‑agnostic and sample‑specific

---

## 🧪 Diverse Sample Visualization

The pipeline automatically selects and visualizes:

- ✅ NORMAL – Correct prediction
- ❌ NORMAL – Incorrect prediction
- ✅ PNEUMONIA – Correct prediction
- ❌ PNEUMONIA – Incorrect prediction

Each sample includes:
- Original image
- Grad‑CAM heatmap & overlay
- LIME superpixel explanation

---

## 📁 Output Files

All outputs are saved under the `results/` directory:

```
results/
├── training_history.png
├── confusion_matrix.png
├── gradcam_*.png
├── lime_*.png
```

The trained model is saved as:
```
densenet121_pneumonia_detector.pth
```

---

## ▶️ How to Run

1. Run **Cell 1** to install dependencies
2. Restart runtime
3. Run **Cells 2–13 sequentially**
4. The full pipeline executes automatically

Main execution entry point:
```python
model, gradcam_explainer, lime_explainer, test_loader = run_pipeline()
```

---

## Highlights

- Robust handling of class imbalance
- Strong generalization via augmentation & regularization
- Transparent model decisions using XAI
- Ideal for **medical imaging research & academic projects**

---



## 🙌 Acknowledgements

- Kaggle Chest X‑Ray Pneumonia Dataset
- PyTorch & TorchVision
- Captum & LIME libraries

---

**Author:** Sai Surya Mada

