# 10-Class AI Everyday Object Recognition System

A production-grade, end-to-end Computer Vision project that classifies 10 common everyday objects using **MobileNetV2 Transfer Learning**, automated data harvesting from **Wikimedia Commons**, perceptual deduplication, stratified data splitting, and an interactive **Streamlit** web application.

---

## ⚡ Quick Start — All Commands at a Glance

> Run these in order from the project root to go from zero to a fully working system.

```bash
# ── 0. Navigate to project ──────────────────────────────────────────────────
cd /home/deepanshu/Projects/ml-clg

# ── 1. Activate virtual environment ─────────────────────────────────────────
source .venv/bin/activate

# ── 2. Install dependencies ──────────────────────────────────────────────────
pip install -r requirements.txt

# ── 3. Download dataset (50 images × 10 classes from Wikimedia Commons) ──────
python scripts/download_images.py --per-class 50

# ── 4. Train the MobileNetV2 model (Phase 1 + Phase 2 fine-tuning) ───────────
python -m src.train --epochs 15 --fine-tune-epochs 10 --batch-size 16

# ── 5. Evaluate model on test set (generates reports & confusion matrix) ──────
python -m src.evaluate

# ── 6. Predict a single image from command line ───────────────────────────────
python -m src.predict --image dataset/scissors/scissors_001.jpg

# ── 7. Launch the Streamlit web app ──────────────────────────────────────────
streamlit run app.py
```

---

## 1. Project Objective & Target Classes

The goal is to accurately recognize 10 distinct object classes with a balanced dataset of 50 images per class (500 images total):

1. **Water Bottle** (`water_bottle`)
2. **Coffee Mug** (`coffee_mug`)
3. **Smartphone** (`smartphone`)
4. **Book** (`book`)
5. **Computer Mouse** (`computer_mouse`)
6. **Keyboard** (`keyboard`)
7. **TV Remote** (`tv_remote`)
8. **Shoe** (`shoe`)
9. **Scissors** (`scissors`)
10. **Wallet** (`wallet`)

---

## 2. Project Directory Structure

```text
ml-clg/
├── dataset/                  # Class image folders and licensing metadata
│   ├── book/
│   ├── coffee_mug/
│   ├── computer_mouse/
│   ├── keyboard/
│   ├── scissors/
│   ├── shoe/
│   ├── smartphone/
│   ├── tv_remote/
│   ├── wallet/
│   ├── water_bottle/
│   └── attribution.csv       # Wikimedia author, license, and source URLs
├── models/                   # Serialized model checkpoints
│   └── mobilenetv2_best.keras
├── reports/                  # Evaluation artifacts and data splits
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── data_splits.json
│   ├── metrics.json
│   └── training_history.png
├── scripts/                  # Standalone utilities
│   └── download_images.py    # Wikimedia Commons API downloader with dHash
├── src/                      # Modular application source code
│   ├── __init__.py
│   ├── config.py             # Hyperparameters, directories, class mapping
│   ├── dataset.py            # Audit, deduplication, tf.data pipeline, splits
│   ├── evaluate.py           # Metrics, confusion matrix, and curve generator
│   ├── model.py              # MobileNetV2 transfer learning architecture
│   ├── predict.py            # Single-image / batch inference & confidence check
│   └── train.py              # Two-stage training and checkpointing loop
├── app.py                    # Streamlit web interface with webcam & dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # Complete system documentation
```

---

## 3. Setup Instructions (Arch Linux / Caelestia)

Follow these exact terminal commands to configure and run the project:

### Step 1: Navigate to the project directory
```bash
cd /home/deepanshu/Projects/ml-clg
```

### Step 2: Activate the virtual environment
```bash
source .venv/bin/activate
```
*(Or create a fresh one: `python3 -m venv .venv && source .venv/bin/activate`)*

### Step 3: Install required dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Workflow & Usage Guide

### Phase 1: Automated Image Downloading

Download up to 50 valid, unique images per category from Wikimedia Commons:

```bash
# Standard — download only what's missing to reach 50/class
python scripts/download_images.py --per-class 50

# Custom target count
python scripts/download_images.py --per-class 100

# Wipe and re-download everything from scratch
python scripts/download_images.py --per-class 50 --overwrite
```

- **Perceptual Deduplication:** Uses Difference Hashing (`dHash`) with a Hamming distance threshold to filter out near-identical photos.
- **Strict Verification:** Rejects corrupt files, images under 224×224, and abnormal aspect ratios.
- **Attribution & Licensing:** Logs authors, source URLs, and Creative Commons / Public Domain license types to `dataset/attribution.csv`.
- **Missing-Only Mode:** Skips already satisfied categories and preserves existing valid images.

---

### Phase 2: Model Training

Train the MobileNetV2 transfer learning classifier (two-stage):

```bash
# Default run (15 initial epochs + 10 fine-tuning epochs, batch size 16)
python -m src.train

# Custom hyperparameters
python -m src.train --epochs 15 --fine-tune-epochs 10 --batch-size 16

# Skip fine-tuning (only Phase 1 — faster, lower accuracy)
python -m src.train --epochs 20 --no-fine-tune

# Larger batch for systems with more RAM
python -m src.train --epochs 15 --fine-tune-epochs 10 --batch-size 32
```

- **Split Ratio:** 70% Training / 15% Validation / 15% Testing (Stratified).
- **Data Leakage Prevention:** Raw image files are partitioned prior to any augmentation.
- **Phase 1 (Frozen Backbone):** Trains custom dense classification head (256 units + Dropout 0.4) using Adam optimizer.
- **Phase 2 (Fine-Tuning):** Unfreezes top 30 MobileNetV2 layers at a reduced learning rate (1e-5) for fine-grained feature extraction.
- **Checkpoints:** Best weights automatically saved to `models/mobilenetv2_best.keras`.

---

### Phase 3: Model Evaluation

Generate comprehensive evaluation metrics on the untouched test split:

```bash
python -m src.evaluate
```

This produces:
- `reports/confusion_matrix.png` — Annotated Seaborn heatmap
- `reports/training_history.png` — Accuracy and loss progression across epochs
- `reports/metrics.json` — Machine-readable per-class precision, recall, F1
- `reports/classification_report.txt` — Human-readable sklearn report

---

### Phase 4: Command-Line Prediction

Classify any image file directly from the terminal:

```bash
# Predict a single image
python -m src.predict --image dataset/coffee_mug/coffee_mug_001.jpg

# Try different classes
python -m src.predict --image dataset/scissors/scissors_001.jpg
python -m src.predict --image dataset/shoe/shoe_001.jpg
python -m src.predict --image dataset/keyboard/keyboard_001.jpg
python -m src.predict --image dataset/wallet/wallet_001.jpg
python -m src.predict --image dataset/smartphone/smartphone_001.jpg
python -m src.predict --image dataset/tv_remote/tv_remote_001.jpg
python -m src.predict --image dataset/water_bottle/water_bottle_001.jpg
python -m src.predict --image dataset/book/book_001.jpg
python -m src.predict --image dataset/computer_mouse/computer_mouse_001.jpg

# Predict your own image
python -m src.predict --image /path/to/any/image.jpg
```

Output: predicted class name, confidence %, top-3 candidates, and a low-confidence warning if below 45%.

---

### Phase 5: Launch the Streamlit Web Application

Run the interactive web application:

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** with three tabs:

| Tab | What it does |
|-----|-------------|
| 📸 Live Object Recognition | Upload an image or use webcam → instant prediction with confidence |
| 📊 Dataset Management Dashboard | View per-class counts, download missing images with one click |
| 📈 Model Evaluation & Reports | Confusion matrix, training curves, per-class F1 scores |
# ml-clg
