# 17-laser project guide

## Purpose

This is a beginner learning project using the KAMP precision-machining resource-optimization dataset. Explain all work in simple Korean, one step at a time, and explain why each code cell is needed before adding more analysis.

## Environment

- Use Python 3.12 from `.venv`.
- Windows interpreter: `.venv\Scripts\python.exe`
- Basic packages are listed in `requirements.txt`.
- Install `requirements-deep-learning.txt` only when the LSTM autoencoder stage begins.
- Work mainly in `1.ipynb`.

## Data

- `data/normal.xlsx`: 92,280 sensor records collected during the normal machining condition.
- `data/anomaly.xlsx`: 11,800 sensor records collected during an intentionally altered laser-focus condition.
- Columns: `Time`, `Intensity`, `Current`.
- One row is one time-point sensor measurement. It is not one product, one hole, or one LOT.
- The files contain no product ID, hole ID, or LOT ID, so do not invent those boundaries.
- Call the file labels normal-condition and anomaly-condition labels. Do not claim that every anomaly row proves a defective finished product.

## Analysis rules

- Preserve time order.
- Split training and evaluation periods before making windows.
- Never make a window across the normal/anomaly files, a large time gap, or a train/evaluation boundary.
- Do not use the date or source filename as model features.
- The guidebook uses 40 consecutive measurements as a window and an LSTM autoencoder trained on normal data. Treat this as a reference procedure, not proof that 40 is optimal.
- Do not change the original Excel files.
- Before modeling, show the beginner-friendly data exploration and preprocessing results.
