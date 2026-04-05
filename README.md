# superai-ss6-mh4.3-sleep-stage-classification

# 😴 MH4.3 — Sleep Stage Classification

**Super AI Engineer Season 6 | Mini Hackathon 4.3**

จำแนกระยะการนอนหลับ (W / N1 / N2 / N3 / R) จากสัญญาณ EEG
ด้วย Feature Engineering + LightGBM

---

## 📊 Result

| Metric | Value |
|--------|-------|
| **Score (F1 weighted)** | **0.46192** |
| **Rank** | **181 / 350+** |
| Baseline | 0.42327 |
| **Status** | ✅ Passed baseline |

---

## 🔧 Tech Stack

`Python` · `LightGBM` · `SciPy` · `NumPy` · `Pandas` · `Google Colab`

---

## 🏗️ Approach

### Signal Processing Pipeline
```
Raw EEG Signal (fs = 16 Hz)
    ↓
Segmentation (fixed window)
    ↓
Feature Extraction per segment
    ↓
LightGBM Classifier
    ↓
Post-processing (Median Filter Smoothing)
    ↓
Submission
```

### Feature Engineering
แปลง raw EEG signal เป็น tabular features ครอบคลุม 3 domain:

| Domain | Features |
|--------|---------|
| **Time domain** | mean, std, min, max, skewness, kurtosis, zero-crossing rate, RMS |
| **Frequency domain** | FFT power bands (delta/theta/alpha/beta/gamma), dominant frequency |
| **Statistical** | percentiles, IQR, peak-to-peak amplitude |

### Model — LightGBM
```python
objective:       multiclass (5 classes)
metric:          multi_logloss
num_leaves:      63
learning_rate:   0.05
feature_fraction: 0.8
bagging_fraction: 0.8
early_stopping:  100 rounds
```

### Post-processing
ใช้ `median_filter` บน predicted sequence ต่อ subject
เพื่อ smooth transitions ระหว่าง sleep stages ที่ไม่น่าเปลี่ยนเร็วเกินไป

---

## 💡 Key Learnings

- **Frequency features** (FFT bands) สำคัญที่สุดใน feature importance
- **Median smoothing** ช่วย score เพราะ sleep stages เปลี่ยนแบบ gradual
- ควรลอง LSTM / Transformer ที่ใช้ temporal context ข้าม segment

---

## 📁 Files

```
mh4.3-colab.py   # Main notebook
```

---
