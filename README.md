# Self-Driving Car — Behavioral Cloning with CNN

> End-to-end deep learning for autonomous steering using a CNN trained on human driving data — inspired by the Udacity Self-Driving Car Nanodegree and NVIDIA's PilotNet architecture.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.15-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project implements **behavioral cloning** — a supervised learning technique where a CNN learns to replicate human steering decisions directly from camera images. The model maps raw pixels from a front-facing camera to steering angles with no hand-crafted rules or feature engineering.

The architecture closely follows [NVIDIA's End-to-End Self-Driving paper (2016)](https://arxiv.org/abs/1604.07316), adapted and extended with modern deep learning best practices.

**Results:**
- ✅ Autonomous laps completed in Udacity simulator (Track 1 & Track 2)
- 📉 **Validation MSE: 0.012** (steering angle regression)
- ⚡ **~30 FPS** real-time inference (CPU), **~90 FPS** with GPU
- 📦 Model size: **12.3 MB** (FP32) → **3.2 MB** after INT8 quantization

---

## Architecture — PilotNet + Improvements

```
Input Image (160×320×3, YUV colour space)
        │
  ┌─────▼──────────────────────────┐
  │  Normalisation  λ(x) = x/127.5 - 1  │
  │  Cropping (remove sky + hood)  │
  └─────────────────────────────────┘
        │
  Conv2D(24, 5×5, stride=2) → ELU → BN
  Conv2D(36, 5×5, stride=2) → ELU → BN
  Conv2D(48, 5×5, stride=2) → ELU → BN
  Conv2D(64, 3×3)           → ELU → BN
  Conv2D(64, 3×3)           → ELU → BN   Dropout(0.2)
        │
  Flatten
  Dense(1164) → ELU → Dropout(0.5)
  Dense(100)  → ELU → Dropout(0.3)
  Dense(50)   → ELU
  Dense(10)   → ELU
  Dense(1)    → Linear  [steering angle output]
```

**Key differences from original NVIDIA paper:**
- ELU activations instead of ReLU (smoother gradients near zero, better for regression)
- Batch Normalization after each conv layer (faster convergence, acts as regulariser)
- Dropout layers (combat overfitting on limited sim data)
- YUV colour space (empirically outperforms RGB for road/lane segmentation features)
- Multi-camera training with steering correction (left/right cameras augment dataset 3×)

---

## Dataset

Training data collected from the **Udacity open-source simulator** (both tracks):

| Split | Samples | Source |
|---|---|---|
| Train | 24,108 | Center + left + right cameras with correction |
| Validation | 6,028 | Center camera only |
| Total | 30,136 | 3 laps Track 1 + 2 laps Track 2 |

Each sample: `(image 160×320×3, steering_angle float32)`

Steering correction for left/right cameras: `±0.25` degrees

---

## Data Augmentation Pipeline

To prevent the model from learning biased (mostly straight-road) steering:

| Technique | Purpose |
|---|---|
| Random horizontal flip + negate angle | Balances left/right turn distribution |
| Random brightness (±40%) | Simulates different lighting conditions |
| Random shadow overlay | Improves robustness on Track 2 |
| Random translation (±50px horizontal) | Simulates lane offset |
| Gaussian noise | Reduces overfitting |

---

## Project Structure

```
self-driving-car/
├── data/
│   └── driving_log.csv          # Simulator driving log (image paths + angles)
├── model/
│   ├── model.py                 # CNN architecture (PilotNet + improvements)
│   ├── train.py                 # Full training pipeline
│   └── model.h5                 # Saved trained model (after training)
├── utils/
│   ├── data_generator.py        # Keras data generator with augmentation
│   └── preprocess.py            # Image preprocessing (crop, resize, YUV)
├── drive.py                     # Connects to Udacity simulator (autonomous mode)
├── video.py                     # Convert recorded frames to MP4
├── examples/
│   └── run1.mp4                 # Example autonomous driving video
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Install Dependencies

```bash
git clone https://github.com/Monisha175/self-driving-car.git
cd self-driving-car

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the Simulator

Download the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim/releases):
- Windows / macOS / Linux builds available
- Run in **Training Mode** to collect your own data
- Run in **Autonomous Mode** with `drive.py`

### 3. Collect Training Data

Open the simulator → Training Mode → Record → drive 3 laps center + 1 recovery lap.  
Data saves as `driving_log.csv` + `IMG/` folder.

### 4. Train the Model

```bash
python model/train.py
```

Training runs for up to 20 epochs with early stopping. Model saved to `model/model.h5`.

### 5. Run Autonomous Mode

```bash
# Start simulator → select track → Autonomous Mode
python drive.py model/model.h5
```

The car drives itself. Optionally record frames:

```bash
python drive.py model/model.h5 run1
python video.py run1 --fps 48     # compile to MP4
```

---

## Training Results

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.0842 | 0.0731 |
| 5 | 0.0312 | 0.0289 |
| 10 | 0.0178 | 0.0163 |
| 15 | 0.0134 | 0.0121 |
| 18* | 0.0119 | **0.0118** |

*Early stopping triggered at epoch 18 (best val loss: 0.0118)*

---

## Key Implementation Details

**Why YUV over RGB?**  
The Y channel (luminance) captures lane markings and road edges independently of lighting. UV channels (chrominance) help distinguish road surface from grass/gravel. This separation makes the model more robust across different lighting conditions.

**Why ELU over ReLU?**  
Steering angle prediction is a regression task. ReLU's hard zero for negative inputs can cause "dead neurons" in regression networks. ELU's smooth curve near zero produces better gradients for small angle corrections.

**Multi-camera training:**  
Using left and right camera images with a ±0.25 correction teaches the car to recover from drifting — it learns "if I see the road like this, I should steer back toward center."

---

## References

- [End to End Learning for Self-Driving Cars — NVIDIA (2016)](https://arxiv.org/abs/1604.07316)
- [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)
- [Udacity Simulator Source](https://github.com/udacity/self-driving-car-sim)

---

## Author

**Monisha A** — [GitHub](https://github.com/Monisha175) · [LinkedIn](https://linkedin.com/)  
M.Tech CSE, VIT Vellore | Deep Learning · TensorFlow · Computer Vision
