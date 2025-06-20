# Occupancy and Flow Prediction Challenge in the Waymo Dataset 🚗

![Screenshot](Waymo_Open_Challenge.jpg)

This repository presents a deep learning framework designed to predict **occupancy grids** and **motion flow fields** using the [Waymo Open Motion Dataset (WOMD)](https://waymo.com/open/). The project sets a strong baseline for solving key tasks in autonomous vehicle scene understanding, focusing on spatial-temporal prediction of multiple agents (vehicles, pedestrians, etc.).

---

## 🧠 Project Overview

Autonomous vehicles rely on accurate motion prediction to make safe, intelligent decisions in dynamic environments. This project addresses the challenge of **scene-level multi-agent prediction** by forecasting:

- **Observed Occupancy**: Future positions of visible agents.
- **Occluded Occupancy**: Predictions of agents currently hidden from sensors.
- **Motion Flow Fields**: Displacement vectors describing movement direction and magnitude.

---

## 🔧 Model Architecture

Three model variants are developed:

1. **U-Net**  
   - A classic encoder-decoder Convolutional Neural Network (CNN) architecture, adapted for dense prediction tasks in autonomous driving. It captures spatial context from Bird’s Eye View (BEV) representations using symmetric skip connections, enabling fine-grained pixel-level predictions

2. **UNet-LSTM**  
   - An enhanced version of U-Net with ConvLSTM layers integrated into the bottleneck. This allows the model to handle spatio-temporal sequences, learning how occupancy and motion evolve over time from past frames

3. **Attention UNet-LSTM**  
   - The most advanced variant, this model adds an attention mechanism before the ConvLSTM layers. It allows the network to focus on the most relevant spatial regions such as active agents, intersections, or occluded zones—improving both interpretability and accuracy

---

## 📥 Input Features

- Agent history: position, velocity, acceleration, dimensions, orientation.
- Road structure: lane boundaries, traffic lights, signs.
- Traffic signal states.
- Occupancy grid maps (256 × 256 resolution).

---

## 📤 Output

- `Observed Occupancy`: `Ob_k ∈ [0, 1]`
- `Occluded Occupancy`: `Oc_k ∈ [0, 1]`
- `Flow Fields`: `Fk ∈ ℝ^{256×256×2}` — where each cell holds `(dx, dy)` displacement values.

---

## 🧪 Experiments

- **Dataset**: Waymo Open Motion Dataset (WOMD)
- **Data Split**: 70% training, 15% validation, 15% test
- **Prediction Horizon**: 8 seconds (sampled at 10 Hz)
- **Evaluation Metrics**:
  - Soft-IoU
  - AUC-PR (Area Under PR Curve)
  - End-Point Error (EPE)
  - Flow-Grounded Occupancy Metrics

---

## 📈 Key Contributions

- A unified multi-task model for occupancy and flow field prediction.
- Effective use of temporal modeling via ConvLSTM and attention mechanisms.
- Superior performance across multiple benchmarks with a focus on both **short-term control** and **long-term planning**.
- Introduced **time-weighted loss functions** for improved long-horizon accuracy.

---

## 📊 Results Snapshot

| Model              | Soft-IoU ↑ | AUC-PR ↑ | EPE ↓ |
|-------------------|------------|----------|--------|
| U-Net             | 0.52       | 0.78     | 1.24   |
| UNet-LSTM         | 0.56       | 0.81     | 1.08   |
| Attention UNet-LSTM | **0.59**  | **0.83** | **0.94** |

> (↑ higher is better, ↓ lower is better)

---


