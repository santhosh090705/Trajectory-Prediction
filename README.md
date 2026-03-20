# 🚗 Byte Riders – Intent & Trajectory Prediction
### MAHE Mobility Challenge 2026 | Track 1 | AI & Computer Vision

---

## 📌 Project Overview

In a **Level 4 urban autonomous driving** environment, simply reacting to where a pedestrian *is* isn't enough — the vehicle must predict where they *will be*.

This project builds a **multi-modal trajectory prediction model** that:
- Takes **2 seconds of past motion** (x, y coordinates + velocity) of pedestrians and cyclists
- Predicts the **next 3 seconds** of movement
- Outputs the **top-3 most likely future paths** (multi-modal prediction)
- Accounts for **social context** — how nearby agents interact and avoid each other

**Dataset:** [nuScenes](https://www.nuscenes.org/) (publicly available autonomous driving dataset)

---

## 👥 Team

| Name | Role |
|---|---|
| **Santhosh S** *(Team Lead)* | Model Architecture & Transformer Design |
| **Sanjay M** | Data Preprocessing & nuScenes Pipeline |
| **Sanjay Kumar P** | Social Pooling & Context Modeling |
| **Varun S** | Evaluation, Metrics & Visualization |

**Institution:** Manipal Institute of Technology Bengaluru (MIT-B), AI & Data Science, 3rd Year

---

## 🧠 Model Architecture

```
nuScenes Input (x,y coords + velocity, 2s history)
        │
        ▼
┌─────────────────────┐
│  Temporal Encoder   │  LSTM/GRU per agent → hidden vector
└─────────────────────┘
        │
        ├────── Neighbour encodings
        ▼
┌─────────────────────┐
│  Social Attention   │  Graph Attention over neighbours
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Fusion MLP        │  [ego_h | social_ctx] → context
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Multi-Modal Decoder │  GRU decoder × K modes + mode classifier
└─────────────────────┘
        │
        ▼
Top-3 predicted trajectories + mode probabilities
```

### Key Components

| Module | Details |
|---|---|
| **Temporal Encoder** | 1-layer LSTM, hidden_dim=64 |
| **Social Attention** | Single-head dot-product attention over up to 20 neighbours |
| **Multi-Modal Decoder** | K=3 GRU decoders + log-softmax mode classifier |
| **Loss** | MSE (best mode) + NLL (mode classification) |
| **Optimiser** | Adam, lr=1e-3, weight_decay=1e-4 |
| **Parameters** | ~350K total |

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **minADE@3** | Mean Euclidean distance between best predicted path and ground truth, averaged over all timesteps |
| **minFDE@3** | Euclidean distance at the final predicted timestep (t+3s) |
| **Miss Rate** | % of samples where minFDE > 2 metres |

---

## 📁 Repository Structure

```
MIT-hackathon/
├── data_loader.py      # nuScenes data loading & preprocessing
├── model.py            # Full model: Encoder + Social Attention + Decoder
├── train.py            # Training script with checkpointing
├── evaluate.py         # Evaluation: minADE, minFDE, Miss Rate
├── inference.py        # Run predictions + save visualisations
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/santhosh090705/MIT-hackathon.git
cd MIT-hackathon
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download nuScenes dataset
- Register at [https://www.nuscenes.org/](https://www.nuscenes.org/)
- Download **v1.0-trainval** (full dataset) or **v1.0-mini** (for quick testing)
- Extract to `./data/nuscenes/`

Expected folder structure:
```
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
│   ├── attribute.json
│   ├── category.json
│   ├── instance.json
│   ├── ...
```

---

## 🚀 How to Run

### Train
```bash
python train.py \
    --dataroot ./data/nuscenes \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --hidden_dim 64 \
    --save_dir ./checkpoints
```

### Evaluate (on test split)
```bash
python evaluate.py \
    --dataroot ./data/nuscenes \
    --checkpoint ./checkpoints/best_model.pt \
    --split test
```

### Run Inference + Visualise
```bash
python inference.py \
    --dataroot ./data/nuscenes \
    --checkpoint ./checkpoints/best_model.pt \
    --num_scenes 10 \
    --output_dir ./outputs
```

### Quick data check
```bash
python data_loader.py ./data/nuscenes
```

### Quick model shape check (no data needed)
```bash
python model.py
```

---

## 📈 Example Output

After running inference, you will find:
- `outputs/predictions.json` — all predicted trajectories in world coordinates
- `outputs/visualisations/scene_XXXX.png` — one plot per scene showing:
  - 🟠 **Observed path** (past 2 seconds)
  - 🟣 **Ground truth future** (next 3 seconds)
  - 🔴🔵🟢 **Top-3 predicted trajectories** with mode probabilities

Example prediction plot:

```
● ● ● ● (observed, orange)
         ↘ ◆ ◆ ◆ ◆ ◆ ◆  (ground truth, purple dashed)
         ↘ ▲ ▲ ▲ ▲ ▲ ▲  (mode 1 – p=0.62, red)
         ↗ ▲ ▲ ▲ ▲ ▲ ▲  (mode 2 – p=0.25, blue)
         → ▲ ▲ ▲ ▲ ▲ ▲  (mode 3 – p=0.13, green)
```

---

## 📋 Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataroot` | `./data/nuscenes` | Path to nuScenes root |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `64` | Samples per batch |
| `--lr` | `0.001` | Learning rate |
| `--hidden_dim` | `64` | LSTM/GRU hidden size |
| `--num_layers` | `1` | Number of LSTM layers |
| `--dropout` | `0.1` | Dropout rate |
| `--K` | `3` | Number of predicted modes |
| `--save_dir` | `./checkpoints` | Where to save model checkpoints |
| `--resume` | `None` | Path to resume training from a checkpoint |

---

## 🔬 Technical Notes

- **Coordinate frame:** All trajectories are normalised to the ego-agent's last observed position (0, 0) before input to the model
- **Neighbour handling:** Up to 20 neighbouring agents are encoded; absent neighbours are masked in the attention layer
- **Multi-modal output:** The best-matching mode is selected using minADE during training; at inference time all 3 modes are returned with their probabilities
- **Gradient clipping:** Max norm = 1.0 for stable training
- **Hardware used:** Google Colab Pro (NVIDIA T4 / A100) + PyTorch

---

## 📜 Declaration

We confirm that all code in this repository is original work developed by **Team Byte Riders** for the MAHE Mobility Challenge 2026. All team members agree to the hackathon rules and evaluation process.

---

*Manipal Institute of Technology Bengaluru | AI & Data Science | 2026*
