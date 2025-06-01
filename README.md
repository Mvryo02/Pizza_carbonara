# Robust Graph Classification under Noisy Labels – DL Hackaton 2024

## Overview

Graph‑structured data are ubiquitous – from molecular graphs to social or financial networks – yet **label noise** severely degrades performance.
Our solution combines a noise‑resilient **GINE‑based Graph Neural Network** with robust loss functions and training tricks to cope with sporadic or systematic mis‑labelling.

**Key ingredients**

| Component                           | Why we chose it                                                         |
| ----------------------------------- | ----------------------------------------------------------------------- |
| **GINEConv (+ edge‑features)**      | Captures rich edge information provided in the datasets.                |
| **Jumping Knowledge (max)**         | Aggregates multi‑scale node representations, mitigating over‑smoothing. |
| **Global Attention pooling**        | Learns graph‑level summaries adaptive to task.                          |
| **Symmetric / Noisy Cross‑Entropy** | Robust to noisy labels (choice depends on dataset).                     |
| **Deep Dropout + Grad‑Clip**        | Reduces variance introduced by noise.                                   |
| **One‑Cycle LR & Early Stopping**   | Fast convergence while preventing over‑fitting noisy samples.           |

The same codebase supports **training** (when a `--train_path` is provided) and **pure inference** using fully reproducible checkpoints.

---

## Repository structure

```
├── checkpoints/        # ⟶ trained *.pth files (one sub‑folder per dataset)
├── logs/               # ⟶ training & validation history + plots
├── src/                # ⟶ all source code
│   ├── loadData.py     #   dataset loader → PyTorch Geometric
│   ├── models.py       #   GINE_GNN definition
│   └── utils.py        #   losses, early‑stop, helpers
├── submission/         # ⟶ generated CSV predictions
├── main.py             # ⟶ entry‑point (train / inference)
├── requirements.txt    # ⟶ python & package versions
└── README.md           # ⟶ you are here
```

---

## Setup

```bash
# 1. set up environment (tested with Python 3.10)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

> **GPU** is recommended; set `CUDA_VISIBLE_DEVICES` accordingly or use `--device` flag.

---

## Usage

### 1 · Train **and** predict

```bash
python main.py \
  --train_path ./datasets/A/train.json.gz \
  --test_path  ./datasets/A/test.json.gz   \
  --epochs 100 --batch_size 32
```

* saves *≥5* checkpoints in `checkpoints/A/`
* logs every epoch to `logs/A/training.log` + loss/acc curves
* writes CSV to `submission/testset_A.csv`

### 2 · Predict **only** (using the best checkpoint)

```bash
python main.py --test_path ./datasets/A/test.json.gz
```

---

## Method details

1. **Data loading** – `GraphDataset` parses gzipped JSON files and converts them to PyG `Data` objects. If node features are missing we assign a single learnable embedding (all‑zero placeholder).
2. **Model** – 5× `GINEConv` layers (embedding dim 256) with BatchNorm, ReLU and dropout 0.3. Multi‑layer features are fused via **Jumping Knowledge‑max** and **Global Attention** pooling creates a fixed‑size graph representation. A two‑layer MLP head outputs logits for the 6 classes.
3. **Learning with noise** –

   * **Dataset B** shows heavy label corruption → we switch to **Symmetric Cross‑Entropy** (α = 0.1, β = 4.0).
   * Other splits use a lightweight **Noisy Cross‑Entropy** (p = 0.2).
4. **Optimisation** – AdamW (`lr 3e‑3`, `wd 0.01`) + **One‑Cycle** scheduler; gradients are clipped to 1.0.
5. **Regularisation** – early stopping with patience 20 prevents memorising noisy labels.
6. **Inference & submission** – the best checkpoint (highest val‑acc) is automatically re‑loaded and predictions are exported following the required naming convention.
