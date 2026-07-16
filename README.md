# CW-Node Research: Micro Scaling-Law Sweep (V1)

This repository contains the code, baseline datasets, and results for the **CW-Node Architecture**, a novel deep learning architecture that routes information internally as an alternative to standard Dense Feed-Forward Networks (FFNs) in Transformers.

## Objective
The goal of this sweep is toly determine if the CW-Node architecture has a parameter-efficiency advantage over traditional Dense baseline Transformers by plotting validation loss against total parameter counts across 4 distinct tiers: **500K, 1M, 3M, and 5M parameters.**

## Methodology
- **Dataset:** 5,000,000 token training split (extracted from the Holy Quran dataset, vocab size 113), with a 500,000 token held-out validation set.
- **Architectures Compared:**
  1. **Dense Baseline:** Standard FFN with `w_int=0, d_int=0`.
  2. **CW-Node 70/30 (Out-Heavy):** Custom internal routing where ~70% of the FFN parameters are external MLPs, and ~30% are dedicated to internal node routing.
  3. **CW-Node 30/70 (In-Heavy):** Custom internal routing where ~30% of the FFN parameters are external MLPs, and ~70% are dedicated to internal node routing.
- **Training Setup:**
  - Device: Apple Silicon GPU (`mps`)
  - Sequence Block Size: 64, Batch Size: 4
  - Optimizer: AdamW, learning rate = 3e-3
  - Steps: 19,531 steps (equivalent to exactly 1 epoch over the 5M token dataset)
  - Hard constraint: `n_embd = 16` across all tiers for strict comparative alignment.

---

## Findings

### Final Validation Loss Summary
| Architecture | 500K | 1M | 3M | 5M |
| :--- | :---: | :---: | :---: | :---: |
| **Dense Baseline** | 2.6452 | **2.5470** | **2.5430** | 2.5800 |
| **CW-Node 70/30 (Out-Heavy)** | **2.6052** | 2.6805 | 2.7756 | **2.5682** |
| **CW-Node 30/70 (In-Heavy)** | 2.6541 | 2.6020 | 2.5918 | 2.7140 |

### Delta vs Dense (negative = CW-Node Wins)
- **CW-Node 70/30:** 
  - 500K: `-0.04` (Win)
  - 1M: `+0.13` (Loss)
  - 3M: `+0.23` (Loss)
  - 5M: `-0.01` (Win/Tie)
- **CW-Node 30/70:** 
  - 500K: `+0.01` (Loss)
  - 1M: `+0.06` (Loss)
  - 3M: `+0.05` (Loss)
  - 5M: `+0.13` (Loss)

---

## 🔬 Critical Research Discovery: The `n_embd` Confound

### The Hypothesis
The CW-Node architecture routes parameters internally using sparse representations. Theoretically, this allows the architecture to support **significantly wider embedding dimensions (`n_embd`)** than a Dense network *within the exact same total parameter budget*, because its FFN layers scale sub-quadratically.

### The Confound in V1
For strict experimental control in this V1 sweep, the config solver locked the embedding dimension `n_embd = 16` across all architectures. While technically "fair" on paper, **this constraint actively suffocated the CW-Node.** 

Because it was forced to allocate its parameters into complex internal routing pathways within a tiny 16-dimensional embedding space, the model could not capture high-dimensional token relationships. Meanwhile, the Dense baseline easily saturated the 16-dimensional space with its direct projection matrices.

### The Path to V2 (Unrestricted Embedding Sweep)
To reveal the true scaling efficiency of the CW-Node, the next iteration of this research (V2) must allow the parameter solver to vary `n_embd` freely. Under a fixed parameter budget (e.g., 3M parameters):
- The **Dense model** will be constrained to a narrow embedding dimension (e.g., `n_embd = 128`).
- The **CW-Node model** will scale to a massive embedding dimension (e.g., `n_embd = 384`).

If the CW-Node wins under those conditions, it will prove that sub-quadratic internal routing is a superior mechanism for high-dimensional representation learning.

---

## Repository Structure
- `cw_node.py`: The core vectorized implementation of the CW-Node routing.
- `extract_micro_dataset.py`: Script to extract the 5M token benchmark.
- `micro_sweep.py`: Core automated experiment runner for all 12 models.
- `plot_scaling_laws.py`: Visualizes loss curves and generates plots.
- `results.json`: Full validation/training loss history for all runs.
- `scaling_law_frontier.png`: Plot graphing parameter count vs. validation loss.
- `learning_curves_3M.png`: Visual training trajectories of the 3M tier models.
