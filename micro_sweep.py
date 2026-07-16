"""
Micro Scaling-Law Sweep — CW-Node vs Dense Architecture
=========================================================

Trains 12 models (4 parameter tiers × 3 architectures) for 1 epoch on 5M tokens,
using the production CWNodeTransformer from dst_lab/backend/cw_node.py.

Architectures:
  dense:   w_int=0, d_int=0               (100% external — standard FFN)
  cw_70_30: ~70% external, ~30% internal  (out-heavy CW-Node)
  cw_30_70: ~30% external, ~70% internal  (in-heavy CW-Node)

Parameter tiers: 500K, 1M, 3M, 5M total params
Device: MPS (Apple Silicon GPU)
Dataset: 5M token Lucid Quran subset

Output: dst_lab/research_paper_data/results.json

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. .venv/bin/python \
    dst_lab/research_paper_data/micro_sweep.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)

# Import production architecture — no modifications
from cw_node import CWNodeTransformer

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
VOCAB_SIZE = 113
BLOCK_SIZE = 64
D_EXT = 2
D_INT = 2
TIERS = [500_000, 1_000_000, 3_000_000, 5_000_000]
ARCHS = ["dense", "cw_70_30", "cw_30_70"]
EXT_FRAC_TARGET = {"dense": 1.0, "cw_70_30": 0.70, "cw_30_70": 0.30}
# More layers for larger models
N_LAYER_MAP = {500_000: 2, 1_000_000: 2, 3_000_000: 3, 5_000_000: 3}


# ---------------------------------------------------------------------------
#  Data Loader (production memmap pattern from dst_lab/backend/data.py)
# ---------------------------------------------------------------------------
class MicroDataset:
    """Memory-mapped dataset wrapping the micro_{train,val}.bin files."""

    def __init__(self, data_dir: str, split: str = "train", block_size: int = 64):
        self.block_size = block_size

        with open(os.path.join(data_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        self.vocab_size = meta["vocab_size"]

        path = os.path.join(data_dir, f"micro_{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run extract_micro_dataset.py first.")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64))
             for i in ix])
        y = torch.stack(
            [torch.from_numpy((self.data[i + 1 : i + self.block_size + 1]).astype(np.int64))
             for i in ix])
        return x, y


# ---------------------------------------------------------------------------
#  Param-Grid Solver
# ---------------------------------------------------------------------------
def count_total_params(n_embd: int, w_ext: int, w_int: int,
                       n_layer: int, vocab_size: int = VOCAB_SIZE,
                       block_size: int = BLOCK_SIZE) -> int:
    """Count total params for a CWNodeTransformer with given settings."""
    model = CWNodeTransformer(
        vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd,
        w_ext=w_ext, d_ext=D_EXT, w_int=w_int, d_int=D_INT,
        block_size=block_size, dtype=torch.float32,
    )
    return sum(p.numel() for p in model.parameters())


def count_ffn_params_per_layer(n_embd: int, w_ext: int, w_int: int) -> Dict[str, int]:
    """Count FFN-only params per layer (external + internal)."""
    # SquareMLP external
    ext = (n_embd * w_ext + w_ext)                                    # in_proj
    if D_EXT > 1:
        ext += (D_EXT - 1) * (w_ext * w_ext + w_ext)                 # hidden
    ext += (w_ext * n_embd + n_embd)                                  # out_proj

    # Internal per-node MLP
    int_p = 0
    if w_int > 0:
        int_p = (2 * n_embd * w_int                                 # in_w + in_b
                 + n_embd * w_int + n_embd                           # out_w + out_b
                 + (D_INT - 1) * (n_embd * w_int * w_int + n_embd * w_int)   # hidden_w + hidden_b
                 + (D_INT - 1) * 2 * n_embd * w_int)               # ln_w + ln_b

    return {"external": ext, "internal": int_p, "total": ext + int_p}


def solve_w_ext(n_embd: int, target_ext_per_layer: float) -> int:
    """Solve w_ext from quadratic: (d-1)·w² + (2n + d)·w + n ≈ target."""
    if target_ext_per_layer <= 0:
        return 0
    a = max(0, D_EXT - 1)
    b = 2 * n_embd + D_EXT
    c = n_embd - target_ext_per_layer
    if a == 0:
        return max(4, round(-c / b)) if b > 0 else 4
    disc = b * b - 4 * a * c
    if disc < 0:
        return 4
    return max(4, round((-b + math.sqrt(disc)) / (2 * a)))


def solve_w_int(n_embd: int, target_int_per_layer: float) -> int:
    """Solve w_int from quadratic for internal per-node params."""
    if target_int_per_layer <= 0:
        return 0
    a = n_embd * max(0, D_INT - 1)
    b = n_embd * (4 + 3 * max(0, D_INT - 1))
    c = n_embd * 1 - target_int_per_layer
    if a == 0:
        return max(2, round(-c / b)) if b > 0 else 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return 2
    return max(2, round((-b + math.sqrt(disc)) / (2 * a)))


def non_ffn_params(n_embd: int, n_layer: int) -> int:
    """Embeddings + head + per-layer attention (biases resolved in FFN count)."""
    # wte + wpe + lm_head
    emb_head = (VOCAB_SIZE + BLOCK_SIZE + VOCAB_SIZE) * n_embd
    # Per-layer attention: c_attn (3n²+3n) + c_proj (n²+n) + ln_attn (2n)
    attn = n_layer * (4 * n_embd * n_embd + 6 * n_embd)
    # SquareMLP has bias terms: in_proj.bias (w), out_proj.bias (n), optional hidden.bias ((d-1)*w)
    # These depend on w_ext and w_int, so they're NOT part of non-ffn.
    # But we estimate them here since w_ext isn't known yet.
    # Actually, the biases ARE part of the FFN count — they get solved along with w_ext/w_int.
    # So non_ffn is just embeddings + head + attention.
    return emb_head + attn


def solve_config(target_params: int, arch: str, n_layer: int) -> dict:
    """
    Find (n_embd, w_ext, w_int) that hits target_params within ±5%.
    Uses the same n_embd as the dense baseline for CW architectures (fair comparison).
    """
    ext_target = EXT_FRAC_TARGET[arch]

    # Step 1: Find best n_embd for dense baseline at this tier
    best_n_embd = None
    for n_embd in range(16, min(800, int(math.sqrt(target_params))) + 1):
        nffn = non_ffn_params(n_embd, n_layer)
        rem = target_params - nffn
        if rem <= 0:
            continue
        w_ext = solve_w_ext(n_embd, rem / n_layer)
        p = count_total_params(n_embd, w_ext, 0, n_layer)
        if abs(p - target_params) / target_params < 0.03:
            best_n_embd = n_embd
            break
        # Track best even if no exact match (<5%)
        if best_n_embd is None:
            best_n_embd = n_embd  # fallback to largest

    if best_n_embd is None:
        best_n_embd = min(128, int(math.sqrt(target_params)))

    n_embd = best_n_embd

    # Step 2: Solve for this architecture using n_embd
    nffn = non_ffn_params(n_embd, n_layer)
    rem = target_params - nffn
    if rem <= 0:
        return {"n_embd": n_embd, "n_layer": n_layer,
                "w_ext": 4, "w_int": 0 if arch == "dense" else 2,
                "total_params": count_total_params(n_embd, 4, 0 if arch == "dense" else 2, n_layer)}

    ext_budget = rem * ext_target
    int_budget = rem * (1 - ext_target)

    w_ext = solve_w_ext(n_embd, ext_budget / n_layer)
    w_int = solve_w_int(n_embd, int_budget / n_layer) if arch != "dense" else 0

    total = count_total_params(n_embd, w_ext, w_int, n_layer)

    return {
        "arch": arch,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "w_ext": w_ext,
        "d_ext": D_EXT,
        "w_int": w_int,
        "d_int": D_INT,
        "target_params": target_params,
        "total_params": total,
    }


def build_config_grid() -> List[dict]:
    """Build all 12 model configs."""
    configs = []
    for tier in TIERS:
        n_layer = N_LAYER_MAP[tier]
        for arch in ARCHS:
            cfg = solve_config(tier, arch, n_layer)
            configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
#  Training Loop
# ---------------------------------------------------------------------------
def train_one_model(cfg: dict, data_dir: str,
                    device: torch.device,
                    steps: int = 19_531) -> dict:
    """Train one model for `steps` on the micro dataset. Returns full history."""

    torch.manual_seed(42)

    model = CWNodeTransformer(
        vocab_size=VOCAB_SIZE,
        n_layer=cfg["n_layer"],
        n_embd=cfg["n_embd"],
        w_ext=cfg["w_ext"],
        d_ext=D_EXT,
        w_int=cfg["w_int"],
        d_int=D_INT,
        block_size=BLOCK_SIZE,
        dtype=torch.float32,
    ).to(device)

    actual_params = sum(p.numel() for p in model.parameters())

    train_ds = MicroDataset(data_dir, "train", BLOCK_SIZE)
    val_ds = MicroDataset(data_dir, "val", BLOCK_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    run_id = f"{cfg['arch']}_tier{cfg['target_params']//1000}K"
    print(f"  [{run_id}] n_embd={cfg['n_embd']} w_ext={cfg['w_ext']} "
          f"w_int={cfg['w_int']} params={actual_params:,} steps={steps}",
          flush=True)

    history: List[dict] = []
    t0 = time.perf_counter()
    model.train()

    for step in range(1, steps + 1):
        X, Y = train_ds.get_batch(4)
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(X, Y)
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == 1 or step == steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(10):
                    Xv, Yv = val_ds.get_batch(16)
                    Xv, Yv = Xv.to(device), Yv.to(device)
                    _, vl = model(Xv, Yv)
                    val_losses.append(vl.item())
            val_loss = float(np.mean(val_losses))
            model.train()

            entry = {
                "step": step,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "val_bpc": val_loss / math.log(2),
                "elapsed_s": time.perf_counter() - t0,
            }
            history.append(entry)

            if step % 2000 == 0 or step == steps:
                print(f"    step {step:6d}/{steps}  train={loss.item():.4f}  "
                      f"val={val_loss:.4f}  bpc={val_loss/math.log(2):.4f}",
                      flush=True)

        # Empty MPS cache periodically
        if device.type == "mps" and step % 200 == 0:
            torch.mps.empty_cache()

    wall_time = time.perf_counter() - t0

    return {
        **cfg,
        "actual_params": actual_params,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "final_val_bpc": history[-1]["val_loss"] / math.log(2) if history else None,
        "wall_time_s": wall_time,
        "history": history,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=str, default=None,
                        help="Single tier: 500K, 1M, 3M, 5M")
    parser.add_argument("--arch", type=str, default=None,
                        choices=ARCHS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=19_531,
                        help="Training steps (default: 19531 = 1 epoch over 5M tokens)")
    args = parser.parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))
    # Output path
    out_path = os.path.join(data_dir, "results.json")

    # Build configs
    configs = build_config_grid()

    # Filter if single tier/arch
    if args.tier:
        tier_map = {"500K": 500_000, "1M": 1_000_000, "3M": 3_000_000, "5M": 5_000_000}
        configs = [c for c in configs if c["target_params"] == tier_map[args.tier]]
    if args.arch:
        configs = [c for c in configs if c["arch"] == args.arch]

    # Display config grid
    print("=" * 75)
    print("MICRO SCALING-LAW SWEEP — CONFIGURATION GRID")
    print("=" * 75)
    for cfg in configs:
        err = abs(cfg["total_params"] - cfg["target_params"]) / cfg["target_params"]
        print(f"  {cfg['arch']:<12s} {cfg['target_params']//1000:>4d}K  "
              f"n_embd={cfg['n_embd']:>4d}  w_ext={cfg['w_ext']:>4d}  "
              f"w_int={cfg['w_int']:>3d}  "
              f"actual={cfg['total_params']:>10,}  ±{err*100:.1f}%")
    print(f"\nTotal: {len(configs)} models, {len(configs) * args.steps} steps each")
    print(f"Device: MPS (torch {torch.__version__})")

    if args.dry_run:
        print("\n[Dry run — no training.]")
        return

    # Run sweep
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nRunning on device: {device}")

    results = []
    t_total = time.perf_counter()
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {cfg['arch']} | {cfg['target_params']//1000}K params")
        try:
            res = train_one_model(cfg, data_dir, device, steps=args.steps)
            results.append(res)
            # Save incrementally
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved to {out_path}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            results.append({**cfg, "error": str(e)})

    total_time = (time.perf_counter() - t_total) / 60
    print(f"\n{'=' * 75}")
    print(f"SWEEP COMPLETE in {total_time:.1f} min")
    print(f"Results: {out_path}")

    # Quick summary
    print(f"\n{'Arch':<12s} {'Tier':>8s} {'Final Val Loss':>16s} {'BPC':>8s} {'Time':>8s}")
    print("-" * 56)
    for r in results:
        if "error" not in r:
            print(f"{r['arch']:<12s} {r['target_params']:>8,} "
                  f"{r.get('final_val_loss', 0):>16.4f} "
                  f"{r.get('final_val_bpc', 0):>8.4f} "
                  f"{r.get('wall_time_s', 0)/60:>7.1f}m")


if __name__ == "__main__":
    main()
