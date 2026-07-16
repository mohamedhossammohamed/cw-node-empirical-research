"""
Extract micro dataset for research paper scaling-law sweep.

Reads from dst_lab/backend/data/{train.bin, val.bin, meta.json}
Writes to dst_lab/research_paper_data/micro_{train,val}.bin + meta.json

Usage:
  .venv/bin/python dst_lab/research_paper_data/extract_micro_dataset.py
"""

import json, os, shutil, numpy as np

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "data")
DST_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_TOKENS = 5_000_000
VAL_TOKENS = 500_000


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    # Copy meta.json
    shutil.copy2(os.path.join(SRC_DIR, "meta.json"), os.path.join(DST_DIR, "meta.json"))
    print(f"[OK] meta.json copied")

    # Extract micro_train.bin
    full_train = np.memmap(os.path.join(SRC_DIR, "train.bin"), dtype=np.uint16, mode="r")
    n_train = min(len(full_train), TRAIN_TOKENS)
    micro_train = np.memmap(os.path.join(DST_DIR, "micro_train.bin"), dtype=np.uint16,
                             mode="w+", shape=(n_train,))
    micro_train[:] = full_train[:n_train]
    micro_train.flush()
    del micro_train, full_train
    print(f"[OK] micro_train.bin: {n_train:,} tokens ({n_train*2/1e6:.1f} MB)")

    # Extract micro_val.bin
    full_val = np.memmap(os.path.join(SRC_DIR, "val.bin"), dtype=np.uint16, mode="r")
    n_val = min(len(full_val), VAL_TOKENS)
    micro_val = np.memmap(os.path.join(DST_DIR, "micro_val.bin"), dtype=np.uint16,
                           mode="w+", shape=(n_val,))
    micro_val[:] = full_val[:n_val]
    micro_val.flush()
    del micro_val, full_val
    print(f"[OK] micro_val.bin: {n_val:,} tokens ({n_val*2/1e6:.1f} MB)")

    print("\n[DONE] Micro dataset ready.")


if __name__ == "__main__":
    main()
