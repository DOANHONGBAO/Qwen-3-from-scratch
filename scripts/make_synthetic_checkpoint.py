#!/usr/bin/env python
"""Write a RANDOM-weight checkpoint in the official HF layout (for dry runs and tests).

    python scripts/make_synthetic_checkpoint.py --size tiny --out /tmp/qwen3-synthetic-tiny
    python scripts/make_synthetic_checkpoint.py --size 0.6b --out /tmp/qwen3-synthetic-0.6b

``--size 0.6b`` has the exact Qwen3-0.6B-Base shapes (1.1 GiB bf16 file) but random
weights and a toy tokenizer: use it to rehearse memory and timing, never as a model.
"""

from __future__ import annotations

import argparse
import dataclasses

from qwen3_scratch.config import QWEN3_0_6B_BASE, QWEN3_TINY
from qwen3_scratch.synthetic import write_synthetic_checkpoint

SIZES = {
    # tiny: vocab large enough for the toy tokenizer (400 ids)
    "tiny": dataclasses.replace(QWEN3_TINY, vocab_size=512, max_position_embeddings=256),
    "0.6b": QWEN3_0_6B_BASE,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", choices=sorted(SIZES), default="tiny")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-std", type=float, default=None)
    args = parser.parse_args()
    path = write_synthetic_checkpoint(
        SIZES[args.size], args.out, seed=args.seed, weight_std=args.weight_std
    )
    print(f"wrote RANDOM-weight checkpoint to {path}")


if __name__ == "__main__":
    main()
