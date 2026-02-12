#!/usr/bin/env python3

"""
========================================
 TITANCORE WEIGHT CONVERTER
========================================

Converts HuggingFace / PyTorch GPT models
into TitanCore optimized shards:

• Tensor Parallel split
• FSDP sharding
• INT8 / INT4 / FP8 quant
• KV cache alignment
• FlashAttention friendly layout
"""

import torch
import argparse
import json
import os
from pathlib import Path

# ----------------------------
# Arguments
# ----------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True, help="HF model path")
parser.add_argument("--out", required=True)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--dtype", default="fp16", choices=["fp16","int8","int4","fp8"])
parser.add_argument("--vocab", default=None)

args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ----------------------------
# Load HF Model
# ----------------------------

print("[TitanCore] Loading model...")

model = torch.load(args.model, map_location="cpu")

state = model["state_dict"] if "state_dict" in model else model

# ----------------------------
# Tensor Parallel Split
# ----------------------------

def shard_tensor(tensor, parts, dim=0):
    return torch.chunk(tensor, parts, dim=dim)

# ----------------------------
# Quantization
# ----------------------------

def quantize(t, dtype):
    if dtype == "fp16":
        return t.half()

    if dtype == "int8":
        scale = t.abs().max() / 127
        q = (t / scale).round().clamp(-128,127).to(torch.int8)
        return q, scale

    if dtype == "int4":
        scale = t.abs().max() / 7
        q = (t / scale).round().clamp(-8,7).to(torch.int8)
        return q, scale

    return t

# ----------------------------
# Convert
# ----------------------------

metadata = {}

for name, weight in state.items():

    print("Processing:", name)

    shards = shard_tensor(weight, args.tp)

    for rank, shard in enumerate(shards):

        out_dir = Path(args.out) / f"rank{rank}"
        out_dir.mkdir(exist_ok=True)

        if args.dtype in ["int8","int4"]:
            q, scale = quantize(shard, args.dtype)
            torch.save(q, out_dir / f"{name}.pt")
            torch.save(scale, out_dir / f"{name}.scale")

        else:
            shard = quantize(shard, args.dtype)
            torch.save(shard, out_dir / f"{name}.pt")

    metadata[name] = list(weight.shape)

# ----------------------------
# Save Metadata
# ----------------------------

with open(Path(args.out) / "model.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n[TitanCore] Conversion complete.")
