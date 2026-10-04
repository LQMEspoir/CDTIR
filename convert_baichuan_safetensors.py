#!/usr/bin/env python3
"""Convert Baichuan2-7B-Chat's pytorch_model.bin (pickle) into model.safetensors.

Usage:python convert_baichuan_safetensors.py <hf_cache_model_dir>
Note: torch.load into memory, delete the .bin to free space, then write safetensors, avoiding the torch<2.6 pickle safety check.
"""
import os
import sys
import torch
from safetensors.torch import save_file


def convert(base: str) -> None:
    snap = os.path.join(base, "snapshots", os.listdir(os.path.join(base, "snapshots"))[0])
    bin_link = os.path.join(snap, "pytorch_model.bin")
    bin_real = os.path.realpath(bin_link)
    out_path = os.path.join(snap, "model.safetensors")

    print("Loading bin:", bin_real, flush=True)
    sd = torch.load(bin_real, map_location="cpu", weights_only=False)
    clean = {k: v.contiguous().cpu() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    print("Deleting bin blob to free space...", flush=True)
    os.remove(bin_real)
    if os.path.lexists(bin_link):
        os.remove(bin_link)
    print("Writing safetensors:", out_path, flush=True)
    save_file(clean, out_path)
    print("DONE size:", os.path.getsize(out_path), flush=True)


if __name__ == "__main__":
    convert(sys.argv[1])
