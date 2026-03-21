"""
Evaluate a final_model.pt from train_recursive.py at multiple loop counts (no TTT).

Usage:
    python eval_loops.py [--model final_model.pt] [--loops 1 2 4 8] [--int8]

    --model   Path to model file. Use .int8.ptz for quantized, .pt for raw.
    --loops   List of loop counts to evaluate. Defaults to 1 2 3 4.
    --int8    Load from final_model.int8.ptz instead of final_model.pt.
"""

from __future__ import annotations

import argparse
import glob
import io
import math
import os
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Import everything we need from train_recursive
from train_recursive import (
    GPT,
    CastedLinear,
    Rotary,
    Hyperparameters,
    build_sentencepiece_luts,
    load_validation_tokens,
    eval_val,
    restore_low_dim_params_to_fp32,
    dequantize_state_dict_int8,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate final_model at multiple loop counts")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (.pt or .int8.ptz). "
                             "Defaults to final_model.int8.ptz if it exists, else final_model.pt.")
    parser.add_argument("--loops", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Loop counts to evaluate (default: 1 2 3 4)")
    parser.add_argument("--int8", action="store_true",
                        help="Force loading from final_model.int8.ptz")
    return parser.parse_args()


def load_model(model_path: str, args: Hyperparameters, device: torch.device) -> GPT:
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
        if isinstance(module, Rotary):
            module.inv_freq.data = module.inv_freq.data.float()
    restore_low_dim_params_to_fp32(base_model)

    if model_path.endswith(".ptz"):
        print(f"Loading quantized model from {model_path}")
        with open(model_path, "rb") as f:
            blob = f.read()
        state = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
        base_model.load_state_dict(dequantize_state_dict_int8(state), strict=True)
    else:
        print(f"Loading raw model from {model_path}")
        state = torch.load(model_path, map_location="cpu")
        base_model.load_state_dict(state, strict=True)

    return base_model


def main():
    cli = parse_args()
    args = Hyperparameters()
    device = torch.device("cuda")

    # Resolve model path
    if cli.model is not None:
        model_path = cli.model
    elif cli.int8 or os.path.exists("final_model.int8.ptz"):
        model_path = "final_model.int8.ptz"
    else:
        model_path = "final_model.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    base_model = load_model(model_path, args, device)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    # Single-rank setup (no DDP)
    rank = 0
    world_size = 1
    grad_accum_steps = max(1, args.val_batch_size // (world_size * args.train_seq_len))

    print(f"\n{'loops':>6}  {'val_loss':>10}  {'val_bpb':>10}")
    print("-" * 32)
    for num_loops in cli.loops:
        torch.cuda.synchronize()
        val_loss, val_bpb = eval_val(
            args,
            compiled_model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            num_loops=num_loops,
        )
        torch.cuda.synchronize()
        print(f"{num_loops:>6}  {val_loss:>10.4f}  {val_bpb:>10.4f}")


if __name__ == "__main__":
    main()
