"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import importlib.metadata
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from wandb_utils import finish_wandb, hyperparameters_to_config, init_wandb, log_wandb, update_summary

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 1 shared depth-recurrent transformer block at width 512
# - 8 attention heads with 4 KV heads (GQA), depth attention, and expert attention
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap


class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    # Retained for config compatibility; the model below always uses a single shared block.
    num_layers = int(os.environ.get("NUM_LAYERS", 1))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    num_experts = int(os.environ.get("NUM_EXPERTS", 8))
    num_active_experts = int(os.environ.get("NUM_ACTIVE_EXPERTS", 2))
    expert_balance_lambda = float(os.environ.get("EXPERT_BALANCE_LAMBDA", 1e-3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    num_recurrent_steps = int(os.environ.get("NUM_RECURRENT_STEPS", 3))
    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Test-time training (LoRA) hyperparameters.
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 64))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    is_control = getattr(sp, "is_control")
    is_unknown = getattr(sp, "is_unknown")
    is_unused = getattr(sp, "is_unused")
    is_byte = getattr(sp, "is_byte")
    id_to_piece = getattr(sp, "id_to_piece")
    for token_id in range(sp_vocab_size):
        if is_control(token_id) or is_unknown(token_id) or is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = cast(str, id_to_piece(token_id))
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_logits = model(x)
                batch_loss = loss_fn(batch_logits, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,depth_attn_scale,depth_attn_scales,expert_scale,expert_scales,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(
    name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]
) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(
            torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None]
        )
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = (
            torch.clamp(torch.round(clipped / scale[:, None]), -127, 127)
            .to(torch.int8)
            .contiguous()
        )
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = (
        float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item())
        if t32.numel()
        else 0.0
    )
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = (
        torch.clamp(
            torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127
        )
        .to(torch.int8)
        .contiguous()
    )
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = cast(dict[str, dict[str, object]], obj.get("qmeta", {}))
    passthrough_orig_dtypes = cast(
        dict[str, str], obj.get("passthrough_orig_dtypes", {})
    )
    quantized = cast(dict[str, Tensor], obj["quantized"])
    dtypes = cast(dict[str, str], obj["dtypes"])
    scales = cast(dict[str, Tensor], obj["scales"])
    passthrough = cast(dict[str, Tensor], obj["passthrough"])
    for name, q in quantized.items():
        dtype = getattr(torch, dtypes[name])
        s = scales[name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (
                (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1))))
                .to(dtype=dtype)
                .contiguous()
            )
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in passthrough.items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes"
        )
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(
        self, global_tokens: int, seq_len: int, grad_accum_steps: int
    ) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    _zero_init: bool

    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, input: Tensor) -> Tensor:
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, self.weight.to(input.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2
                or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Precomputes RoPE tables up to a fixed maximum length to stay torch.compile-friendly.
    cos: Tensor
    sin: Tensor

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.max_seq_len = max_seq_len
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds configured max_seq_len={self.max_seq_len}"
            )
        return (
            self.cos[:, :, :seq_len, :].to(device=device, dtype=dtype),
            self.sin[:, :, :seq_len, :].to(device=device, dtype=dtype),
        )


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, max_seq_len=max_seq_len, base=rope_base)

    def forward(self, x: Tensor, q_delta=None, v_delta=None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x) + (q_delta if q_delta is not None else 0)
        k = self.c_k(x)
        v = self.c_v(x) + (v_delta if v_delta is not None else 0)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class DepthSelfAttention(nn.Module):
    depth_cos: Tensor
    depth_sin: Tensor
    depth_rev_cos: Tensor
    depth_rev_sin: Tensor

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_depth: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.max_depth = max_depth
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        inv_freq = 1.0 / (
            rope_base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        depth_positions = torch.arange(max_depth, dtype=torch.float32)
        reverse_positions = torch.arange(max_depth - 1, -1, -1, dtype=torch.float32)
        forward_freqs = torch.outer(depth_positions, inv_freq)
        reverse_freqs = torch.outer(reverse_positions, inv_freq)
        self.register_buffer(
            "depth_cos", forward_freqs.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "depth_sin", forward_freqs.sin()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "depth_rev_cos", reverse_freqs.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "depth_rev_sin", reverse_freqs.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x: Tensor, depth_history: list[Tensor]) -> Tensor:
        bsz, seqlen, dim = x.shape
        kv_source = torch.stack([*depth_history, x], dim=2)
        depth = kv_source.size(2)
        q = self.c_q(x).reshape(bsz * seqlen, 1, self.num_heads, self.head_dim)
        k = self.c_k(kv_source).reshape(
            bsz * seqlen, depth, self.num_kv_heads, self.head_dim
        )
        v = self.c_v(kv_source).reshape(
            bsz * seqlen, depth, self.num_kv_heads, self.head_dim
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        half = self.head_dim // 2
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q_cos = self.depth_cos[:, :, depth - 1 : depth, :].to(device=x.device, dtype=q.dtype)
        q_sin = self.depth_sin[:, :, depth - 1 : depth, :].to(device=x.device, dtype=q.dtype)
        rq_cos = self.depth_rev_cos[:, :, depth - 1 : depth, :].to(
            device=x.device, dtype=q.dtype
        )
        rq_sin = self.depth_rev_sin[:, :, depth - 1 : depth, :].to(
            device=x.device, dtype=q.dtype
        )
        k_cos = self.depth_cos[:, :, :depth, :].to(device=x.device, dtype=k.dtype)
        k_sin = self.depth_sin[:, :, :depth, :].to(device=x.device, dtype=k.dtype)
        rk_cos = self.depth_rev_cos[:, :, :depth, :].to(device=x.device, dtype=k.dtype)
        rk_sin = self.depth_rev_sin[:, :, :depth, :].to(device=x.device, dtype=k.dtype)
        q = torch.cat((q1 * q_cos + q2 * q_sin, q1 * (-rq_sin) + q2 * rq_cos), dim=-1)
        k = torch.cat((k1 * k_cos + k2 * k_sin, k1 * (-rk_sin) + k2 * rk_cos), dim=-1)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class ExpertMLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class ExpertAttention(nn.Module):
    depth_cos: Tensor
    depth_sin: Tensor
    depth_rev_cos: Tensor
    depth_rev_sin: Tensor
    expert_bias: Tensor

    def __init__(
        self,
        dim: int,
        mlp_mult: int,
        num_experts: int,
        num_active_experts: int,
        max_depth: int,
        rope_base: float,
        balance_lambda: float = 1e-3,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if num_active_experts <= 0:
            raise ValueError(
                f"num_active_experts must be positive, got {num_active_experts}"
            )
        if num_active_experts > num_experts:
            raise ValueError(
                f"num_active_experts={num_active_experts} exceeds num_experts={num_experts}"
            )
        if dim % 2 != 0:
            raise ValueError("model_dim must be even for expert-attention RoPE")
        self.dim = dim
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.balance_lambda = balance_lambda
        self.router = CastedLinear(dim, dim, bias=False)
        self.expert_keys = nn.Parameter(torch.empty(num_experts, dim, dtype=torch.float32))
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.experts = nn.ModuleList(
            [ExpertMLP(dim, mlp_mult) for _ in range(num_experts)]
        )

        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        depth_positions = torch.arange(max_depth, dtype=torch.float32)
        reverse_positions = torch.arange(max_depth - 1, -1, -1, dtype=torch.float32)
        forward_freqs = torch.outer(depth_positions, inv_freq)
        reverse_freqs = torch.outer(reverse_positions, inv_freq)
        self.register_buffer(
            "depth_cos", forward_freqs.cos(), persistent=False
        )
        self.register_buffer(
            "depth_sin", forward_freqs.sin(), persistent=False
        )
        self.register_buffer(
            "depth_rev_cos", reverse_freqs.cos(), persistent=False
        )
        self.register_buffer(
            "depth_rev_sin", reverse_freqs.sin(), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.dim)
        with torch.no_grad():
            self.expert_keys.uniform_(-bound, bound)

    def _apply_depth_rope(self, q: Tensor, depth_idx: int) -> Tensor:
        half = self.dim // 2
        q1, q2 = q[..., :half], q[..., half:]
        q_cos = self.depth_cos[depth_idx : depth_idx + 1, :].to(
            device=q.device, dtype=q.dtype
        )
        q_sin = self.depth_sin[depth_idx : depth_idx + 1, :].to(
            device=q.device, dtype=q.dtype
        )
        rq_cos = self.depth_rev_cos[depth_idx : depth_idx + 1, :].to(
            device=q.device, dtype=q.dtype
        )
        rq_sin = self.depth_rev_sin[depth_idx : depth_idx + 1, :].to(
            device=q.device, dtype=q.dtype
        )
        return torch.cat(
            (q1 * q_cos + q2 * q_sin, q1 * (-rq_sin) + q2 * rq_cos), dim=-1
        )

    def forward(self, x: Tensor, depth_idx: int) -> Tensor:
        bsz, seqlen, dim = x.shape
        flat_x = x.reshape(bsz * seqlen, dim)
        q = self.router(flat_x)
        q = F.rms_norm(q, (q.size(-1),))
        q = self._apply_depth_rope(q, depth_idx)
        logits = (q @ self.expert_keys.to(dtype=q.dtype).T) / math.sqrt(dim)
        gate_values = torch.sigmoid(logits)
        biased_logits = logits + self.expert_bias.to(dtype=logits.dtype)
        _, topk_idx = torch.topk(biased_logits, k=self.num_active_experts, dim=-1)
        topk_gates = gate_values.gather(-1, topk_idx)
        topk_gates = (topk_gates / topk_gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)).to(
            dtype=gate_values.dtype
        )
        gates = torch.zeros_like(gate_values)
        gates.scatter_(-1, topk_idx, topk_gates)
        if self.training and self.balance_lambda > 0:
            expert_counts = gates.gt(0).float().sum(dim=0)
            delta = self.balance_lambda * (expert_counts.median() - expert_counts).sign()
            self.expert_bias.add_(delta.to(dtype=self.expert_bias.dtype))

        expert_outputs = torch.stack(
            [expert(flat_x) for expert in self.experts], dim=1
        )
        y = (expert_outputs * gates.to(dtype=expert_outputs.dtype).unsqueeze(-1)).sum(
            dim=1
        )
        return y.reshape(bsz, seqlen, dim)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        num_experts: int,
        num_active_experts: int,
        max_seq_len: int,
        max_depth: int,
        rope_base: float,
        qk_gain_init: float,
        expert_balance_lambda: float = 1e-3,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.depth_attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, max_seq_len, rope_base, qk_gain_init
        )
        self.depth_attn = DepthSelfAttention(
            dim, num_heads, num_kv_heads, max_depth, rope_base, qk_gain_init
        )
        self.expert_attn = ExpertAttention(
            dim, mlp_mult, num_experts, num_active_experts, max_depth, rope_base,
            balance_lambda=expert_balance_lambda,
        )
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.depth_attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.expert_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(
        self, x: Tensor, depth_history: list[Tensor], q_delta_fn=None, v_delta_fn=None
    ) -> tuple[Tensor, list[Tensor]]:
        seq_in = self.attn_norm(x)
        qd = q_delta_fn(seq_in) if q_delta_fn is not None else None
        vd = v_delta_fn(seq_in) if v_delta_fn is not None else None
        attn_out = self.attn(seq_in, qd, vd)
        depth_in = self.depth_attn_norm(x)
        depth_attn_out = self.depth_attn(depth_in, depth_history)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = (
            x
            + self.depth_attn_scale.to(dtype=x.dtype)[None, None, :]
            * depth_attn_out
        )
        x = x + self.expert_scale.to(dtype=x.dtype)[None, None, :] * self.expert_attn(
            self.mlp_norm(x), len(depth_history)
        )
        return F.rms_norm(x, (x.size(-1),)), [*depth_history, depth_in]


class RecursiveGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        num_experts: int,
        num_active_experts: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        max_seq_len: int,
        rope_base: float,
        qk_gain_init: float,
        num_recurrent_steps: int,
        expert_balance_lambda: float = 1e-3,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if num_recurrent_steps <= 0:
            raise ValueError(
                f"num_recurrent_steps must be positive, got {num_recurrent_steps}"
            )
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_recurrent_steps = num_recurrent_steps
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_layers = 1
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    num_experts,
                    num_active_experts,
                    max_seq_len,
                    num_recurrent_steps,
                    rope_base,
                    qk_gain_init,
                    expert_balance_lambda=expert_balance_lambda,
                )
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _compute_logits(self, y: Tensor, lora=None) -> Tensor:
        y = self.final_norm(y)
        if self.tie_embeddings:
            logits = F.linear(y, self.tok_emb.weight)
        else:
            assert self.lm_head is not None
            logits = self.lm_head(y)
        logits = logits + (lora.lm_head_lora(y) if lora else 0)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def recurrent_steps(
        self, input_ids: Tensor, y: Tensor | None = None, lora=None
    ) -> tuple[list[Tensor], Tensor]:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if y is not None:
            x = F.rms_norm(x + y.to(dtype=x.dtype), (x.size(-1),))

        depth_history: list[Tensor] = []
        step_logits: list[Tensor] = []
        for _ in range(self.num_recurrent_steps):
            qd = lora.q_loras[0] if lora else None
            vd = lora.v_loras[0] if lora else None
            x, depth_history = self.blocks[0](x, depth_history, qd, vd)
            step_logits.append(self._compute_logits(x, lora))
        return step_logits, x

    def forward(
        self,
        input_ids: Tensor,
        y: Tensor | None = None,
        lora=None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        step_logits, y = self.recurrent_steps(input_ids, y, lora)
        logits = step_logits[-1]
        if return_state:
            return logits, y.detach()
        return logits


def loss_fn(logits, target_ids, lora=None):
    if lora:
        bsz, sl, V = logits.shape
        return F.cross_entropy(
            logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none"
        ).reshape(bsz, sl)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction="mean",
    )


def weighted_step_loss(step_logits: list[Tensor], target_ids: Tensor) -> Tensor:
    weights = torch.arange(
        1,
        len(step_logits) + 1,
        device=step_logits[0].device,
        dtype=torch.float32,
    )
    weights = weights / weights.sum()
    losses = torch.stack(
        [loss_fn(logits, target_ids).to(torch.float32) for logits in step_logits]
    )
    return (losses * weights).sum().to(dtype=step_logits[0].dtype)


# -----------------------------
# TEST-TIME TRAINING (LoRA)
# -----------------------------
#
# At evaluation time, we adapt per-document low-rank adapters on the validation data.
# Each document gets its own adapter, so there is no inter-document dependency.

BOS_ID = 1


class BatchedLinearLoRA(nn.Module):
    """LoRA for a linear layer, with independent weights per batch element.
    Computes x @ Aᵀ @ Bᵀ  =  x @ (BA)ᵀ,  i.e. the LoRA delta is ΔW = BA."""

    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))  # down-projection
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))  # up-projection
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)  # (bsz, T, out)

    def reset(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)  # kaiming-uniform
            self.B.zero_()


class BatchedTTTLoRA(nn.Module):
    """All LoRA adapters for one batch: LM head and Q/V per block."""

    def __init__(self, bsz: int, model: RecursiveGPT, rank: int):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for module in model.blocks:
            block = cast(Block, module)
            self.q_loras.append(
                BatchedLinearLoRA(bsz, dim, block.attn.c_q.out_features, rank)
            )
            self.v_loras.append(
                BatchedLinearLoRA(bsz, dim, block.attn.c_v.out_features, rank)
            )

    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()


def _reset_ttt_optimizer(opt):
    for group in opt.param_groups:
        for p in group["params"]:
            s = opt.state.get(p)
            if not s:  # Fresh state.
                continue
            s["exp_avg"].zero_()
            s["exp_avg_sq"].zero_()
            s["step"].fill_(0)


def _build_ttt_optimizer(lora, args: Hyperparameters):
    return torch.optim.AdamW(
        lora.parameters(),
        lr=args.ttt_lora_lr,
        betas=(args.beta1, args.beta2),
        eps=1e-10,
    )


def _find_docs(
    all_tokens: Tensor, include_next_bos: bool = True
) -> list[tuple[int, int]]:
    """Return (start_offset, length) for each document, identified by BOS boundaries.

    If include_next_bos is True, include next document's BOS (to match continuous-stream
    eval token count exactly).
    """
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = (
            int(bos_positions[i + 1])
            if i + 1 < len(bos_positions)
            else all_tokens.numel()
        )
        if include_next_bos and i + 1 < len(bos_positions):
            end += 1
        assert end - start >= 2
        docs.append((start, end - start))
    return docs


def _compute_chunk_window(
    ci: int, pred_len: int, num_chunks: int, chunk_size: int, eval_seq_len: int
):
    """Return (win_start, win_len, chunk_offset, chunk_len) for chunk `ci` of a doc."""
    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


def _accumulate_bpb(
    ptl: Tensor,
    x: Tensor,
    y: Tensor,
    batch_i: int,
    chunk_offset: int,
    chunk_len: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    loss_sum: Tensor,
    byte_sum: Tensor,
    token_count: Tensor,
):
    """Add one doc-chunk's contribution to the running BPB accumulators."""
    lbl = ptl[batch_i, chunk_offset : chunk_offset + chunk_len].to(torch.float64)
    prev = x[batch_i, chunk_offset : chunk_offset + chunk_len]
    tgt = y[batch_i, chunk_offset : chunk_offset + chunk_len]
    tok_bytes = base_bytes_lut[tgt].to(torch.float64)
    tok_bytes += has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]
    loss_sum += lbl.sum()
    byte_sum += tok_bytes.sum()
    token_count += chunk_len


def eval_val_ttt_lora(
    args: Hyperparameters,
    base_model: RecursiveGPT,
    rank: int,
    world_size: int,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Evaluate with batched LoRA test-time training. Returns (val_loss, val_bpb)."""
    # Load validation tokens and find document boundaries
    files = sorted(glob.glob(args.val_files))
    all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs = _find_docs(all_tokens)

    # Each rank takes a contiguous slice of documents
    rank_docs = docs[
        (len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size
    ]
    chunk_size = args.ttt_chunk_size
    eval_seq_len = args.ttt_eval_seq_len
    batch_size = args.ttt_batch_size
    lora_rank = args.ttt_lora_rank

    rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    lora = BatchedTTTLoRA(batch_size, base_model, lora_rank).to(device)
    opt = _build_ttt_optimizer(lora, args)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    for bi in range(0, len(rank_docs), batch_size):
        batch = rank_docs[bi : bi + batch_size]
        bsz = len(batch)

        if bsz == batch_size:
            cur_lora, cur_opt = lora, opt
            cur_lora.reset()
            _reset_ttt_optimizer(cur_opt)
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, lora_rank).to(device)
            cur_opt = _build_ttt_optimizer(cur_lora, args)

        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)

        for ci in range(max_nc):
            chunk_stats = _compute_chunk_window(
                ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len
            )
            context_size, chunk_offset = chunk_stats[1], chunk_stats[2]

            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc - 1 for nc in num_chunks)

            x = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            doc_info = []  # (chunk_offset, chunk_len) per doc
            for b in range(bsz):
                if not active[b]:
                    doc_info.append((0, 0))
                    continue
                ds, dl = batch[b]
                ws, wl, co, cl = _compute_chunk_window(
                    ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len
                )
                chunk = all_tokens[ds + ws : ds + ws + wl + 1]
                toks = chunk.to(dtype=torch.int64, device=device)
                x[b, :wl] = toks[:-1]
                y[b, :wl] = toks[1:]
                doc_info.append((co, cl))

            # Forward pass (keep grad graph alive only when we need to train)
            if needs_train:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = loss_fn(base_model(x, lora=cur_lora), y, lora=cur_lora)
            else:
                with (
                    torch.no_grad(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    ptl = loss_fn(base_model(x, lora=cur_lora), y, lora=cur_lora)

            # Score: accumulate loss and byte counts for BPB (before training on chunk)
            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]:
                        continue
                    co, cl = doc_info[b]
                    _accumulate_bpb(
                        ptl,
                        x,
                        y,
                        b,
                        co,
                        cl,
                        base_bytes_lut,
                        has_leading_space_lut,
                        is_boundary_token_lut,
                        loss_sum,
                        byte_sum,
                        token_count,
                    )

            # Train: one Adam step on the LoRA params using this chunk's loss
            if needs_train:
                mask = torch.tensor(
                    [float(ci < num_chunks[b] - 1) for b in range(bsz)], device=device
                )
                per_doc = ptl[:, chunk_offset : chunk_offset + chunk_size].mean(dim=-1)
                cur_opt.zero_grad()
                (per_doc * mask).sum().backward()
                cur_opt.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    return val_loss, val_bpb


# -----------------------------
# TRAINING
# -----------------------------


def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = cast(Any, torch.compile(zeropower_via_newtonschulz5))

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral"
        )
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(
            f"Script only setup for SentencePiece .model file: {args.tokenizer_path}"
        )
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )
    log0(
        f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}"
    )
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = (
        RecursiveGPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            num_experts=args.num_experts,
            num_active_experts=args.num_active_experts,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            max_seq_len=args.train_seq_len,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            num_recurrent_steps=args.num_recurrent_steps,
            expert_balance_lambda=args.expert_balance_lambda,
        )
        .to(device)
        .bfloat16()
    )
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = cast(
        nn.Module, torch.compile(base_model, dynamic=False, fullgraph=True)
    )
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )
    ddp_model = cast(DDP | None, model if distributed else None)
    train_model = cast(
        RecursiveGPT, ddp_model.module if ddp_model is not None else model
    )

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2
        or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=1e-3,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=1e-3,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [
        optimizer_tok,
        optimizer_muon,
        optimizer_scalar,
    ]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [
                {
                    "params": [base_model.lm_head.weight],
                    "lr": args.head_lr,
                    "base_lr": args.head_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}"
    )
    log0(
        f"expert_attention:num_experts:{args.num_experts} "
        f"num_active_experts:{args.num_active_experts} expert_hidden_mult:{args.mlp_mult}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    if args.num_layers != 1:
        log0(f"note: NUM_LAYERS={args.num_layers} ignored; using one shared recurrent block")
    log0(f"num_recurrent_steps:{args.num_recurrent_steps}")
    log0(f"seed:{args.seed}")
    wandb_run = None
    if master_process:
        wandb_run = init_wandb(
            run_id=args.run_id,
            backend="pytorch",
            config=hyperparameters_to_config(args),
            extra_config={
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "wandb_backend": "pytorch",
                "distributed": distributed,
                "world_size": world_size,
                "grad_accum_steps": grad_accum_steps,
                "dataset_name": dataset_dir.name,
                "actual_train_files": actual_train_files,
                "val_tokens": int(val_tokens.numel() - 1),
                "tokenizer_path": args.tokenizer_path,
                "train_files_glob": args.train_files,
                "val_files_glob": args.val_files,
                "n_params": int(n_params),
                "pytorch_version": importlib.metadata.version("torch"),
                "logfile": logfile,
            },
        )
        update_summary(
            wandb_run,
            {
                "dataset/name": dataset_dir.name,
                "dataset/train_shards": actual_train_files,
                "dataset/val_tokens": int(val_tokens.numel() - 1),
                "model/n_params": int(n_params),
                "runtime/logfile": logfile,
            },
        )

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return (
            remaining_ms / max(warmdown_ms, 1e-9)
            if remaining_ms <= warmdown_ms
            else 1.0
        )

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    assert ddp_model is not None
                    ddp_model.require_backward_grad_sync = (
                        micro_step == grad_accum_steps - 1
                    )
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    warmup_step_logits, _ = train_model.recurrent_steps(x)
                    warmup_loss = weighted_step_loss(warmup_step_logits, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if (
                args.warmup_steps <= 20
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == args.warmup_steps
            ):
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            assert ddp_model is not None
            ddp_model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device
        )

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )

        should_validate = last_step or (
            args.val_loss_every > 0 and step % args.val_loss_every == 0
        )
        if should_validate:
            if last_step:
                if stop_after_step is not None and step < args.iterations:
                    log0(
                        f"finalizing: reached wallclock cap, running final validation at step:{step}"
                    )
                else:
                    log0(f"finalizing: running terminal validation at step:{step}")
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            log_wandb(
                wandb_run,
                {
                    "val/loss": val_loss,
                    "val/bpb": val_bpb,
                    "train/time_ms": training_time_ms,
                    "train/step_avg_ms": training_time_ms / max(step, 1),
                },
                step=step,
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()

        batch_xy = [
            train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            for _ in range(grad_accum_steps)
        ]
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                assert ddp_model is not None
                ddp_model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps - 1
                )
            x, y_true = batch_xy[micro_step]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                step_logits, _ = train_model.recurrent_steps(x)
                loss = weighted_step_loss(step_logits, y_true)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = (
            min(step / args.muon_momentum_warmup_steps, 1.0)
            if args.muon_momentum_warmup_steps > 0
            else 1.0
        )
        muon_momentum = (
            1 - frac
        ) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (
            step <= 10
            or step % args.train_log_every == 0
            or stop_after_step is not None
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
            log_wandb(
                wandb_run,
                {
                    "train/loss": train_loss.item(),
                    "train/time_ms": approx_training_time_ms,
                    "train/step_avg_ms": approx_training_time_ms / step,
                    "train/lr_scale": scale,
                    "optimizer/muon_momentum": muon_momentum,
                },
                step=step,
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = (
            max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        )
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
            log0(
                f"wallclock_cap_reached: step:{step} "
                f"train_time:{approx_training_time_ms:.0f}ms entering finalization"
            )

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        log0("finalizing: serializing raw model")
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")
        update_summary(
            wandb_run,
            {
                "artifact/raw_model_bytes": model_bytes,
                "artifact/code_bytes": code_bytes,
                "artifact/raw_total_bytes": model_bytes + code_bytes,
            },
        )

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        log0("finalizing: writing int8+zlib artifact")
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(
            quant_stats["int8_payload_bytes"], 1
        )
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")
        update_summary(
            wandb_run,
            {
                "artifact/int8_zlib_bytes": quant_file_bytes,
                "artifact/int8_payload_bytes": quant_stats["int8_payload_bytes"],
                "artifact/int8_raw_torch_bytes": quant_raw_bytes,
                "artifact/int8_payload_ratio": ratio,
                "artifact/int8_total_bytes": quant_file_bytes + code_bytes,
            },
        )

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu"
    )
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    log0("finalizing: running int8 roundtrip validation")
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(
        f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}"
    )
    update_summary(
        wandb_run,
        {
            "final/int8_roundtrip_val_loss": q_val_loss,
            "final/int8_roundtrip_val_bpb": q_val_bpb,
        },
    )

    # LoRA test-time training evaluation (the competition score)
    torch._dynamo.reset()
    torch.cuda.synchronize()
    log0("finalizing: running TTT LoRA evaluation")
    t_ttt = time.perf_counter()
    ttt_val_loss, ttt_val_bpb = eval_val_ttt_lora(
        args,
        base_model,
        rank,
        world_size,
        device,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_ttt_lora val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
    )
    update_summary(
        wandb_run,
        {
            "final/ttt_lora_val_loss": ttt_val_loss,
            "final/ttt_lora_val_bpb": ttt_val_bpb,
        },
    )
    finish_wandb(wandb_run)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
