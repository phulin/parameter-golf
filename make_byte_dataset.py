"""Convert fineweb10B_sp1024 tokenized shards to raw byte shards.

Each BOS-delimited document is decoded from sp1024 tokens back to UTF-8 text,
then written as 0x01 (BOS) + utf8_bytes. Output files are flat binary with no
header, readable by load_raw_bytes_shard() in train_hybrid.py.

Usage:
    python make_byte_dataset.py [src_dir [dst_dir [tokenizer_path]]]

Defaults:
    src_dir   = ./data/datasets/fineweb10B_sp1024
    dst_dir   = ./data/datasets/fineweb10B_bytes
    tokenizer = ./data/tokenizers/fineweb_1024_bpe.model
"""

import glob
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

BOS_ID = 1
DEFAULT_SRC = "./data/datasets/fineweb10B_sp1024"
DEFAULT_DST = "./data/datasets/fineweb10B_bytes"
DEFAULT_TOKENIZER = "./data/tokenizers/fineweb_1024_bpe.model"


def convert_shard(src: Path, dst: Path, sp: spm.SentencePieceProcessor) -> tuple[int, int]:
    """Decode a tokenized shard to a raw byte shard. Returns (num_docs, num_bytes)."""
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(src, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header in {src}")
    num_tokens = int(header[2])
    tokens = np.fromfile(src, dtype="<u2", count=num_tokens, offset=header_bytes)

    bos_positions = np.where(tokens == BOS_ID)[0]
    if len(bos_positions) == 0:
        print(f"  WARNING: no BOS found in {src.name}, writing empty shard")
        dst.write_bytes(b"")
        return 0, 0

    out_chunks: list[bytes] = []
    for i, bos_pos in enumerate(bos_positions):
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else len(tokens)
        doc_tokens = tokens[int(bos_pos) + 1 : end].tolist()
        text = sp.decode(doc_tokens)
        out_chunks.append(b"\x01" + text.encode("utf-8"))

    out_bytes = b"".join(out_chunks)
    dst.write_bytes(out_bytes)
    return len(bos_positions), len(out_bytes)


def main() -> None:
    src_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC)
    dst_dir = Path(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DST)
    tokenizer_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_TOKENIZER

    dst_dir.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    src_files = sorted(src_dir.glob("fineweb_*.bin"))
    if not src_files:
        raise FileNotFoundError(f"No fineweb_*.bin files found in {src_dir}")
    print(f"Converting {len(src_files)} shards: {src_dir} -> {dst_dir}")

    total_docs = 0
    total_bytes = 0
    for src in src_files:
        dst = dst_dir / src.name
        ndocs, nbytes = convert_shard(src, dst, sp)
        total_docs += ndocs
        total_bytes += nbytes
        print(f"  {src.name}: {ndocs:,} docs, {nbytes:,} bytes")

    print(f"\nDone: {total_docs:,} docs, {total_bytes / 1e9:.2f} GB total")


if __name__ == "__main__":
    main()
