#!/usr/bin/env python3
"""Count one bits in reconstructed quantized linear weights.

This script targets PrefixQuant fake-quantized checkpoints. The saved weights are
floating point tensors after quantize/dequantize; for each quantized linear layer
we reconstruct integer codes from weight / scale, then count one bits. By default
signed negative values are counted by magnitude, so -1 contributes the same one
bit as +1 instead of eight one bits in int8 two's-complement.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

LAYER_WEIGHT_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)\.weight$")


@dataclass
class BitStats:
    one_bits: int = 0
    total_bits: int = 0
    values: int = 0

    @property
    def one_ratio(self) -> float:
        return self.one_bits / self.total_bits if self.total_bits else float("nan")

    def add(self, one_bits: int, total_bits: int, values: int) -> None:
        self.one_bits += int(one_bits)
        self.total_bits += int(total_bits)
        self.values += int(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count bit=1 ratio in reconstructed quantized linear weights."
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Quantized model directory containing model.safetensors.index.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <model_dir>/bit_stats.",
    )
    parser.add_argument(
        "--storage-bits",
        type=int,
        default=8,
        help="Binary width used as the denominator, e.g. 8 for int8.",
    )
    parser.add_argument(
        "--count-mode",
        choices=("magnitude", "twos-complement"),
        default="magnitude",
        help=(
            "How to count one bits for signed negative codes. 'magnitude' counts "
            "popcount(abs(q)); 'twos-complement' counts the storage representation."
        ),
    )
    parser.add_argument(
        "--code-bits",
        type=int,
        default=None,
        help="Quantizer bit width. Defaults to prefixequant_config.json wbits.",
    )
    parser.add_argument(
        "--signed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether quantized codes are signed. Defaults to not w_asym from config.",
    )
    parser.add_argument(
        "--include-lm-head",
        action="store_true",
        help="Also include lm_head.weight if it has a matching weight_quantizer.scale.",
    )
    parser.add_argument(
        "--module-regex",
        default=None,
        help="Only include module names matching this regex, e.g. 'self_attn|mlp'.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_popcount_lut(storage_bits: int) -> torch.Tensor:
    if storage_bits <= 0 or storage_bits > 16:
        raise ValueError("storage_bits must be in [1, 16]")
    return torch.tensor([int(i).bit_count() for i in range(1 << storage_bits)], dtype=torch.int16)


class ShardedSafetensors:
    def __init__(self, model_dir: Path, weight_map: dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._open_shards = {}

    def get_tensor(self, key: str) -> torch.Tensor:
        shard_name = self.weight_map[key]
        if shard_name not in self._open_shards:
            self._open_shards[shard_name] = safe_open(
                str(self.model_dir / shard_name), framework="pt", device="cpu"
            )
        return self._open_shards[shard_name].get_tensor(key)


def iter_quantized_weight_keys(
    weight_map: dict[str, str], include_lm_head: bool, module_re: re.Pattern | None
) -> Iterable[tuple[int | str, str, str, str, str | None]]:
    suffix = ".weight"
    for key in sorted(weight_map):
        if not key.endswith(suffix):
            continue
        base = key[: -len(suffix)]
        scale_key = f"{base}.weight_quantizer.scale"
        if scale_key not in weight_map:
            continue
        zero_key = f"{base}.weight_quantizer.zero_point"
        if zero_key not in weight_map:
            zero_key = None

        m = LAYER_WEIGHT_RE.match(key)
        if m:
            layer: int | str = int(m.group(1))
            module = m.group(2)
        elif include_lm_head and key == "lm_head.weight":
            layer = "lm_head"
            module = "lm_head"
        else:
            continue

        if module_re is not None and not module_re.search(module):
            continue
        yield layer, module, key, scale_key, zero_key


def reconstruct_integer_codes(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    code_bits: int,
    signed: bool,
) -> torch.Tensor:
    if signed:
        qmin = -(1 << (code_bits - 1))
        qmax = (1 << (code_bits - 1)) - 1
    else:
        qmin = 0
        qmax = (1 << code_bits) - 1

    x = weight.float().reshape(-1, weight.shape[-1])
    s = scale.float()
    q = torch.round(x / s)
    if zero_point is not None:
        q = q + torch.round(zero_point.float())
    return q.clamp(qmin, qmax).to(torch.int32)


def count_one_bits(
    codes: torch.Tensor, storage_bits: int, lut: torch.Tensor, count_mode: str
) -> tuple[int, int, int]:
    if count_mode == "magnitude":
        unsigned_codes = torch.abs(codes).to(torch.long)
        max_code = int(unsigned_codes.max().item()) if unsigned_codes.numel() else 0
        if max_code >= (1 << storage_bits):
            raise ValueError(
                f"abs(code)={max_code} does not fit in storage_bits={storage_bits}"
            )
    elif count_mode == "twos-complement":
        mask = (1 << storage_bits) - 1
        unsigned_codes = torch.bitwise_and(codes, mask).to(torch.long)
    else:
        raise ValueError(f"Unsupported count_mode: {count_mode}")
    one_bits = lut[unsigned_codes].sum().item()
    values = codes.numel()
    return int(one_bits), int(values * storage_bits), int(values)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    out_dir = args.out_dir or model_dir / "bit_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(model_dir / "prefixequant_config.json")
    index = load_json(model_dir / "model.safetensors.index.json")
    weight_map = index["weight_map"]

    code_bits = args.code_bits if args.code_bits is not None else int(config["wbits"])
    signed = args.signed if args.signed is not None else not bool(config.get("w_asym", False))
    storage_bits = args.storage_bits
    if code_bits > storage_bits:
        raise ValueError("code_bits cannot exceed storage_bits for this binary interpretation")

    module_re = re.compile(args.module_regex) if args.module_regex else None
    lut = build_popcount_lut(storage_bits)
    tensors = ShardedSafetensors(model_dir, weight_map)

    detail_rows: list[dict] = []
    layer_totals: dict[int | str, BitStats] = defaultdict(BitStats)
    total = BitStats()

    for layer, module, weight_key, scale_key, zero_key in iter_quantized_weight_keys(
        weight_map, args.include_lm_head, module_re
    ):
        weight = tensors.get_tensor(weight_key)
        scale = tensors.get_tensor(scale_key)
        zero_point = tensors.get_tensor(zero_key) if zero_key is not None else None
        codes = reconstruct_integer_codes(weight, scale, zero_point, code_bits, signed)
        one_bits, total_bits, values = count_one_bits(codes, storage_bits, lut, args.count_mode)
        ratio = one_bits / total_bits

        detail_rows.append(
            {
                "layer": layer,
                "module": module,
                "weight_key": weight_key,
                "shape": "x".join(str(x) for x in weight.shape),
                "code_bits": code_bits,
                "storage_bits": storage_bits,
                "count_mode": args.count_mode,
                "signed": signed,
                "int_values": values,
                "one_bits": one_bits,
                "total_bits": total_bits,
                "one_ratio": ratio,
            }
        )
        layer_totals[layer].add(one_bits, total_bits, values)
        total.add(one_bits, total_bits, values)

    def layer_sort_key(item: tuple[int | str, BitStats]) -> tuple[int, int | str]:
        layer = item[0]
        return (0, layer) if isinstance(layer, int) else (1, layer)

    layer_rows = []
    for layer, stats in sorted(layer_totals.items(), key=layer_sort_key):
        layer_rows.append(
            {
                "layer": layer,
                "code_bits": code_bits,
                "storage_bits": storage_bits,
                "count_mode": args.count_mode,
                "signed": signed,
                "int_values": stats.values,
                "one_bits": stats.one_bits,
                "total_bits": stats.total_bits,
                "one_ratio": stats.one_ratio,
            }
        )

    summary = {
        "model_dir": str(model_dir),
        "code_bits": code_bits,
        "storage_bits": storage_bits,
        "count_mode": args.count_mode,
        "signed": signed,
        "num_modules": len(detail_rows),
        "num_layers": len(layer_rows),
        "total": {
            "int_values": total.values,
            "one_bits": total.one_bits,
            "total_bits": total.total_bits,
            "one_ratio": total.one_ratio,
        },
    }

    detail_path = out_dir / "quantized_weight_bit_ones_by_module.csv"
    layer_path = out_dir / "quantized_weight_bit_ones_by_layer.csv"
    summary_path = out_dir / "quantized_weight_bit_ones_summary.json"
    write_csv(detail_path, detail_rows)
    write_csv(layer_path, layer_rows)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"modules: {len(detail_rows)}")
    print(f"layers: {len(layer_rows)}")
    print(f"total one ratio: {total.one_ratio:.8f} ({total.one_ratio * 100:.4f}%)")
    print(f"by layer: {layer_path}")
    print(f"by module: {detail_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
