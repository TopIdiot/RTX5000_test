#!/usr/bin/env python3
"""Benchmark only the WeLMv4 80B kernels selected for optimization.

Run from an SGLang checkout with PYTHONPATH pointing to its python directory.
The script is standalone and does not depend on a precision-test module.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


HIDDEN_SIZE = 2048
HEAD_DIM = 256
TP_SIZE = 4
TP4_Q_HEADS = 6
TP4_KV_HEADS = 1
NUM_EXPERTS = 512
TOPK = 10
SHARED_INTERMEDIATE_SIZE = 512
ROPE_DIM = 64
MAX_POSITION = 262144
LOCAL_WINDOW = 512
ROPE_ORIGINAL_MAX_POSITION = 32768
ROPE_SCALING_FACTOR = 8.0
ROPE_THETA = 500000
RMS_EPS = 1e-5
OE_DIM = 512
OE_VOCAB_SIZES = (16000008, 16000016, 16000024, 16000032)

DEFAULT_PEAK_BANDWIDTH_GBS = 1344.0
DEFAULT_PEAK_BF16_TFLOPS = 268.0

CASES = (
    "rms_norm",
    "k_rms_norm",
    "rope_yarn_partial",
    "attention_gate_projection",
    "attention_gate_sigmoid_mul",
    "attention_prefill_sink_local",
    "attention_prefill_sink_global",
    "o_norm",
    "router_linear",
    "expert_bias_topk",
    "shared_gate_up_projection",
    "silu_and_mul",
    "shared_down_projection",
    "oe_lookup_concat",
)


@dataclass
class Run:
    fn: Callable[[], Any]
    flops: float
    logical_bytes: float


class _FakeShard:
    def __init__(self, end: int):
        self.org_vocab_start_index = 0
        self.org_vocab_end_index = end
        self.num_org_vocab_padding = 0
        self.num_added_elements_padded = 0


class _FakeEmbed:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.shard_indices = _FakeShard(weight.shape[0])
        self.quant_method = None


_OE_TABLES: list[torch.Tensor] | None = None


def csv_positive_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(nbytes(item) for item in value)
    return 0


def randn(shape, *, dtype=torch.bfloat16, scale=1.0) -> torch.Tensor:
    return (torch.randn(shape, dtype=torch.float32, device="cuda") * scale).to(dtype)


def configure_sglang() -> None:
    from sglang.srt.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def load_ops(path: Path):
    configure_sglang()
    spec = importlib.util.spec_from_file_location("welmv4_optimized_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load kernel file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attention_scale() -> float:
    mscale = 0.1 * math.log(ROPE_SCALING_FACTOR) + 1.0
    return HEAD_DIM**-0.5 * mscale * mscale


def make_yarn_cache() -> torch.Tensor:
    from sglang.srt.layers.rotary_embedding import (
        _yarn_find_correction_range,
        _yarn_linear_ramp_mask,
        yarn_get_mscale,
    )

    inv_freq_extra = 1.0 / (
        ROPE_THETA
        ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    )
    inv_freq_inter = inv_freq_extra / ROPE_SCALING_FACTOR
    low, high = _yarn_find_correction_range(
        32.0, 1.0, ROPE_DIM, ROPE_THETA, ROPE_ORIGINAL_MAX_POSITION
    )
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
        low, high, ROPE_DIM // 2, dtype=torch.float32
    )
    inv_freq = inv_freq_inter * (1 - inv_freq_mask) + inv_freq_extra * inv_freq_mask
    positions = torch.arange(MAX_POSITION, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    mscale = yarn_get_mscale(ROPE_SCALING_FACTOR, 1.0) / yarn_get_mscale(
        ROPE_SCALING_FACTOR, 1.0
    )
    cache = torch.cat(
        (torch.cos(freqs) * mscale, torch.sin(freqs) * mscale), dim=-1
    )
    return cache.to("cuda")


def make_sink_prefill(seq: int, window: int, implementation: str) -> Run:
    q = randn((seq, TP4_Q_HEADS, HEAD_DIM))
    k = randn((seq, TP4_KV_HEADS, HEAD_DIM))
    v = randn((seq, TP4_KV_HEADS, HEAD_DIM))
    sinks = randn((TP4_Q_HEADS,))
    scale = attention_scale()

    if implementation == "triton":
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd_unified,
        )

        output = torch.empty_like(q)
        indptr = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
        indices = torch.arange(seq, dtype=torch.int64, device="cuda")
        prefix = torch.zeros(1, dtype=torch.int32, device="cuda")

        def fn():
            extend_attention_fwd_unified(
                q,
                output,
                k,
                v,
                1.0,
                1.0,
                indptr,
                indptr,
                indices,
                prefix,
                seq,
                sm_scale=scale,
                is_causal=True,
                sliding_window_size=window,
                sinks=sinks,
            )
            return output

    elif implementation == "fa4":
        from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func

        cu_seqlens = torch.tensor([0, seq], dtype=torch.int32, device="cuda")

        def fn():
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq,
                max_seqlen_k=seq,
                softmax_scale=scale,
                causal=True,
                window_size=(window, 0),
                sinks=sinks,
                pack_gqa=False,
            )

    else:
        raise ValueError(f"unsupported attention implementation: {implementation}")

    attended = min(seq, window)
    if attended < seq:
        pairs = seq * attended - attended * (attended - 1) / 2
    else:
        pairs = seq * (seq + 1) / 2
    flops = 4.0 * pairs * TP4_Q_HEADS * HEAD_DIM
    logical_bytes = nbytes(q) + nbytes(k) + nbytes(v) + nbytes(sinks) + nbytes(q)
    return Run(fn, flops, float(logical_bytes))


def make_oe_lookup(tokens: int) -> Run:
    global _OE_TABLES

    from sglang.srt.models.welm_perf_opt import (
        _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512,
    )

    if _OE_TABLES is None:
        _OE_TABLES = [
            torch.empty(
                (math.ceil(vocab / TP_SIZE), OE_DIM),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for vocab in OE_VOCAB_SIZES
        ]
    hashed = [
        torch.randint(0, table.shape[0], (tokens,), dtype=torch.int64, device="cuda")
        for table in _OE_TABLES
    ]
    modules = [_FakeEmbed(table) for table in _OE_TABLES]

    def fn():
        return _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512(
            hashed_inputs=hashed, oe_embed_modules=modules
        )

    logical_bytes = tokens * 4 * (8 + 4 * OE_DIM)
    return Run(fn, 0.0, float(logical_bytes))


def make_run(name: str, size: int, implementation: str, ops) -> Run:
    import torch.nn.functional as F

    if name in {"rms_norm", "o_norm"}:
        x = randn((size, HIDDEN_SIZE))
        weight = randn((HIDDEN_SIZE,), scale=0.1) + 1
        module = ops.WelmV4FusedRMSNorm(
            HIDDEN_SIZE, eps=RMS_EPS, weight_dtype=torch.bfloat16
        ).cuda()
        module.weight.data.copy_(weight)
        return Run(
            lambda: module.forward_cuda(x)[0],
            float(x.numel() * 5),
            float(nbytes(x) * 2 + nbytes(weight)),
        )

    if name == "k_rms_norm":
        x = randn((size, TP4_KV_HEADS, HEAD_DIM)).contiguous()
        weight = randn((HEAD_DIM,), scale=0.1) + 1
        return Run(
            lambda: ops.mmq_style_k_rms_norm(x, weight, RMS_EPS),
            float(x.numel() * 5),
            float(nbytes(x) * 2 + nbytes(weight)),
        )

    if name == "rope_yarn_partial":
        positions = torch.linspace(0, MAX_POSITION - 1, size, device="cuda").to(
            torch.int64
        )
        q = randn((size, TP4_Q_HEADS * HEAD_DIM)).contiguous()
        k = randn((size, TP4_KV_HEADS * HEAD_DIM)).contiguous()
        module = ops.WelmV4InplaceRotaryEmbedding(
            HEAD_DIM, ROPE_DIM, 1, ROPE_THETA, True, torch.bfloat16
        ).cuda()
        module.cos_sin_cache = make_yarn_cache()
        rotated = size * (TP4_Q_HEADS + TP4_KV_HEADS) * ROPE_DIM
        logical_bytes = (
            nbytes(positions)
            + 2 * nbytes(q)
            + 2 * nbytes(k)
            + size * ROPE_DIM * 4
        )
        return Run(
            lambda: module.forward_cuda(positions, q, k),
            float(rotated * 3),
            float(logical_bytes),
        )

    linear_shapes = {
        "attention_gate_projection": (HIDDEN_SIZE, TP4_Q_HEADS),
        "shared_gate_up_projection": (
            HIDDEN_SIZE,
            2 * SHARED_INTERMEDIATE_SIZE,
        ),
        "shared_down_projection": (SHARED_INTERMEDIATE_SIZE, HIDDEN_SIZE),
    }
    if name in linear_shapes:
        input_width, output_width = linear_shapes[name]
        x = randn((size, input_width))
        weight = randn((output_width, input_width), scale=0.02)
        flops = 2.0 * size * input_width * output_width
        output_bytes = size * output_width * torch.bfloat16.itemsize
        return Run(
            lambda: F.linear(x, weight),
            flops,
            float(nbytes(x) + nbytes(weight) + output_bytes),
        )

    if name == "attention_gate_sigmoid_mul":
        gate = randn((size, TP4_Q_HEADS, 1)).contiguous()
        value = randn((size, TP4_Q_HEADS, HEAD_DIM)).contiguous()
        return Run(
            lambda: ops.inplace_sigmoid_mul(gate, value),
            float(value.numel() * 5),
            float(nbytes(gate) + 2 * nbytes(value)),
        )

    if name.startswith("attention_prefill_sink_"):
        window = LOCAL_WINDOW if name.endswith("local") else MAX_POSITION
        return make_sink_prefill(size, window, implementation)

    if name == "router_linear":
        x = randn((size, HIDDEN_SIZE))
        weight = randn(
            (NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float32, scale=0.02
        )
        output_bytes = size * NUM_EXPERTS * torch.float32.itemsize
        return Run(
            lambda: ops.mmq_style_router_linear(x, weight),
            float(2 * size * HIDDEN_SIZE * NUM_EXPERTS),
            float(nbytes(x) + nbytes(weight) + output_bytes),
        )

    if name == "expert_bias_topk":
        scores = torch.sigmoid(
            randn((size, NUM_EXPERTS), dtype=torch.float32)
        )
        bias = randn((NUM_EXPERTS,), dtype=torch.float32, scale=0.05)
        output_bytes = size * TOPK * (
            torch.float32.itemsize + torch.int64.itemsize
        )
        return Run(
            lambda: ops.mmq_style_expert_bias_topk(scores, bias, TOPK),
            0.0,
            float(nbytes(scores) + nbytes(bias) + output_bytes),
        )

    if name == "silu_and_mul":
        from sglang.srt.layers.activation import SiluAndMul

        x = randn((size, 2 * SHARED_INTERMEDIATE_SIZE))
        module = SiluAndMul()
        output_numel = size * SHARED_INTERMEDIATE_SIZE
        output_bytes = output_numel * torch.bfloat16.itemsize
        return Run(
            lambda: module(x),
            float(output_numel * 5),
            float(nbytes(x) + output_bytes),
        )

    if name == "oe_lookup_concat":
        return make_oe_lookup(size)

    raise KeyError(name)


def benchmark_once(fn: Callable[[], Any], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for index in range(repeat):
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    return statistics.median(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    )


def environment() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = "unknown"
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton_version,
        "gpu": props.name,
        "compute_capability": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "memory_bytes": props.total_memory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel-file",
        type=Path,
        default=Path(__file__).with_name("welmv4_kernels.py"),
    )
    parser.add_argument("--cases", help="comma-separated subset; use --list-cases")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--tokens", type=csv_positive_ints, default=[1, 32, 128, 512]
    )
    parser.add_argument(
        "--attention-seq-lens",
        type=csv_positive_ints,
        default=[512, 2048, 4096],
    )
    parser.add_argument(
        "--attention-impl", choices=("triton", "fa4", "all"), default="triton"
    )
    parser.add_argument(
        "--include-oe",
        action="store_true",
        help="allocate approximately 16 GiB of real OE embedding tables",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--peak-bandwidth-gbs", type=float, default=1344.0)
    parser.add_argument("--peak-bf16-tflops", type=float, default=268.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/welmv4_optimized_kernel_benchmark.json"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be >= 0 and repeat must be > 0")
    if args.peak_bandwidth_gbs <= 0 or args.peak_bf16_tflops <= 0:
        parser.error("peak values must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.list_cases:
        print("\n".join(CASES))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    selected = CASES
    if args.cases:
        requested = tuple(item.strip() for item in args.cases.split(",") if item.strip())
        unknown = sorted(set(requested) - set(CASES))
        if unknown:
            raise ValueError(f"unknown cases: {', '.join(unknown)}")
        selected = tuple(name for name in CASES if name in requested)

    torch.manual_seed(args.seed)
    ops = load_ops(args.kernel_file.resolve())
    ridge = args.peak_bf16_tflops * 1000.0 / args.peak_bandwidth_gbs
    results: list[dict[str, Any]] = []

    print(
        f"GPU: {torch.cuda.get_device_name(0)} | "
        f"roofline ridge: {ridge:.1f} FLOP/Byte"
    )
    for name in selected:
        if name == "oe_lookup_concat" and not args.include_oe:
            print("SKIP oe_lookup_concat: pass --include-oe (requires about 16 GiB)")
            results.append(
                {
                    "case": name,
                    "status": "skipped",
                    "reason": "pass --include-oe to allocate real OE tables",
                }
            )
            continue

        sizes = (
            args.attention_seq_lens
            if name.startswith("attention_prefill_sink_")
            else args.tokens
        )
        implementations = (
            ("triton", "fa4")
            if name.startswith("attention_prefill_sink_")
            and args.attention_impl == "all"
            else (args.attention_impl,)
            if name.startswith("attention_prefill_sink_")
            else ("model",)
        )

        for size in sizes:
            for implementation in implementations:
                run = None
                try:
                    run = make_run(name, size, implementation, ops)
                    ms = benchmark_once(run.fn, args.warmup, args.repeat)
                    intensity = run.flops / run.logical_bytes if run.logical_bytes else 0.0
                    bound = "compute" if intensity >= ridge else "bandwidth"
                    tflops = run.flops / (ms * 1e9) if run.flops else 0.0
                    bandwidth = run.logical_bytes / (ms * 1e6) if run.logical_bytes else 0.0
                    efficiency = 100.0 * (
                        tflops / args.peak_bf16_tflops
                        if bound == "compute"
                        else bandwidth / args.peak_bandwidth_gbs
                    )
                    row = {
                        "case": name,
                        "implementation": implementation,
                        "tokens": None
                        if name.startswith("attention_prefill_sink_")
                        else size,
                        "sequence": size
                        if name.startswith("attention_prefill_sink_")
                        else None,
                        "ms": ms,
                        "flops": run.flops,
                        "logical_bytes": run.logical_bytes,
                        "arithmetic_intensity_flop_per_byte": intensity,
                        "bound": bound,
                        "achieved_tflops": tflops,
                        "logical_bandwidth_gbs": bandwidth,
                        "roofline_efficiency_pct": efficiency,
                        "status": "ok",
                    }
                    results.append(row)
                    metric = (
                        f"{tflops:.2f} TFLOPS"
                        if bound == "compute"
                        else f"{bandwidth:.1f} GB/s"
                    )
                    print(
                        f"OK   {name:<34} {implementation:<7} size={size:<5} "
                        f"{ms:>8.4f} ms  {metric:<18} {efficiency:>5.1f}%"
                    )
                except Exception as exc:
                    results.append(
                        {
                            "case": name,
                            "implementation": implementation,
                            "size": size,
                            "status": "unsupported",
                            "error": repr(exc),
                        }
                    )
                    print(
                        f"FAIL {name:<34} {implementation:<7} size={size:<5} "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if args.strict:
                        raise
                finally:
                    del run
                    gc.collect()
                    torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "environment": environment(),
        "model_shape": {
            "hidden_size": HIDDEN_SIZE,
            "head_dim": HEAD_DIM,
            "tp_size": TP_SIZE,
            "tp4_q_heads": TP4_Q_HEADS,
            "tp4_kv_heads": TP4_KV_HEADS,
            "num_experts": NUM_EXPERTS,
            "topk": TOPK,
            "shared_intermediate_size": SHARED_INTERMEDIATE_SIZE,
            "rope_dim": ROPE_DIM,
        },
        "method": {
            "timing": "median CUDA-event latency",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "tokens": args.tokens,
            "attention_seq_lens": args.attention_seq_lens,
            "attention_impl": args.attention_impl,
            "include_oe": args.include_oe,
            "traffic": "logical tensor bytes, not hardware-counter DRAM bytes",
        },
        "official_peaks": {
            "memory_bandwidth_gbs": args.peak_bandwidth_gbs,
            "bf16_tensor_tflops": args.peak_bf16_tflops,
        },
        "ridge_point_flop_per_byte": ridge,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Report: {args.report}")
    failed = any(row["status"] == "unsupported" for row in results)
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
