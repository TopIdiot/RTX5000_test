#!/usr/bin/env python3
"""Benchmark all executable ops used by the supplied WeLMv4 80B text model.

Run from an SGLang checkout with ``PYTHONPATH=python``. Shapes are the TP=4
local-rank shapes from 80B_config.json. Custom ops are loaded from the adjacent
candidate file; other ops use installed open-source SGLang/PyTorch kernels.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from test_welmv4_op_precision import (
    HEAD_DIM,
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    OE_DIM,
    ROPE_DIM,
    TOPK,
    TP4_KV_HEADS,
    TP4_Q_HEADS,
    environment,
    load_ops,
    make_inputs,
    model_shape,
    run_case,
    selected_cases,
)


DEFAULT_PEAK_BANDWIDTH_GBS = 1344.0
DEFAULT_PEAK_BF16_TFLOPS = 268.0
PEAK_SOURCE = (
    "NVIDIA RTX PRO 5000 Blackwell Server Edition product specification: "
    "https://www.nvidia.com/en-us/data-center/rtx-pro-5000-blackwell-server-edition/"
)


def nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nbytes(item) for item in value)
    return 0


def workload(case, inputs, params, outputs) -> tuple[float, float]:
    """Return algorithmic FLOPs and logical bytes read/written."""
    name = case.name
    logical_bytes = float(nbytes(inputs) + nbytes(outputs))
    flops = 0.0

    if name == "token_embedding":
        rows = inputs["ids"].numel()
        logical_bytes = float(rows * (8 + 4 * HIDDEN_SIZE))
    elif name in {"rms_norm", "o_norm"}:
        logical_bytes = float(nbytes(inputs["x"]) * 2 + nbytes(inputs["weight"]))
        flops = float(inputs["x"].numel() * 5)
    elif name == "rms_norm_residual":
        logical_bytes = float(
            nbytes(inputs["x"])
            + nbytes(inputs["residual"])
            + nbytes(inputs["weight"])
            + nbytes(outputs["output"])
            + nbytes(outputs["residual"])
        )
        flops = float(inputs["x"].numel() * 6)
    elif name == "rms_norm_ppln":
        logical_bytes = float(
            nbytes(inputs["x"])
            + nbytes(inputs["weight"])
            + nbytes(outputs["output"])
            + nbytes(outputs["residual"])
            + nbytes(outputs["fp32_output"])
        )
        flops = float(inputs["x"].numel() * 5)
    elif name == "norm_after_attn":
        logical_bytes = float(
            nbytes(inputs["hidden"])
            + nbytes(inputs["residual"])
            + nbytes(inputs["o_weight"])
            + nbytes(inputs["post_weight"])
            + nbytes(outputs)
        )
        flops = float(inputs["hidden"].numel() * 11)
    elif name == "add_residual":
        logical_bytes = float(
            nbytes(inputs["hidden"])
            + nbytes(inputs["residual"])
            + nbytes(outputs["output"])
        )
        flops = float(inputs["hidden"].numel())
    elif name == "k_rms_norm":
        logical_bytes = float(nbytes(inputs["x"]) * 2 + nbytes(inputs["weight"]))
        flops = float(inputs["x"].numel() * 5)
    elif name == "router_linear":
        tokens, hidden = inputs["x"].shape
        experts = inputs["weight"].shape[0]
        flops = float(2 * tokens * hidden * experts)
    elif name == "attention_gate_sigmoid_mul":
        logical_bytes = float(
            nbytes(inputs["gate"])
            + nbytes(inputs["value"])
            + nbytes(outputs["output"])
        )
        flops = float(inputs["value"].numel() * 5)
    elif name == "rope_yarn_partial":
        cache_bytes = inputs["positions"].numel() * ROPE_DIM * 4
        logical_bytes = float(
            nbytes(inputs["positions"])
            + nbytes(inputs["q"]) * 2
            + nbytes(inputs["k"]) * 2
            + cache_bytes
        )
        rotated = (
            inputs["positions"].numel()
            * (TP4_Q_HEADS + TP4_KV_HEADS)
            * ROPE_DIM
        )
        flops = float(rotated * 3)
    elif "x" in inputs and "weight" in inputs:
        rows, inner = inputs["x"].shape
        output_width = params.get("weight_spec", {}).get(
            "logical_shape", list(inputs["weight"].shape)
        )[0]
        flops = float(2 * rows * inner * output_width)
    elif name.startswith("attention_prefill"):
        seq = params["seq_len"]
        attended = min(seq, params["window"])
        if attended < seq:
            pairs = seq * attended - attended * (attended - 1) / 2
        else:
            pairs = seq * (seq + 1) / 2
        flops = float(4 * pairs * TP4_Q_HEADS * HEAD_DIM)
    elif name.startswith("attention_decode"):
        flops = float(
            4 * params["batch"] * params["context"] * TP4_Q_HEADS * HEAD_DIM
        )
    elif name == "fused_moe":
        flops = float(
            6
            * inputs["x"].shape[0]
            * TOPK
            * HIDDEN_SIZE
            * MOE_INTERMEDIATE_SIZE
        )
        active_experts = int(inputs["topk_ids"].unique().numel())
        active_weight_bytes = (
            active_experts
            * 3
            * HIDDEN_SIZE
            * MOE_INTERMEDIATE_SIZE
            * 2
        )
        logical_bytes = float(
            active_weight_bytes
            + nbytes(inputs["x"])
            + nbytes(inputs["topk_ids"])
            + nbytes(inputs["topk_weights"])
            + nbytes(outputs["output"])
        )
    elif name == "router_sigmoid":
        flops = float(inputs["logits"].numel() * 4)
    elif name == "oe_lookup_concat":
        tokens = inputs["hashed_inputs"][0].numel()
        logical_bytes = float(tokens * 4 * (8 + 4 * OE_DIM))
    elif name == "oe_blend":
        flops = float(inputs["base"].numel() * 2)
    elif name == "silu_and_mul":
        flops = float(outputs["output"].numel() * 5)
    elif name == "shared_expert_add":
        flops = float(outputs["output"].numel())

    return flops, logical_bytes


def benchmark_once(fn, warmup: int, repeat: int) -> float:
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
    timings = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return statistics.median(timings)


def csv_positive_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op-file", type=Path, default=Path(__file__).with_name("welmv4_op.py")
    )
    parser.add_argument("--tokens", type=csv_positive_ints, default=[1, 32, 128, 512])
    parser.add_argument(
        "--attention-seq-lens",
        type=csv_positive_ints,
        default=[512, 2048, 4096],
    )
    parser.add_argument(
        "--decode-batches", type=csv_positive_ints, default=[1, 8, 32, 128]
    )
    parser.add_argument(
        "--decode-contexts", type=csv_positive_ints, default=[512, 4096, 32768]
    )
    parser.add_argument(
        "--attention-impl",
        choices=("triton", "fa3", "fa4", "all"),
        default="triton",
    )
    parser.add_argument("--materialize-large", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--cases", help="comma-separated subset of operation names")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--peak-bandwidth-gbs", type=float, default=DEFAULT_PEAK_BANDWIDTH_GBS)
    parser.add_argument("--peak-bf16-tflops", type=float, default=DEFAULT_PEAK_BF16_TFLOPS)
    parser.add_argument(
        "--report", type=Path, default=Path("/tmp/welmv4_80b_ops_benchmark.json")
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be non-negative and --repeat must be positive")
    if args.peak_bandwidth_gbs <= 0 or args.peak_bf16_tflops <= 0:
        parser.error("peak values must be positive")
    args.op_file = args.op_file.resolve()
    args.case_filter = set(args.cases.split(",")) if args.cases else None
    return args


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args = parse_args()
    ops = load_ops(args.op_file)
    ridge = args.peak_bf16_tflops * 1000.0 / args.peak_bandwidth_gbs
    results = []

    print(
        f"GPU: {torch.cuda.get_device_name(0)} | ridge: {ridge:.1f} FLOP/byte | "
        f"warmup/repeat: {args.warmup}/{args.repeat}"
    )
    for case in selected_cases(args.case_filter):
        if case.large and not args.materialize_large:
            results.append(
                {
                    "case": case.name,
                    "group": case.group,
                    "status": "skipped",
                    "reason": "pass --materialize-large for a valid weight benchmark",
                }
            )
            print(f"SKIP {case.name}: pass --materialize-large")
            continue
        if case.name.startswith("attention_prefill"):
            variants = [
                {
                    "tokens": 1,
                    "attention_seq": seq,
                    "decode_batch": args.decode_batches[0],
                    "decode_context": args.decode_contexts[0],
                }
                for seq in args.attention_seq_lens
            ]
        elif case.name.startswith("attention_decode"):
            contexts = (
                [min(args.decode_contexts)]
                if case.name.endswith("local")
                else args.decode_contexts
            )
            variants = [
                {
                    "tokens": 1,
                    "attention_seq": args.attention_seq_lens[0],
                    "decode_batch": batch,
                    "decode_context": context,
                }
                for batch in args.decode_batches
                for context in contexts
            ]
        elif not case.token_dependent:
            variants = [
                {
                    "tokens": args.tokens[0],
                    "attention_seq": args.attention_seq_lens[0],
                    "decode_batch": args.decode_batches[0],
                    "decode_context": args.decode_contexts[0],
                }
            ]
        else:
            variants = [
                {
                    "tokens": tokens,
                    "attention_seq": args.attention_seq_lens[0],
                    "decode_batch": args.decode_batches[0],
                    "decode_context": args.decode_contexts[0],
                }
                for tokens in args.tokens
            ]
        attention_impls = (
            ["triton", "fa4"]
            if args.attention_impl == "all"
            else [args.attention_impl]
        )
        implementations = (
            attention_impls
            if case.name.startswith("attention_") and "_sink_" in case.name
            else ["model"]
        )
        for variant in variants:
            for implementation in implementations:
                inputs = None
                first = None
                tokens = variant["tokens"]
                try:
                    inputs, params = make_inputs(
                        case,
                        tokens,
                        args.seed,
                        materialize_large=args.materialize_large,
                        attention_seq_len=variant["attention_seq"],
                        decode_batch=variant["decode_batch"],
                        decode_context=variant["decode_context"],
                    )
                    impl = implementation if implementation != "model" else "triton"
                    first = run_case(
                        ops, case, inputs, params, attention_impl=impl
                    )
                    torch.cuda.synchronize()
                    ms = benchmark_once(
                        lambda: run_case(
                            ops, case, inputs, params, attention_impl=impl
                        ),
                        args.warmup,
                        args.repeat,
                    )
                    flops, logical_bytes = workload(case, inputs, params, first)
                    intensity = flops / logical_bytes if logical_bytes else 0.0
                    bound = "compute" if intensity >= ridge else "bandwidth"
                    tflops = flops / (ms * 1e9) if flops else 0.0
                    bandwidth = (
                        logical_bytes / (ms * 1e6) if logical_bytes else 0.0
                    )
                    efficiency = 100.0 * (
                        tflops / args.peak_bf16_tflops
                        if bound == "compute"
                        else bandwidth / args.peak_bandwidth_gbs
                    )
                    row = {
                        "case": case.name,
                        "group": case.group,
                        "custom": case.custom,
                        "implementation": implementation,
                        "tokens": tokens if case.token_dependent else None,
                        "attention_seq": (
                            params.get("seq_len")
                            if case.name.startswith("attention_prefill")
                            else None
                        ),
                        "decode_batch": (
                            params.get("batch")
                            if case.name.startswith("attention_decode")
                            else None
                        ),
                        "decode_context": (
                            params.get("context")
                            if case.name.startswith("attention_decode")
                            else None
                        ),
                        "ms": ms,
                        "flops": flops,
                        "logical_bytes": logical_bytes,
                        "arithmetic_intensity_flop_per_byte": intensity,
                        "bound": bound,
                        "achieved_tflops": tflops,
                        "logical_bandwidth_gbs": bandwidth,
                        "roofline_efficiency_pct": efficiency,
                        "status": "ok",
                    }
                    results.append(row)
                    metric = (
                        f"{tflops:8.2f} TFLOPS"
                        if bound == "compute"
                        else f"{bandwidth:8.1f} GB/s"
                    )
                    shape = (
                        f"seq={params['seq_len']}"
                        if "seq_len" in params
                        else (
                            f"batch={params['batch']} ctx={params['context']}"
                            if "batch" in params and "context" in params
                            else f"tokens={tokens}"
                        )
                    )
                    print(
                        f"{case.name:<31} {implementation:<6} {shape:<21} "
                        f"{ms:8.4f} ms  {bound:<9} {metric}  "
                        f"{efficiency:6.2f}% peak"
                    )
                except Exception as exc:
                    results.append(
                        {
                            "case": case.name,
                            "group": case.group,
                            "custom": case.custom,
                            "implementation": implementation,
                            "tokens": tokens if case.token_dependent else None,
                            "attention_seq": (
                                variant["attention_seq"]
                                if case.name.startswith("attention_prefill")
                                else None
                            ),
                            "decode_batch": (
                                variant["decode_batch"]
                                if case.name.startswith("attention_decode")
                                else None
                            ),
                            "decode_context": (
                                variant["decode_context"]
                                if case.name.startswith("attention_decode")
                                else None
                            ),
                            "status": "unsupported",
                            "error": repr(exc),
                        }
                    )
                    print(
                        f"UNSUPPORTED {case.name}/{implementation} "
                        f"tokens={tokens}: {exc!r}"
                    )
                finally:
                    if case.large:
                        inputs = None
                        first = None
                        gc.collect()
                        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "environment": environment(),
        "model_shape": {
            **model_shape(),
            "tp_size": 4,
            "global_q_heads": 24,
            "global_kv_heads": 2,
        },
        "method": {
            "timing": "median of back-to-back CUDA event latency samples",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "tokens": args.tokens,
            "attention_seq_lens": args.attention_seq_lens,
            "decode_batches": args.decode_batches,
            "decode_contexts": args.decode_contexts,
            "attention_impl": args.attention_impl,
            "materialize_large": args.materialize_large,
            "traffic": "logical tensor bytes, not hardware-counter DRAM bytes",
        },
        "official_peaks": {
            "memory_bandwidth_gbs": args.peak_bandwidth_gbs,
            "bf16_tensor_tflops": args.peak_bf16_tflops,
            "source": PEAK_SOURCE,
        },
        "ridge_point_flop_per_byte": ridge,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {args.report}")
    failures = [row for row in results if row["status"] == "unsupported"]
    return int(args.strict and bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
