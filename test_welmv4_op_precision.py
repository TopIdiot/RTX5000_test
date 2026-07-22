#!/usr/bin/env python3
"""Cross-GPU precision tests for all executable ops used by WeLMv4 80B.

Run ``dump`` on H20 to save exact inputs and reference outputs, then run
``validate`` on the target GPU with the same dump directory. The candidate
``welmv4_op.py`` is loaded from this file's directory; non-custom operations
use the open-source implementations installed in the SGLang environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


HIDDEN_SIZE = 2048
HEAD_DIM = 256
TP_SIZE = 4
GLOBAL_Q_HEADS = 24
GLOBAL_KV_HEADS = 2
TP4_Q_HEADS = 6
TP4_KV_HEADS = 1
NUM_EXPERTS = 512
TOPK = 10
VOCAB_SIZE = 155648
VOCAB_SHARD = 38912
OE_GRAMS = (2, 2, 3, 3)
OE_DIM = 512
OE_VOCAB_SIZES = (16000008, 16000016, 16000024, 16000032)
MOE_INTERMEDIATE_SIZE = 512
SHARED_INTERMEDIATE_SIZE = 512
ROPE_DIM = 64
MAX_POSITION = 262144
LOCAL_WINDOW = 512
ROPE_ORIGINAL_MAX_POSITION = 32768
ROPE_SCALING_FACTOR = 8.0
ROPE_THETA = 500000
RMS_EPS = 1e-5
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    description: str
    rtol: float = 2e-2
    atol: float = 2e-2
    token_dependent: bool = True
    large: bool = False
    custom: bool = False


CASES = (
    Case("token_embedding", "embedding", "TP4 vocabulary embedding", large=True),
    Case("oe_hash_decode", "embedding", "OE decode hash CUDA kernel", 0.0, 0.0, False),
    Case("oe_hash_segments", "embedding", "OE prefill hash CUDA kernel", 0.0, 0.0, False),
    Case("oe_hash_mtp_history", "embedding", "OE MTP history CUDA kernel", 0.0, 0.0, False),
    Case("oe_lookup_concat", "embedding", "Four-branch OE lookup/concat", large=True),
    Case("oe_projection", "embedding", "OE concat-to-hidden GEMM"),
    Case("oe_blend", "embedding", "OE/base embedding blend"),
    Case("rms_norm", "norm", "Fused RMSNorm", custom=True),
    Case("rms_norm_residual", "norm", "Residual add plus RMSNorm", custom=True),
    Case("rms_norm_ppln", "norm", "PPLN RMSNorm with FP32 clone", custom=True),
    Case("norm_after_attn", "norm", "Fused o-norm/residual/post-norm", custom=True),
    Case("add_residual", "norm", "FP32 residual add", 1e-5, 1e-5, custom=True),
    Case("k_rms_norm", "attention", "Per-KV-head RMSNorm", custom=True),
    Case("rope_yarn_partial", "attention", "Partial YaRN RoPE", 3e-2, 3e-2, custom=True),
    Case("qkv_imitated_projection", "linear", "KV-mirror producer QKV GEMM"),
    Case("qkv_standard_projection", "linear", "Standard QKV GEMM"),
    Case("q_mirror_projection", "linear", "KV-mirror consumer Q GEMM"),
    Case("attention_gate_projection", "linear", "Per-head attention gate GEMM"),
    Case("attention_gate_sigmoid_mul", "attention", "In-place attention gate", custom=True),
    Case("attention_prefill_sink_local", "attention", "Sink prefill, 512 window", 3e-2, 3e-2, False),
    Case("attention_prefill_sink_global", "attention", "Sink prefill, global window", 3e-2, 3e-2, False),
    Case("attention_decode_sink_local", "attention", "Sink decode, 512 window", 3e-2, 3e-2, False),
    Case("attention_decode_sink_global", "attention", "Sink decode, global window", 3e-2, 3e-2, False),
    Case("o_projection", "linear", "TP4 local attention output GEMM"),
    Case("o_norm", "norm", "Attention output RMSNorm", custom=True),
    Case("router_linear", "moe", "FP32-output router GEMM", 3e-2, 8e-2, custom=True),
    Case("router_sigmoid", "moe", "Router sigmoid"),
    Case("expert_bias_topk", "moe", "Expert-bias TopK", 1e-6, 1e-6, custom=True),
    Case("fused_moe", "moe", "512-expert top-10 fused MoE", 4e-2, 4e-2, large=True),
    Case("shared_gate_up_projection", "moe", "Shared-expert gate/up GEMM"),
    Case("silu_and_mul", "moe", "Shared-expert SwiGLU"),
    Case("shared_down_projection", "moe", "Shared-expert down GEMM"),
    Case("shared_expert_add", "moe", "Routed/shared expert add"),
    Case("lm_head", "output", "TP4 vocabulary LM-head GEMM", large=True),
)


_MODULE_CACHE: dict[tuple[Any, ...], Any] = {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    spec = importlib.util.spec_from_file_location("welmv4_op_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def environment() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    try:
        import triton

        triton_version = triton.__version__
    except Exception as exc:
        triton_version = f"unavailable: {exc!r}"
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton_version,
        "gpu": props.name,
        "compute_capability": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
    }


def model_shape() -> dict[str, Any]:
    return {
        "tp_size": TP_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "head_dim": HEAD_DIM,
        "global_q_heads": GLOBAL_Q_HEADS,
        "global_kv_heads": GLOBAL_KV_HEADS,
        "tp4_q_heads": TP4_Q_HEADS,
        "tp4_kv_heads": TP4_KV_HEADS,
        "num_experts": NUM_EXPERTS,
        "topk": TOPK,
        "vocab_size": VOCAB_SIZE,
        "vocab_shard": VOCAB_SHARD,
        "oe_grams": list(OE_GRAMS),
        "oe_dim": OE_DIM,
        "oe_vocab_sizes": list(OE_VOCAB_SIZES),
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "shared_intermediate_size": SHARED_INTERMEDIATE_SIZE,
        "rope_dim": ROPE_DIM,
        "max_position": MAX_POSITION,
        "local_window": LOCAL_WINDOW,
    }


def case_seed(name: str, seed: int) -> int:
    return seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def randn(shape, *, dtype, device, scale=1.0):
    return (torch.randn(shape, dtype=torch.float32, device=device) * scale).to(dtype)


def compact_or_full(shape, *, dtype, device, materialize: bool):
    if materialize:
        return randn(shape, dtype=dtype, device=device, scale=0.02).contiguous(), {
            "expanded": False
        }
    base = randn((1, *shape[1:]), dtype=dtype, device=device, scale=0.02).contiguous()
    return base, {"expanded": True, "logical_shape": list(shape)}


def logical_tensor(value: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
    if spec.get("expanded"):
        return value.expand(tuple(spec["logical_shape"]))
    return value


def attention_scale() -> float:
    mscale = 0.1 * math.log(ROPE_SCALING_FACTOR) + 1.0
    return HEAD_DIM**-0.5 * mscale * mscale


def make_yarn_cache(device: str) -> torch.Tensor:
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
        32.0,
        1.0,
        ROPE_DIM,
        ROPE_THETA,
        ROPE_ORIGINAL_MAX_POSITION,
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
    return torch.cat((torch.cos(freqs) * mscale, torch.sin(freqs) * mscale), dim=-1).to(
        device
    )


def make_inputs(
    case: Case,
    tokens: int,
    seed: int,
    device: str = "cuda",
    *,
    materialize_large: bool = False,
    attention_seq_len: int = 1024,
    decode_batch: int = 8,
    decode_context: int = 4096,
):
    torch.manual_seed(case_seed(case.name, seed))
    bf16 = torch.bfloat16
    weight = lambda size: randn((size,), dtype=bf16, device=device, scale=0.1) + 1
    name = case.name
    params: dict[str, Any] = {"tp_size": TP_SIZE}

    if name == "token_embedding":
        table, spec = compact_or_full(
            (VOCAB_SHARD, HIDDEN_SIZE),
            dtype=bf16,
            device=device,
            materialize=materialize_large,
        )
        ids = torch.randint(0, VOCAB_SHARD, (tokens,), device=device)
        return {"ids": ids, "weight": table}, {**params, "weight_spec": spec}
    if name in {"oe_hash_decode", "oe_hash_segments"}:
        count = 16
        ids = torch.randint(0, VOCAB_SIZE, (count,), dtype=torch.int64, device=device)
        prefixes = [
            [int((17 * lag + index) % VOCAB_SIZE) for index in range(count)]
            for lag in range(max(OE_GRAMS) - 1)
        ]
        inputs: dict[str, Any] = {"input_ids": ids, "prefixes": prefixes}
        if name == "oe_hash_segments":
            inputs.update(
                {
                    "extend_start_loc": torch.tensor(
                        [0, 7], dtype=torch.int32, device=device
                    ),
                    "extend_seq_lens": torch.tensor(
                        [7, 9], dtype=torch.int32, device=device
                    ),
                }
            )
            inputs["prefixes"] = [[row[0], row[7]] for row in prefixes]
        return inputs, {
            **params,
            "oe_grams": list(OE_GRAMS),
            "oe_vocab_sizes": list(OE_VOCAB_SIZES),
            "vocab_size": VOCAB_SIZE,
        }
    if name == "oe_hash_mtp_history":
        batch = 16
        history = max(OE_GRAMS) - 1
        prefixes = torch.randint(
            0, VOCAB_SIZE, (batch * history,), dtype=torch.int64, device="cpu"
        ).tolist()
        return {"prefixes": prefixes}, {
            **params,
            "batch": batch,
            "history": history,
        }
    if name == "oe_lookup_concat":
        tables, specs, hashed = [], [], []
        for vocab in OE_VOCAB_SIZES:
            shard = math.ceil(vocab / TP_SIZE)
            table, spec = compact_or_full(
                (shard, OE_DIM),
                dtype=bf16,
                device=device,
                materialize=materialize_large,
            )
            tables.append(table)
            specs.append(spec)
            hashed.append(
                torch.randint(0, shard, (tokens,), dtype=torch.int64, device=device)
            )
        return {"hashed_inputs": hashed, "weights": tables}, {
            **params,
            "weight_specs": specs,
        }
    if name == "oe_projection":
        width = len(OE_GRAMS) * OE_DIM
        return {
            "x": randn((tokens, width), dtype=bf16, device=device),
            "weight": randn(
                (HIDDEN_SIZE, width), dtype=bf16, device=device, scale=0.02
            ),
        }, params
    if name == "oe_blend":
        return {
            "base": randn((tokens, HIDDEN_SIZE), dtype=bf16, device=device),
            "oe": randn((tokens, HIDDEN_SIZE), dtype=bf16, device=device),
        }, params

    if name in {"rms_norm", "rms_norm_ppln", "o_norm"}:
        return {
            "x": randn((tokens, HIDDEN_SIZE), dtype=bf16, device=device),
            "weight": weight(HIDDEN_SIZE),
        }, params
    if name == "rms_norm_residual":
        return {
            "x": randn((tokens, HIDDEN_SIZE), dtype=bf16, device=device),
            "residual": randn((tokens, HIDDEN_SIZE), dtype=torch.float32, device=device),
            "weight": weight(HIDDEN_SIZE),
        }, params
    if name == "norm_after_attn":
        return {
            "hidden": randn((tokens, HIDDEN_SIZE), dtype=bf16, device=device),
            "residual": randn((tokens, HIDDEN_SIZE), dtype=torch.float32, device=device),
            "o_weight": weight(HIDDEN_SIZE),
            "post_weight": weight(HIDDEN_SIZE),
        }, params
    if name == "add_residual":
        return {
            "hidden": randn((tokens, HIDDEN_SIZE), dtype=bf16, device=device).contiguous(),
            "residual": randn((tokens, HIDDEN_SIZE), dtype=torch.float32, device=device).contiguous(),
        }, params
    if name == "k_rms_norm":
        return {
            "x": randn((tokens, TP4_KV_HEADS, HEAD_DIM), dtype=bf16, device=device).contiguous(),
            "weight": weight(HEAD_DIM),
        }, params
    if name == "rope_yarn_partial":
        positions = torch.linspace(0, MAX_POSITION - 1, tokens, device=device).to(
            torch.int64
        )
        return {
            "positions": positions,
            "q": randn(
                (tokens, TP4_Q_HEADS * HEAD_DIM), dtype=bf16, device=device
            ).contiguous(),
            "k": randn(
                (tokens, TP4_KV_HEADS * HEAD_DIM), dtype=bf16, device=device
            ).contiguous(),
            "cos_sin_cache": make_yarn_cache(device),
        }, params

    linear_widths = {
        "qkv_imitated_projection": (HIDDEN_SIZE, 2560),
        "qkv_standard_projection": (HIDDEN_SIZE, 2048),
        "q_mirror_projection": (HIDDEN_SIZE, 1536),
        "attention_gate_projection": (HIDDEN_SIZE, TP4_Q_HEADS),
        "o_projection": (TP4_Q_HEADS * HEAD_DIM, HIDDEN_SIZE),
        "shared_gate_up_projection": (HIDDEN_SIZE, 2 * SHARED_INTERMEDIATE_SIZE),
        "shared_down_projection": (SHARED_INTERMEDIATE_SIZE, HIDDEN_SIZE),
    }
    if name in linear_widths:
        input_width, output_width = linear_widths[name]
        return {
            "x": randn((tokens, input_width), dtype=bf16, device=device),
            "weight": randn(
                (output_width, input_width), dtype=bf16, device=device, scale=0.02
            ),
        }, params
    if name == "attention_gate_sigmoid_mul":
        return {
            "gate": randn(
                (tokens, TP4_Q_HEADS, 1), dtype=bf16, device=device
            ).contiguous(),
            "value": randn(
                (tokens, TP4_Q_HEADS, HEAD_DIM), dtype=bf16, device=device
            ).contiguous(),
        }, params
    if name.startswith("attention_prefill_sink_"):
        window = LOCAL_WINDOW if name.endswith("local") else MAX_POSITION
        seq = attention_seq_len
        return {
            "q": randn((seq, TP4_Q_HEADS, HEAD_DIM), dtype=bf16, device=device),
            "k": randn((seq, TP4_KV_HEADS, HEAD_DIM), dtype=bf16, device=device),
            "v": randn((seq, TP4_KV_HEADS, HEAD_DIM), dtype=bf16, device=device),
            "sinks": randn((TP4_Q_HEADS,), dtype=bf16, device=device),
        }, {
            **params,
            "kind": "prefill",
            "seq_len": seq,
            "window": window,
            "scale": attention_scale(),
        }
    if name.startswith("attention_decode_sink_"):
        context = min(decode_context, LOCAL_WINDOW) if name.endswith("local") else decode_context
        return {
            "q": randn(
                (decode_batch, TP4_Q_HEADS, HEAD_DIM), dtype=bf16, device=device
            ),
            "k": randn(
                (decode_batch * context, TP4_KV_HEADS, HEAD_DIM),
                dtype=bf16,
                device=device,
            ),
            "v": randn(
                (decode_batch * context, TP4_KV_HEADS, HEAD_DIM),
                dtype=bf16,
                device=device,
            ),
            "sinks": randn((TP4_Q_HEADS,), dtype=bf16, device=device),
        }, {
            **params,
            "kind": "decode",
            "batch": decode_batch,
            "context": context,
            "window": LOCAL_WINDOW if name.endswith("local") else MAX_POSITION,
            "scale": attention_scale(),
        }
    if name == "router_linear":
        return {
            "x": randn((tokens, HIDDEN_SIZE), dtype=bf16, device=device),
            "weight": randn((NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float32, device=device, scale=0.02),
        }, params
    if name == "router_sigmoid":
        return {
            "logits": randn(
                (tokens, NUM_EXPERTS), dtype=torch.float32, device=device
            )
        }, params
    if name == "expert_bias_topk":
        return {
            "scores": torch.sigmoid(randn((tokens, NUM_EXPERTS), dtype=torch.float32, device=device)),
            "bias": randn((NUM_EXPERTS,), dtype=torch.float32, device=device, scale=0.05),
        }, {**params, "topk": TOPK}
    if name == "fused_moe":
        w1, w1_spec = compact_or_full(
            (NUM_EXPERTS, 2 * MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
            dtype=bf16,
            device=device,
            materialize=materialize_large,
        )
        w2, w2_spec = compact_or_full(
            (NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
            dtype=bf16,
            device=device,
            materialize=materialize_large,
        )
        ids = (
            torch.arange(tokens * TOPK, device=device).reshape(tokens, TOPK) * 37
        ) % NUM_EXPERTS
        return {
            "x": randn((tokens, HIDDEN_SIZE), dtype=bf16, device=device),
            "w1": w1,
            "w2": w2,
            "topk_ids": ids.to(torch.int64),
            "topk_weights": torch.sigmoid(
                randn((tokens, TOPK), dtype=torch.float32, device=device)
            ),
        }, {
            **params,
            "w1_spec": w1_spec,
            "w2_spec": w2_spec,
            "num_experts": NUM_EXPERTS,
            "intermediate": MOE_INTERMEDIATE_SIZE,
            "topk": TOPK,
        }
    if name == "silu_and_mul":
        return {
            "x": randn(
                (tokens, 2 * SHARED_INTERMEDIATE_SIZE), dtype=bf16, device=device
            )
        }, params
    if name == "shared_expert_add":
        return {
            "routed": randn((tokens, HIDDEN_SIZE), dtype=bf16, device=device),
            "shared": randn((tokens, HIDDEN_SIZE), dtype=bf16, device=device),
        }, params
    if name == "lm_head":
        table, spec = compact_or_full(
            (VOCAB_SHARD, HIDDEN_SIZE),
            dtype=bf16,
            device=device,
            materialize=materialize_large,
        )
        return {
            "x": randn((tokens, HIDDEN_SIZE), dtype=bf16, device=device),
            "weight": table,
        }, {**params, "weight_spec": spec}
    raise KeyError(name)


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


def run_case(
    ops,
    case: Case,
    inputs: dict[str, Any],
    params: dict[str, Any],
    attention_impl: str = "triton",
) -> dict[str, torch.Tensor]:
    import torch.nn.functional as F

    name = case.name
    if name == "token_embedding":
        weight = logical_tensor(inputs["weight"], params["weight_spec"])
        return {"output": F.embedding(inputs["ids"], weight)}
    if name in {"oe_hash_decode", "oe_hash_segments"}:
        from sglang.jit_kernel.welm_oe import (
            welm_oe_hash_decode_from_prefixes_cuda,
            welm_oe_hash_segments_from_prefixes_cuda,
        )

        output = torch.empty(
            (len(params["oe_grams"]), inputs["input_ids"].numel()),
            dtype=torch.int64,
            device=inputs["input_ids"].device,
        )
        prefixes = [item for row in inputs["prefixes"] for item in row]
        if name == "oe_hash_decode":
            welm_oe_hash_decode_from_prefixes_cuda(
                inputs["input_ids"],
                prefixes,
                params["oe_grams"],
                params["oe_vocab_sizes"],
                output,
                params["vocab_size"],
            )
        else:
            welm_oe_hash_segments_from_prefixes_cuda(
                inputs["input_ids"],
                inputs["extend_start_loc"],
                inputs["extend_seq_lens"],
                prefixes,
                params["oe_grams"],
                params["oe_vocab_sizes"],
                output,
                params["vocab_size"],
            )
        return {"output": output}
    if name == "oe_hash_mtp_history":
        from sglang.jit_kernel.welm_oe import (
            welm_oe_hash_mtp_init_history_from_prefixes_cuda,
        )

        output = torch.empty(
            (params["batch"], params["history"]),
            dtype=torch.int64,
            device="cuda",
        )
        welm_oe_hash_mtp_init_history_from_prefixes_cuda(
            inputs["prefixes"], output
        )
        return {"output": output}
    if name == "oe_lookup_concat":
        from sglang.srt.models.welm_perf_opt import (
            _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512,
        )

        weights = [
            logical_tensor(value, spec)
            for value, spec in zip(inputs["weights"], params["weight_specs"])
        ]
        modules = [_FakeEmbed(value) for value in weights]
        return {
            "output": _compute_welm_oe_concat_local_partials_prehashed_specialized_4x512(
                hashed_inputs=inputs["hashed_inputs"], oe_embed_modules=modules
            )
        }
    linear_cases = {
        "oe_projection",
        "qkv_imitated_projection",
        "qkv_standard_projection",
        "q_mirror_projection",
        "attention_gate_projection",
        "o_projection",
        "shared_gate_up_projection",
        "shared_down_projection",
    }
    if name in linear_cases:
        return {"output": F.linear(inputs["x"], inputs["weight"])}
    if name == "oe_blend":
        return {"output": (inputs["base"] + inputs["oe"]) / 2.0}
    if name in {"rms_norm", "rms_norm_residual", "rms_norm_ppln", "o_norm"}:
        key = ("rms", name, inputs["weight"].data_ptr())
        module = _MODULE_CACHE.get(key)
        if module is None:
            module = ops.WelmV4FusedRMSNorm(
                HIDDEN_SIZE, eps=RMS_EPS, weight_dtype=inputs["weight"].dtype
            ).to(inputs["x"].device)
            module.weight.data.copy_(inputs["weight"])
            _MODULE_CACHE[key] = module
        if name == "rms_norm_ppln":
            out, residual, fp32 = module.forward_cuda(
                inputs["x"], residual_after_layernorm=True, clone_fp32_out=True
            )
            return {"output": out, "residual": residual, "fp32_output": fp32}
        out, residual = module.forward_cuda(inputs["x"], residual=inputs.get("residual"))
        return {"output": out, "residual": residual}
    if name == "norm_after_attn":
        out, residual, fp32 = ops.mmq_style_norm_after_attn(
            inputs["hidden"], inputs["residual"], inputs["o_weight"], inputs["post_weight"], RMS_EPS
        )
        return {"output": out, "residual": residual, "fp32_output": fp32}
    if name == "add_residual":
        return {"output": ops.mmq_style_add_residual(inputs["hidden"], inputs["residual"])}
    if name == "k_rms_norm":
        return {"output": ops.mmq_style_k_rms_norm(inputs["x"], inputs["weight"], RMS_EPS)}
    if name == "router_linear":
        return {"output": ops.mmq_style_router_linear(inputs["x"], inputs["weight"])}
    if name == "expert_bias_topk":
        weights, ids = ops.mmq_style_expert_bias_topk(inputs["scores"], inputs["bias"], TOPK)
        return {"weights": weights, "ids": ids}
    if name == "attention_gate_sigmoid_mul":
        ops.inplace_sigmoid_mul(inputs["gate"], inputs["value"])
        return {"output": inputs["value"]}
    if name.startswith("attention_") and "_sink_" in name:
        return {"output": run_attention(inputs, params, attention_impl)}
    if name == "rope_yarn_partial":
        key = ("rope", id(ops), str(inputs["q"].device))
        module = _MODULE_CACHE.get(key)
        if module is None:
            module = ops.WelmV4InplaceRotaryEmbedding(
                HEAD_DIM, ROPE_DIM, 1, ROPE_THETA, True, torch.bfloat16
            ).to(inputs["q"].device)
            _MODULE_CACHE[key] = module
        module.cos_sin_cache = inputs["cos_sin_cache"]
        q, k = module.forward_cuda(inputs["positions"], inputs["q"], inputs["k"])
        return {"q": q, "k": k}
    if name == "router_sigmoid":
        return {"output": torch.sigmoid(inputs["logits"])}
    if name == "fused_moe":
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        w1 = logical_tensor(inputs["w1"], params["w1_spec"])
        w2 = logical_tensor(inputs["w2"], params["w2_spec"])
        topk = StandardTopKOutput(
            topk_weights=inputs["topk_weights"],
            topk_ids=inputs["topk_ids"],
            router_logits=None,
        )
        runner = MoeRunnerConfig(
            num_experts=params["num_experts"],
            num_local_experts=params["num_experts"],
            top_k=params["topk"],
            hidden_size=HIDDEN_SIZE,
            intermediate_size_per_partition=params["intermediate"],
            params_dtype=inputs["x"].dtype,
            activation="silu",
            inplace=False,
        )
        return {"output": fused_experts(inputs["x"], w1, w2, topk, runner)}
    if name == "silu_and_mul":
        from sglang.srt.layers.activation import SiluAndMul

        key = ("silu_and_mul", str(inputs["x"].device))
        module = _MODULE_CACHE.get(key)
        if module is None:
            module = SiluAndMul()
            _MODULE_CACHE[key] = module
        return {"output": module(inputs["x"])}
    if name == "shared_expert_add":
        return {"output": inputs["routed"] + inputs["shared"]}
    if name == "lm_head":
        weight = logical_tensor(inputs["weight"], params["weight_spec"])
        return {"output": F.linear(inputs["x"], weight)}
    raise KeyError(name)


def run_attention(
    inputs: dict[str, torch.Tensor], params: dict[str, Any], implementation: str
) -> torch.Tensor:
    q, k, v, sinks = (
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["sinks"],
    )
    cache_key = (
        "attention",
        implementation,
        params["kind"],
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        tuple(q.shape),
        tuple(k.shape),
    )
    cached = _MODULE_CACHE.get(cache_key)

    if implementation in {"fa3", "fa4"}:
        if implementation == "fa3":
            from sglang.jit_kernel.flash_attention_v3 import flash_attn_varlen_func
        else:
            from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func

        if params["kind"] == "prefill":
            seq = params["seq_len"]
            if cached is None:
                cached = torch.tensor([0, seq], dtype=torch.int32, device=q.device)
                _MODULE_CACHE[cache_key] = cached
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cached,
                cu_seqlens_k=cached,
                max_seqlen_q=seq,
                max_seqlen_k=seq,
                softmax_scale=params["scale"],
                causal=True,
                window_size=(params["window"], 0),
                sinks=sinks,
                pack_gqa=False,
            )
        batch, context = params["batch"], params["context"]
        if cached is None:
            cached = (
                torch.arange(0, batch + 1, dtype=torch.int32, device=q.device),
                torch.arange(
                    0,
                    (batch + 1) * context,
                    context,
                    dtype=torch.int32,
                    device=q.device,
                ),
            )
            _MODULE_CACHE[cache_key] = cached
        qptr, kvptr = cached
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=qptr,
            cu_seqlens_k=kvptr,
            max_seqlen_q=1,
            max_seqlen_k=context,
            softmax_scale=params["scale"],
            causal=True,
            window_size=(params["window"], 0),
            sinks=sinks,
            pack_gqa=False,
        )

    if implementation != "triton":
        raise ValueError(f"Unsupported attention implementation: {implementation}")
    if params["kind"] == "prefill":
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd_unified,
        )

        seq = params["seq_len"]
        if cached is None:
            cached = (
                torch.empty_like(q),
                torch.tensor([0, seq], dtype=torch.int32, device=q.device),
                torch.arange(seq, dtype=torch.int64, device=q.device),
                torch.zeros(1, dtype=torch.int32, device=q.device),
            )
            _MODULE_CACHE[cache_key] = cached
        output, indptr, indices, prefix = cached
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
            sm_scale=params["scale"],
            is_causal=True,
            sliding_window_size=params["window"],
            sinks=sinks,
        )
        return output

    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd,
    )

    batch, context = params["batch"], params["context"]
    max_splits = 8
    if cached is None:
        cached = (
            torch.empty_like(q),
            torch.arange(
                0,
                (batch + 1) * context,
                context,
                dtype=torch.int32,
                device=q.device,
            ),
            torch.arange(batch * context, dtype=torch.int64, device=q.device),
            torch.empty(
                (batch, q.shape[1], max_splits, v.shape[-1]),
                dtype=torch.float32,
                device=q.device,
            ),
            torch.empty(
                (batch, q.shape[1], max_splits),
                dtype=torch.float32,
                device=q.device,
            ),
            torch.full((batch,), 4, dtype=torch.int32, device=q.device),
        )
        _MODULE_CACHE[cache_key] = cached
    output, kvptr, indices, attn_logits, attn_lse, splits = cached
    decode_attention_fwd(
        q,
        k,
        v,
        output,
        kvptr,
        indices,
        attn_logits,
        attn_lse,
        splits,
        max_splits,
        params["scale"],
        1.0,
        1.0,
        sinks=sinks,
    )
    return output


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def to_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {key: to_cuda(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cuda(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cuda(item) for item in value)
    return value


def tensor_error(actual, expected, rtol: float, atol: float) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "pass": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "actual_dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
        }
    if not actual.is_floating_point():
        equal = torch.equal(actual, expected)
        return {"pass": equal, "mismatch_count": int((actual != expected).sum().item())}
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    finite = torch.isfinite(actual_f) & torch.isfinite(expected_f)
    mismatch = (~finite) | (diff > atol + rtol * expected_f.abs())
    relative = diff / expected_f.abs().clamp_min(1e-8)
    return {
        "pass": not bool(mismatch.any().item()),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float(relative.max().item()) if relative.numel() else 0.0,
        "mismatch_count": int(mismatch.sum().item()),
        "numel": actual.numel(),
    }


def selected_cases(case_filter: set[str] | None):
    known = {case.name for case in CASES}
    if case_filter:
        unknown = case_filter - known
        if unknown:
            raise ValueError(f"Unknown cases: {sorted(unknown)}")
    return [case for case in CASES if case_filter is None or case.name in case_filter]


def dump_reference(args, ops) -> int:
    gpu = torch.cuda.get_device_name(0)
    if "H20" not in gpu.upper() and not args.allow_any_reference_gpu:
        raise RuntimeError(f"Reference dump requires H20, found {gpu!r}")
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    manifest_path = args.dump_dir / "manifest.json"
    existing_cases = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("model_shape") != model_shape():
            raise RuntimeError("Existing reference model shape does not match")
        if existing.get("attention_impl") != args.attention_impl:
            raise RuntimeError("Use one attention implementation per dump directory")
        existing_cases = existing.get("cases", {})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "environment": env,
        "op_source_sha256": file_sha256(args.op_file),
        "model_shape": model_shape(),
        "tokens": args.tokens,
        "attention_impl": args.attention_impl,
        "attention_seq": args.attention_seq,
        "decode_batch": args.decode_batch,
        "decode_context": args.decode_context,
        "materialize_large": args.materialize_large,
        "cases": existing_cases,
    }
    failures = 0
    for case in selected_cases(args.case_filter):
        if case.large and not args.materialize_large:
            manifest["cases"][case.name] = {
                "status": "skipped",
                "group": case.group,
                "custom": case.custom,
                "reason": "rerun selected large cases with --materialize-large",
            }
            print(f"SKIP dump {case.name}: pass --materialize-large")
            continue
        try:
            inputs, params = make_inputs(
                case,
                args.tokens,
                args.seed,
                materialize_large=args.materialize_large,
                attention_seq_len=args.attention_seq,
                decode_batch=args.decode_batch,
                decode_context=args.decode_context,
            )
            saved_inputs = to_cpu(inputs)
            outputs = run_case(
                ops, case, inputs, params, attention_impl=args.attention_impl
            )
            torch.cuda.synchronize()
            payload = {
                "schema_version": SCHEMA_VERSION,
                "case": case.name,
                "inputs": saved_inputs,
                "params": params,
                "outputs": to_cpu(outputs),
                "rtol": case.rtol,
                "atol": case.atol,
                "environment": env,
                "op_source_sha256": manifest["op_source_sha256"],
                "model_shape": model_shape(),
                "tokens": args.tokens,
                "attention_impl": args.attention_impl,
            }
            torch.save(payload, args.dump_dir / f"{case.name}.pt")
            manifest["cases"][case.name] = {
                "status": "ok",
                "file": f"{case.name}.pt",
                "group": case.group,
                "custom": case.custom,
                "compact_large_input": case.large and not args.materialize_large,
            }
            print(f"PASS dump {case.name}")
        except Exception as exc:
            failures += 1
            manifest["cases"][case.name] = {"status": "error", "error": repr(exc)}
            print(f"ERROR dump {case.name}: {exc!r}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Manifest: {manifest_path}")
    return 1 if failures else 0


def validate_reference(args, ops) -> int:
    manifest_path = args.dump_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model_shape") != model_shape():
        raise RuntimeError("Reference model shape does not match this test")
    results = {}
    failures = 0
    for case in selected_cases(args.case_filter):
        reference_status = manifest.get("cases", {}).get(case.name, {}).get("status")
        if reference_status == "skipped":
            results[case.name] = {"status": "skipped_reference"}
            print(f"SKIP {case.name}: reference was not materialized")
            continue
        path = args.dump_dir / f"{case.name}.pt"
        if not path.exists():
            failures += 1
            results[case.name] = {"status": "missing_reference"}
            print(f"MISSING {case.name}")
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            inputs = to_cuda(payload["inputs"])
            actual = run_case(
                ops,
                case,
                inputs,
                payload["params"],
                attention_impl=args.attention_impl,
            )
            expected = to_cuda(payload["outputs"])
            metrics = {}
            passed = True
            for key, expected_tensor in expected.items():
                metric = tensor_error(actual[key], expected_tensor, case.rtol, case.atol)
                metrics[key] = metric
                passed &= bool(metric["pass"])
            status = "pass" if passed else "fail"
            results[case.name] = {
                "status": status,
                "group": case.group,
                "custom": case.custom,
                "reference_attention_impl": payload.get("attention_impl"),
                "target_attention_impl": args.attention_impl,
                "tensors": metrics,
            }
            failures += int(not passed)
            print(f"{status.upper():4s} {case.name}: {json.dumps(metrics, sort_keys=True)}")
        except Exception as exc:
            failures += 1
            results[case.name] = {"status": "error", "error": repr(exc)}
            print(f"ERROR {case.name}: {exc!r}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "reference_environment": manifest["environment"],
        "target_environment": environment(),
        "reference_op_source_sha256": manifest.get("op_source_sha256"),
        "target_op_source_sha256": file_sha256(args.op_file),
        "reference_attention_impl": manifest.get("attention_impl"),
        "target_attention_impl": args.attention_impl,
        "results": results,
    }
    report_path = args.report or args.dump_dir / "validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {report_path}")
    return 1 if failures else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dump", "validate"))
    parser.add_argument("--op-file", type=Path, default=Path(__file__).with_name("welmv4_op.py"))
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cases")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--attention-seq", type=int, default=1024)
    parser.add_argument("--decode-batch", type=int, default=8)
    parser.add_argument("--decode-context", type=int, default=4096)
    parser.add_argument(
        "--attention-impl", choices=("triton", "fa3", "fa4"), default="triton"
    )
    parser.add_argument("--materialize-large", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-any-reference-gpu", action="store_true")
    args = parser.parse_args()
    args.op_file = args.op_file.resolve()
    args.dump_dir = args.dump_dir.resolve()
    args.case_filter = set(args.cases.split(",")) if args.cases else None
    if min(args.tokens, args.attention_seq, args.decode_batch, args.decode_context) <= 0:
        parser.error("token and attention dimensions must be positive")
    return args


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.getenv("WELM_USE_PREVIOUS_PRECISION", "0").lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("Unset WELM_USE_PREVIOUS_PRECISION for this 80B config")
    args = parse_args()
    ops = load_ops(args.op_file)
    if args.mode == "dump":
        return dump_reference(args, ops)
    return validate_reference(args, ops)


if __name__ == "__main__":
    raise SystemExit(main())
