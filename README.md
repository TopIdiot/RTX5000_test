# WeLMv4 待优化 Kernel

## 未支持的Kernel

### FA4 与 Attention Sink
- 当前环境的 FA4 + Attention Sink 不可运行。
- 不带 Sink 的 FA4 varlen 在 `Q heads=6`、`KV heads=1`、`head_dim=256` shape 下也会编译失败。
- SGLang Triton Attention Sink 可以运行，但性能较低。

## 需要优化的 Kernel

关键单卡 shape：`hidden=2048`、`Q/KV heads=6/1`、`head_dim=256`、
`RoPE dim=64`、`experts/topk=512/10`、Shared Expert intermediate `128`。

| Kernel | 实现/来源 | 代表 Shape | 延迟 | 当前性能 | 峰值利用率 | Benchmark 位置 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **RMSNorm** | WeLM custom `WelmV4FusedRMSNorm` | `tokens=512` | 0.0235 ms | 179.0 GB/s | 13.3% | `benchmark_welmv4_kernels.py:L280` |
| **K RMSNorm** | WeLM custom `mmq_style_k_rms_norm` | `tokens=512` | 0.0352 ms | 14.9 GB/s | 1.1% | `benchmark_welmv4_kernels.py:L293` |
| **Partial YaRN RoPE** | WeLM custom `WelmV4InplaceRotaryEmbedding` | `tokens=512` | 0.0371 ms | 28.3 GB/s | 2.1% | `benchmark_welmv4_kernels.py:L302` |
| **Attention gate projection** | PyTorch BF16 GEMM | `tokens=512` | 0.0243 ms | 87.7 GB/s | 6.5% | `benchmark_welmv4_kernels.py:L324` |
| **Attention gate sigmoid multiply** | WeLM custom `inplace_sigmoid_mul` | `tokens=512` | 0.0178 ms | 177.5 GB/s | 13.2% | `benchmark_welmv4_kernels.py:L344` |
| **Sink prefill local** | SGLang Triton Attention | `sequence=2048, window=512` | 0.5111 ms | 11.05 TFLOPS | 4.1% | `benchmark_welmv4_kernels.py:L164` |
| **Sink prefill global** | SGLang Triton Attention | `sequence=4096, window=262144` | 3.1503 ms | 16.36 TFLOPS | 6.1% | `benchmark_welmv4_kernels.py:L164` |
| **O norm** | WeLM custom `WelmV4FusedRMSNorm` | `tokens=512` | 0.0244 ms | 172.1 GB/s | 12.8% | `benchmark_welmv4_kernels.py:L280` |
| **Router linear** | WeLM custom `mmq_style_router_linear` | `tokens=512` | 0.0583 ms | 197.8 GB/s | 14.7% | `benchmark_welmv4_kernels.py:L357` |
| **Expert-bias TopK** | WeLM custom `mmq_style_expert_bias_topk` | `tokens=512` | 0.0494 ms | 22.5 GB/s | 1.7% | `benchmark_welmv4_kernels.py:L375` |
| **Shared gate/up projection** | PyTorch BF16 GEMM | `tokens=512, out=256` | 0.0226 ms | 150.6 GB/s | 11.2% | `benchmark_welmv4_kernels.py:L324` |
| **SwiGLU / SiLU-and-Mul** | SGLang JIT CUDA | `tokens=512, in/out=256/128` | 0.0358 ms | 11.0 GB/s | 0.8% | `benchmark_welmv4_kernels.py:L389` |
| **Shared down projection** | PyTorch BF16 GEMM | `tokens=512, in=128` | 0.0158 ms | 173.9 GB/s | 12.9% | `benchmark_welmv4_kernels.py:L324` |
| **OE lookup concat** | WeLM custom OE kernel | `tokens=512, TP4` | 0.0290 ms | 90.7 GB/s | 6.7% | `benchmark_welmv4_kernels.py:L239` |

14 类测项中有 7 个独立的 WeLM custom kernel；`RMSNorm` 和 `O norm`
共用实现。其余测项直接调用 PyTorch 或 SGLang 开源实现。

## 复现

```bash
# 核心项；默认测试 Triton Attention Sink
python benchmark_welmv4_kernels.py \
  --strict \
  --report /tmp/welmv4_optimized.json

# OE lookup，需要约 16 GiB 额外显存
python benchmark_welmv4_kernels.py \
  --cases oe_lookup_concat \
  --include-oe \
  --strict \
  --report /tmp/welmv4_oe.json

# FA4 + Attention Sink
python benchmark_welmv4_kernels.py \
  --cases attention_prefill_sink_local,attention_prefill_sink_global \
  --attention-impl fa4 \
  --attention-seq-lens 512 \
  --report /tmp/welmv4_fa4_sink.json
