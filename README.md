# WeLMv4 待优化 Kernel 说明

## 需要支持的算子

### FA4 + Attention Sink

RTX PRO 5000 上普通 FA4 可以运行，但当前 FA4 + Attention Sink
不支持。

SGLang Triton Attention Sink 在 RTX PRO 5000 上可以运行，但性能
较差。

## 需要优化的 Kernel

带宽型算子对比 1344 GB/s，计算型算子对比 268 BF16 TFLOPS。

| Kernel | 来源 | 代表 Shape | 延迟 | 当前性能 | 峰值利用率 | 优化原因 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **RMSNorm** | WeLM custom | `tokens=512` | 0.0249 ms | 168.5 GB/s | 12.5% | custom kernel 带宽利用率低 |
| **K RMSNorm** | WeLM custom | `tokens=512` | 0.0332 ms | 15.8 GB/s | 1.2% | 单 KV head 小规模归一化效率低 |
| **Partial YaRN RoPE** | WeLM custom | `tokens=512` | 0.0216 ms | 176.2 GB/s | 13.1% | partial rotary 数据访问效率低 |
| **Attention gate projection** | PyTorch BF16 GEMM | `tokens=512` | 0.0248 ms | 86.0 GB/s | 6.4% | `[512,2048] x [2048,6]` 窄 GEMM 效率低 |
| **Attention gate sigmoid multiply** | WeLM custom | `tokens=512` | 0.0279 ms | 113.1 GB/s | 8.4% | elementwise kernel 带宽利用率低 |
| **O norm** | WeLM custom | `tokens=512` | 0.0226 ms | 185.7 GB/s | 13.8% | 与 RMSNorm 共用同一个低效 kernel |
| **Router linear** | WeLM custom | `tokens=512` | 0.0827 ms | 88.7 GB/s | 6.6% | `[512,2048] x [2048,512]` custom matmul 效率低 |
| **Expert-bias TopK** | WeLM custom | `tokens=512` | 0.0271 ms | 41.0 GB/s | 3.1% | 512 experts、`TopK=10` 选择效率低 |
| **Shared gate/up projection** | PyTorch BF16 GEMM | `tokens=512` | 0.0307 ms | 69.98 TFLOPS | 26.1% | Shared Expert GEMM 计算利用率低 |
| **SwiGLU / SiLU-and-Mul** | SGLang open source | `tokens=512` | 0.0262 ms | 60.0 GB/s | 4.5% | elementwise kernel 带宽利用率低 |
| **Shared down projection** | PyTorch BF16 GEMM | `tokens=512` | 0.0205 ms | 52.43 TFLOPS | 19.6% | Shared Expert GEMM 计算利用率低 |
| **OE lookup concat** | SGLang open source | `tokens=512` | 0.0278 ms | 151.3 GB/s | 11.3% | 四路 embedding lookup/concat 搬运效率低 |

`RMSNorm` 和 `O norm` 共用实现，因此 12 类测项对应 6 个独立 WeLM
custom kernel。其余测项直接调用 PyTorch 或 SGLang 开源实现，未复制其源码。

## 复现

```bash
export PYTHONPATH=/path/to/sglang/python
export WELM_USE_PREVIOUS_PRECISION=0
cd /path/to/nvidia_welmv4_optimized_kernels_20260724

# 核心待优化项
python benchmark_welmv4_kernels.py \
  --strict \
  --report /tmp/welmv4_optimized.json

# OE lookup，需要约 16 GiB 额外显存
python benchmark_welmv4_kernels.py \
  --cases oe_lookup_concat \
  --include-oe \
  --strict \
  --report /tmp/welmv4_oe.json

# 复现 FA4 + Sink 不支持
python benchmark_welmv4_kernels.py \
  --cases attention_prefill_sink_local,attention_prefill_sink_global \
  --attention-impl fa4 \
  --attention-seq-lens 512 \
  --report /tmp/welmv4_fa4_sink.json
```
