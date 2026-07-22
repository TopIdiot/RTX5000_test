# WeLMv4 80B 算子评测说明

## 1. 交付内容

本目录包含五个文件：

| 文件 | 用途 |
| --- | --- |
| `welmv4_op.py` | WeLMv4 自定义 Triton 算子源码，供 NVIDIA 修改优化 |
| `test_welmv4_op_precision.py` | 在 H20 生成输入/输出，再到目标 GPU 验证精度 |
| `benchmark_welmv4_op.py` | 按 80B 配置测试延迟、TFLOPS 和逻辑带宽 |
| `requirements.txt` | 调用方核心 Python/GPU 依赖版本 |
| `REPORT.md` | 覆盖范围、当前结论和复现命令 |

`welmv4_op.py` 与当前模型环境中的源码完全一致。

SHA256：

```text
2c7290feb988a63ff3303b9a864a98225dd4a08a52e8385c6a4ea342dd31f609
```

其他算子直接调用 SGLang、PyTorch、Triton、FA3/FA4 和
sglang-kernel 的开源实现，不再复制复杂的 `kernel_sources`。

### 调用方环境

本报告结果来自以下环境：

| 项目 | 已测版本 |
| --- | --- |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell，SM120 |
| NVIDIA Driver | 580.126.20 |
| CUDA Runtime | 12.8 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| sglang-kernel | 0.4.2.post2 |
| FA4 | `>=4.0.0b9` |
| Cutlass DSL | 4.5.1，CUDA 13 variant |

测试会直接 import SGLang 源码中的 OE、FA3/FA4、Triton Attention 和 MoE
实现，因此调用方必须准备与待测模型匹配的 SGLang 源码，不能只安装
`requirements.txt`。推荐配置方式：

```bash
export BUNDLE=/path/to/nvidia_welmv4_custom_ops_20260722
export SGLANG_ROOT=/path/to/sglang

python3.12 -m venv /path/to/welmv4-bench-env
source /path/to/welmv4-bench-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$SGLANG_ROOT/python"
python -m pip install -r "$BUNDLE/requirements.txt"

export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export WELM_USE_PREVIOUS_PRECISION=0
```

H20 生成 reference 时使用该 SGLang 源码的 FA3；RTX PRO 5000 验证和
benchmark 使用同一源码中的 Triton Attention，并额外探测 FA4。

## 2. 80B 模型配置

评测 shape 来自真实的 `80B_config.json`：

| 参数 | 数值 |
| --- | ---: |
| Transformer 层数 | 48 层 + 1 个 MTP 层 |
| Hidden size | 2048 |
| Q/KV heads | 24/2 |
| TP size | 4 |
| 单卡 Q/KV heads | 6/1 |
| Head dim | 256 |
| RoPE dim | 64 |
| 最大序列长度 | 262144 |
| Experts / TopK | 512/10 |
| MoE intermediate size | 512 |
| Attention Sink | 49/49 层开启 |
| Attention window | 512 和 262144 交替 |

## 3. 覆盖范围

精度和性能脚本共覆盖 34 类模型实际使用的算子：

| 类型 | 数量 | 主要算子 |
| --- | ---: | --- |
| Embedding/OE | 7 | Token embedding、OE hash、OE lookup、OE projection |
| Norm/Residual | 6 | RMSNorm、PPLN、post-attention norm、residual add |
| Attention/Matmul | 12 | K norm、RoPE、QKV/O GEMM、gate GEMM、Sink Attention |
| MoE | 8 | Router、TopK、Fused MoE、Shared Expert |
| Output | 1 | TP4 LM head |

其中 11 个 case 会执行本目录的 `welmv4_op.py`，其余 23 个 case 调用
模型当前使用的开源实现。

以下内容属于端到端运行时行为，没有伪装成单 GPU 算子：

- TP/DP collective
- KV mirror 的状态传递和收缩
- Paged KV cache 和 `ForwardBatch` 调度
- Logits processor 和 sampling

### Matmul/GEMM 覆盖

模型中的 Matmul 已经测试。脚本使用 `torch.nn.functional.linear` 调用
PyTorch 当前的 BF16 GEMM 实现，而 Router 使用 `welmv4_op.py` 中的自定义
Triton matmul。所有普通 GEMM 都测试 `M=1,32,128,512`。

| Case | 实际矩阵 Shape | 实现 |
| --- | --- | --- |
| `oe_projection` | `[M,2048] x [2048,2048]` | PyTorch BF16 GEMM |
| `qkv_imitated_projection` | `[M,2048] x [2048,2560]` | PyTorch BF16 GEMM |
| `qkv_standard_projection` | `[M,2048] x [2048,2048]` | PyTorch BF16 GEMM |
| `q_mirror_projection` | `[M,2048] x [2048,1536]` | PyTorch BF16 GEMM |
| `attention_gate_projection` | `[M,2048] x [2048,6]` | PyTorch BF16 GEMM |
| `o_projection` | `[M,1536] x [1536,2048]` | PyTorch BF16 GEMM |
| `router_linear` | `[M,2048] x [2048,512]` | WeLM Triton custom matmul |
| `shared_gate_up_projection` | `[M,2048] x [2048,1024]` | PyTorch BF16 GEMM |
| `shared_down_projection` | `[M,512] x [512,2048]` | PyTorch BF16 GEMM |
| `lm_head` | `[M,2048] x [2048,38912]` | PyTorch BF16 GEMM |
| `fused_moe` gate/up | 512 组 `[M,2048] x [2048,1024]`，TopK=10 | SGLang Triton MoE |
| `fused_moe` down | 512 组 `[M,512] x [512,2048]`，TopK=10 | SGLang Triton MoE |

这些 GEMM 同时包含在 H20 精度 dump/RTX validate 和性能 benchmark 中。
计算密集型 GEMM 使用 268 BF16 TFLOPS 作为峰值对比，而不是用显存带宽
评价。

## 4. 精度验证

精度流程如下：

1. 在 H20 上运行真实算子，保存完整输入和输出。
2. 将 dump 目录传到 RTX PRO 5000。
3. 使用完全相同的输入运行候选 kernel。
4. 对比输出误差、TopK ID 和 OE hash 结果。

当前 RTX 本机 dump/validate 已覆盖全部 34 类算子，结果全部通过且误差为
0。这只能证明测试脚本正确，H20 到 RTX 的最终跨架构精度仍需要在 H20
生成 reference 后确认。

四个大权重算子需要真实物化权重：

- Token embedding：约 153 MiB
- OE lookup 四张表：约 16 GiB
- Fused MoE：约 3.1 GiB
- LM head：约 153 MiB

完整 H20 reference 目录约为 19 GiB。

### H20 生成 reference

```bash
export BUNDLE=/path/to/nvidia_welmv4_custom_ops_20260722
export SGLANG_ROOT=/path/to/sglang
export PYTHON=/envs/venv/bin/python
cd "$SGLANG_ROOT"
export PYTHONPATH=python

CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  "$BUNDLE/test_welmv4_op_precision.py" dump \
  --dump-dir /tmp/welmv4_h20_reference \
  --attention-impl fa3 \
  --tokens 128 \
  --attention-seq 1024 \
  --decode-batch 8 \
  --decode-context 4096
```

继续向同一目录追加四个大权重算子：

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  "$BUNDLE/test_welmv4_op_precision.py" dump \
  --dump-dir /tmp/welmv4_h20_reference \
  --attention-impl fa3 \
  --cases token_embedding,oe_lookup_concat,fused_moe,lm_head \
  --tokens 128 \
  --materialize-large
```

### RTX 验证

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  "$BUNDLE/test_welmv4_op_precision.py" validate \
  --op-file /path/to/candidate_welmv4_op.py \
  --dump-dir /path/to/welmv4_h20_reference \
  --attention-impl triton \
  --report /tmp/welmv4_validation.json
```

## 5. 性能结论

RTX PRO 5000 参考峰值：

- BF16 Tensor Core：268 TFLOPS
- 显存带宽：1344 GB/s
- Roofline 分界点：199.4 FLOP/Byte

当前正式测试包含：

- 117 个成功的核心算子测点
- 16 个成功的大权重算子测点
- 22 个 FA4 + Attention Sink 不支持测点

下面两张表合计覆盖全部 34 类算子，每个算子只出现一次。表中位置指向
`test_welmv4_op_precision.py` 内的实际 kernel 调用；所有 case 均由
`benchmark_welmv4_op.py:L346-L356` 调用并计时。

### 需要优化的算子（14 类）

| 算子 | 最佳测试 Shape | 延迟 | 瓶颈 | 当前性能 | 峰值利用率 | 需要优化的原因 | 测试代码位置 |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| **RMSNorm** | tokens=512 | 0.0232 ms | 带宽 | 180.8 GB/s | 13.5% | custom kernel 带宽利用率低 | `L627-L642` |
| **K RMSNorm** | tokens=512 | 0.0214 ms | 带宽 | 24.5 GB/s | 1.8% | 小 head 归一化效率很低 | `L650-L651` |
| **Partial YaRN RoPE** | tokens=512 | 0.0310 ms | 带宽 | 122.8 GB/s | 9.1% | partial rotary 数据访问效率低 | `L662-L672` |
| **Attention gate projection** | tokens=512 | 0.0327 ms | 带宽 | 65.1 GB/s | 4.8% | 窄输出 projection 效率低 | `L613-L624` |
| **Attention gate sigmoid multiply** | tokens=512 | 0.0195 ms | 带宽 | 161.7 GB/s | 12.0% | elementwise kernel 带宽利用率低 | `L657-L659` |
| **Sink prefill local** | sequence=2048 | 0.5153 ms | 计算 | 10.94 TFLOPS | 4.1% | Triton Sink attention 计算利用率低 | `L660-L661, L794-L826` |
| **Sink prefill global** | sequence=4096 | 3.2867 ms | 计算 | 15.69 TFLOPS | 5.9% | 长序列 Triton Sink attention 利用率低 | `L660-L661, L794-L826` |
| **O norm** | tokens=512 | 0.0233 ms | 带宽 | 180.3 GB/s | 13.4% | 与 RMSNorm 共用低效 custom kernel | `L627-L642` |
| **Router linear** | tokens=512 | 0.0770 ms | 带宽 | 95.3 GB/s | 7.1% | MoE router 小 GEMM 效率低 | `L652-L653` |
| **Expert-bias TopK** | tokens=512 | 0.0286 ms | 带宽 | 38.9 GB/s | 2.9% | 512 experts 的选择 kernel 效率低 | `L654-L656` |
| **Shared gate/up projection** | tokens=512 | 0.0383 ms | 计算 | 56.11 TFLOPS | 20.9% | Shared Expert GEMM 计算利用率低 | `L613-L624` |
| **SwiGLU / SiLU-and-Mul** | tokens=512 | 0.0249 ms | 带宽 | 63.1 GB/s | 4.7% | 大 intermediate elementwise kernel 效率低 | `L700-L708` |
| **Shared down projection** | tokens=512 | 0.0225 ms | 计算 | 47.73 TFLOPS | 17.8% | Shared Expert GEMM 计算利用率低 | `L613-L624` |
| **OE lookup concat** | tokens=512 | 0.0316 ms | 带宽 | 133.5 GB/s | 9.9% | embedding lookup 与 concat 数据搬运效率低 | `L598-L612` |

### 其余算子（20 类）

| 算子 | 最佳测试 Shape | 延迟 | 瓶颈 | 性能 | 峰值利用率 | 测试代码位置 |
| --- | --- | ---: | --- | ---: | ---: | --- |
| OE hash decode | 固定小规模 | 0.0341 ms | Launch | - | - | `L563-L571` |
| OE hash segments | 固定小规模 | 0.0236 ms | Launch | - | - | `L572-L582` |
| OE MTP history | 固定小规模 | 0.0269 ms | Launch | - | - | `L584-L597` |
| OE projection | tokens=512 | 0.0399 ms | 计算 | 107.72 TFLOPS | 40.2% | `L613-L624` |
| OE blend | tokens=512 | 0.0247 ms | 带宽 | 254.7 GB/s | 18.9% | `L625-L626` |
| RMSNorm + residual | tokens=512 | 0.0260 ms | 带宽 | 484.4 GB/s | 36.0% | `L627-L642` |
| PPLN RMSNorm | tokens=512 | 0.0274 ms | 带宽 | 382.5 GB/s | 28.5% | `L627-L640` |
| Norm after attention | tokens=512 | 0.0284 ms | 带宽 | 590.7 GB/s | 44.0% | `L643-L647` |
| Add residual | tokens=512 | 0.0212 ms | 带宽 | 494.2 GB/s | 36.8% | `L648-L649` |
| QKV KV-mirror projection | tokens=512 | 0.0356 ms | 计算 | 150.67 TFLOPS | 56.2% | `L613-L624` |
| QKV standard projection | tokens=512 | 0.0389 ms | 计算 | 110.51 TFLOPS | 41.2% | `L613-L624` |
| Q mirror projection | tokens=512 | 0.0296 ms | 计算 | 108.88 TFLOPS | 40.6% | `L613-L624` |
| Sink decode local | batch=128, context=512 | 0.0750 ms | 带宽 | 905.8 GB/s | 67.4% | `L660-L661, L738-L790, L828-L875` |
| Sink decode global | batch=128, context=512 | 0.0718 ms | 带宽 | 945.1 GB/s | 70.3% | `L660-L661, L738-L790, L828-L875` |
| O projection | tokens=512 | 0.0257 ms | 计算 | 125.52 TFLOPS | 46.8% | `L613-L624` |
| Router sigmoid | tokens=512 | 0.0120 ms | 带宽 | 174.1 GB/s | 13.0% | `L673-L674` |
| Shared expert add | tokens=512 | 0.0120 ms | 带宽 | 525.7 GB/s | 39.1% | `L709-L710` |
| Token embedding | tokens=512 | 0.0244 ms | 带宽 | 172.0 GB/s | 12.8% | `L548-L550` |
| Fused MoE | tokens=32 | 2.1112 ms | 带宽 | 953.7 GB/s | 71.0% | `L675-L699` |
| LM head | tokens=512 | 0.3562 ms | 计算 | 229.08 TFLOPS | 85.5% | `L711-L713` |

当前 Fused MoE 没有 RTX PRO 5000 对应的 `E=512,N=512` Triton tuning
配置，测试使用默认配置。

### 核心 benchmark

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  "$BUNDLE/benchmark_welmv4_op.py" \
  --op-file /path/to/candidate_welmv4_op.py \
  --tokens 1,32,128,512 \
  --attention-seq-lens 512,2048,4096 \
  --decode-batches 1,8,32,128 \
  --decode-contexts 512,4096,32768 \
  --attention-impl all \
  --warmup 10 \
  --repeat 50 \
  --report /tmp/welmv4_core.json
```

### 大权重 benchmark

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  "$BUNDLE/benchmark_welmv4_op.py" \
  --op-file /path/to/candidate_welmv4_op.py \
  --cases token_embedding,oe_lookup_concat,fused_moe,lm_head \
  --tokens 1,32,128,512 \
  --materialize-large \
  --warmup 10 \
  --repeat 50 \
  --strict \
  --report /tmp/welmv4_large.json
```

## 6. Attention Sink 和 FA4

- H20 使用 SGLang FA3，可以传入 Attention Sink。
- RTX PRO 5000 上 SGLang Triton Sink prefill/decode 可以运行。
- 普通 FA4 可以在 SM120 上运行。
- 当前 FA4 + Sink 首先遇到 Cutlass DSL `Int32` window 参数接口错误。
- 即使修复接口，当前 SM120 FA4 forward 也没有实例化 learnable sink 支持。
- WeLMv4 没有自己的 FA4 Sink custom kernel。

因此当前模型在 RTX PRO 5000 上应使用 Triton Attention Sink，FA4 + Sink
需要 NVIDIA/FA4 上游继续适配。
