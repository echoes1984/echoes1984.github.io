---
title: 'MiniMax-H3 Transformer Block：真实矩阵变换逐步拆解'
description: '按官方 H3 配置的真实维度，从 packed hidden states 走完一个 Block：AdaLN、QKV、QK Norm、3D RoPE、全注意力、SwiGLU FFN 与两次门控残差。'
pubDate: '2026-10-19'
heroImage: '../../assets/hero-dit.png'
tags: ['minimax-h3']
---

> 本文有[交互版](/interactive/minimax-h3-dit-block.html)（每一步可展开/收起细节），建议对照阅读。

按官方 H3 配置中的真实维度，从 packed hidden states 开始，走过 AdaLN、QKV、QK Norm、3D RoPE、全注意力、FFN 和两次门控残差。

## 先看关键配置

| 指标 | 数值 | 说明 |
|---|---|---|
| hidden_size | 5,376 | 残差流宽度 H |
| 注意力头 | 56 × 128 | 头数 × 每头维度 |
| attention inner | 7,168 | 56 × 128 |
| FFN 中间维度 | 14,336 | SwiGLU |
| Transformer blocks | 50 | 主干 |

## 先看完整数据流

令 packed 序列长度为 **T**。T 随视频帧数、分辨率、音频长度和输入内容变化，所以不固定；H3 的 block 主干始终把每个 token 表示为 5,376 维。

```text
X [T, 5376]
  → AdaLN 调制 → H [T, 5376]
  → QKV 投影 [T, 21504] → Q,K,V 各 [T, 56, 128]
  → Q/K RMSNorm + 3D MM-RoPE
  → packed non-causal self-attention → [T, 56, 128]
  → flatten [T, 7168] → output projection [T, 5376]
  → X1 = X + gate_msa ⊙ attn_out
  → AdaLN 调制 → H2 [T, 5376]
  → SwiGLU FFN: 5376 → 28672 → split 2×14336 → 14336 → 5376
  → Xout = X1 + gate_mlp ⊙ ffn_out
```

> 主模型配置：hidden_size=5376、num_attention_heads=56、attention_head_dim=128、ffn_dim=14336。注意：H3 的注意力内部宽度 56×128=7168，**大于** hidden_size 5376。

## 逐步展开真实矩阵形状

### 1. AdaLN：从 timestep 条件生成六组调制向量

每个主 block 有自己的 AdaLN 投影。官方配置的 timestep MLP 输出维度是 **2,688**；AdaLN 一次输出 96,768 个数，恰好是 **3 modality × 6 组 × 5,376 hidden**。

```text
E_t [N_t, 2688] × W_adaln [2688, 96768] + b [96768]
→ A [N_t, 96768]
→ reshape [N_t, 3, 6, 5376]
```

reshape 后每行六个 5,376 维向量，含义是：`shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`。官方代码按这个顺序排列六组。token 的 modality tag 和 timestep index 决定查哪一行。

> PyTorch Linear 的 weight 通常以 [out_features, in_features] 存储，因此 checkpoint 参数张量对应 [96768, 2688]；上式 W 按数学乘法方向书写。

若一次 forward 的 packed tokens 有 T 行，索引查表后，六个调制张量各自为 `[T, 5376]`。同一 timestep、同一 modality 的 token 可以复用同一行调制参数。

> **不要把 AdaLN 参数理解成 token 内容。** 它是由扩散 timestep 条件投影出的控制向量；索引步骤把相应控制向量分配给 packed 序列里的每个 token。三种 modality 的具体 tag 编码顺序应以所用版本的模型代码为准。

### 2. Norm1 + AdaLN：逐通道缩放、平移

残差流维持 `[T, 5376]`；归一化不改变矩阵形状。每个 token 的 5,376 个通道独立参与 RMSNorm 归一化，然后乘以 (1+scale_msa)，再加 shift。

```text
H = RMSNorm(X) ⊙ (1 + scale_msa[T,5376]) + shift_msa[T,5376]
→ H [T,5376]

H[i,j] = RMSNorm(X[i,:])[j] × (1 + scale_msa[i,j]) + shift_msa[i,j]
其中 i ∈ [0,T)，j ∈ [0,5376)
```

### 3. QKV 投影：从残差宽度映射到注意力宽度

注意力每个头 128 维，56 个头拼接后宽度为 **7,168**。所以 Q、K、V 的总投影宽度是 **21,504**。

```text
H [T, 5376] × W_qkv [5376, 21504]
→ QKV [T, 21504]
→ split(last dim, [7168,7168,7168])
→ Q [T,7168], K [T,7168], V [T,7168]
→ reshape
→ Q,K,V 各 [T,56,128]
```

逻辑矩阵 W 按输入维×输出维表示为 [5376,21504]。若查看 PyTorch Linear 的 weight 参数存储，常见形状是 [21504,5376]。原始 Diffusers 权重也可能分别存 Q/K/V 投影，运行时再融合。

### 4. Q/K RMSNorm + 3D MM-RoPE：处理每个头的内容和位置

**Q/K RMSNorm**：让每个注意力头的 Q 和 K 在进入点积前按各自的 RMSNorm 规则归一化（V 不做这一步）。

```text
Q [T,56,128] → 每个 (token, head) 对 128 维归一化
K [T,56,128] → 同样处理
Norm 权重：每个 Q/K 通道向量 [128]
```

**3D MM-RoPE**：把时空位置信息注入 Q 和 K，使注意力分数能反映 token 的相对位置。

```text
位置坐标：每个 token 有 (t, h, w)
rope_freq_dim = 16 / 轴
3 轴 × 16 频率 × 每频率 2 个旋转通道 = 96
每头 128 维中：96 维参与旋转，32 维不旋转
```

形状保持不变：`Q′, K′, V: [T,56,128]`。

### 5. Packed self-attention：每个 token 从序列读取信息

对某个 head (h)，Q、K、V 的形状分别是 `[T,128]`。先对 token 两两做 QK 点积，再对分数做 softmax，最后用概率加权 V。

```text
每个 head h：
Q′_h [T,128] × K′_hᵀ [128,T]
→ scores_h [T,T] / √128
→ softmax(scores_h) = P_h [T,T]
P_h [T,T] × V_h [T,128]
→ O_h [T,128]

56 个 head 合并：O [T,56,128]
flatten heads → [T,7168]
```

> **cu_seqlens 的意义**：它标出 packed batch 中每段序列的边界。若是一条 H3 请求，概念上可看作一个长度 T 的 attention document；若多个独立样本被打包，段与段之间不能互相注意。H3 主注意力是 full self-attention、non-causal；模态 token 位于统一 packed 序列中。

真实 kernel 通常按 tile 计算 score/softmax/value，不会把完整 `[56,T,T]` 保存到显存。这个矩阵是理解数学关系的概念形状。

### 6. Attention 输出投影 + 第一次门控残差

7,168 → 5,376，然后按 gate_msa 逐元素加回原 X：

```text
O [T,7168] × W_o [7168,5376]
→ attn_out [T,5376]

X1 = X + gate_msa ⊙ attn_out
X [T,5376], gate_msa [T,5376], attn_out [T,5376]
→ X1 [T,5376]
```

`gate_msa` 也是逐 token、逐 hidden channel 的门控向量，控制注意力分支更新的大小。

### 7. Norm2 + AdaLN：准备 FFN 输入

结构与 Attention 前归一化类似，但使用 MLP 对应的 scale/shift：

```text
RMSNorm(X1) [T,5376]
H2 = RMSNorm(X1) ⊙ (1 + scale_mlp[T,5376]) + shift_mlp[T,5376]
→ H2 [T,5376]
```

这里的 scale/shift 与步骤 2 的 Attention 参数不同。

### 8. SwiGLU FFN：扩展通道、门控非线性、压回残差宽度

5376 → 28672 → 两路各 14336 → 5376。FFN 对每个 token 独立做特征变换；这里不在 token 之间交换信息。第一层同时产生 gate 与 up 两路。

```text
H2 [T,5376] × W_fc1 [5376,28672]
→ Z [T,28672]
→ split Z = [G | U]
G [T,14336], U [T,14336]

Y = SiLU(G) ⊙ U
→ Y [T,14336]

Y [T,14336] × W_fc2 [14336,5376]
→ ffn_out [T,5376]
```

> 数学写法中的权重按 [in,out] 排列；PyTorch Linear weight 参数一般按 [out,in] 存储，因此 fc1 weight 常见为 [28672,5376]、fc2 weight 常见为 [5376,14336]。

### 9. 第二次门控残差：得到 block 输出

```text
Xout = X1 + gate_mlp ⊙ ffn_out
X1 [T,5376], gate_mlp [T,5376], ffn_out [T,5376]
→ Xout [T,5376]
```

block 的输入输出形状一致，都是 `[T,5376]`。50 个主 Transformer blocks 依次堆叠时，一个 block 的 `Xout` 就是下一个 block 的 `X`。

## 本页涉及的矩阵尺寸总表

| 位置 | 逻辑运算 / 张量 | 真实形状 | 解释 |
|---|---|---|---|
| 残差流 | X、X1、Xout | [T, 5376] | 每个 packed token 的隐藏表示 |
| AdaLN 输入 | timestep embedding | [N_t, 2688] | 时间条件 MLP 的输出宽度 |
| AdaLN 投影 | linear 输出 | [N_t, 96768] | 3 modality × 6 调制组 × 5376 |
| AdaLN 查表 | 每组 scale/shift/gate | 各 [T, 5376] | 按 token 的 timestep/modality 索引选取 |
| QKV | 融合 QKV 输出 | [T, 21504] | 三份 [T,7168] |
| Heads | Q、K、V 各自 | [T, 56, 128] | 56 个头，每头 128 维 |
| Attention | 单头 score（概念形状） | [T, T] | 实际通常分块算，不完整物化 |
| Attention 输出 | 合并 heads | [T, 7168] | 56×128 |
| FFN 第一层 | gate + up | [T, 28672] | 2×14336 |
| SwiGLU 中间态 | SiLU(gate) ⊙ up | [T, 14336] | FFN 中间宽度 |
| FFN 输出 | fc2 | [T, 5376] | 回到残差流宽度 |

## 阅读时记住三件事

- **残差流**：一直是 5,376 维。两次子层都把结果投影回 5,376，再加回残差。
- **Attention**：注意力宽度是 7,168。它由 56 个 128 维头组成；与 5,376 不同。
- **AdaLN**：输出控制向量。每 token 查到六个 5,376 维向量，分别调制 Attention 与 FFN。
- **动态长度**：T 不是模型常量。输入分辨率、视频帧数、音频长度和 packed layout 会改变 T。

> **关于多卡切分**：上面是单个 block 的全局逻辑形状。TP/USP 下每张卡只持有切分后的权重或 token/head 分片，局部矩阵形状会变化；全局模型配置不变。若要画具体每卡矩阵，需要再指定 TP、Ulysses degree 和是否 padding。

## 参考配置与实现

1. [MiniMax-H3 官方 FL2VA transformer config](https://huggingface.co/MiniMaxai/MiniMax-H3/blob/main/FL2VA/transformer/config.json)：hidden size、head 数、head dim、FFN dim、AdaLN 输出维度等
2. [Diffusers MiniMax-H3 实现](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_minimax_h3.py)：AdaLN 调制顺序、QKV/heads、QK Norm、3D RoPE、packed self-attention 和 block 数据流

---

*矩阵形状按 H3 主 DiT Transformer block 的配置展示；忽略 batch 维、并行分片和具体权重数值。权重矩阵公式以 [in,out] 数学方向书写，框架参数存储常见为转置形状。*
