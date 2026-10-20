---
title: 'LLM Inference Internals：从 Prompt 到 Token 的缓存与调度机制'
description: '跟着三个请求走完一次 LLM 推理：Tokenize、Prefix Cache、KV Cache、Chunked Prefill 与调度时序，以及 vLLM 与 SGLang 的差异。'
pubDate: '2026-10-19'
heroImage: '../../assets/hero-cyber.png'
---

重点不再是"Cache 的名称列表"，而是：**请求在什么时间到达，什么时候被 Tokenize，何时命中前缀，何时新建 KV，何时生成第一个 Token，什么时候回收，下一请求又怎么复用。**

> 交互教学模拟：Block Size=4，每轮 Token 预算=4。Token ID、Block ID、时序与调度顺序均为便于理解的虚构示例，不代表任何特定模型或线上运行记录。
>
> 本文有[交互版](/interactive/llm-inference-internals.html)（含单步调试器、Token 轨道动画与物理 Block Pool 可视化），建议对照阅读。

## 01｜先确定输入：模型还没开始计算时，发生了什么？

假设服务使用 Chat Completions 接口，用户在三个时间点发送请求（仅作教学）：

- **A · t0 到达**："请总结：甲乙丙丁戊己" —— 经过 Chat Template 和 Tokenizer，得到 8 个 Prompt Token ID
- **B · t1 到达**："请总结：甲乙丙丁庚辛" —— 前 4 个 Token 与 A 相同，后半不同
- **C · t3 到达**："你好" —— 3 个 Token，不能复用 A 的前缀

### Tokenize 是不是第一步？

在典型 Chat API 流程里：**JSON 解析 → Chat Template 拼接角色/特殊标记 → Tokenizer 生成 token IDs → Request 入 EngineCore → Scheduler**。对于某些 API，可以直接传 token IDs 或使用不同预处理路径，因此这不是所有入口唯一的固定顺序。

```text
HTTP messages: [{role:"user",content:"请总结：甲乙丙丁戊己"}]
  → chat template: [BOS] user ... [结束标记] assistant ...
  → tokenizer.encode(...)
  → prompt_token_ids = [101, 102, 103, 104, 105, 106, 107, 108]  # 虚构 ID
  → Request(request_id=A, prompt_token_ids, sampling_params)
  → Scheduler WAITING 队列
```

> 真实 tokenizer 不会保证一个汉字对应一个 Token；Chat Template 的特殊标记也会占 Token。所有模拟 token IDs 都是虚构的，我们关心的是位置和前缀关系。

### 此时有哪些 Cache？

| 对象 | 状态 |
|---|---|
| 模型权重 / lm_head | 服务启动时已加载；**不是 KV Cache**，通常整个服务期间保持常驻 |
| KV Cache Pool | 服务初始化时预分配了设备缓存空间；目前请求 A 还没有写入 KV |
| Prefix Cache 索引 | 可能已经有旧请求的完整块条目，也可能为空；只影响可复用前缀 |
| Block Table / Slot Mapping | 请求调度后才为当前批次建立/更新所需映射；它们是**元数据**，不是另一份 K/V |

## 02｜逐轮交互：三个请求如何共享、计算、生成、释放？

这里刻意将 A、B 的前 4 Token 设置相同，**Block Size=4**。B 在 A 的首个 4-Token Block 已经被正确计算、登记为可复用之后才被调度，从而形成明确的 Prefix Cache Hit。C 则作为完全无关的新请求展示 Cache Miss。

整个时间线可以浓缩为 8 步（[交互版](/interactive/llm-inference-internals.html)里可以逐步单步执行）：

| 步骤 | 发生了什么 |
|---|---|
| 0 | 服务已启动，KV Cache Pool 就绪；尚无任何请求的有效 KV |
| 1 | A 到达 → Chat Template → Tokenizer，进入 WAITING；Prefix Cache Miss |
| 2 | A 首轮 Chunked Prefill：预算 4 不够算完 8 个 Prompt，先算位置 0–3，分配物理块 7 写 KV |
| 3 | A 第二轮 Prefill 位置 4–7；最后位置经 lm_head → Sampler 采样出首个输出 a₁（注意 a₁ 自己的 KV 还没算） |
| 4 | B 到达：前 4 Token 命中物理块 7，**跳过重算**，只算自己的后半；A 同时可以 Decode |
| 5 | A、B Decode；C 到达但不共享前缀，Cache Miss，需独立分配 KV Block |
| 6 | C 独立 Prefill 完成并采样 c₁；A/B 的 Decode 可与 C 的 Prefill 在预算内混合同批 |
| 7 | A 结束释放引用；B 仍持有共享块 7，该块不能被不兼容的新请求覆写 |
| 8 | B、C 结束；完整可复用块留在缓存池，后续请求 D 若前缀匹配可直接命中 |

### 在这个例子中，几个机制到底何时出现？

| 时间 | Prefix Cache | Chunked Prefill | KV Cache | Decode |
|---|---|---|---|---|
| A 首轮 | Miss | Prompt 一轮预算不够，先处理部分 | 分配 Block 写 K/V | 未开始 |
| A 后续轮 | 完成的可缓存前缀进入索引 | 完成剩余 Prompt | 继续追加 KV | Prefill 完成才采样首 Token |
| B 到达 | 前 4 Token Hit，跳过重算 | 只处理未缓存的尾部 | 共享前缀 Block，写新 Block | 随后开始 |
| C 到达 | Miss | 短 Prompt 可以一轮完成 | 独立分配新 Block | 随后开始 |
| 结束 | 完整可复用块可留存 | 无关 | 引用释放 / Block 可重新利用 | EOS / length / stop |

## 03｜把"运行时间"映射到每一种机制的作用

**① 服务初始化（所有请求之前）**：加载 Embedding、Attention/FFN 权重、Final Norm、`lm_head`；建立 KV Cache Pool，初始化 Block 元数据。缓存空间可能预分配，但**没有针对某个 Prompt 的有效 KV**。权重不是所谓"lm_head cache"；计算图、workspace 是另一种运行时资源。

**② 请求预处理**：Chat Template 和 Tokenizer 获得 token IDs，Scheduler 按可调度预算和已有缓存决定当前应计算多少。Prefix Cache 命中匹配的是有效 token 序列及必要上下文，不是"字符串长得像"。

**③ Prefill / Chunked Prefill**：当前 Token 做 Embedding → 各层 QKV → **写入每层新 K/V** → Attention → FFN；长度超过预算就分多轮。下一轮 `positions` 必须从正确 offset 开始；缓存完整块以后才具备跨请求复用意义。

**④ 第一个输出 Token**：最后位置的 Hidden State 经 Final Norm → `lm_head` → Logits → Sampler，采样出首个输出 ID。它此时通常尚未产生自己的 KV。**模型选出了 Token ≠ 新 Token 的 K/V 已经写入。**

**⑤ 后续 Decode**：下一轮把已采样 Token 作为输入，计算它的 QKV，Slot Mapping 决定写入位置，Attention 从历史 Block Table 读 K/V；结束后采样下一枚 Token。Decode 一轮可能与别人的 Chunked Prefill 同批执行。

**⑥ 请求结束**：EOS / Stop / 达到 max_tokens / Abort。Request 引用释放，Block Pool 决定哪些块可以复用；Prefix Cache 开启时，合格的完整前缀 KV 可以作为缓存保留。异步执行时必须确保设备上的访存已经与生命周期处理正确协调。

> **重要关系**：KV Cache 是真实 K/V 数据；Paged KV 是存储管理方式；Prefix Cache 是跨请求复用策略；Chunked Prefill 是计算预算切分；Async Scheduling 是执行时序优化。后三者不是三种额外的 KV Tensor。

## 04｜从 Token Position 到 KV 写地址：问题通常出现在哪一跳？

```text
logical_block = floor(position / block_size)
offset        = position mod block_size
physical_block = block_table[logical_block]
slot          = physical_block × block_size + offset
```

教学示例中 Block Size=4、请求 A 的 Block Table=[7,2]，位置 5 的 `logical_block=1`，物理块 2，Offset=1，所以 Slot=9。

一轮完整调用链：

```text
Scheduler → KV Cache Manager → Block Table → Slot Mapping
Embedding / QKV → KV Cache 写入 → Attention / FFN → Final Norm / lm_head / Sampler
```

两行不是"一行做完才执行下一行"，而是控制元数据在每一轮指导数据计算。**Slot Mapping 错误会通过错误 KV 影响 Attention、Hidden States 和最终的 lm_head 输出，即便 lm_head 自己的权重与 GEMM 都正确。**

### 三种故障需要分开

| 出错位置 | 表现 | 是否一定写出 KV Pool |
|---|---|---|
| Block Table 行索引错 / 旧 Block 被误引用 | 写到或读到其他合法物理块 | 否 |
| Slot ID 直接越过有效 KV Pool | 可能非法访存 / silent corruption / 硬件异常 | 可能 |
| KV 均正确，lm_head / INT8 GEMM 错 | Logits 异常、输出异常 | 否 |

## 05｜同样三个 Prompt，SGLang 与 vLLM 哪些地方不一样？

**共同点**：二者都需要 Tokenize、Schedule、KV Cache、Prefill、Decode、Attention、lm_head 和采样。区别主要在于**缓存前缀如何建立索引、如何查找与保留、怎样结合调度策略**，而不是 SGLang 不需要 Block Table 或不会出现 KV 管理错误。

| 阶段 | vLLM V1（经典全注意力 / Block Hash Prefix Cache） | SGLang（RadixAttention / RadixCache） |
|---|---|---|
| 第一个请求 A | 正常 Prefill；完成的可缓存完整 Block 建立 hash 链关联 | 正常 Prefill；可复用前缀以 Radix 树/前缀索引方式管理 |
| B 与 A 共享一段前缀 | 按完整 KV Block 的 hash 匹配已计算前缀；可跳过命中 Block 的计算 | 沿压缩 Radix 树查找最长共享前缀；具体复用粒度受 page size 和实现配置限制 |
| 复杂前缀分叉 | hash 链表示各 Block 的前缀依赖，逐块匹配 | 树节点可在共享前缀分叉处拆分，共享公共路径 |
| 请求结束 | 释放活跃引用；完整可复用 Block 可留在 Cache Pool 等待淘汰 | 更新树节点引用或锁定/可回收状态；淘汰可复用 KV 节点 |
| 调度与命中率 | 调度器按其版本配置和预算决策，使用 Prefix Cache 提供的命中结果 | 提供 cache-aware 调度策略（如 LPM 等），是否启用取决于配置，并非永远开启 |
| 设备内核 | 不同 Attention Backend / 设备插件影响 Slot 与 KV 访存实现 | 同样因硬件、版本和 Kernel Backend 而不同 |

vLLM 的 Block Hash 索引（示意）：

```text
A: [甲乙丙丁] → [戊己··]        B: [甲乙丙丁] → [庚辛··]
     hash0      hash1                ↑命中      ↑新计算

hash(block tokens, prefix hash, context) → 已计算完整块定位
```

这是经典单类全 Attention 的概念模型；新版 vLLM 存在更复杂的 Hybrid KV Cache 实现，不应原样套到所有架构。

SGLang 的 Radix 前缀树（示意）：

```text
root
 └── 甲乙丙丁  [共享前缀 KV]
      ├── 戊己 ...  (A)
      └── 庚辛 ...  (B)

match_prefix(B) → 查找最长共享路径 → 新尾部单独计算
```

RadixCache 不表示始终能精确复用任意 Token 数，实际边界取决于 page size、RadixKey 和模型约束。

> **结论**：对于两个 Prompt 是否命中缓存，vLLM 常见路径更适合从"完整 Block Hash"理解；SGLang 更适合从"Radix 树最长前缀匹配"理解。但二者共享一个不变量：命中的必须是与当前计算条件匹配、内容有效且仍可安全使用的 KV。
>
> 对比范围：vLLM V1 的经典 Block Hash 设计与 SGLang 的 RadixAttention 思路。不同版本（尤其 Hybrid Attention、Mamba、Sliding Window 或 Multi-Modal）可能扩展或更换缓存机制。

## 06｜从推理机制到正确性：缓存链路容易在哪些环节出错？

了解缓存机制不仅是为了性能调优，也是为了理解 **Serving Runtime Correctness（推理运行时正确性）**。把常见风险放回一次请求的执行时间线中：区分逻辑错误、数据错误与物理越界，而不是把所有问题笼统称作"KV Cache 污染"。

| 运行阶段 | 关键状态 | 可能出现的错误 | 如何验证 |
|---|---|---|---|
| 请求调度 | Token Position、已计算 Token 数、调度批次 | 进度或 Chunk 边界不一致 | 比较调度进度与实际计算 Position |
| 前缀复用 | Prefix Hash、有效 KV 块、引用关系 | 错误命中或共享不当 | 检查 Hash、前缀长度与引用生命周期 |
| 物理 Block 分配 | Block Pool、Block Table、有效块数 | 过期映射、过早复用、错误索引 | 检查有效长度与合法共享条件 |
| 本轮 KV 写入 | Position、Slot Mapping、设备缓存布局 | 写入错误的有效 Slot 或越界 Slot | 与独立参考映射逐 Token 对照 |
| Attention 读取 | Block Table、序列长度、缓存内容 | 读取其他位置的 KV，造成错误 Attention | 对照 KV 快照和 Attention 输出 |
| 后续计算 | Hidden States、Final Norm、lm_head、Logits | 上游错误传播，或输出投影本身异常 | 逐层对照最早出现差异的 Tensor |

**Prefix Cache 未命中，不代表没有 KV Cache**：即使某个请求没有命中已有前缀，也仍然要分配 KV Block、建立映射、写入新 K/V，并在 Decode 时读取缓存。"无 Prefix Hit"和"没有经过 KV Cache 管理"是两个完全不同的概念。

**Block Table 中存在旧值，是否必然出错？** 不必然。很多实现借助有效长度或掩码界定可访问范围，表尾留有旧值本身不等于错误。只有有效 Token 在不符合约定的情况下读到了无效列，才构成索引与生命周期问题。同样，多个请求引用相同物理 Block，也可能是合法的前缀共享。

**为什么异步执行需要额外关注？** Continuous Batching 和 Async Scheduling 允许不同阶段重叠。正确性要求 CPU 的调度元数据、Host→Device 传输以及设备 Kernel 的消费顺序具备有效依赖；异步传输本身是正常优化，不能仅凭 non_blocking 标志认定存在竞态。

### 通用的排查顺序

1. 先验证 Scheduler 的 Token 进度与实际请求 Position 是否一致
2. 再检查 KV Block 的有效范围、共享及释放语义
3. 对照 Block Table 与 Slot Mapping，确认每个有效 Token 的 KV 写入位置
4. 比较 KV 内容、Attention 输出、最后 Hidden States、Final Norm 和 lm_head Logits，定位首次偏差
5. 对低概率问题采用隔离环境定向测试，避免把"未复现"误写成"彻底排除"

> 上述内容是通用机制分析，不代表特定版本的 vLLM、SGLang 或硬件后端存在对应缺陷。实际数据结构与不变量应以目标版本源码和 Kernel 契约为准。

## 07｜参考资料与实现边界

本教程结合 vLLM V1 与 SGLang 的公开设计，采用同一组教学请求解释缓存与调度的共性和差异。框架 API、后端实现、缓存粒度、调度优化会随版本变化；文中的示意图和伪数据不能代替具体版本的源码验证。

1. [vLLM V1 Prefix Caching Design](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)：Block Hash、引用计数、Free Block Queue
2. [vLLM V1 Scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)：num_computed_tokens、计算预算、KV 分配
3. [vLLM Scheduler 配置](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)：Chunked Prefill / Async Scheduling
4. [SGLang RadixCache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)：Radix 树前缀索引
5. [SGLang Unified Radix Cache 官方技术说明](https://www.sglang.io/blog/unified-radix-cache)

---

*所有 Token ID、请求步骤与 Block ID 都是教学模拟数据。*
