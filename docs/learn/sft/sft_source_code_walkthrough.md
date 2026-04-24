# Slime SFT (Supervised Fine-Tuning) 源码解读

## 概述

Slime 框架虽然以 RL 后训练为核心设计目标，但通过巧妙复用 RL 训练管线中的组件，也原生支持 SFT。SFT 模式本质上是将 RL 管线中的「rollout 数据生成 → 训练」流程简化为「直接从数据集构造 token + loss mask → 训练」，跳过了 SGLang 推理引擎、reward 计算、advantage 计算等 RL 专有步骤。

## 核心源码文件一览

| 文件 | 职责 |
|------|------|
| `slime/rollout/sft_rollout.py` | SFT 数据生成函数，负责将多轮对话 tokenize 并生成 loss mask |
| `slime/utils/mask_utils.py` | `MultiTurnLossMaskGenerator`：多轮对话 loss mask 生成核心类 |
| `slime/backends/megatron_utils/loss.py` | `sft_loss_function`：SFT 损失函数（负对数似然） |
| `slime/backends/megatron_utils/cp_utils.py` | `get_sum_of_sample_mean`：loss 归约函数，处理 loss mask 加权与 context parallelism |
| `slime/utils/ppo_utils.py` | `compute_log_probs` / `calculate_log_probs_and_entropy`：从 logits 计算 log 概率 |
| `slime/backends/megatron_utils/actor.py` | `train_actor` 方法：当 `compute_advantages_and_returns=False` 时跳过 RL 相关计算 |
| `slime/ray/rollout.py` | `RolloutManager`：`debug_train_only` 模式下跳过 SGLang 引擎创建 |

---

## 一、SFT 模式的启用方式

通过一组 CLI 参数组合启用 SFT 模式，典型配置（参考 `scripts/run-qwen3-4B-base-sft.sh`）：

```bash
SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout  # 指定SFT数据生成函数
   --prompt-data /path/to/data.parquet    # 数据路径
   --input-key messages                   # 数据中的对话字段名
   --rollout-shuffle                      # 随机打乱数据
   --num-epoch 3                          # 训练轮数
   --rollout-batch-size 128               # 每批样本数
   --global-batch-size 128                # 全局batch大小

   --loss-type sft_loss                   # 使用SFT损失函数
   --calculate-per-token-loss             # 按token粒度计算loss
   --disable-compute-advantages-and-returns  # 禁用RL的advantage/return计算
   --debug-train-only                     # 不启动SGLang推理引擎，仅做训练
)
```

这些参数的关键作用：

1. **`--rollout-function-path slime.rollout.sft_rollout.generate_rollout`**：将默认的 SGLang 推理 rollout 替换为纯 CPU 端的 tokenize + mask 生成。
2. **`--loss-type sft_loss`**：将损失函数从 `policy_loss`（PPO）切换为 `sft_loss`（NLL）。
3. **`--disable-compute-advantages-and-returns`**：跳过 advantage estimation（GRPO/PPO 等）。
4. **`--debug-train-only`**：不创建 SGLang 引擎实例，节省 GPU 资源——所有 GPU 都分配给 Megatron 训练。
5. **`--calculate-per-token-loss`**：按 token 级别而非 sample 级别归一化 loss，这是 SFT 的标准做法。

---

## 二、SFT 数据生成：`sft_rollout.py`

> 源文件：`slime/rollout/sft_rollout.py`

这是 SFT 的核心数据处理函数。在 RL 模式中，`generate_rollout` 函数负责调用 SGLang 引擎生成响应并计算 reward；而在 SFT 模式中，这个函数被替换为纯粹的 tokenization 和 loss mask 构造。

### 函数签名

```python
def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
```

- `args`：全局参数
- `rollout_id`：当前 rollout 的步数 ID
- `data_buffer`：数据源，从中按批次获取样本
- `evaluation`：SFT rollout 不支持 evaluation 模式（`assert not evaluation`）

### 核心流程

```
1. 初始化 tokenizer 和 MultiTurnLossMaskGenerator（全局单例，只初始化一次）
2. 从 data_buffer 获取 rollout_batch_size 个样本
3. 对每个样本：
   a. 读取 sample.prompt（多轮对话 messages 列表）
   b. 调用 MASK_GENERATOR.get_loss_mask(messages) 得到 token_ids 和 loss_mask
   c. 计算 response_length（从第一个 mask=1 的位置到末尾的长度）
   d. 将结果写回 sample 对象
4. 返回处理后的 samples 列表
```

### 关键代码解析

```python
token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)
```

这一行将多轮对话转换为：
- `token_ids`：完整的 token 序列（包含所有轮次的 prompt + response）
- `loss_mask`：与 `token_ids` 等长的 0/1 列表，只有 assistant 回复的 token 位置为 1

```python
response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
sample.loss_mask = loss_mask[-response_length:]
```

`response_length` 的定义：从 loss_mask 中**第一个 1** 出现的位置到末尾的长度。注意这并不等于 "assistant 回复的 token 数"——在多轮对话中，中间的 user 轮次的 token 也被包含在这个区间内（它们的 loss_mask 值为 0，不参与 loss 计算），但它们属于 response_length 的范围。

#### 直观示例：response_length 与实际训练 token 的关系

假设一个两轮对话 tokenize 后的完整序列和 loss_mask：

```
tokens:    [SYS][U1][U1][U1] [A1][A1][A1] [U2][U2] [A2][A2][A2][A2]
loss_mask: [ 0 ][ 0][ 0][ 0] [ 1][ 1][ 1] [ 0][ 0] [ 1][ 1][ 1][ 1]
                               ↑ 第一个 1 出现的位置
           |← prompt_length →||←────── response_length = 9 ──────→|
```

- `response_length = 9`（从首个 1 到末尾的总长度）
- 实际参与 loss 计算的 token 数 = `sum(loss_mask) = 7`（只有 mask=1 的位置）
- 中间的 `[U2][U2]` 属于 response_length 区间，但 mask=0，不贡献 loss

这种设计的目的：Megatron 训练需要知道"从哪里开始切 response"（即 prompt 和 response 的分界点），而 `loss_mask` 在这个区间内进一步精细控制哪些 token 真正参与损失计算。

Sample 上设置的字段：
- `sample.tokens`：完整 token 序列
- `sample.response_length`：从首个 loss=1 位置到末尾的长度
- `sample.reward = 0`：SFT 模式不需要 reward，固定为 0
- `sample.loss_mask`：只保留 response 部分的 mask（长度 = response_length）

---

## 三、Loss Mask 生成：`mask_utils.py`

> 源文件：`slime/utils/mask_utils.py`

`MultiTurnLossMaskGenerator` 是 SFT 的核心工具类，负责将多轮对话结构转换为 token 级别的 loss mask。它支持多种 tokenizer 类型。

### 支持的 tokenizer 类型

通过 `--loss-mask-type` 参数指定，目前支持：

| 类型 | 方法 | 适用模型 |
|------|------|----------|
| `qwen` | `gen_multi_turn_loss_mask_qwen` | Qwen 系列（默认） |
| `qwen3` | `gen_multi_turn_loss_mask_qwen3` | Qwen3 系列 |
| `qwen3_5` | `gen_multi_turn_loss_mask_qwen3_5` | Qwen3.5 系列 |
| `distill_qwen` | `gen_multi_turn_loss_mask_distill_qwen` | DeepSeek 蒸馏 Qwen |

### Qwen 系列的 Mask 生成原理

以 `gen_multi_turn_loss_mask_qwen` 为例，核心逻辑：

```python
for i, message in enumerate(messages):
    # 1. 对每条消息单独 apply_chat_template 得到 token 序列
    message_ids = tokenizer.apply_chat_template([message], tokenize=True)

    # 2. 去掉重复的 system prefix（第一条消息之后的消息都会被加上系统前缀）
    if message["role"] != "system" and i > 0:
        message_ids = message_ids[self.system_message_length:]

    # 3. 根据角色生成 mask
    if message["role"] == "assistant":
        # assistant 回复：generation prompt token 不计入 loss，内容 token 计入 loss
        loss_mask = [0] * gen_token_length + [1] * (len(message_ids) - gen_token_length)
    else:
        # user/system 消息：全部不计入 loss
        loss_mask = [0] * len(message_ids)

    # 4. 支持 step_loss_mask 字段控制特定轮次不参与 loss
    if message.get("step_loss_mask", 1) != 1:
        loss_mask = [0] * len(message_ids)
```

#### 关键技巧：`system_message_length` 的计算

由于 `apply_chat_template` 会为每条消息自动添加系统前缀（如 `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n`），而多轮对话中只需要一次系统前缀。因此在初始化时通过探测两条测试消息来计算系统前缀的 token 长度：

```python
def get_system_message_length(self):
    # 用两条相同的测试消息来推算系统前缀长度
    test_messages = [
        {"role": "user", "content": "FOR TESTING ONLY"},
        {"role": "user", "content": "FOR TESTING ONLY"},
    ]
    # ... 通过比对 token 位置差异推算 system_message_length 和 gen_token_length
```

**展开说明：为什么需要这个探测？**

Hugging Face 的 `apply_chat_template` 设计初衷是渲染一段**完整**对话——它会自动在开头插入系统提示词。如果我们对多轮对话的每条消息分别调用 `apply_chat_template`，每条消息都会各自带上一份系统前缀，导致拼接后系统前缀被重复多次。

探测方法的思路：用两条一模一样的测试消息调用 `apply_chat_template`，然后在 token 序列中找到两次测试文本出现的位置。两次出现之间的"多余部分"就是每条非首消息会被自动添加的系统前缀。在后续处理中对非首消息做 `message_ids[system_message_length:]` 裁剪，就能消除重复。

类似地，`gen_token_length` 通过对比 `add_generation_prompt=True/False` 两种情况来测算 `<|im_start|>assistant\n` 等 generation prompt 的 token 数，这些 token 是模板自动添加的"角色标记"而非模型实际生成的内容，所以不应参与 loss 计算。

### Qwen3 的处理

Qwen3 (`gen_multi_turn_loss_mask_qwen3`) 的逻辑与 Qwen 基本相同，但针对 Qwen3 tokenizer 的一个特性做了适配：Qwen3 的 `apply_chat_template` 无法通过简单裁剪来去掉系统前缀（因为不同消息的 tokenization 结果可能因上下文不同而有差异）。

解决方案：对每条消息都拼接一个已知的"锚点消息"（`prefix_message`），用锚点消息的已知 token 序列作为参照来精确切割出当前消息的 token：

```python
# 第一条消息：在后面拼锚点，然后裁掉锚点部分
tailed_message_ids = tokenizer.apply_chat_template(
    [message, prefix_message], tokenize=True, tools=tools)
message_ids = tailed_message_ids[:-len(prefix_token_ids)]

# 后续消息：在前面拼锚点，然后裁掉锚点部分
prefixed_message_ids = tokenizer.apply_chat_template(
    [prefix_message, message], tokenize=True)
message_ids = prefixed_message_ids[len(prefix_token_ids):]
```

这种"锚点拼接-裁剪"的方法避免了直接依赖 `system_message_length` 常量，对 tokenizer 的 chat template 实现更加鲁棒。

### Qwen3.5 的特殊处理

Qwen3.5 (`gen_multi_turn_loss_mask_qwen3_5`) 采用了完全不同的策略——**字符级定位 + offset_mapping 映射**：

```
步骤 1: 渲染完整对话为纯文本
         apply_chat_template(messages, tokenize=False) → rendered_text

步骤 2: 在字符级别构建 mask
         遍历 rendered_text，定位每个 <|im_start|>assistant\n ... <|im_end|> 区间
         将 assistant 内容区间的字符标记为 1，其余为 0

步骤 3: 字符级 mask → token 级 mask
         利用 tokenizer 的 offset_mapping（每个 token 对应原文的 [start, end) 字符区间）
         如果一个 token 覆盖的字符区间中有任何 mask=1 的字符，该 token 的 loss mask = 1
```

**为什么 Qwen3.5 不能用前两种方法？**

Qwen3.5 的 chat template 包含 `<think>` 思考链标记，这些标记会改变 tokenization 的上下文，使得"逐条消息分别 tokenize 再拼接"的结果与"整体 tokenize"的结果不一致。因此必须先整体 tokenize，再通过字符级定位来回标 mask。

```python
# 思考链前缀不计入 loss
if rendered_text[content_start:content_start + len(think_prefix)] == think_prefix:
    mask_start = content_start + len(think_prefix)  # 跳过 <think>\n
else:
    mask_start = content_start
```

这意味着在 Qwen3.5 SFT 中，模型的思考过程（`<think>...</think>` 块）**不参与 loss 计算**，只有思考之后的实际回复内容才会被训练。这在训练"思考型"模型时是关键设计：我们希望模型学会生成正确答案，但不强制其复现特定的思考过程。

**offset_mapping 映射的细节**：tokenizer 在 tokenize 时返回每个 token 对应的原始字符区间 `(start, end)`。通过前缀和技巧（prefix sum）高效判断每个 token 是否覆盖了 mask=1 的字符：

```python
# 字符级 mask 的前缀和，用于 O(1) 区间查询
char_mask_prefix_sum = [0]
for value in char_mask:
    char_mask_prefix_sum.append(char_mask_prefix_sum[-1] + value)

# 对每个 token，检查其字符区间内是否有 mask=1
for start, end in offset_mapping:
    if end <= start:
        loss_mask.append(0)
    else:
        # prefix_sum[end] - prefix_sum[start] > 0 说明区间内存在 mask=1 的字符
        loss_mask.append(1 if char_mask_prefix_sum[end] - char_mask_prefix_sum[start] > 0 else 0)
```

### `step_loss_mask` 字段

数据中的每条消息可以包含 `step_loss_mask` 字段（默认值 1）。当某条 assistant 消息的 `step_loss_mask != 1` 时，该消息的所有 token 的 loss mask 将被强制设为 0，即该轮回复不参与 loss 计算。这在需要选择性训练特定轮次时非常有用。

### 多模态支持

`get_loss_mask_with_multimodal_alignment` 方法用于视觉语言模型（VLM）的 SFT。它先对纯文本生成 loss mask，然后通过前补零的方式对齐到包含图像 token 的完整 input_ids 序列：

```python
diff = len(input_ids) - len(loss_mask_text)  # 图像token引入的额外长度
loss_mask = [0] * diff + loss_mask_text       # 图像token不参与loss
```

---

## 四、SFT 损失函数：数学原理与代码实现

> 源文件：`slime/backends/megatron_utils/loss.py`，第 892-940 行

### 4.1 数学公式

SFT 的损失函数是标准的**负对数似然（Negative Log-Likelihood, NLL）**，又叫**交叉熵损失（Cross-Entropy Loss）**。

#### 直觉理解

想象你在教一个学生回答问题。你给出标准答案"2+2 等于 4"，模型会对词表中每个词产生一个"猜测概率"。SFT 的目标就是让模型在每个位置上，对正确答案词的猜测概率尽可能高。

#### 单个 token 的损失

对于序列中的第 $t$ 个 token，模型输出 logits 向量 $z_t \in \mathbb{R}^{|V|}$（$|V|$ 是词表大小），通过 softmax 转换为概率分布：

$$P(w | z_t) = \frac{\exp(z_t[w])}{\sum_{w' \in V} \exp(z_t[w'])}$$

其中 $z_t[w]$ 表示 logits 向量中对应词 $w$ 的分量。

该位置的**对数概率（log probability）** 为：

$$\log P(y_t | z_t) = z_t[y_t] - \log \sum_{w' \in V} \exp(z_t[w'])$$

其中 $y_t$ 是该位置的真实目标 token。右边的第二项就是 log-sum-exp，也叫 log partition function。

#### 整体 SFT 损失

设一个 batch 中有 $N$ 个样本，第 $i$ 个样本的 response 区间内有 $R_i$ 个 token，loss mask 为 $m_i \in \{0, 1\}^{R_i}$，则：

**Per-sample 归一化**（`--calculate-per-token-loss` 关闭时）：

$$\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{N} \frac{\sum_{t=1}^{R_i} m_{i,t} \cdot \log P(y_{i,t} | x_{i,<t})}{\max\left(\sum_{t=1}^{R_i} m_{i,t},\ 1\right)}$$

每个样本内部按有效 token 数平均（分母 clamp 到 1 防止除零），然后各样本的 loss 直接相加。

**Per-token 归一化**（`--calculate-per-token-loss` 开启时，SFT 推荐）：

$$\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{N} \sum_{t=1}^{R_i} m_{i,t} \cdot \log P(y_{i,t} | x_{i,<t})$$

各 token 的 loss 直接相加（不在样本层面做平均），最终由 Megatron 的梯度累积机制统一除以全局有效 token 总数。

**为什么 SFT 推荐 per-token？** 如果用 per-sample 归一化，一个 500-token 的回复和一个 10-token 的回复对总 loss 的贡献权重相同（都是"一个样本"）。这意味着短回复中每个 token 的梯度贡献被放大了 50 倍。Per-token 归一化让每个 token 的贡献权重相等，训练更加稳定。

### 4.2 从 logits 到 log probability 的计算路径

代码中 log probability 的计算经过了三层调用：

```
sft_loss_function
  → get_log_probs_and_entropy          (loss.py: 提取 response 区间的 logits)
    → calculate_log_probs_and_entropy  (ppo_utils.py: 分 chunk 处理避免 OOM)
      → compute_log_probs             (ppo_utils.py: 调用 Megatron fused kernel)
        → fused_vocab_parallel_cross_entropy  (Megatron: GPU fused 实现)
```

最底层调用的是 Megatron 的 `fused_vocab_parallel_cross_entropy`，这是一个 fused CUDA kernel，它将 softmax + cross-entropy 合并为一步计算（而不是先算 softmax 概率再取 log），避免了大词表下的数值不稳定和显存浪费。

> **为什么需要 fused kernel？**
>
> 朴素实现需要先将 logits 过 softmax 得到一个 $|V|$ 维的概率向量（如 Qwen3 的词表大小是 151,936），然后再取 log。这个中间概率向量非常大，需要额外的显存存储。Fused kernel 在一次 GPU kernel 调用中完成"数值稳定的 log-softmax → 取目标 token 位置"，既省显存又更快。

在张量并行（Tensor Parallel）环境下，词表被切分到不同 GPU 上（例如 4 卡 TP 时每张卡只持有 ~38K 个词的 logits）。`fused_vocab_parallel_cross_entropy` 自动处理了跨卡的 log-sum-exp 归约——先在每张卡上算局部 max 和局部 exp-sum，再通过 all-reduce 得到全局结果。

### 4.3 get_responses：如何从拼接序列中精确切出 response logits

`get_responses` 函数（`loss.py:34`）是理解 SFT loss 计算的关键。Megatron 在前向传播时将一个 micro-batch 内所有样本的 token 拼接为一个长序列输入（形状 `[1, T, V]`），`get_responses` 负责从中切出每个样本 response 区间的 logits。

#### 自回归语言模型的 offset-by-one

一个需要注意的细节是 **logits 和 tokens 之间存在一位偏移**：

```
位置:     0     1     2     3     4
tokens:   [A]   [B]   [C]   [D]   [EOS]
logits:  P(·|∅) P(·|A) P(·|AB) P(·|ABC) P(·|ABCD)
```

位置 $t$ 的 logits 预测的是位置 $t+1$ 的 token。因此在切取 response 区间时，代码做了 `-1` 偏移：

```python
logits_chunk = logits[start - 1 : end - 1]  # logits 偏移 -1
tokens_chunk = tokens[-response_length:]     # tokens 不偏移
```

如果 response 从位置 5 开始到位置 9 结束，那么需要的 logits 是位置 4~8（预测位置 5~9 的 token）。

#### Context Parallelism 的处理

当启用 Context Parallelism（CP）时，长序列被切分到多个 GPU 上并行处理。这意味着一个 response 区间可能跨越多个 CP rank 的边界。`get_responses` 支持三种 CP 模式：
- **CP=1**：直接切片
- **allgather_cp**：全局拼接后按 contiguous chunk 切分，每个 rank 持有全局序列的一个连续片段
- **zigzag ring attention**：序列按锯齿形分布到 2 个 chunk 上，需要分别从两个 chunk 中提取对应区间再拼接

### 4.4 sum_of_sample_mean：loss mask 加权归约

> 源文件：`slime/backends/megatron_utils/cp_utils.py`

`sum_of_sample_mean` 是连接 loss mask 和最终 loss 值的关键归约函数。它由 `get_sum_of_sample_mean` 工厂函数根据配置生成。

**Per-sample 模式**（`calculate_per_token_loss=False`）：

```python
def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
    return sum([
        (x_i * loss_mask_i).sum() / torch.clamp_min(loss_mask_i.sum(), 1)
        for x_i, loss_mask_i in zip(x.split(response_lengths), loss_masks)
    ])
```

对每个样本 $i$：将 log_probs 乘以 loss_mask（屏蔽不参与 loss 的 token），求和后除以该样本有效 token 数。然后所有样本的结果相加。`clamp_min(..., 1)` 防止某个样本的 loss_mask 全为 0 时出现除零。

**Per-token 模式**（`calculate_per_token_loss=True`）：

```python
def sum_of_token(x: torch.Tensor) -> torch.Tensor:
    return sum([
        (x_i * loss_mask_i).sum()
        for x_i, loss_mask_i in zip(x.split(response_lengths), loss_masks)
    ])
```

直接对所有样本的所有有效 token 的 log_probs 求和，不在样本级别做归一化。最终在 `loss_function` 外层由 Megatron 的 `num_tokens` normalizer 完成全局归一化。

### 4.5 loss_function 外层：Megatron 梯度累积的适配

> 源文件：`slime/backends/megatron_utils/loss.py`，第 943-1031 行

`sft_loss_function` 返回的 loss 还需要经过外层 `loss_function` 的 rescale，才能正确对接 Megatron 的梯度累积（gradient accumulation）机制：

```python
if not args.calculate_per_token_loss:
    # Per-sample 模式：loss 除以 global_batch_size，乘以 DP 并行度
    loss = loss * num_microbatches / global_batch_size * dp_size
else:
    # Per-token 模式：loss 乘以 CP 并行度
    loss = loss * cp_size
```

**为什么需要这些 rescale？**

Megatron 在梯度累积时会对多个 micro-batch 的 loss 取平均。但 Slime 需要对整个 global batch 取平均（而非 micro-batch），所以需要反向补偿 Megatron 的自动除法。

对于 per-token 模式，还需要将 `num_tokens`（当前 micro-batch 的有效 token 总数）作为 normalizer 传回 Megatron，由其在 backward 时自动完成 $\frac{1}{\text{num\_tokens}}$ 的归一化。CP 的 `cp_size` 倍率是因为 CP 切分后每个 rank 只持有部分序列的 loss，需要乘回来得到完整的 loss。

### 4.6 梯度为零时的保护

```python
# make sure the gradient could backprop correctly.
if log_probs.numel() == 0:
    loss += 0 * logits.sum()
```

当某个 micro-batch 在当前 pipeline stage 没有 response token 时（可能出现在 pipeline parallelism 的非末尾 stage），`log_probs` 为空，loss 也为 0。但直接返回常数 0 会导致 `logits` 没有梯度流过，梯度在 pipeline 的某些 stage 会"断裂"。`0 * logits.sum()` 是一个优雅的技巧——值仍然是 0（不影响 loss），但让 autograd 知道"loss 依赖于 logits"，从而保证梯度能正确地反向传播穿过整个 pipeline。

### 4.7 完整公式总结

将上述过程串起来，SFT 的损失计算可以用如下公式描述：

**输入**：模型输出 logits $z \in \mathbb{R}^{T \times |V|}$，目标 token 序列 $y$，loss mask $m$

**步骤 1**：计算每个位置的 log probability（fused kernel 内完成）

$$\ell_t = \log P(y_t | z_{t-1}) = z_{t-1}[y_t] - \underbrace{\log \sum_{w \in V} \exp(z_{t-1}[w])}_{\text{log partition function}}$$

**步骤 2**：应用 loss mask 并归约

$$\mathcal{L} = -\text{reduce}(\ell \odot m)$$

其中 $\odot$ 表示逐元素乘法，reduce 策略取决于 `--calculate-per-token-loss`。

**步骤 3**：rescale 以适配 Megatron 梯度累积

最终梯度更新：$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$

### 4.8 与 RL policy_loss 的区别

| 方面 | SFT (`sft_loss`) | RL (`policy_loss`) |
|------|-------------------|---------------------|
| 目标 | 最大化目标 token 的 $\log P(y_t)$ | PPO clipped surrogate: $\min(r_t A_t, \text{clip}(r_t) A_t)$ |
| 输入 | 仅需要 logits 和目标 token | 还需要 advantages、old log_probs、ref log_probs |
| 参考信号 | 人工标注的正确答案 | reward model 给出的奖励信号 |
| KL 约束 | 无 | 通过 `kl_coef` 控制与 ref model 的偏离 |
| Entropy bonus | 无 | 通过 `entropy_coef` 鼓励探索 |
| Clipping | 无 | `eps_clip` 限制策略更新幅度 |
| 典型 lr | 1e-5 ~ 5e-6 | 1e-6 ~ 5e-7（更小，避免破坏已有能力）|

> **直觉**：SFT 相当于"老师把标准答案写在黑板上让学生抄"——模型直接模仿正确输出。RL 则是"老师只给分数不给答案"——模型需要自己探索并根据奖励信号调整行为，因此需要更多的约束（KL、clipping）来防止"学歪了"。

---

## 五、训练流程：SFT 如何复用 RL 管线

### 5.1 资源分配（`placement_group.py`）

当 `--debug-train-only` 时，不分配 rollout GPU：

```python
if args.debug_train_only:
    num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    rollout_offset = 0
```

所有 GPU 都分配给 Megatron 训练 actor。

### 5.2 RolloutManager 初始化（`ray/rollout.py`）

```python
if self.args.debug_train_only:
    self.servers = {}  # 不创建 SGLang 引擎
```

但 RolloutManager 仍然会被创建——它负责加载数据集和调用 `generate_rollout` 函数。

### 5.3 训练循环

整体流程（以 `train_async.py` 为例）：

```
for rollout_id in range(num_rollout):
    rollout_data = rollout_manager.generate(rollout_id)
        → RolloutManager._get_rollout_data()
            → call_rollout_fn(sft_rollout.generate_rollout, ...)  # tokenize + mask
        → RolloutManager._convert_samples_to_train_data()          # 转换为训练格式
        → RolloutManager._split_train_data_by_dp()                 # 按 DP 切分

    actor_model.train(rollout_id, rollout_data)
        → actor._get_rollout_data()        # 从 Ray 拉取数据到 GPU
        → actor.train_actor()
            # compute_advantages_and_returns=False → 跳过以下步骤：
            #   - ref model forward
            #   - log_probs 计算
            #   - advantage/return 计算
            → train()                      # 直接进入 Megatron 训练
                → loss_function()
                    → sft_loss_function()  # NLL loss
```

### 5.4 Actor 的 `train_actor` 方法

> 源文件：`slime/backends/megatron_utils/actor.py`，第 406-500 行

关键分支：

```python
def train_actor(self, rollout_id, rollout_data):
    data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

    with inverse_timer("train_wait"), timer("train"):
        if self.args.compute_advantages_and_returns:
            # --- RL 路径：计算 ref log_probs、advantages、returns ---
            # ... 很多步骤 ...
        # --- SFT 路径：compute_advantages_and_returns=False，直接跳到这里 ---

        # 直接执行 Megatron 训练步
        train(rollout_id, self.model, self.optimizer, self.opt_param_scheduler,
              data_iterator, num_microbatches)
```

由于 `--disable-compute-advantages-and-returns`，SFT 模式完全跳过了 ref model forward、log_probs 计算、advantage estimation 等 RL 开销。

### 5.5 Loss 调度

> 源文件：`slime/backends/megatron_utils/loss.py`，第 943-1031 行

```python
def loss_function(args, batch, num_microbatches, logits):
    match args.loss_type:
        case "policy_loss": func = policy_loss_function
        case "value_loss":  func = value_loss_function
        case "sft_loss":    func = sft_loss_function    # ← SFT 走这里
        case "custom_loss": func = load_function(args.custom_loss_function_path)
```

---

## 六、SFT 数据格式要求

### 输入数据

支持 JSONL 或 Parquet 格式。每条数据需要包含一个由 `--input-key` 指定的字段（默认 `messages`），值为标准的多轮对话列表：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "3+3 equals 6."}
  ]
}
```

可选字段：
- `tools`：工具定义列表（会传递给 `apply_chat_template` 的 `tools` 参数）
- `step_loss_mask`：消息级别的字段，设为非 1 值可屏蔽该轮次的 loss

### 处理后的训练数据结构

经过 `sft_rollout.generate_rollout` 和 `_convert_samples_to_train_data` 后：

```python
train_data = {
    "tokens": [[token_id, ...]],          # 完整 token 序列
    "response_lengths": [int],             # 从首个 loss=1 位置到末尾的长度
    "rewards": [0.0],                      # 固定为 0
    "loss_masks": [[0/1, ...]],            # response 部分的 loss mask
    "total_lengths": [int],                # = len(tokens)
    ...
}
```

---

## 七、典型 SFT 运行脚本解析

以 `scripts/run-qwen3-4B-base-sft.sh` 为例：

```bash
# 1. 指定 SFT 相关参数
--rollout-function-path slime.rollout.sft_rollout.generate_rollout
--loss-type sft_loss
--disable-compute-advantages-and-returns
--debug-train-only

# 2. 数据配置
--prompt-data /root/openhermes2_5.parquet
--input-key messages
--rollout-shuffle
--num-epoch 3
--rollout-batch-size 128
--global-batch-size 128

# 3. 性能配置
--use-dynamic-batch-size        # 动态 batch 组装（按 token 数）
--max-tokens-per-gpu 9216       # 每 GPU 最大 token 数
--calculate-per-token-loss      # 按 token 粒度归一化 loss

# 4. 优化器配置
--lr 1e-5
--lr-decay-style cosine
--min-lr 1e-6
--lr-warmup-fraction 0.1

# 5. 通过 train_async.py 启动（SFT 仍走异步入口）
ray job submit ... -- python3 train_async.py ...
```

对于更大的模型（如 `run-qwen3.5-35B-A3B-sft.sh`），额外配置包括：
- `--loss-mask-type qwen3_5`：使用 Qwen3.5 专属的 mask 生成策略
- `--use-distributed-optimizer` + `--optimizer-cpu-offload`：分布式优化器 + CPU offload 节省显存
- `--moe-token-dispatcher-type flex` + `--moe-enable-deepep`：MoE 模型专属优化

---

## 八、架构总结

```
数据流（SFT 模式）：

Parquet/JSONL 文件
    │
    ▼
DataSource (slime/rollout/data_source.py)
    │  按 rollout_batch_size 采样
    ▼
sft_rollout.generate_rollout (slime/rollout/sft_rollout.py)
    │  tokenize + loss mask 生成（CPU 端）
    │  使用 MultiTurnLossMaskGenerator
    ▼
RolloutManager._convert_samples_to_train_data
    │  samples → train_data dict
    ▼
RolloutManager._split_train_data_by_dp
    │  按 data parallel 维度切分
    ▼
MegatronTrainRayActor.train_actor (actor.py)
    │  跳过 advantage 计算
    │  直接进入 Megatron train loop
    ▼
loss_function → sft_loss_function (loss.py)
    │  ┌──────────────────────────────────────────┐
    │  │ 1. get_responses: 切出 response logits   │
    │  │ 2. compute_log_probs: fused CE kernel     │
    │  │ 3. sum_of_sample_mean: mask 加权归约      │
    │  │ 4. rescale: 适配 Megatron 梯度累积        │
    │  └──────────────────────────────────────────┘
    ▼
Megatron gradient accumulation + optimizer step
    │
    ▼
Checkpoint 保存
```

### 设计理念

Slime 的 SFT 实现体现了框架的核心设计哲学：**通过插件化和参数开关复用 RL 管线**。SFT 并非一个独立的训练模式，而是通过以下替换实现的：

1. **替换 rollout 函数**：`sglang_rollout → sft_rollout`（`--rollout-function-path`）
2. **替换 loss 函数**：`policy_loss → sft_loss`（`--loss-type`）
3. **关闭 RL 特有组件**：advantage 计算（`--disable-compute-advantages-and-returns`）、SGLang 引擎（`--debug-train-only`）

这种设计使得 SFT 和 RL 训练共享了 Megatron 训练 backend、分布式通信、checkpoint 管理、数据管线等所有基础设施。
