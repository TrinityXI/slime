# PPO 算法学习笔记 — 基于 slime 代码库

本笔记对照 PPO 论文的核心公式，逐一映射到 slime 代码库中的实现，帮助读者从代码角度理解 PPO 算法。

## 0. 先用人话讲清楚 PPO 在做什么

把大模型想象成一个正在学写答案的学生。每次训练时，slime 会先让当前模型自己做一批题，得到一批回答；然后用 reward model、规则 verifier 或自定义 reward 函数给这些回答打分。分数高的回答，模型以后应该更愿意生成；分数低的回答，模型以后应该少生成。

如果只按这个思路做策略梯度，会很容易把模型推得太猛：某一批样本里偶然得高分的写法，可能一下子被放大很多；某些低分 token 的概率也可能被压得过头。PPO 的核心就是给这个更新加一个“安全带”：**让好答案的概率上升、坏答案的概率下降，但每一步都不能离生成这批数据时的旧模型太远。**

在 LLM 后训练里，可以把 PPO 的关键对象对应成下面几件事：

| PPO 概念 | 在语言模型里的直观含义 | slime 里的位置 |
|---|---|---|
| state | 已经看到的 prompt 和前文 token | rollout / training batch 中的 token 序列 |
| action | 当前要生成的下一个 token | response token |
| policy | 大模型给下一个 token 的概率分布 | actor model |
| old policy | 生成 rollout 数据时的模型概率 | rollout 或训练前记录的 `log_probs` |
| reward | 整个回答的质量分数，外加 KL 惩罚 | `rewards` + `kl_coef` |
| advantage | 这个 token/回答比预期好多少 | `advantages` |
| clip | 限制新旧概率比值变化范围 | `compute_policy_loss()` |

用一句话概括 slime 中 PPO 的主链路：

```
模型先生成回答 → 回答被打分 → 把分数拆到 token 级别 → 估计每个 token 值不值得鼓励 → 用 clip 限制策略更新幅度 → 把新权重同步给下一轮 rollout
```

学习时可以抓住三个问题：

1. **奖励从哪里来？**<br>
   来自 rollout 阶段的 reward/verifier，并可叠加 reference model KL 惩罚。
2. **为什么要有 advantage？**<br>
   因为绝对 reward 不够稳定，我们更关心“这个 token/回答比当前预期好还是坏”。
3. **PPO 为什么要 clip？**<br>
   因为 rollout 数据来自旧策略，更新太大后这批数据就不再可靠；clip 用概率比值把更新限制在一个可信范围里。

注意：slime 当前默认 `--advantage-estimator` 是 `grpo`，不是 `ppo`。只有显式设置 `--advantage-estimator ppo` 时，才会启用 critic/value model 路线；默认 GRPO 路线不训练 critic。本文重点讲 PPO，同时也在第 9 节对 slime 支持的 GRPO、GSPO、REINFORCE++ 等变体做对照。

---

## 1. 整体训练流程

PPO 属于 on-policy 的 Actor-Critic 算法。slime 中的同步训练循环 (`train.py`) 每一轮的流程为：

```
Rollout（生成数据）→ 计算 Advantage → 训练 Critic → 训练 Actor → 同步权重
```

具体来说：
1. **Rollout**: 用当前策略 `π_old` 生成 response，获取 reward
2. **Advantage 计算**: 用 GAE 估计每个 token 的优势值
3. **训练步切分**: 将 `rollout_batch_size * n_samples_per_prompt` 条样本按 `global_batch_size` 切成若干训练 step；也可以用 `--num-steps-per-rollout` 反推 `global_batch_size`
4. **权重同步**: 将更新后的模型权重推回 rollout 引擎

源码中没有 `--ppo-epoch` 这个参数；如果想控制每次 rollout 后训练多少步，主要看 `--global-batch-size`、`--rollout-batch-size`、`--n-samples-per-prompt` 和 `--num-steps-per-rollout`。

---

## 2. 策略梯度与 PPO Clipping

### 2.1 原始策略梯度

策略梯度的基本形式：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \hat{A}(s,a) \right]$$

其中 importance sampling ratio 为：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

### 2.2 PPO-Clip 目标函数

PPO 的核心思想是限制每次更新的步长，防止策略变化过大：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

**代码对应**: `slime/utils/ppo_utils.py` → `compute_policy_loss()`

```python
# ppo_utils.py:124-148
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    # ratio = π_θ / π_old = exp(log π_θ - log π_old)
    # 这里 ppo_kl = log π_old - log π_θ，所以 ratio = exp(-ppo_kl)
    ratio = (-ppo_kl).exp()

    # 未裁剪的目标: -ratio * advantage（取负号因为要做梯度下降）
    pg_losses1 = -ratio * advantages

    # 裁剪后的目标
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages

    # 取两者的最大值（因为带负号，max 等价于原公式的 min）
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
```

关键点：
- `eps_clip` 默认 0.2，即 ratio 被限制在 [0.8, 1.2]
- slime 支持**非对称裁剪**：`eps_clip_high` 可以和 `eps_clip` 不同
- `clipfrac` 记录了被裁剪的比例，是重要的训练监控指标

### 2.3 Dual-Clip PPO

slime 的 `compute_policy_loss()` 底层函数支持 dual-clip PPO：

$$L^{DUAL}(\theta) = \begin{cases} \max\left(L^{CLIP}, c \cdot \hat{A}_t\right), & \text{if } \hat{A}_t < 0 \\ L^{CLIP}, & \text{otherwise} \end{cases}$$

当 advantage 为负（即希望减少某个 action 的概率）时，额外设置一个下界 `c * A`，防止过度惩罚。

```python
    # ppo_utils.py:138-144
    if eps_clip_c is not None:
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

源码 review 提醒：`slime/utils/arguments.py` 中定义了 `--eps-clip-c`，但当前 `slime/backends/megatron_utils/loss.py` 的 `policy_loss_function()` 调用是：

```python
pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
```

也就是说，当前 actor loss 路径没有把 `args.eps_clip_c` 传给 `compute_policy_loss()`。如果要实际启用 dual-clip，需要先确认这里是否是遗漏。

---

## 3. Generalized Advantage Estimation (GAE)

### 3.1 公式

GAE 通过引入参数 λ 在偏差和方差之间做权衡：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}$$

等价的递推形式：

$$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$$

- `γ = 1, λ = 1`（slime 默认值）：退化为 Monte Carlo 估计（无偏但方差大）
- `γ = 1, λ = 0`：退化为 TD(0)（偏差大但方差小）

### 3.2 代码实现 — 朴素版本

**代码对应**: `ppo_utils.py` → `vanilla_gae()`

```python
# ppo_utils.py:482-503
def vanilla_gae(rewards, values, gamma, lambd):
    B, T = rewards.shape
    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    # 从后往前递推
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]  # δ_t
        lastgaelam = delta + gamma * lambd * lastgaelam             # A_t = δ_t + γλ A_{t+1}
        adv_rev.append(lastgaelam)

    full_advantages = torch.stack(adv_rev[::-1], dim=1)
    full_returns = full_advantages + values  # Return = A + V
```

这是教科书式的实现，从序列末尾反向递推。

### 3.3 代码实现 — 分块并行版本 (Chunked GAE)

**代码对应**: `ppo_utils.py` → `chunked_gae()`

这是一个性能优化版本，灵感来自 FlashLinearAttention 的 chunk-parallel 思想：

1. 将序列翻转（把反向递推变成正向扫描）
2. 分成 chunk_size=128 的块
3. 每个块内通过矩阵乘法并行计算（构造下三角矩阵 M，其中 `M[i,j] = (γλ)^(j-i)`）
4. 块间串行传播状态

这将序列依赖长度从 O(T) 降低到 O(T/chunk_size)，显著提升长序列的 GAE 计算效率。

---

## 4. KL 散度

### 4.1 公式

KL 散度用于度量新旧策略的差异。slime 支持多种 KL 估计器：

| 类型 | 公式 | 说明 |
|------|------|------|
| `k1` | $\log\frac{\pi_\theta}{\pi_{\text{base}}}$ | 最简单，可能为负 |
| `k2` | $\frac{1}{2}\left(\log\frac{\pi_\theta}{\pi_{\text{base}}}\right)^2$ | 平方形式，恒非负 |
| `k3` / `low_var_kl` | $\frac{\pi_{\text{base}}}{\pi_\theta} - 1 - \log\frac{\pi_{\text{base}}}{\pi_\theta}$ | Schulman 提出的低方差估计 |

这里的 `base` 取决于调用场景：做 reward shaping 时通常是 reference model 的 `ref_log_probs`；做 PPO clipping 时则用生成 rollout 时记录下来的 old policy `log_probs`。

**代码对应**: `ppo_utils.py` → `compute_approx_kl()`

```python
# ppo_utils.py:11-51
def compute_approx_kl(log_probs, log_probs_base, kl_loss_type, importance_ratio=None):
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio  # 即 π_base/π_θ - 1 - log(π_base/π_θ)
```

### 4.2 KL 在训练中的两种用法

1. **作为 reward 的惩罚项**（`--kl-coef`）：在 GAE 计算前，将 KL 作为负奖励叠加到每个 token 上
   ```
   r'_t = r_t - kl_coef * KL_t
   ```
2. **作为 loss 的正则项**（`--kl-loss-coef`）：直接加到策略 loss 上
   ```
   L = L_policy + kl_loss_coef * KL
   ```

slime 会断言 `--kl-coef` 和 `--kl-loss-coef` 不能同时非零，避免同一类 KL 约束被重复施加。loss 侧 KL 还需要显式打开 `--use-kl-loss`。

---

## 5. Reward → Advantage 的完整链路

在 LLM 场景下，reward 通常只在序列结束时有一个标量值。完整的链路：

```
sequence-level reward
    ↓  加入 KL 惩罚得到 token-level reward
    ↓  token_reward[t] = -kl_coef * KL[t], 最后一个 token 加上 reward
    ↓
GAE 计算 advantage 和 return
    ↓
advantage 白化（normalize_advantages）
    ↓  跨 data-parallel group 做 mean/std 归一化
    ↓
送入 compute_policy_loss
```

Advantage 白化公式：

$$\hat{A}_t^{norm} = \frac{\hat{A}_t - \mu(\hat{A})}{\sigma(\hat{A}) + \epsilon}$$

在分布式场景下，slime 通过 `distributed_masked_whiten` 跨所有 data-parallel rank 计算全局均值和标准差。

**代码对应**: `slime/backends/megatron_utils/loss.py` → `compute_advantages_and_returns()`

当 `args.advantage_estimator == "ppo"` 时，slime 会把 KL 惩罚写进 token-level reward：

```python
kl_coef = -args.kl_coef
for reward, k in zip(old_rewards, kl, strict=False):
    k *= kl_coef
    if cp_rank == 0:
        k[-1] += reward
    rewards.append(k)
```

也就是说，大部分 token 的 reward 是 `-kl_coef * KL`，最后一个 response token 额外加上序列级 reward。

---

## 6. Value Function Loss

当 `--advantage-estimator ppo` 时，slime 会启用 critic。Critic 网络预测每个 token 的 value $V(s_t)$，用于计算 GAE。其 loss 是 PPO 风格的 clipped value loss：

$$L^{VF} = \mathbb{E}\left[\max\left((V_\theta - R_t)^2, (V_{clip} - R_t)^2\right)\right]$$

其中 $V_{clip} = \text{clip}(V_\theta, V_{old} - \epsilon_{vf}, V_{old} + \epsilon_{vf})$，在 slime 中参数名是 `--value-clip`。

```python
# slime/backends/megatron_utils/loss.py:value_loss_function()
values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
surr1 = (values_clipped - returns) ** 2
surr2 = (values - returns) ** 2
loss = torch.max(surr1, surr2)
loss = sum_of_sample_mean(loss)
```

---

## 7. Entropy Bonus

为了鼓励探索，PPO 会在 loss 中减去一个熵奖励：

$$L = L^{CLIP} - c_2 \cdot H[\pi_\theta]$$

熵越大说明策略越"随机"，减去熵等价于鼓励更高熵的策略。

```python
# slime/backends/megatron_utils/loss.py:policy_loss_function()
entropy_loss = sum_of_sample_mean(entropy)
loss = pg_loss - args.entropy_coef * entropy_loss
```

**代码对应**: `ppo_utils.py` → `compute_entropy_from_logits()`

slime 通过 `_VocabParallelEntropy` 自定义 autograd Function 来高效计算并行词表下的熵，支持梯度反传。

---

## 8. 最终 Loss 汇总

Actor 的总 loss 由三部分组成：

$$L^{total} = L^{CLIP} + c_1 \cdot L^{KL} - c_2 \cdot H[\pi_\theta]$$

| 组件 | 对应参数 | 默认值 |
|------|----------|--------|
| Policy Loss (clipped) | `--eps-clip` | 0.2 |
| KL Loss | `--use-kl-loss` + `--kl-loss-coef` | 关闭 / 0.0 |
| Entropy Bonus | `--entropy-coef` | 0.0 |
| Value Loss | `--advantage-estimator ppo` 时启用 critic | — |

源码阅读时要区分两类 KL：

- `ppo_kl = old_log_probs - log_probs`：用于 PPO clipping，衡量“新策略相对 rollout 旧策略变了多少”。
- `compute_approx_kl(log_probs, ref_log_probs, ...)`：用于 reference KL，衡量“当前策略相对 reference model 偏离多少”。

这两类 KL 变量名很像，但角色不同。前者服务于 PPO 的 trust region，后者服务于 RLHF 中常见的“别离开参考模型太远”约束。

---

## 9. 算法变体

slime 除了标准 PPO 外，还实现了以下变体：

### 9.1 GRPO (Group Relative Policy Optimization)

简化版本，不需要 Critic。直接用 reward 作为 advantage：

```python
# ppo_utils.py:201-208
def get_grpo_returns(rewards, kl):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

每个 token 的 return 直接等于序列级别的 reward（已经过 group 内归一化）。

这是 slime 当前默认的 `--advantage-estimator`。因此读默认训练脚本时，经常会先看到 GRPO 路线，而不是标准 PPO critic 路线。

### 9.2 GSPO

GSPO 与 GRPO 一样不依赖 critic，但在 policy loss 中使用序列级 KL：`compute_gspo_kl()` 会把整条 response 的平均 KL 扩展到每个 token 上。直觉上，它更关注“整条回答作为一个整体偏离了多少”，而不是只看单 token 的局部变化。

### 9.3 REINFORCE++

带 baseline 和折扣的 REINFORCE 变体。在每个 token 位置计算折扣回报：

$$G_t = r_t + \gamma G_{t+1}$$

其中 $r_t = -\text{kl\_coef} \cdot KL_t$，最后一个 token 加上序列 reward。

```python
# ppo_utils.py:261-266
returns_for_seq = torch.zeros_like(token_level_rewards)
running_return = 0.0
for t in reversed(range(token_level_rewards.size(0))):
    running_return = token_level_rewards[t] + gamma * running_return
    returns_for_seq[t] = running_return
```

### 9.4 REINFORCE++-Baseline

在 REINFORCE++ 基础上引入 group baseline（类似 GRPO 的思路），将 reward 减去组内均值：

$$A_t = (R - \bar{R}_{group}) - \text{kl\_coef} \cdot KL_t$$

slime 对 `reinforce_plus_plus` 和 `reinforce_plus_plus_baseline` 有额外约束：必须打开 `--normalize-advantages`。

### 9.5 OPSM (Off-Policy Sequence Masking)

在异步训练中，策略更新后旧数据可能变得 off-policy。OPSM 通过检测序列级 KL 来动态屏蔽：

$$\text{mask} = \begin{cases} 0, & \text{if } \hat{A} < 0 \text{ and } KL_{seq} > \delta \\ 1, & \text{otherwise} \end{cases}$$

即：如果一个序列的优势为负（不好的回答）且策略已经偏移很大（KL 超过阈值），则跳过该序列的训练。

---

## 10. 启动 PPO 训练需要准备的参数输入

slime 的默认 RL 路线是 GRPO。要启动真正的 PPO，需要显式选择 `--advantage-estimator ppo`，并额外准备 critic 相关资源。可以把启动参数分成 7 组来看。

### 10.1 模型与 checkpoint 输入

| 参数 | 是否必需 | 作用 |
|---|---|---|
| `--hf-checkpoint` | 必需 | HF 模型目录，主要提供 tokenizer、config 等信息 |
| `--ref-load` | KL 非零时必需；通常建议提供 | Megatron 格式 reference model checkpoint |
| `--load` | 可选 | actor checkpoint；不存在有效 checkpoint 时会从 `--ref-load` 初始化 |
| `--save` | 保存时必需 | actor checkpoint 保存目录 |
| `--save-interval` | 可选 | 每隔多少 rollout 保存 actor |
| `--critic-load` | PPO 建议提供 | critic checkpoint；默认值会跟随 `--load` |
| `--critic-save` | PPO 建议提供 | critic checkpoint 保存目录 |

Megatron checkpoint 目录通常需要包含 `latest_checkpointed_iteration.txt`。如果 `--kl-coef != 0` 或打开 `--use-kl-loss`，slime 会检查 `--ref-load` 是否存在。

### 10.2 模型结构与并行参数

需要通过 `scripts/models/` 中的模型配置提供 Megatron 结构参数，例如：

```bash
source scripts/models/qwen3-4B.sh
```

这些参数包括但不限于：

| 参数 | 作用 |
|---|---|
| `--tensor-model-parallel-size` | Tensor Parallel 大小 |
| `--pipeline-model-parallel-size` | Pipeline Parallel 大小 |
| `--context-parallel-size` | Context Parallel 大小 |
| `--expert-model-parallel-size` | MoE Expert Parallel 大小 |
| `--sequence-parallel` | 是否启用 sequence parallel |
| 模型结构参数 | 层数、hidden size、attention heads、rotary 配置等 |

关键点：Megatron 不能完全从 checkpoint 自动推断模型结构，所以这些参数必须和模型 checkpoint 对齐。

### 10.3 数据输入

PPO 需要 prompt 数据，rollout 阶段会用当前 actor 生成 response，再计算 reward。

| 参数 | 作用 |
|---|---|
| `--prompt-data` | 训练 prompt 数据路径，支持 JSONL/Parquet 等 |
| `--input-key` | prompt 字段名 |
| `--label-key` | label/答案字段名，取决于 reward 函数是否需要 |
| `--apply-chat-template` | 当输入是 messages 或需要套 chat template 时启用 |
| `--rollout-shuffle` | rollout 读取数据时打乱 |

例如数学 verifier 任务常见格式是：

```bash
--prompt-data /path/to/dapo-math-17k.jsonl
--input-key prompt
--label-key label
--apply-chat-template
```

### 10.4 Reward 输入

PPO 的 advantage 来自 reward 和 critic value 的差值；没有有效 reward，PPO 就没有学习方向。

| 参数 | 作用 |
|---|---|
| `--rm-type` | 使用 slime 内置 reward 类型 |
| `--custom-rm-path` | 加载自定义 reward 函数 |
| `--reward-key` | 从 rollout/sample 中读取 reward 的字段名 |

两种常见方式：

```bash
# 使用内置 reward
--rm-type deepscaler
```

```bash
# 使用自定义 reward 函数
--custom-rm-path your_package.reward:reward_func
```

如果任务是数学、代码、搜索或工具调用，通常需要自定义 reward/verifier，确保它能返回每条 response 的标量 reward。

### 10.5 PPO 算法参数

这是从 GRPO 切到 PPO 的最小核心：

```bash
--advantage-estimator ppo
--eps-clip 0.2
--value-clip 0.2
--gamma 1.0
--lambd 1.0
--normalize-advantages
```

KL 有两种使用方式，二选一：

```bash
# 方式 1：把 KL 当作 reward penalty，进入 GAE 前的 reward
--kl-coef 0.01
```

```bash
# 方式 2：把 KL 当作 actor loss 的正则项
--use-kl-loss
--kl-loss-coef 0.01
```

源码里会断言 `--kl-coef` 和 `--kl-loss-coef` 不能同时非零。

### 10.6 Actor / Critic / Rollout 资源输入

PPO 会额外创建 critic 训练组，因此 GPU 资源由三部分组成：

```text
总 GPU = actor GPUs + critic GPUs + rollout GPUs
```

示例：

```bash
--actor-num-nodes 1
--actor-num-gpus-per-node 4

--critic-num-nodes 1
--critic-num-gpus-per-node 4

--rollout-num-gpus 8
```

这个配置总共需要 16 张 GPU。若不显式配置 critic 资源，slime 会默认让 critic 使用和 actor 相同的 `num_nodes` / `num_gpus_per_node`。

critic 相关训练参数：

| 参数 | 作用 |
|---|---|
| `--critic-lr` | critic 学习率；默认跟 actor `--lr` 一样 |
| `--critic-lr-warmup-iters` | critic warmup step 数 |
| `--num-critic-only-steps` | 训练开始时只训练 critic 的 step 数 |
| `--critic-train-only` | 只训练 critic，调试 critic 时使用 |

### 10.7 Rollout 与训练 batch 对齐

每轮 rollout 产生的样本数必须和训练消耗的样本数对齐：

```text
rollout_batch_size * n_samples_per_prompt = global_batch_size * num_steps_per_rollout
```

常见配置：

```bash
--rollout-batch-size 16
--n-samples-per-prompt 8
--num-steps-per-rollout 1
--global-batch-size 128
```

如果设置了 `--num-steps-per-rollout`，slime 可以根据它反推 `--global-batch-size`；如果两者都设置，则会校验这个等式。

### 10.8 最小 PPO 参数骨架

下面是一个只保留关键字段的骨架，真实训练还需要拼上具体模型配置、Megatron 并行参数、SGLang 参数和集群启动命令：

```bash
CKPT_ARGS=(
  --hf-checkpoint /path/to/hf_model
  --ref-load /path/to/ref_megatron_ckpt
  --load /path/to/actor_ckpt
  --save /path/to/actor_save
  --critic-load /path/to/critic_ckpt
  --critic-save /path/to/critic_save
)

DATA_ARGS=(
  --prompt-data /path/to/train.jsonl
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
)

RM_ARGS=(
  --rm-type deepscaler
  # 或者：--custom-rm-path your_package.reward:reward_func
)

PPO_ARGS=(
  --advantage-estimator ppo
  --eps-clip 0.2
  --value-clip 0.2
  --gamma 1.0
  --lambd 1.0
  --normalize-advantages
  --kl-coef 0.0
)

BATCH_ARGS=(
  --num-rollout 3000
  --rollout-batch-size 16
  --n-samples-per-prompt 8
  --num-steps-per-rollout 1
  --global-batch-size 128
)

RESOURCE_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node 4
  --critic-num-nodes 1
  --critic-num-gpus-per-node 4
  --rollout-num-gpus 8
)
```

---

## 11. 关键超参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator` | `grpo` | 选择 `grpo`、`gspo`、`ppo`、`reinforce_plus_plus` 等 advantage 估计器 |
| `--eps-clip` | 0.2 | PPO clip 范围 |
| `--eps-clip-high` | None（= eps-clip） | 非对称裁剪上界 |
| `--eps-clip-c` | None | Dual-Clip PPO 的额外裁剪系数 |
| `--kl-coef` | 0.0 | reward 中的 KL 惩罚系数 |
| `--use-kl-loss` | False | 是否在 actor loss 中加入 reference KL |
| `--kl-loss-coef` | 0.0 | loss 中的 KL 正则系数 |
| `--kl-loss-type` | `k1` | KL 估计器类型：`k1`、`k2`、`k3`、`low_var_kl` |
| `--entropy-coef` | 0.0 | 熵奖励系数 |
| `--gamma` | 1.0 | 折扣因子 |
| `--lambd` | 1.0 | GAE lambda |
| `--num-steps-per-rollout` | None | 每轮 rollout 对应多少个训练 step；会反推 `global_batch_size` |
| `--normalize-advantages` | — | 是否做 advantage 白化 |
| `--value-clip` | 0.2 | value loss 的裁剪范围 |
| `--use-opsm` | False | 是否启用 Off-Policy Sequence Masking |
| `--opsm-delta` | 1e-4 | OPSM 的序列级 KL 阈值 |

---

## 12. 推荐阅读顺序

1. `slime/utils/ppo_utils.py` — 核心算法函数
2. `slime/backends/megatron_utils/loss.py` — Loss 组装和 advantage 计算
3. `slime/utils/arguments.py` — 先确认默认参数，尤其是默认 `advantage_estimator=grpo`
4. `slime/ray/rollout.py` — 数据生成（rollout）
5. `train.py` — 同步训练主循环

---

## 参考文献

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using GAE (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438)
- [Approximating KL Divergence (Schulman blog)](http://joschu.net/blog/kl-approx.html)
- [REINFORCE++ (2025)](https://arxiv.org/pdf/2501.03262)
