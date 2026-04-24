# PPO 算法学习笔记 — 基于 slime 代码库

本笔记对照 PPO 论文的核心公式，逐一映射到 slime 代码库中的实现，帮助读者从代码角度理解 PPO 算法。

---

## 1. 整体训练流程

PPO 属于 on-policy 的 Actor-Critic 算法。slime 中的同步训练循环 (`train.py`) 每一轮的流程为：

```
Rollout（生成数据）→ 计算 Advantage → 训练 Critic → 训练 Actor → 同步权重
```

具体来说：
1. **Rollout**: 用当前策略 `π_old` 生成 response，获取 reward
2. **Advantage 计算**: 用 GAE 估计每个 token 的优势值
3. **多轮 PPO epoch**: 在同一批数据上多次更新策略（由 `--ppo-epoch` 控制）
4. **权重同步**: 将更新后的模型权重推回 rollout 引擎

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

slime 还实现了 dual-clip PPO（通过 `--eps-clip-c` 参数启用）：

$$L^{DUAL}(\theta) = \begin{cases} \max\left(L^{CLIP}, c \cdot \hat{A}_t\right), & \text{if } \hat{A}_t < 0 \\ L^{CLIP}, & \text{otherwise} \end{cases}$$

当 advantage 为负（即希望减少某个 action 的概率）时，额外设置一个下界 `c * A`，防止过度惩罚。

```python
    # ppo_utils.py:138-144
    if eps_clip_c is not None:
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

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
| `k1` | $\log\frac{\pi_\theta}{\pi_{\text{old}}}$ | 最简单，可能为负 |
| `k2` | $\frac{1}{2}\left(\log\frac{\pi_\theta}{\pi_{\text{old}}}\right)^2$ | 平方形式，恒非负 |
| `k3` / `low_var_kl` | $\frac{\pi_{\text{old}}}{\pi_\theta} - 1 - \log\frac{\pi_{\text{old}}}{\pi_\theta}$ | Schulman 提出的低方差无偏估计 |

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
        kl = log_ratio.exp() - 1 - log_ratio  # 即 π_old/π_θ - 1 - log(π_old/π_θ)
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

---

## 6. Value Function Loss

Critic 网络预测每个 token 的 value $V(s_t)$，用于计算 GAE。其 loss 为 MSE：

$$L^{VF} = \frac{1}{2} \mathbb{E}\left[\max\left((V_\theta - R_t)^2, (V_{clip} - R_t)^2\right)\right]$$

其中 $V_{clip} = \text{clip}(V_\theta, V_{old} - \epsilon_{vf}, V_{old} + \epsilon_{vf})$

```python
# loss.py:817
v_loss = (values - returns) ** 2
if args.vf_clip:
    v_loss_clipped = (values_clipped - returns) ** 2
    v_loss = torch.max(v_loss, v_loss_clipped)
loss = 0.5 * v_loss.mean()
```

---

## 7. Entropy Bonus

为了鼓励探索，PPO 会在 loss 中减去一个熵奖励：

$$L = L^{CLIP} - c_2 \cdot H[\pi_\theta]$$

熵越大说明策略越"随机"，减去熵等价于鼓励更高熵的策略。

```python
# loss.py:743
entropy_loss = -entropy * args.entropy_coef
loss = pg_loss + entropy_loss
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
| KL Loss | `--kl-loss-coef` | 0.0 |
| Entropy Bonus | `--entropy-coef` | 0.0 |
| Value Loss | Critic 独立训练 | — |

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

### 9.2 REINFORCE++

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

### 9.3 REINFORCE++-Baseline

在 REINFORCE++ 基础上引入 group baseline（类似 GRPO 的思路），将 reward 减去组内均值：

$$A_t = (R - \bar{R}_{group}) - \text{kl\_coef} \cdot KL_t$$

### 9.4 OPSM (Off-Policy Sequence Masking)

在异步训练中，策略更新后旧数据可能变得 off-policy。OPSM 通过检测序列级 KL 来动态屏蔽：

$$\text{mask} = \begin{cases} 0, & \text{if } \hat{A} < 0 \text{ and } KL_{seq} > \delta \\ 1, & \text{otherwise} \end{cases}$$

即：如果一个序列的优势为负（不好的回答）且策略已经偏移很大（KL 超过阈值），则跳过该序列的训练。

---

## 10. 关键超参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--eps-clip` | 0.2 | PPO clip 范围 |
| `--eps-clip-high` | None（= eps-clip） | 非对称裁剪上界 |
| `--kl-coef` | 0.0 | reward 中的 KL 惩罚系数 |
| `--kl-loss-coef` | 0.0 | loss 中的 KL 正则系数 |
| `--entropy-coef` | 0.0 | 熵奖励系数 |
| `--gamma` | 1.0 | 折扣因子 |
| `--lambd` | 1.0 | GAE lambda |
| `--ppo-epoch` | 1 | 每批数据的训练轮数 |
| `--normalize-advantages` | — | 是否做 advantage 白化 |
| `--vf-clip` | — | 是否裁剪 value loss |

---

## 11. 推荐阅读顺序

1. `slime/utils/ppo_utils.py` — 核心算法函数
2. `slime/backends/megatron_utils/loss.py` — Loss 组装和 advantage 计算
3. `train.py` — 同步训练主循环
4. `slime/ray/rollout.py` — 数据生成（rollout）
5. `slime/utils/arguments.py` — 所有超参数定义

---

## 参考文献

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using GAE (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438)
- [Approximating KL Divergence (Schulman blog)](http://joschu.net/blog/kl-approx.html)
- [REINFORCE++ (2025)](https://arxiv.org/pdf/2501.03262)
