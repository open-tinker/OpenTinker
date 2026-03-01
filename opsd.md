# Self-Distilled Reasoner: On-Policy Self-Distillation (OPSD) 方法分析

## 1. 核心思想

OPSD 的核心洞察是：一个足够强的 LLM，在看到正确答案（特权信息）后，能够"合理化"推理过程，从而指导没有看到答案的自身版本。这类似于学生做错题后，对照答案回溯思路、理解解法的学习方式。

关键创新点在于：**同一个模型同时扮演教师和学生**，仅通过不同的输入上下文来区分两个角色，无需外部教师模型。

## 2. 方法架构

### 2.1 双策略设计

从同一个 LLM（参数 θ）实例化两个条件分布：

- **学生策略** `pS(· | x) := pθ(· | x)` — 仅看到问题 x，模拟推理时的真实条件
- **教师策略** `pT(· | x, y*) := pθ(· | x, y*)` — 同时看到问题 x 和参考答案 y*（特权信息）

两者共享同一组参数 θ，仅在 prompt 构造上不同。

### 2.2 训练流程（Algorithm 1）

```
对于每个 mini-batch (x, y*):
  1. 学生策略采样：ŷ ~ pS(· | x)              // on-policy rollout
  2. 两个策略分别对 ŷ 的每个位置 n 产生 next-token 分布：
     - pS(yn | x, ŷ<n)    // 学生视角
     - pT(yn | x, y*, ŷ<n) // 教师视角（含特权信息）
  3. 计算 token 级散度并求平均：
     loss = (1/|ŷ|) * Σ D(pT(· | x, y*, ŷ<n) ‖ pS(· | x, ŷ<n))
  4. 梯度仅通过学生 logits 反向传播，更新 θ
```

### 2.3 Prompt 设计

**学生 Prompt：**
```
Problem: <问题描述>
Answer:
```

**教师 Prompt：**
```
Problem: <问题描述>
Here is a reference solution:
<参考解答 y*>
After understanding the reference solution, please try to solve this problem
using your own approach below:
Answer:
```

教师 prompt 的关键设计：先给出参考答案，再要求模型用自己的方式重新解题。这让教师在评估学生 rollout 时能自然过渡，而不是简单复述答案。

## 3. 训练目标

### 3.1 全词表 logit 蒸馏（主要目标）

在每个 token 位置上计算教师和学生完整词表分布的散度：

```
D(pT ‖ pS)(ŷ | x) = (1/|ŷ|) Σ_{n=1}^{|ŷ|} D(pT(· | x, y*, ŷ<n) ‖ pS(· | x, ŷ<n))
```

散度函数 D 使用 **广义 Jensen-Shannon 散度（JSD_β）**，β=0.5：

```
JSD_β(pT ‖ pS) = β · KL(pT ‖ m) + (1-β) · KL(pS ‖ m)
    其中 m = β·pT + (1-β)·pS
```

JSD 相比纯 KL 散度的优势在于：对称性更好、数值更稳定，避免了当 pS 在 pT 有概率质量处接近零时的梯度爆炸问题。

### 3.2 采样 token 策略梯度（备选目标）

仅在学生采样的 token ŷn 上评估教师/学生的 log 概率差作为 advantage：

```
An(x, ŷ) = log pT(ŷn | x, y*, ŷ<n) - log pS(ŷn | x, ŷ<n)
```

然后用策略梯度优化：

```
L(θ) = -E[ (1/|ŷ|) Σ An(x, ŷ) · log pS(ŷn | x, ŷ<n) ]
```

An 在反向传播时视为常数（stop gradient）。

实验表明全词表蒸馏优于采样 token 方式（AIME25: 84.1% vs 82.1%），但内存开销更大。

## 4. 关键实现细节

### 4.1 教师策略冻结

教师策略固定为**初始策略**的参数，而非训练中不断更新的参数。这有两个作用：
- 稳定训练过程
- 隐式起到正则化效果，防止模型过度偏离初始分布

### 4.2 采样效率

| 方法 | 每个问题采样数 | 生成长度 | token 效率 |
|------|-------------|---------|-----------|
| GRPO | 8 | 16384 | 基准 |
| OPSD | 1 | 2048 | 4-8× 优于 GRPO |

OPSD 的高效率来源于：密集的 token 级监督使得短序列即可提供足够学习信号，不需要像 GRPO 那样用长序列+多采样来获取稀疏的序列级奖励。

### 4.3 生成长度的影响

- 1024 tokens: 基线性能
- 2048 tokens: 显著提升
- 4096 tokens: 进一步提升但边际递减

作者假设早期 token 对蒸馏更重要，因为它们代表更关键的推理分支点。

### 4.4 模型规模的影响

OPSD 在 1.7B 上效果有限（与 GRPO 持平），在 4B 和 8B 上效果递增。这说明自蒸馏要求模型具备足够的"合理化"能力——能在看到答案后真正理解推理过程，而非简单记忆。

## 5. 代码实现思路

### 5.1 整体训练循环伪代码

```python
class OPSDTrainer:
    def __init__(self, model, tokenizer, dataset, config):
        self.model = model                    # 同一个 LLM
        self.teacher_model = copy.deepcopy(model)  # 冻结的初始参数作为教师
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.tokenizer = tokenizer
        self.dataset = dataset  # [(x_i, y_star_i), ...]
        self.beta = 0.5         # JSD 权重

    def build_student_prompt(self, problem):
        return f"Problem: {problem}\nAnswer:"

    def build_teacher_prompt(self, problem, solution):
        return (
            f"Problem: {problem}\n"
            f"Here is a reference solution:\n{solution}\n"
            f"After understanding the reference solution, "
            f"please try to solve this problem using your own approach below:\n"
            f"Answer:"
        )

    def train_step(self, batch):
        total_loss = 0
        for (x, y_star) in batch:
            # Step 1: 构建 prompt
            student_prompt = self.build_student_prompt(x)
            teacher_prompt = self.build_teacher_prompt(x, y_star)

            # Step 2: 学生 on-policy 采样（不计算梯度）
            with torch.no_grad():
                y_hat = self.model.generate(
                    student_prompt, max_new_tokens=2048, temperature=1.2
                )

            # Step 3: 对 rollout 的每个位置计算两个策略的 logits
            student_input = concat(student_prompt, y_hat)
            teacher_input = concat(teacher_prompt, y_hat)

            student_logits = self.model(student_input)       # 需要梯度
            with torch.no_grad():
                teacher_logits = self.teacher_model(teacher_input)  # 不需要梯度

            # Step 4: 对齐位置（仅取 y_hat 对应的 token 位置）
            # 注意: teacher prompt 更长，需要正确对齐 y_hat 部分的 logits
            student_probs = softmax(student_logits_at_yhat_positions)
            teacher_probs = softmax(teacher_logits_at_yhat_positions)

            # Step 5: 计算 JSD loss
            loss = self.compute_jsd(teacher_probs, student_probs)
            total_loss += loss

        return total_loss / len(batch)
```

### 5.2 JSD 散度计算

```python
def compute_jsd(self, p_teacher, p_student, beta=0.5):
    """
    p_teacher: [seq_len, vocab_size] 教师的 next-token 分布
    p_student: [seq_len, vocab_size] 学生的 next-token 分布
    """
    # 混合分布
    m = beta * p_teacher + (1 - beta) * p_student

    # JSD = β·KL(pT‖m) + (1-β)·KL(pS‖m)
    kl_teacher = F.kl_div(m.log(), p_teacher, reduction='none').sum(-1)  # per-token
    kl_student = F.kl_div(m.log(), p_student, reduction='none').sum(-1)

    jsd_per_token = beta * kl_teacher + (1 - beta) * kl_student  # [seq_len]
    return jsd_per_token.mean()  # 序列平均
```

### 5.3 Logits 对齐的关键问题

由于教师和学生的 prompt 长度不同，需要正确对齐 `y_hat` 部分的 logits：

```python
def align_logits(student_logits, teacher_logits,
                 student_prompt_len, teacher_prompt_len, y_hat_len):
    """
    student_logits: 来自 [student_prompt + y_hat] 的前向传播
    teacher_logits: 来自 [teacher_prompt + y_hat] 的前向传播

    对齐策略：取各自 prompt 之后的 y_hat 部分
    注意：logits[i] 预测的是 position i+1 的 token
    """
    # 学生: 从 prompt 最后一个 token 开始（预测 y_hat 的第一个 token）
    s_start = student_prompt_len - 1
    s_end = s_start + y_hat_len

    # 教师: 同理
    t_start = teacher_prompt_len - 1
    t_end = t_start + y_hat_len

    return student_logits[s_start:s_end], teacher_logits[t_start:t_end]
```

### 5.4 与 LoRA 集成

实验使用 LoRA 进行参数高效训练：

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM"
)

# 学生模型: 带 LoRA（可训练）
student_model = get_peft_model(base_model, lora_config)

# 教师模型: 原始基座模型（冻结）
teacher_model = base_model  # 无 LoRA，参数冻结
```

这样教师始终是初始策略（base instruct model），学生通过 LoRA 适配逐步学习。

## 6. 与其他方法的对比总结

| 特性 | SFT | GRPO | On-Policy 蒸馏 | OPSD |
|------|-----|------|--------------|------|
| On-Policy 数据 | ✗ | ✓ | ✓ | ✓ |
| 密集学习信号 | ✓ | ✗ | ✓ | ✓ |
| 低采样成本 | ✓ | ✗ | ✓ | ✓ |
| 无需外部教师 | ✓ | ✓ | ✗ | ✓ |

OPSD 是唯一同时具备以上四个优势的方法。

## 7. 局限性与未来方向

- **模型规模要求**：太小的模型（如 1.7B）合理化能力不足，自蒸馏效果有限
- **未利用正确性验证**：当前框架没有显式利用生成答案的正确性信号
- **问题难度**：超出模型理解能力的问题，即使给出答案教师策略也无法提供有效监督
- **课程学习**：根据模型能力动态调整问题难度是一个重要的扩展方向
