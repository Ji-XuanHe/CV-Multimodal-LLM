"""nano-mllm · D07 后训练/RL 从零实现：GRPO toy（DeepSeek-R1 同款思路）
============================================================================
为什么有这个文件：很多初学者卡在「RLHF/PPO/GRPO 到底在干嘛」。这里用 ~50 行
纯 numpy，把 RL 后训练的核心一次说清，运行就能看到「奖励自己往上爬」。

核心思想（4 个零件，和真正的 RLHF/GRPO 一一对应）：
  1. 策略 π            —— 模型本身（这里简化成「每个位置一个字符分布」的 logits 表）
  2. rollout（采样）   —— 让策略生成一批完成（completions），就像 LLM 自回归生成
  3. 奖励 r            —— 一个【规则】给每条完成打分。注意：没有人工标注、没有学出来的
                          reward model —— 这正是 GRPO / 「可验证奖励 RLVR」的精髓：
                          奖励来自规则（答案对不对 / 格式对不对），便宜、稳、不被 hack
  4. 优势 advantage    —— GRPO 不用价值网络(critic)：同一题采 G 条，谁高于这组的平均
                          谁就是「相对更好」，advantage = (r − 组均值) / 组标准差
  5. KL 拴住参考模型    —— 别为了刷奖励就跑飞（reward hacking）。β 越大越保守。

把「规则奖励」换成「数学答案对不对」「代码能不能跑」，这就是训练推理模型的 RL。
============================================================================
运行: python nano-mllm/finetune/toy_rlhf.py
依赖: numpy
"""
import numpy as np

VOCAB = "abcdefgh"
VOWELS = set("ae")              # 我们「偏好」元音多的串（这就是我们的规则偏好）
T, V, G = 6, len(VOCAB), 64     # 序列长 6，词表 8，每步采样 G=64 条完成


def reward(seq):
    """【规则奖励 / 可验证奖励】：元音个数越多越好。无需任何人工标注。"""
    return sum(VOCAB[i] in VOWELS for i in seq)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def train(beta=0.0, lr=0.5, steps=120, seed=0):
    rng = np.random.default_rng(seed)
    logits = np.zeros((T, V))        # 策略 π：初始均匀（也就是「参考模型」）
    ref_logp = np.log(softmax(logits.copy()) + 1e-9)
    random_baseline = T * len(VOWELS) / V

    for step in range(steps + 1):
        probs = softmax(logits)                                  # (T, V)
        # 1) rollout：从当前策略采 G 条完成
        seqs = np.stack([[rng.choice(V, p=probs[t]) for t in range(T)] for _ in range(G)])
        rews = np.array([reward(s) for s in seqs], dtype=float)
        # 2) GRPO 优势：组内相对（无需 critic / value network）
        adv = (rews - rews.mean()) / (rews.std() + 1e-8)
        # 3) REINFORCE 策略梯度：把高优势完成里出现的字符概率推高
        grad = np.zeros_like(logits)
        for g in range(G):
            for t in range(T):
                onehot = np.zeros(V)
                onehot[seqs[g, t]] = 1.0
                grad[t] += adv[g] * (onehot - probs[t])
        grad /= G
        # 4) KL 拴住参考模型（β=0 时关闭；调大 β 策略会更贴近初始均匀）
        grad -= beta * (np.log(probs + 1e-9) - ref_logp)
        logits += lr * grad

        if step % 20 == 0:
            print(f"step {step:3d}  平均奖励 {rews.mean():.2f}/{T}   (随机基线 ≈ {random_baseline:.2f})")

    best = "".join(VOCAB[i] for i in softmax(logits).argmax(1))
    print(f"\n训练后每位置最可能字符: {best!r}  (β={beta}) "
          f"-> {'几乎全是元音，学会了高奖励行为 ✅' if sum(c in VOWELS for c in best) >= T - 1 else ''}")


if __name__ == "__main__":
    print("=== 纯奖励最大化 (β=0)：策略会一路冲向高奖励 ===")
    train(beta=0.0)
    print("\n=== 加 KL 约束 (β=0.3)：被参考模型拴住，更保守（防 reward hacking）===")
    train(beta=0.3)
