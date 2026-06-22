"""nano-mllm · 后训练前沿：On-Policy Distillation (OPD) 从零实现
============================================================================
OPD 是 2025 最火的后训练范式之一，正好补在 SFT 和 RL 中间：
  · SFT/离策略蒸馏：信号稠密(每个 token 一个标签)，但在【教师的轨迹】上学 —— 推理时学生
                    自己走偏到没学过的状态会「错上加错」(exposure bias)。
  · RL/GRPO：在【学生自己的轨迹】上学(on-policy)，但奖励稀疏(一整条才一个标量分)。
  · OPD：两者之长 —— 学生【自己生成】轨迹(on-policy)，教师对【每一个 token】给出完整分布、
         用 reverse-KL 打分(稠密)。结果：≈10× 比 GRPO 省(Thinking Machines: AIME'24 74.4%
         只用 RL 的 1/10 GPU 时)，已成主流蒸馏配方。

这个 toy：学生在一个「计数」序列任务上，靠在自己 rollout 的状态上匹配教师的逐 token 分布
(reverse-KL)学会教师的规则。运行就能看到 reverse-KL 掉到 ~0、规则跟随率爬到 ~100%。

核心 4 点 ↔ 真 OPD：
  1. 教师 q(·|state)：固定的「好分布」(这里规则=下一个≈(prev+1)%V)，给出【整条分布】(稠密)
  2. on-policy rollout：序列从【学生自己】采样 —— 学生会访问到它推理时真会遇到的状态
  3. reverse-KL：在学生访问的每个状态上，min KL(student || teacher) —— mode-seeking
  4. 稠密信号：每个 token 都有教师的完整分布做监督，不是 RL 的一个标量
运行: python nano-mllm/finetune/opd.py
依赖: numpy
============================================================================
"""
import numpy as np

V, L, B = 6, 10, 48          # 词表 6，rollout 长度 10，每步 48 条
PEAK = 0.75                  # 教师把 0.75 概率放在「正确的下一个」上，其余均摊


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def teacher():
    """教师分布 q(next|prev)：规则 = 下一个 token ≈ (prev+1)%V（带一点熵）。返回 (V,V)。"""
    q = np.full((V, V), (1 - PEAK) / (V - 1))
    for s in range(V):
        q[s, (s + 1) % V] = PEAK
    return q


def train(steps=150, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    q = teacher()
    W = np.zeros((V, V))                              # 学生 logits：p(·|prev)=softmax(W[prev])，初始均匀

    for step in range(steps + 1):
        grad = np.zeros((V, V))
        visits = np.zeros(V)
        for _ in range(B):
            s = rng.integers(V)                      # 随机起点
            for _ in range(L):                       # on-policy rollout：用学生自己采样往前走
                p = softmax(W[s])
                logr = np.log(p + 1e-9) - np.log(q[s] + 1e-9)
                kl = (p * logr).sum()                # reverse-KL(student||teacher) at state s
                grad[s] += p * (logr - kl)           # KL 对 logits 的闭式梯度
                visits[s] += 1
                s = rng.choice(V, p=p)               # 走到学生自己采的下一个状态
        W -= lr * grad / max(visits.sum(), 1)

        if step % 30 == 0:
            P = softmax(W)
            kl_avg = float((P * (np.log(P + 1e-9) - np.log(q + 1e-9))).sum(-1).mean())
            follow = float((P.argmax(-1) == (np.arange(V) + 1) % V).mean())
            print(f"step {step:3d}  reverse-KL(均) {kl_avg:.3f}   规则跟随率 {follow * 100:4.0f}%")

    P = softmax(W)
    print("\n学生学到的「下一个」:", [int(P[s].argmax()) for s in range(V)],
          " 教师规则:", [(s + 1) % V for s in range(V)],
          "->", "学生在自己的轨迹上学会了教师 ✅" if (P.argmax(-1) == (np.arange(V) + 1) % V).all() else "未完全收敛")


if __name__ == "__main__":
    train()
