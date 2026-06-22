"""nano-mllm · D32/D38 退化感知视觉 token 压缩（你的研究种子）
对应路线图 D32（提出想法）+ D38（做成可消融的「开关」）。
核心论点：低光/噪声等退化会改变「哪些视觉 token 重要」，而 FastV 等现有方法
几乎只按清晰图上的 attention 剪枝。本模块在剪枝前用一个【质量/退化感知】信号
重排 token 重要性，并做成「一个开关」：enable=False 时逐位等于基线，便于干净消融。

与 token_prune.py 的关系：token_prune.py = 基线 FastV 剪枝 + 「退化会改变重要性」的现象演示；
本文件 = 真正用质量信号去重排的方法模块（带 enable 开关，给 D38 的消融用）。
运行: python nano-mllm/efficiency/degradation_aware.py
依赖: numpy
"""
import numpy as np


def quality_map(tokens):
    """估计每个 token 的局部质量：用特征方差当代理 —— 退化（低光/噪声压平）会降低局部对比 → 方差小。"""
    return tokens.var(axis=-1)


def degradation_aware_importance(attn_importance, tokens, enable=True, gamma=1.0):
    """把基线 attention 重要性，用质量信号重加权。
    enable=False -> 原样返回（= 基线 FastV），这就是给消融用的「开关」。"""
    if not enable:
        return attn_importance
    q = quality_map(tokens)
    q = (q - q.min()) / (q.max() - q.min() + 1e-8)        # 归一化到 [0,1]
    return attn_importance * (1.0 + gamma * q)            # 清晰 token 加权、退化 token 相对降权


def keep_topk(importance, k):
    return set(int(i) for i in np.argsort(importance)[-k:])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_tok, dim, keep = 16, 32, 6
    tokens = rng.standard_normal((n_tok, dim))
    tokens[n_tok // 2:] *= 0.2                            # 模拟退化：后半区 token 被压平（细节丢失）
    attn_imp = rng.random(n_tok)                          # 基线 attention 重要性

    base = keep_topk(degradation_aware_importance(attn_imp, tokens, enable=False), keep)
    ours = keep_topk(degradation_aware_importance(attn_imp, tokens, enable=True), keep)
    print(f"基线(开关=关) 保留: {sorted(base)}")
    print(f"退化感知(开关=开) 保留: {sorted(ours)}")
    changed = keep - len(base & ours)
    print(f"切换开关改变了 {changed}/{keep} 个保留 token "
          f"-> enable 是干净的消融开关（关=基线，开=你的方法）✅")
