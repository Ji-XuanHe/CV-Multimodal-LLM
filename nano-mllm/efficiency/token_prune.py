"""nano-mllm · D32 视觉 Token 压缩（你的研究方向）：FastV FLOPs + 退化感知
对应路线图 D32「视觉 Token 压缩」。
研究钩子：退化（低光/噪声）会改变「哪些 token 重要」，现有方法几乎只在清晰图上设计。
运行: python nano-mllm/efficiency/token_prune.py
依赖: numpy
"""
import numpy as np


def fastv_flops_saving(n_text=64, n_vis=576, n_layer=32, keep_after_layer=2, prune_ratio=0.5):
    """FastV：前 keep_after_layer 层用全部视觉 token，之后剪掉一部分。
    大模型 FLOPs 主要正比于「序列长 × 层」（FFN/线性主导）。"""
    def cost(nv):
        return n_text + nv
    full = n_layer * cost(n_vis)
    kept = int(n_vis * (1 - prune_ratio))
    fastv = keep_after_layer * cost(n_vis) + (n_layer - keep_after_layer) * cost(kept)
    return (1 - fastv / full) * 100


def degradation_changes_importance(seed=0):
    """退化是否改变「最该保留的 token」？返回 (清晰 top4, 退化 top4, 重叠数)。"""
    imp_clean = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    degrade = np.array([-0.5, -0.4, -0.3, 0.0, 0.0, 0.4, 0.5, 0.6])   # 压制细节、抬高平坦区
    imp_noisy = imp_clean + degrade
    top_clean = set(np.argsort(imp_clean)[-4:])
    top_noisy = set(np.argsort(imp_noisy)[-4:])
    return sorted(int(i) for i in top_clean), sorted(int(i) for i in top_noisy), len(top_clean & top_noisy)


if __name__ == "__main__":
    print(f"FastV 式剪枝省 FLOPs ≈ {fastv_flops_saving():.0f}%  （论文 ~45% 同量级）")
    c, n, overlap = degradation_changes_importance()
    print(f"清晰图最该留: {c}   退化图最该留: {n}")
    print(f"重叠 {overlap}/4 -> 退化改变了 token 重要性排序 = 你的研究空白 🎯")
