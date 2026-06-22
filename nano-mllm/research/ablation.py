"""nano-mllm · D35–D40 研究骨架：退化感知视觉 token 压缩，多 seed with/without 消融
把你的研究问题做成能跑的实验：在【清晰 vs 低光】图上、给定 token 预算，比较四种「保留哪些
视觉 token」的策略，看哪种在退化下最能保住「看图说话」准确率：
  full(不压缩,上界) / blind(退化盲,按 token 范数) / degaware(退化感知,按图像局部对比度) / random(下界)

★ 关键纪律（D38）：跑【多个 seed】报 mean±std，而不是单 seed。因为单 seed 的「提升」常常是噪声。
   实测就踩到了这个坑：某个 seed 上 degaware 比 blind 高 +10pp，但跨 5 个 seed 一平均，
   差距落进噪声里、符号还会翻。诚实结论：toy 合成数据上无法断定方法更好，要靠真实低光 VQA。

输出 experiments/baselines.csv（每个 seed 的原始数据）。这里如实报数，不为「赢」造数据。
运行: python nano-mllm/research/ablation.py   （CPU 约 3~4 分钟：5 seed × 训练+消融）
依赖: torch, numpy
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from efficiency.token_select import make_compressor
from train_vlm import BOS, COLORS, EOS, ID2W, IMG, SHAPES, render, train_model

N_TOK, BUDGET = 9, 4                    # 9 个视觉 token，推理期保留 4（≈55% 压缩）
SEEDS = [0, 1, 2, 3, 4]
SETTINGS = [(1.0, "清晰"), (0.25, "低光x0.25")]
MODES = [("full", None), ("blind", BUDGET), ("degaware", BUDGET), ("random", BUDGET)]


def caption_acc(vlm, dev, dim, compressor, n_each=25, seed=2024):
    rng = np.random.default_rng(seed)
    vlm.compress = compressor
    prefix = torch.tensor([[BOS, IMG]], device=dev)
    correct = total = 0
    with torch.no_grad():
        for c in range(3):
            for s in range(3):
                for _ in range(n_each):
                    img = torch.tensor(render(c, s, rng, dim=dim)[None]).to(dev)
                    out = vlm.generate(img, prefix, max_new=5, eos_id=EOS)[0]
                    cap = " ".join(ID2W.get(i, "?") for i in out)
                    correct += (cap == f"a {COLORS[c]} {SHAPES[s]}")
                    total += 1
    vlm.compress = None
    return correct / total


def main():
    print(f"多 seed 消融：{len(SEEDS)} 个 seed，各训一个 token-dropout 鲁棒的 tiny VLM 再评测...")
    raw = []                                            # 每个 (seed, 图像, 策略) 一行
    for seed in SEEDS:
        vlm, dev = train_model(steps=800, verbose=False, drop_to=BUDGET, seed=seed)
        for dim, dname in SETTINGS:
            for mode, k in MODES:
                comp = None if mode == "full" else make_compressor(k, mode)
                acc = caption_acc(vlm, dev, dim, comp)
                raw.append({"seed": seed, "image": dname, "method": mode,
                            "kept_tokens": N_TOK if mode == "full" else k, "accuracy": round(acc, 3)})
        print(f"  seed {seed} 完成")

    # —— 汇总 mean±std ——
    print(f"\n{'图像':<11}{'策略':<11}{'保留':<7}{'准确率 mean±std (5 seeds)'}")
    print("-" * 52)
    agg = {}
    for dim, dname in SETTINGS:
        for mode, _ in MODES:
            accs = np.array([r["accuracy"] for r in raw if r["image"] == dname and r["method"] == mode])
            agg[(dname, mode)] = (accs.mean(), accs.std())
            kept = N_TOK if mode == "full" else BUDGET
            print(f"{dname:<12}{mode:<12}{kept:<7}{accs.mean() * 100:5.1f}% ± {accs.std() * 100:4.1f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "baselines.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "image", "method", "kept_tokens", "accuracy"])
        w.writeheader()
        w.writerows(raw)
    print(f"\n已写入 {os.path.relpath(csv_path)}（{len(raw)} 行原始数据）")

    # —— 诚实判定：低光下 degaware vs blind ——
    gaps = np.array([
        next(r["accuracy"] for r in raw if r["seed"] == s and r["image"] == "低光x0.25" and r["method"] == "degaware")
        - next(r["accuracy"] for r in raw if r["seed"] == s and r["image"] == "低光x0.25" and r["method"] == "blind")
        for s in SEEDS])
    m, sd = gaps.mean(), gaps.std()
    noisy = abs(m) <= sd                                # 0 落在 ±1σ 内 -> 不能断定
    print(f"\n低光 degaware−blind 每 seed: {[round(float(g), 3) for g in gaps]}")
    print(f"  gap = {m:+.3f} ± {sd:.3f}  ->  "
          + ("⚠️ 0 在 ±1σ 内，toy 上无法断定退化感知更好（这就是 D38：单 seed 提升≈噪声）。"
             "真结论要靠真实低光 VQA + 多 seed。" if noisy
             else "跨 seed 稳健，值得在真实数据上验证。"))
    print("\n但骨架本身是真的：研究模块已能插进真 VLM、能消融、能出可信区间 —— "
          "把 render 换成真实低光图、把 TinyViT 换成 SigLIP，这套流程照搬即用。")


if __name__ == "__main__":
    main()
