"""nano-mllm · D14/D15 端到端：训练 tiny VLM，让它「看图说话」🎉（capstone 多模态毕业作品）
自包含合成数据（彩色形状），训练 视觉塔+projector+nano-GPT，然后对没见过的图生成「a {颜色} {形状}」。
不下载任何权重，CPU 约 20~40 秒。这是 nano-mllm 里「M」真正成立的时刻。

注：真 LLaVA 是两阶段（先冻结预训练视觉塔+LLM 只训 projector，再联合微调）；这里所有零件都是
从零的 tiny 模型，直接联合训练更直接。把 TinyViT 换成真 SigLIP 后即可照搬两阶段配方（见 encoder.py）。
运行: python nano-mllm/train_vlm.py
依赖: torch, numpy
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

from model.mllm import TinyVLM
from model.transformer import GPT, GPTConfig
from vision.encoder import TinyViT

# —— 微型词表 ——
PAD, BOS, EOS, IMG, A = 0, 1, 2, 3, 4
COLORS = ["red", "green", "blue"]          # id 5,6,7
SHAPES = ["square", "circle", "triangle"]  # id 8,9,10
ID2W = {4: "a", 5: "red", 6: "green", 7: "blue", 8: "square", 9: "circle", 10: "triangle"}
VOCAB = 11


def render(color, shape, rng, dim=1.0):
    """画一张 12×12 的图：颜色=哪个通道亮，形状=亮区的空间样式。每次带噪声，不可逐像素记忆。
    dim<1 模拟低光退化（整体变暗、信号沉向噪声底）—— 你研究方向的核心场景。"""
    img = rng.normal(0, 0.05, (3, 12, 12)).astype("float32")
    yy, xx = np.mgrid[0:12, 0:12]
    if shape == 0:                                       # 方块：中央实心块
        m = (yy >= 2) & (yy < 10) & (xx >= 2) & (xx < 10)
    elif shape == 1:                                     # 圆：中央圆盘
        m = (yy - 5.5) ** 2 + (xx - 5.5) ** 2 <= 4.0 ** 2
    else:                                                # 三角：左下三角
        m = (yy >= 2) & (xx >= 2) & (xx <= yy)
    img[color][m] += 1.0
    return img * dim


def make_batch(B, rng):
    colors, shapes = rng.integers(0, 3, B), rng.integers(0, 3, B)
    imgs = np.stack([render(c, s, rng) for c, s in zip(colors, shapes)])
    ids = np.stack([[BOS, IMG, A, 5 + c, 8 + s, EOS] for c, s in zip(colors, shapes)])  # [BOS,<image>,a,颜色,形状,EOS]
    return torch.tensor(imgs), torch.tensor(ids), colors, shapes


def train_model(steps=600, B=64, lr=3e-3, seed=0, verbose=True, drop_to=None):
    """训练 tiny VLM（清晰图），返回 (vlm, dev)。供 research/ablation.py 复用。
    drop_to!=None 时做 token dropout（每步随机保留 [drop_to, N] 个视觉 token），让模型学会用任意子集，
    这样推理期「保留哪些 token」才真正影响准确率（消融才有意义）。"""
    from efficiency.token_select import make_compressor
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpt = GPT(GPTConfig(vocab_size=VOCAB, n_layer=2, n_head=4, d_model=128, block_size=64))
    vlm = TinyVLM(gpt, TinyViT(), image_token_id=IMG).to(dev)
    opt = torch.optim.AdamW(vlm.parameters(), lr=lr)
    if verbose:
        print(f"TinyVLM 参数量 ≈ {sum(p.numel() for p in vlm.parameters()) / 1e6:.2f}M  设备 {dev}")
    for step in range(steps + 1):
        imgs, ids, _, _ = make_batch(B, rng)
        imgs, ids = imgs.to(dev), ids.to(dev)
        if drop_to is not None:                          # 随机 token dropout：每步保留 [drop_to, N] 个
            vlm.compress = make_compressor(int(rng.integers(drop_to, vlm.vision.n_patches + 1)), "random")
        logits, n_vis, pos = vlm(imgs, ids)
        tgt = ids[:, pos + 1:]                            # 监督目标：[a, 颜色, 形状, EOS]
        lg = logits[:, pos + n_vis - 1: pos + n_vis - 1 + tgt.size(1)]
        loss = F.cross_entropy(lg.reshape(-1, VOCAB), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and step % 150 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}")
    vlm.compress = None
    return vlm, dev


def main(steps=600):
    vlm, dev = train_model(steps=steps)
    rng = np.random.default_rng(123)                     # held-out 用新 rng
    # —— 看图说话：对没见过的新图（9 类各 10 张）生成 caption ——
    print("\n--- 看图说话（held-out）---")
    correct, shown = 0, 0
    prefix = torch.tensor([[BOS, IMG]], device=dev)
    for c in range(3):
        for s in range(3):
            for _ in range(10):
                img = torch.tensor(render(c, s, rng)[None]).to(dev)
                ids_out = vlm.generate(img, prefix, max_new=5, eos_id=EOS)[0]
                cap = " ".join(ID2W.get(i, "?") for i in ids_out)
                ok = (cap == f"a {COLORS[c]} {SHAPES[s]}")
                correct += ok
                if shown < 6:                            # 抽样展示前 6 条
                    print(f"  图=({COLORS[c]} {SHAPES[s]})  模型说: '{cap}'  {'✅' if ok else '❌'}")
                    shown += 1
    print(f"\n看图说话准确率: {correct}/90 = {correct / 90 * 100:.0f}%  -> nano-GPT 真的有了眼睛 🎉")


if __name__ == "__main__":
    main()
