"""nano-mllm · D14 合体时刻：视觉塔 + Projector + nano-GPT = 第一个完整 VLM 🎉
对应路线图 D14。装配逻辑 = LLaVA：图像 → 视觉塔(D12) → projector(D13) → 替换文本里的
<image> 占位符 → 和文本词向量一起进 nano-GPT(W1) 自回归生成。nano-GPT 终于有了眼睛。
端到端训练 + 看图说话见 train_vlm.py。
运行: python nano-mllm/model/mllm.py   （打印一次前向的形状）
依赖: torch
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from connector.projector import Projector
from model.transformer import GPT
from vision.encoder import TinyViT


class TinyVLM(nn.Module):
    def __init__(self, gpt: GPT, vision: TinyViT, image_token_id: int):
        super().__init__()
        self.gpt = gpt
        self.vision = vision
        self.proj = Projector(vision.dim, gpt.cfg.d_model)
        self.image_token_id = image_token_id

    def _merge(self, images, input_ids):
        """把 <image> 占位符那一格，替换成 N 个投影后的视觉 token（LLaVA 的核心动作）。"""
        vis = self.proj(self.vision(images))                      # (B, Nv, d)
        tok = self.gpt.tok_emb(input_ids)                         # (B, T, d)
        # 简化：假设整个 batch 的 <image> 在同一列（真 LLaVA 支持逐样本任意位置）
        pos = (input_ids[0] == self.image_token_id).nonzero()[0, 0].item()
        merged = torch.cat([tok[:, :pos], vis, tok[:, pos + 1:]], dim=1)
        return merged, vis.size(1), pos                           # (B, T-1+Nv, d)

    def _run_lm(self, x):
        for blk in self.gpt.blocks:
            x = blk(x)
        return self.gpt.lm_head(self.gpt.ln_f(x))

    def forward(self, images, input_ids):
        x, n_vis, pos = self._merge(images, input_ids)
        return self._run_lm(x), n_vis, pos                        # logits, Nv, <image>位置

    @torch.no_grad()
    def generate(self, images, prefix_ids, max_new, eos_id):
        """prefix_ids 形如 [BOS, <image>]；从视觉+BOS 出发自回归生成答案 token。"""
        B = images.size(0)
        x, _, _ = self._merge(images, prefix_ids)
        outs, done = [[] for _ in range(B)], [False] * B
        for _ in range(max_new):
            nxt = self._run_lm(x)[:, -1].argmax(-1)               # 贪心
            x = torch.cat([x, self.gpt.tok_emb(nxt[:, None])], dim=1)
            for b in range(B):
                if not done[b]:
                    if nxt[b].item() == eos_id:
                        done[b] = True
                    else:
                        outs[b].append(nxt[b].item())
            if all(done):
                break
        return outs


if __name__ == "__main__":
    from model.transformer import GPTConfig
    gpt = GPT(GPTConfig(vocab_size=20, n_layer=2, n_head=4, d_model=128))
    vlm = TinyVLM(gpt, TinyViT(), image_token_id=3)
    imgs = torch.randn(2, 3, 12, 12)
    ids = torch.tensor([[1, 3, 5, 6, 7, 2], [1, 3, 8, 9, 10, 2]])     # [BOS,<image>,..,EOS]
    logits, nv, pos = vlm(imgs, ids)
    print(f"TinyVLM 前向: 文本 {tuple(ids.shape)} + 图像9token -> logits {tuple(logits.shape)}")
    print(f"  <image> 在第 {pos} 列，被替换成 {nv} 个视觉 token（序列 6 -> {logits.shape[1]}）✅")
