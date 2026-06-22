"""nano-mllm · D03 完整 GPT 架构：embedding + N×decoder block + lm_head
对应路线图 D03「GPT 完整架构」。把 attention(P03/D01) + RoPE(D04) 拼成一个可训练的 nano-GPT。
运行: python nano-mllm/model/transformer.py   （打印参数量）
依赖: torch
"""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from model.attention import CausalSelfAttention


@dataclass
class GPTConfig:
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    block_size: int = 128       # 最大上下文长度


class Block(nn.Module):
    """Pre-Norm decoder block：x → LN → Attn → +残差 → LN → FFN → +残差。"""

    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)   # RoPE 已带位置信息，无需 pos_emb
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight                 # 权重共享，省参数
        self.apply(self._init_weights)                           # GPT-2 式初始化（起始 loss ≈ ln(vocab)）

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):                                       # idx: (B, T)
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))                         # logits: (B, T, vocab)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.block_size:])[:, -1] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


if __name__ == "__main__":
    m = GPT(GPTConfig())
    n = sum(p.numel() for p in m.parameters())
    print(f"nano-GPT 参数量 ≈ {n / 1e6:.2f}M  (n_layer={m.cfg.n_layer}, d_model={m.cfg.d_model})")
    print("前向输出 logits:", tuple(m(torch.zeros(1, 16, dtype=torch.long)).shape))
