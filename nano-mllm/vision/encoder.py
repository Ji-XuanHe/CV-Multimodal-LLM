"""nano-mllm · D12 视觉塔：把图像切 patch → 编码成一串「视觉特征」(grid token，非 CLS)
对应路线图 D12。这是 LLaVA 三件套的第①件。本仓库默认用一个自包含、可训练的 TinyViT
（不下载权重，便于端到端跑通）；要换成真 CLIP/SigLIP，见文件末 load_pretrained 的说明。
D14 的 model/mllm.py 直接 import 这里的 TinyViT。
运行: python nano-mllm/vision/encoder.py
依赖: torch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """ViT 编码层：双向（无 causal mask）多头自注意力 + FFN，pre-norm。"""

    def __init__(self, dim, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.h, self.dk = heads, dim // heads
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        B, N, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(d, dim=2)
        q, k, v = (t.view(B, N, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v)        # 双向：patch 互相都能看
        a = a.transpose(1, 2).reshape(B, N, d)
        x = x + self.proj(a)
        x = x + self.ffn(self.ln2(x))
        return x


class TinyViT(nn.Module):
    """最小视觉塔：Conv patchify → +位置嵌入 → N 层 Transformer → 返回 patch token 序列。"""

    def __init__(self, img_size=12, patch=4, in_ch=3, dim=64, depth=2, heads=4):
        super().__init__()
        self.n_patches = (img_size // patch) ** 2
        self.dim = dim
        self.patch = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)  # 切 patch + 线性嵌入
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads) for _ in range(depth)])

    def forward(self, imgs):                               # imgs: (B, 3, H, W)
        z = self.patch(imgs).flatten(2).transpose(1, 2)    # (B, n_patches, dim)
        z = z + self.pos
        for blk in self.blocks:
            z = blk(z)
        return z                                           # grid 特征（像 LLaVA，不取 CLS）


# 想用真视觉塔（带语义的预训练特征）时，把 TinyViT 换成下面这种壳即可（需 pip install open_clip_torch）：
#   import open_clip
#   model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
#   feats = model.visual.trunk.forward_features(imgs)   # 取倒数第二层 grid 特征，冻结
# 接口保持「(B,3,H,W) -> (B, n_patches, dim)」不变，D13/D14 无需改动。


if __name__ == "__main__":
    vit = TinyViT()
    out = vit(torch.randn(2, 3, 12, 12))
    print(f"TinyViT: 图像(2,3,12,12) -> 视觉 token {tuple(out.shape)}  "
          f"({vit.n_patches} 个 patch，每个 {vit.dim} 维)")
