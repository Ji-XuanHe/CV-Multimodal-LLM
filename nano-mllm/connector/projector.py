"""nano-mllm · D13 连接器 / Projector：把视觉 token 对齐到 LLM 的词向量空间
对应路线图 D13。LLaVA 三件套的第②件，也是整个架构里唯一为「对齐」而新增、必须训练的部件。
LLaVA-1 用单层 Linear，LLaVA-1.5 换成 2 层 MLP（这里用后者）。
可选：把 num_compress<1 来顺带压缩视觉 token 数量（接你 D32 的研究方向）。
运行: python nano-mllm/connector/projector.py
依赖: torch
"""
import torch
import torch.nn as nn


class Projector(nn.Module):
    """视觉特征维度 vis_dim → LLM 隐藏维度 llm_dim 的 2 层 MLP。"""

    def __init__(self, vis_dim, llm_dim, hidden=None):
        super().__init__()
        hidden = hidden or llm_dim
        self.net = nn.Sequential(
            nn.Linear(vis_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, z):                 # z: (B, N, vis_dim)
        return self.net(z)                # -> (B, N, llm_dim)，变成 LLM 能吃的「伪词向量」


if __name__ == "__main__":
    proj = Projector(vis_dim=64, llm_dim=128)
    vis_tokens = torch.randn(2, 9, 64)    # 9 个视觉 token，每个 64 维
    out = proj(vis_tokens)
    print(f"Projector: 视觉特征 {tuple(vis_tokens.shape)} -> LLM 维度 {tuple(out.shape)}  "
          f"（{out.shape[1]} 个「视觉词」，可直接和文本词向量拼接）")
