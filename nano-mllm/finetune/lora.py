"""nano-mllm · D08 LoRA：冻结大矩阵，只训低秩旁路
对应路线图 D08「LoRA」。B 零初始化 -> 起步 ΔW=0，从原模型无损开始。
运行: python nano-mllm/finetune/lora.py
依赖: torch
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                        # ❄️ 冻结原权重 W
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # 零初始化
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T   # Wx + BAx


if __name__ == "__main__":
    lin = nn.Linear(4096, 4096)
    lora = LoRALinear(lin)
    full = sum(p.numel() for p in lin.parameters())
    train = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    x = torch.randn(2, 4096)
    print(f"可训练 {train / 1e3:.0f}K  vs  全量 {full / 1e6:.1f}M  -> 省 {full / train:.0f}×")
    print(f"B=0 时输出与原模型一致? {torch.allclose(lora(x), lin(x))} ✅")
