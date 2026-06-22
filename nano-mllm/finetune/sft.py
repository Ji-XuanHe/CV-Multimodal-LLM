"""nano-mllm · D06 SFT：监督微调的核心 = loss mask（只学回答）
对应路线图 D06「预训练→SFT→RLHF」。
关键：prompt(system+user) 不算 loss，只对 response(assistant) 算 loss。
运行: python nano-mllm/finetune/sft.py
依赖: torch
"""
import torch
import torch.nn.functional as F


def sft_loss(logits, labels, prompt_len):
    """logits: (T, V); labels: (T,); 前 prompt_len 个 token 不计 loss。"""
    masked = labels.clone()
    masked[:prompt_len] = -100                  # -100 被 CrossEntropy 忽略
    return F.cross_entropy(logits, masked, ignore_index=-100)


if __name__ == "__main__":
    torch.manual_seed(0)
    T, V, prompt_len = 8, 1000, 5               # 前 5 个 token 是 system+user，后 3 个是回答
    logits = torch.randn(T, V)
    labels = torch.randint(0, V, (T,))

    sft = sft_loss(logits, labels, prompt_len)
    resp_only = F.cross_entropy(logits[prompt_len:], labels[prompt_len:])
    print(f"SFT loss = {sft:.4f}   仅 response = {resp_only:.4f}")
    print(f"两者一致? {torch.allclose(sft, resp_only)}  -> SFT 确实只学回答 ✅")
