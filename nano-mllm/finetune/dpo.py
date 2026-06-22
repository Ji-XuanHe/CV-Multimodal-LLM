"""nano-mllm · D07 DPO：不需要 reward model 的对齐
对应路线图 D07「DPO」。直觉：抬高 chosen、压低 rejected，β 拴住不偏离参考模型。
运行: python nano-mllm/finetune/dpo.py
依赖: torch
"""
import torch
import torch.nn.functional as F


def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta=0.1):
    """log 概率：*_w=chosen, *_l=rejected; logp=当前策略 π_θ, ref=冻结的参考模型。"""
    pi_logratio = logp_w - logp_l
    ref_logratio = ref_w - ref_l
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio))


if __name__ == "__main__":
    rw, rl = torch.tensor(-2.0), torch.tensor(-2.0)            # 参考模型对好/坏一视同仁
    a = dpo_loss(torch.tensor(-1.0), torch.tensor(-3.0), rw, rl)   # 策略更偏好「好回答」
    b = dpo_loss(torch.tensor(-3.0), torch.tensor(-1.0), rw, rl)   # 策略更偏好「坏回答」
    print(f"更偏好好回答 loss = {a.item():.3f}（小）")
    print(f"更偏好坏回答 loss = {b.item():.3f}（大）-> 梯度把它往 A 推 ✅")
