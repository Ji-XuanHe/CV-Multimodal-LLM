"""nano-mllm · D01 因果注意力：给注意力加上三角 mask（GPT 与 ViT 唯一却最关键的区别）
对应路线图 D01。这是 D01 的最小教学演示——只演示「因果 mask」一个点；
可复用的多头生产版本见 model/attention.py 的 CausalSelfAttention（P03/D01 合并模块）。
运行: python nano-mllm/model/causal_attention.py
依赖: torch
"""
import torch
import torch.nn.functional as F


def causal_attention(Q, K, V):
    """带因果 mask 的 scaled dot-product：每个位置只能看自己和左边。"""
    T = Q.size(-2)
    scores = Q @ K.transpose(-2, -1) / Q.size(-1) ** 0.5
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)  # 上三角=未来，屏蔽掉
    return F.softmax(scores + mask, dim=-1) @ V


if __name__ == "__main__":
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 5, 64) for _ in range(3))
    out = causal_attention(q, k, v)
    print("因果注意力输出:", tuple(out.shape), "（每个位置只看自己和左边）")

    # 验证因果性：篡改未来 token，不应影响更早位置的输出
    k2 = k.clone()
    k2[0, 4] += 10.0                                  # 改最后一个位置
    out2 = causal_attention(q, k2, v)
    same = torch.allclose(out[0, 0], out2[0, 0], atol=1e-6)
    print(f"改 t=4 后，t=0 的输出是否不变? {same}  -> 因果性成立（看不到未来）✅")
