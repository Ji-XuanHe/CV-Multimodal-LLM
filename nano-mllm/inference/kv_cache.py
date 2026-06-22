"""nano-mllm · D10 KV Cache：缓存历史 K/V，生成提速（不改结果）
对应路线图 D10「KV Cache」。验证「逐步缓存」与「一次性全算」逐元素一致。
运行: python nano-mllm/inference/kv_cache.py
依赖: torch
"""
import torch
import torch.nn.functional as F


def full_causal_attention(Q, K, V):
    T = Q.size(0)
    S = Q @ K.transpose(-2, -1) / Q.size(-1) ** 0.5
    S = S + torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    return F.softmax(S, -1) @ V


def kv_cache_attention(Q, K, V):
    ck, cv, outs = [], [], []
    for t in range(Q.size(0)):
        ck.append(K[t])
        cv.append(V[t])                 # 追加当前步的 K、V（之前算过的不重算）
        Kc, Vc = torch.stack(ck), torch.stack(cv)
        s = Q[t] @ Kc.T / Q.size(-1) ** 0.5
        outs.append(F.softmax(s, -1) @ Vc)
    return torch.stack(outs)


if __name__ == "__main__":
    torch.manual_seed(0)
    Q, K, V = (torch.randn(6, 32) for _ in range(3))
    same = torch.allclose(full_causal_attention(Q, K, V), kv_cache_attention(Q, K, V), atol=1e-5)
    print(f"KV Cache 与 full attention 逐元素一致? {same}  -> 只省算力(免去重算历史 K/V 投影，整体生成 O(T³)→O(T²)) ✅")
