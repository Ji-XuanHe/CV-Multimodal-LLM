"""nano-mllm · D32/D34 视觉 token 压缩器：插进 VLM 的 compress 钩子，做 with/without 消融
对应你的研究方向。提供三种「保留 K 个视觉 token」的策略，签名统一为 fn(vis, images)->kept：
  · full      不压缩（上界）
  · blind     退化盲：按 token 幅值(L2 范数)选 —— 低光整体变暗/噪声会误导它
  · degaware  退化感知：用图像局部对比度选「信号 patch」—— 不被整体变暗迷惑（你的方法雏形）
  · random    随机保留（训练期做 token dropout，让模型学会用任意子集；也是消融的下界）
运行: python nano-mllm/efficiency/token_select.py
依赖: torch
"""
import torch


def patch_quality(images, n_side):
    """每个 patch 的局部对比度（通道平均的空间 std）：平坦背景/整体变暗->低，形状边缘->高。"""
    B, C, H, W = images.shape
    p = H // n_side
    x = images.unfold(2, p, p).unfold(3, p, p)            # (B, C, n_side, n_side, p, p)
    x = x.reshape(B, C, n_side * n_side, p * p)
    return x.std(dim=-1).mean(dim=1)                      # (B, n_side*n_side)


def select_tokens(vis, images, k, mode):
    """vis:(B,N,d), images:(B,3,H,W) -> 保留的 (B,k,d)。"""
    B, N, d = vis.shape
    if mode == "full" or k >= N:
        return vis
    if mode == "blind":
        imp = vis.norm(dim=-1)                            # token 幅值（退化盲）
    elif mode == "degaware":
        imp = patch_quality(images, int(round(N ** 0.5)))  # 图像局部对比度（退化感知）
    elif mode == "random":
        imp = torch.rand(B, N, device=vis.device)
    else:
        raise ValueError(mode)
    idx = imp.topk(k, dim=-1).indices                     # (B, k)
    idx = idx.sort(dim=-1).values                         # 保留原 patch 顺序
    return torch.gather(vis, 1, idx[:, :, None].expand(-1, -1, d))


def make_compressor(k, mode):
    """返回可挂到 vlm.compress 的闭包。"""
    return lambda vis, images: select_tokens(vis, images, k, mode)


if __name__ == "__main__":
    torch.manual_seed(0)
    vis = torch.randn(2, 9, 128)
    imgs = torch.randn(2, 3, 12, 12)
    for mode in ["full", "blind", "degaware", "random"]:
        out = select_tokens(vis, imgs, k=4, mode=mode)
        print(f"  {mode:9s} 9 token -> 保留 {out.shape[1]} 个  {tuple(out.shape)}")
