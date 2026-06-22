"""nano-mllm · D05 训练 nano-GPT（W1 毕业作品）
在 data/sample.txt 上训一个字符级 GPT，然后生成文本。CPU 上约 1~2 分钟。
对应路线图 D05「语言模型训练」。这是把 P01~D04 的零件合体跑通的「会写字」时刻。

运行: python nano-mllm/train_lm.py
依赖: torch, numpy
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from data.tokenizer import CharTokenizer
from model.transformer import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])   # 标签右移一位
    return x.to(device), y.to(device)


def main(iters=2000, block_size=64, batch_size=32, lr=3e-3):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    text = open(os.path.join(HERE, "data", "sample.txt"), encoding="utf-8").read()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"语料 {len(text)} 字符, 词表 {tok.vocab_size}, 设备 {dev}")

    cfg = GPTConfig(vocab_size=tok.vocab_size, n_layer=4, n_head=4, d_model=128, block_size=block_size)
    model = GPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for it in range(iters + 1):
        x, y = get_batch(data, block_size, batch_size, dev)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 250 == 0:
            print(f"iter {it:4d}  loss {loss.item():.3f}")

    print("\n--- 生成 (训练后) ---")
    start = torch.tensor([tok.encode("To be")], dtype=torch.long, device=dev)
    out = model.generate(start, max_new_tokens=200, temperature=0.8, top_k=20)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
