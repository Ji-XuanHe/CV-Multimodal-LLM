"""nano-mllm · D02 Tokenizer：字符级 tokenizer（train_lm 用）+ BPE 合并教学
对应路线图 D02「Tokenization」。
运行: python nano-mllm/data/tokenizer.py
依赖: numpy
"""
from collections import Counter


class CharTokenizer:
    """最简单的字符级 tokenizer —— nano-GPT 训练直接用它（够学习、零依赖）。"""

    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def bpe_best_merge(corpus):
    """走一轮 BPE：统计相邻 pair 频率，返回最高频的那一对（教学版）。"""
    pairs = Counter()
    for word, freq in corpus.items():
        syms = word.split()
        for a, b in zip(syms, syms[1:]):
            pairs[(a, b)] += freq
    (a, b), c = pairs.most_common(1)[0]
    return a, b, c


if __name__ == "__main__":
    tok = CharTokenizer("hello nano-mllm")
    ids = tok.encode("nano")
    print(f"字符级: 'nano' -> {ids} -> '{tok.decode(ids)}'  词表大小={tok.vocab_size}")

    corpus = {"h u g": 10, "p u g": 5, "h u g s": 5, "b u n": 4}
    a, b, c = bpe_best_merge(corpus)
    print(f"BPE 首次合并: {a!r}+{b!r} 出现 {c} 次 -> 合并成 {a + b!r}")
