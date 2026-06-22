"""nano-mllm · D29 RAG：余弦相似度检索 Top-K
对应路线图 D29「RAG」。检索内核 = 向量 + 余弦（正是 CLIP 的归一化点积）。
运行: python nano-mllm/rag/retriever.py
依赖: numpy
"""
import numpy as np


class Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.vocab = sorted(set(" ".join(docs).split()))
        self.mat = np.stack([self._embed(d) for d in docs])

    def _embed(self, text):
        v = np.array([text.split().count(w) for w in self.vocab], float)   # 玩具词袋
        n = np.linalg.norm(v)
        return v / n if n else v

    def search(self, query, top_k=2):
        q = self._embed(query)
        scores = self.mat @ q                          # 已归一化 -> 点积即余弦
        order = np.argsort(scores)[::-1][:top_k]
        return [(round(float(scores[i]), 3), self.docs[i]) for i in order]


if __name__ == "__main__":
    r = Retriever([
        "low light image enhancement",
        "a cat sat on the mat",
        "object detection with yolo",
    ])
    for score, doc in r.search("how to enhance low light photos"):
        print(f"{score:.3f}  {doc}")
    print("-> 最相关文档排第一 ✅  （真实 RAG 把它拼进 prompt 前/后，避开中间盲区，见 D29）")
