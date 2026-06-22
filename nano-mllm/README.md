# nano-mllm —— 配套学习路线的「能跑」代码仓库

> [CV → 多模态大模型学习路线](https://ji-xuanhe.github.io/CV-Multimodal-LLM/) 的配套代码。
> 路线图里每天说的「今天产出 `xxx.py`」，在这里都能**直接运行、一步步跑通**。
> 从一行 numpy 反向传播，搭到 nano-GPT → 微调（含 RLHF/GRPO）→ CLIP → Agent → 你的研究方向。

## 快速开始

```bash
cd nano-mllm
pip install -r requirements.txt          # 只要 numpy + torch
```

然后**按这个顺序一步步跑**（每步都会打印一个让你「看到它真的在工作」的结果）：

```bash
# —— 基础：训练引擎 + 注意力 + nano-GPT —— 
python engine/autograd_demo.py     # P01 手写反向 + 梯度检查（<1e-6 = 你懂反向了）
python data/tokenizer.py           # D02 字符 tokenizer + BPE 合并
python model/attention.py          # P03 手写注意力 vs 官方 SDPA 对拍
python model/causal_attention.py   # D01 因果 mask：验证「看不到未来」
python model/rope.py               # D04 RoPE：验证只依赖相对位置
python model/transformer.py        # D03 nano-GPT 参数量
python train_lm.py                 # D05 ★ 训练 nano-GPT 并生成文本（CPU ~1-2 分钟）

# —— 后训练 / 强化学习（初学者重点，先读 finetune/README.md）—— 
python finetune/sft.py             # D06 SFT：loss 只学回答
python finetune/toy_rlhf.py        # ★ 从零 GRPO：无标注，奖励自己往上爬
python finetune/dpo.py             # D07 DPO：不用 RL 的对齐捷径
python finetune/lora.py            # D08 LoRA：省 128× 参数

# —— 多模态核心：把 nano-GPT 变成会看图的 VLM —— 
python vision/clip.py              # D11 CLIP 对比损失单 batch 过拟合自检
python vision/encoder.py           # D12 视觉塔（TinyViT）：图 -> 视觉 token
python connector/projector.py      # D13 projector：视觉 token -> LLM 维度
python model/mllm.py               # D14 合体 VLM：<image> 占位符替换
python train_vlm.py                # D14/D15 ★ 训 tiny VLM 看图说话（CPU ~30s，准确率 100%）

# —— 推理 / Agent / 效率 —— 
python inference/kv_cache.py       # D10 KV Cache 不改结果只提速
python efficiency/token_prune.py        # D32 FastV 剪枝 + 「退化改变重要性」现象
python efficiency/degradation_aware.py  # D32/D38 ★ 退化感知压缩（你的研究种子，带消融开关）
python agent/react.py              # D26 ReAct Agent 跑出决策轨迹
python rag/retriever.py            # D29 余弦检索 Top-K

# —— 研究方向：把退化感知 token 压缩接进真 VLM，跑真实消融 —— 
python efficiency/token_select.py  # D32/D34 三种 token 压缩器（full / blind / degaware）
python research/ablation.py        # D35-D40 ★ 多 seed 消融 -> experiments/baselines.csv（CPU ~3-4 分钟）
```

> 想一键全跑：`for f in engine/autograd_demo.py model/attention.py ... ; do python $f; done`

## 目录 = 学习路线的 capstone 主线

| 目录 / 文件 | 对应天 | 干什么 |
|---|---|---|
| `engine/autograd_demo.py` | P01 | 手写前向+反向+梯度检查 |
| `data/tokenizer.py` | D02 | 字符 tokenizer + BPE |
| `model/attention.py` | P03 | 注意力（SDPA+多头+causal+RoPE），**全项目复用** |
| `model/causal_attention.py` | D01 | 因果 mask 最小演示（验证看不到未来） |
| `model/rope.py` | D04 | 旋转位置编码 |
| `model/transformer.py` | D03 | nano-GPT 模型 |
| `train_lm.py` | D05 | **训练 nano-GPT + 生成** |
| `finetune/` | D06–D08 | **后训练教程：SFT / GRPO / DPO / LoRA**（见 `finetune/README.md`）|
| `inference/kv_cache.py` | D10 | KV Cache 加速生成 |
| `vision/clip.py` | D11 | CLIP 对比损失 + 过拟合自检 |
| `vision/encoder.py` | D12 | 视觉塔 TinyViT（grid 视觉 token） |
| `connector/projector.py` | D13 | 视觉 token → LLM 维度（MLP） |
| `model/mllm.py` | D14 | **VLM 合体**：视觉塔+projector+nano-GPT |
| `train_vlm.py` | D14/D15 | **★ 训 tiny VLM 看图说话（held-out 100%）** |
| `efficiency/token_prune.py` | D32 | FastV 剪枝 + 「退化改变重要性」现象 |
| `efficiency/degradation_aware.py` | D32/D38 | **退化感知压缩（研究种子，带消融开关）** |
| `efficiency/token_select.py` | D32/D34 | 视觉 token 压缩器（接进 VLM 的 `compress` 钩子） |
| `research/ablation.py` | D35–D40 | **★ 多 seed 退化消融 → `experiments/baselines.csv`** |
| `agent/react.py` | D26/D30 | ReAct Agent 循环 |
| `rag/retriever.py` | D29 | 向量检索 |

## 重点：后训练 / RLHF / GRPO 初学者教程

很多人卡在「RLHF / PPO / GRPO 到底在干嘛」。`finetune/` 是一套**能直接运行**的最小教程，
跑一下 `python finetune/toy_rlhf.py` 就能看到「**无任何标注，只给一个规则奖励，模型自己
学会高奖励行为**」—— 这正是训练 DeepSeek-R1 / o1 这类推理模型的 RL 核心。

👉 **先读 [`finetune/README.md`](finetune/README.md)**，它把 SFT → 奖励来源 → PPO vs GRPO →
DPO 用大白话 + 可运行代码串起来。

## 重点：从 nano-GPT 到「会看图」的 tiny VLM 🎉

`vision/` + `connector/` + `model/mllm.py` 把语言大脑接上眼睛：**图像 → 视觉塔 → projector →
替换 `<image>` 占位符 → 和文本一起进 nano-GPT**（正是 LLaVA 三件套）。

👉 跑一下 `python train_vlm.py`：自包含合成「彩色形状」数据，CPU 上 ~30 秒训出一个 tiny VLM，
对**没见过**的图生成「a red square」之类的描述，**held-out 准确率 100%** —— 这是 nano-mllm 里
「M」真正成立的时刻。不下载任何权重；要换真 SigLIP/CLIP 视觉塔，见 `vision/encoder.py` 末尾说明。

## 重点：你的研究方向，做成能跑的实验骨架 🔬

`model/mllm.py` 留了个 `compress` 钩子，`efficiency/token_select.py` 提供三种视觉 token 压缩器
（不压缩 / 退化盲 / **退化感知**）。`research/ablation.py` 把它们接进真 VLM，在**清晰 vs 低光**图上、
给定 token 预算跑 **with/without 消融**，输出 `experiments/baselines.csv`。

👉 它顺带演示了一条**做研究的纪律**：单 seed 上「退化感知」一度比基线高 +10pp，但跑 **5 个 seed
报 mean±std**，差距落进噪声里（`+0.01 ± 0.09`，符号还会翻）—— 这正是路线图 D38 的教训：
**单 seed 的提升常是噪声**。脚本如实报数、不为「赢」造数据。把 `render` 换成真实低光图、
`TinyViT` 换成 SigLIP，这套流程照搬即用，就是你真正研究的起点。

## 状态

这是配套代码的 **v1**：capstone 主线的**可运行核心轨**已打通 —— 前置周 → W1 nano-GPT →
后训练（SFT/RLHF/GRPO/DPO/LoRA）→ **多模态核心（D11–D14：视觉塔 + projector + VLM，端到端会看图说话）**
→ Agent / 效率 / 研究种子 demo。其余天的「真模型大件」（用预训练 LLaVA/Qwen-VL 权重推理、
完整评测 pipeline、grounding / video 等）请对照路线图各天代码块 + 注释指向的官方库
（timm / transformers / open_clip / VLMEvalKit）。

## License

MIT —— 自由用于学习、教学、二次开发。
