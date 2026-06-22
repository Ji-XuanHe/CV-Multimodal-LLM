# 立项文档 · 退化感知视觉 Token 压缩
### Degradation-Aware Visual Token Compression for Efficient Multimodal LLMs

> 配套可运行骨架：`model/mllm.py`（compress 钩子）· `efficiency/token_select.py`（压缩器）·
> `research/ablation.py`（多 seed 消融）· `experiments/baselines.csv`（原始数据）。
> 本文档遵守：引用全部经 arXiv 核实；初步结果如实报（含负结果）；实验设计预注册、多 seed 报 CI。

---

## 0. 一句话（TL;DR）

视觉 token 压缩是 VLM 提速的主流手段，但现有方法（FastV/ToMe/PruMerge）几乎都在**清晰图**上设计与评测。
真实部署常遇退化（低光/噪声/模糊），而**退化会改变「哪些视觉 token 重要」**——
在清晰图上学到的重要性度量在退化下可能失效。本课题研究**退化感知的视觉 token 压缩**：
在压缩时显式利用图像质量/退化信号，使「在固定 token 预算下、退化输入上」尽量少掉点。

## 1. 背景与动机

**瓶颈。** LLaVA 式 VLM 把一张图编码成数百个视觉 token（LLaVA-1.5 用 576 个 [2,3]），
它们和文本拼成长序列进 LLM，是推理 FLOPs/显存的大头。视觉 token 压缩因此成为提速主线：
FastV 在第 2 层后按 attention 剪掉低注意力视觉 token [1]；ToMe 合并相似 token [4]；
PruMerge 自适应减少 token [5]。

**空白。** 这些方法的设计与评测**几乎都基于清晰图**。但真实输入有退化——
R-Bench 系统评测显示主流 LMM 在真实世界 corruption 下显著掉点 [8]，
LVLM corruption robustness 也被进一步基准化 [9]。

**关键洞察（本人 CV 护城河：低光 / 图像质量评估）。**
退化不是「整体变难」这么简单——它会**改变 token 之间的相对重要性**：
低光把细节压向噪声底、噪声抬高平坦区的"伪显著性"。
因此「在清晰图上按 attention/范数选 token」的策略，在退化下可能保留了被噪声点亮的背景、丢掉了真正的信号。
**假设：把退化/质量感知显式引入 token 选择，能在退化输入上更好地保住任务表现。**

**初步实验已搭好骨架并给出诚实信号（见 §5）。**

## 2. 研究问题与假设（预注册）

> 先注册假设与指标，再跑完整评测，避免看到结果后再改度量（anti-p-hacking [10]）。

- **H1**：在固定 token 预算下，退化（低光/噪声/模糊）显著降低 compressed-VLM 的任务准确率（相对清晰图）。
- **H2**：在同一预算下，**退化感知选择 > 退化盲选择**，且差异在多 seed 下统计稳健（95% CI 不含 0）。
- **H3**：退化感知模块可作为「开关」干净消融——关闭时逐位等于 baseline，开启才是本方法（便于归因）。

主指标：退化 VQA 任务准确率 @ token keep-ratio ∈ {25%, 50%}。次指标：FLOPs、latency。

## 3. 方法

**架构。** 在标准 VLM 数据流 `图像 → 视觉塔 → projector → [视觉 token] + [文本] → LLM` 中，
于视觉 token 进入 LLM 前插入一个**压缩模块**（骨架里已是 `model/mllm.py` 的 `compress` 钩子）。

**退化感知重要性。** 不只用 attention / token 范数，而是融合一个**质量/退化信号**重排 token 重要性：
- 退化估计：局部对比度、亮度、噪声水平（接本人低光/IQA 经验，可用无参考 IQA 或 Retinex/Zero-DCE [7] 式分解的中间量）；
- 重排：`importance = f(attention, quality)`，使退化区域的伪显著性被抑制、真实信号 token 被保留。

**两条路线（消融对比）。**
- *训练无关（training-free）*：像 FastV 一样即插即用，便于和 [1,4,5] 公平比；
- *轻量可学习*：projector 后接一个小的 token scorer，用少量退化数据微调。

**「一个开关」。** `enable=False` 时退化感知项关闭、退化为 baseline 选择，保证 §H3 的干净消融。

## 4. 实验设计（严格）

**数据。** 在标准 VQA（VQAv2 / GQA / TextVQA / POPE）上施加可控退化（低光、高斯/泊松噪声、模糊、JPEG 压缩，
R-Bench [8] / ImageNet-C 式），并尽量纳入**真实低光**样本；清晰版本作为上界对照。

**模型。** LLaVA-1.5 [3]（CLIP [6] 视觉塔；同时试 SigLIP [11] 塔），先 training-free，再轻量可学习。

**Baselines（contribution 顺序对齐）。** full（不压，上界）· random（下界）· FastV [1] · ToMe [4] · PruMerge [5]
· **degaware（本方法）**。可选：先做低光增强 [7] 再压缩（检验"增强 vs 感知压缩"哪条更省更准）。

**评测协议（anti-p-hacking [10] + 消融纪律）。**
1. 预注册 H1–H3 与指标后再跑 test；超参在 val 上定死。
2. **固定 N=5 seeds 全报**，不挑 seed；报 mean ± std 与 95% CI（单 seed 的提升常是噪声 [12]）。
3. 消融**一次只动一个组件**，其余（seed/数据/预算）冻结；报 **DELTA** 而非只报绝对值。
4. 负结果照报（写进正文/附录）。

## 5. 初步结果（nano-mllm 玩具，诚实报）

已在自包含玩具（合成「彩色形状」VQA）上把方法接进真 VLM 跑通多 seed 消融
（`research/ablation.py` → `experiments/baselines.csv`）：

| 输入 | 策略 | 保留 token | 准确率（5 seed, mean±std） |
|---|---|---|---|
| 清晰 | full | 9 | 86.7% ± 26.7 |
| 低光×0.25 | full | 9 | **53.5% ± 20.1** |
| 低光×0.25 | blind（退化盲） | 4 | 48.4% ± 16.7 |
| 低光×0.25 | degaware（退化感知） | 4 | 49.7% ± 16.4 |

- **H1 在玩具上成立**：低光把 full-token 准确率从 86.7% 拉到 53.5%——退化确实损害。
- **H2 在玩具上「不成立 / 落在噪声内」**：低光下 degaware − blind = **+0.013 ± 0.090**
  （每 seed：+0.06, +0.13, −0.13, −0.05, +0.05）。**单个 seed 上 degaware 一度高 +10.8pp，但跨 5 seed 一平均就落进噪声、符号还翻。**
  这正是评测纪律的现场提醒 [9-RL, 10]：单 seed 的"提升"常是噪声，不能据此下结论。

**结论（诚实）**：玩具数据太易、信号太弱，**无法判定方法是否有效**——这恰恰说明
（a）退化损害是真的、值得做；（b）必须在**真实退化 VQA + 多 seed CI** 上验证，而非玩具/单 seed。
**但骨架是真的**：方法已能插进真 VLM、能消融、能出可信区间；把 `render` 换成真实低光图、
`TinyViT` 换成 SigLIP/CLIP，§4 流程照搬即用。

## 6. 风险与局限（诚实）

- **方法可能无效**：玩具已提示 degaware ≈ blind。准备好"负结果也是结果"——
  退路：若"选哪些 token"无增益，转向"退化区域 token 的修复/增强后再压缩"。
- **冗余风险**：退化信号可能与 attention 重要性高度相关，边际增益有限 → 需做相关性分析。
- **数据**：真实低光 VQA 标注稀缺；合成退化与真实退化存在 gap（需混合验证）。
- **算力**：4×4090 跑 LLaVA-1.5 **推理 + training-free 消融**充裕；轻量可学习只训 token scorer/projector，可行。

## 7. 里程碑

| 阶段 | 目标 | 周期 |
|---|---|---|
| M1 | 真 LLaVA-1.5 上复现 FastV，接 degaware `compress` 钩子，对齐手写 merge 逻辑 | 2 周 |
| M2 | 退化 VQA 评测集（R-Bench [8] 式 corruption + 真实低光），固定协议 | 2 周 |
| M3 | 多 seed 消融（full/random/FastV/ToMe/PruMerge/degaware），得 H1–H3 结论 | 2–3 周 |
| M4 | H2 成立 → 扩展 + 投稿（CVPR / ICCV / ECCV，效率+鲁棒性双卖点）；不成立 → 转向或诚实负结果报告 | 持续 |

## 8. 预期贡献

1. 指出并量化"退化改变视觉 token 重要性"这一被现有压缩方法忽视的问题；
2. 一个即插即用、可消融的退化感知 token 压缩模块；
3. 一套**退化 × token 预算 × 多 seed CI** 的严格评测协议与可复现骨架（本仓库）。

## 参考文献（均经 arXiv 核实）

1. FastV — *An Image is Worth 1/2 Tokens After Layer 2*. arXiv:2403.06764
2. LLaVA — *Visual Instruction Tuning*. arXiv:2304.08485
3. LLaVA-1.5 — *Improved Baselines with Visual Instruction Tuning*. arXiv:2310.03744
4. ToMe — *Token Merging: Your ViT But Faster*. arXiv:2210.09461
5. PruMerge — *LLaVA-PruMerge: Adaptive Token Reduction for Efficient LMMs*. arXiv:2403.15388
6. CLIP — *Learning Transferable Visual Models From Natural Language Supervision*. arXiv:2103.00020
7. Zero-DCE — *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*. arXiv:2001.06826
8. R-Bench — *Are your Large Multimodal Model Robust to Real-world Corruptions?* arXiv:2410.05474
9. *Benchmarking Corruption Robustness of LVLMs*. arXiv:2511.19032
10. *Troubling Trends in Machine Learning Scholarship*（评测/表述纪律）. arXiv:1807.03341
11. SigLIP — *Sigmoid Loss for Language Image Pre-Training*. arXiv:2303.15343
12. *Deep Reinforcement Learning that Matters*（多 seed / 报 CI 的纪律来源）. arXiv:1709.06560
