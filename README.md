# CV → 多模态大模型 · 学习路线

一份精细到每天的交互式学习计划 —— 从深度学习基础开始，经由 LLM、CLIP 到多模态大模型（MLLM）与视觉 Agent。

> 前置 + 8 周系统学习路线，从零基础到 MLLM + Agent。

## 🌐 在线查看（渲染版）

> ⚠️ 在 GitHub 上直接点 `.html` 文件，只会看到**源码** —— 出于安全，GitHub 不会把仓库里的 HTML 当网页渲染。请用下面任一方式看到真正的页面。

### 方式一：GitHub Pages（推荐，体验最佳）

开启后访问 👉 **https://ji-xuanhe.github.io/CV-Multimodal-LLM/cv-to-mllm-roadmap_2.html**

开启步骤（一次性，约 1 分钟）：

1. 仓库页面 → **Settings** → 左侧 **Pages**
2. **Source** 选 `Deploy from a branch`
3. **Branch** 选 `main` / `(root)` → **Save**
4. 等 1–2 分钟构建完成，即可访问上面的链接（之后每次 push 自动更新）

> Pages 是 https 环境，还能顺带解决本地 `file://` 下 YouTube 无法内嵌播放（error 153）的问题。

### 方式二：htmlpreview（零配置，即点即看）

[▶ 立即在线预览](https://htmlpreview.github.io/?https://github.com/Ji-XuanHe/CV-Multimodal-LLM/blob/main/cv-to-mllm-roadmap_2.html)

通过第三方代理实时渲染，无需任何设置；偶有兼容性差异，长期使用建议用 Pages。

### 方式三：本地打开

下载后用浏览器直接打开，或克隆后在目录里起一个本地服务（推荐，能让侧窗/视频正常工作）：

```bash
# 直接打开（macOS / Linux / Windows）
open cv-to-mllm-roadmap_2.html   # macOS
xdg-open cv-to-mllm-roadmap_2.html   # Linux
start cv-to-mllm-roadmap_2.html   # Windows

# 或起本地服务（https 同源，体验更完整）
python3 -m http.server
# 然后浏览器访问 http://localhost:8000/cv-to-mllm-roadmap_2.html
```

## ✨ 特性

- **共 9 周 · 45 天**（含前置周），精细到每天的学习安排
- **15+ 处数学推导**，标注「需要手推」的关键公式
- **10 个动手实践**，所有代码可直接运行
- **进度跟踪**：点击圆圈标记完成，顶部 / 侧边栏进度条实时更新（保存在浏览器本地）
- **报刊式排版**：衬线字体 + 纸张质感，建议配合纸笔阅读 ✎

## 🧭 阅读器导航

为避免「一根超长滚动条」的阅读负担，页面内置了一套阅读器导航：

- **左侧目录**：9 周 → 每天的树状目录，点击任意条目直达；当前阅读位置自动高亮（scroll-spy）
- **一次聚焦一周**：默认只显示当前一周，把 45 天的长滚动拆成 9 个短页
- **顶栏控制**：`‹ 上一周 / 下一周 ›` 切换、`展开 / 折叠本周` 一键开合所有卡片
- **快捷键**：`←` `→` 翻周，`j` `k` 在「天」之间跳转，`Esc` 关闭目录
- **回到顶部**：右下角浮动按钮
- **进度记忆**：自动记住上次看到的周；目录链接（`#day-1` 等）可直接定位到某天
- **移动端**：目录收起为抽屉，点 `☰ 目录` 唤出

> 这些都只改变「怎么看」，不改动任何学习内容。

## 🔗 论文链接 + 并排侧窗

每天的「阅读材料」链接、以及精选补充的**顶级公司技术报告 / 顶会论文**，都可以一键在**右侧并排侧窗**打开，边看路线边读原文：

- **venue 标签**：每个链接自动标注来源（`arXiv` / `NeurIPS` / `CVPR` / `OpenAI` / `Qwen` / `DeepSeek` …），一眼看清分量
- **arXiv 官方 PDF 原版**：arXiv 论文在侧窗内直接以官方 PDF 呈现（公式排版完美），可滚动阅读、左右对照
- **PDF / 卡片 双视图**：侧窗顶部 `☰ 卡片` 可切换 —— 卡片视图汇总该论文的多种阅读入口（PDF 原版、ar5iv / alphaXiv 的 HTML 自适应版、arXiv 摘要、Connected Papers 关系图谱），窄屏想 reflow 阅读点一下即可
- **优雅回退**：对禁止嵌入的站点（X-Frame-Options），侧窗显示信息卡片 + 「↗ 在新标签打开」
- **可拖拽调宽**：拖侧窗左缘自由调节宽度；`Esc` 或 `✕` 关闭
- **延伸资源**：在对应天追加了一批必读，如 GPT-3 / GPT-4 / Llama 3 / Qwen-VL / DeepSeek-V3·R1 / Gemini 1.5 / BLIP-2 / Flamingo / SAM / FlashAttention 等

> 想用浏览器原生「新标签」？按住 `⌘/Ctrl` 再点链接即可，侧窗逻辑会自动让行。

## 🗺️ 路线总览

| 阶段 | 主题 |
|------|------|
| 前置 | 深度学习与视觉 Transformer |
| Week 1 | Transformer 与语言模型基础 |
| Week 2 | LLM 训练范式与高效微调 |
| Week 3 | 视觉-语言对齐基础：CLIP 与视觉编码器 |
| Week 4 | 前沿多模态大模型深入 |
| Week 5 | 视觉 Grounding 与细粒度理解 |
| Week 6 | Agent 基础与工具调用 |
| Week 7 | 前沿主题与研究方向探索 |
| Week 8 | 实战：启动你的第一个多模态项目 |

## 📐 使用建议

- **推导标注「需要手推」**：建议拿出纸笔亲手推导，不要只看不写。
- **重点标注「必须掌握」**：这些是后续内容的基石，务必吃透。
- **代码可直接运行**：边学边跑，遇到报错就是最好的学习机会。

## 📁 仓库结构

```
CV-Multimodal-LLM/
├── cv-to-mllm-roadmap_2.html   # 交互式学习路线（主文件）
└── README.md
```

---

*进度数据保存在浏览器本地，不会上传到任何服务器。换设备或清除缓存后需要重新标记。*
