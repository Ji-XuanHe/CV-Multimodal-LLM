"""nano-mllm · D26/D30 ReAct：Thought -> Action -> Observation 循环
对应路线图 D26「Agent 框架」、D30「视觉 Agent Demo」。
Agent 的全部秘密：把「历史 + 工具结果」喂回去，让模型自己接着想。
运行: python nano-mllm/agent/react.py
依赖: 无（纯标准库）
"""

# 注册工具（真实场景就是把你的 CV 模型/API 包成函数）
TOOLS = {
    "measure_quality": lambda img: "PSNR=18.3, 低光退化严重",
    "enhance_image": lambda img: "已增强 -> enhanced.jpg",
    "detect_objects": lambda img: "检测到: 1辆车, 2个人",
}


def scripted_brain(scratch):
    """真实场景这里是 MLLM；这里用规则模拟「看历史 -> 决定下一步」。"""
    n = scratch.count("Observation:")
    plan = [
        {"thought": "先评估图像质量", "action": "measure_quality"},
        {"thought": "太暗，先增强", "action": "enhance_image"},
        {"thought": "再检测物体", "action": "detect_objects"},
    ]
    if n < len(plan):
        return plan[n]
    return {"answer": "图中有 1 车 2 人（先增强了低光再识别）"}


def react(brain, question, max_steps=6):
    scratch, trace = question, []
    for _ in range(max_steps):
        step = brain(scratch)
        if "answer" in step:
            return step["answer"], trace
        act = step["action"]
        obs = TOOLS[act]("dark.jpg")               # 执行工具 -> Observation
        trace.append(act)
        scratch += f"\nThought: {step['thought']}\nAction: {act}\nObservation: {obs}"
    return "达到最大步数", trace


if __name__ == "__main__":
    ans, trace = react(scripted_brain, "任务：这张暗图里有什么？")
    print("决策轨迹:", " -> ".join(trace))
    print("最终回答:", ans)
