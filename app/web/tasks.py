"""Web 任务注册表：把 CLI 的每个能力包装成「可流式输出的事件生成器」。

为什么要单独一层
----------------
CLI 的能力函数是「跑完一次性返回 `AgentRun`」，界面需要的是「边跑边推事件」。
把转换逻辑集中在注册表里，好处是：

* 界面与终端共用同一套提示词和工具装配（`agents/*.py` 里的 SYSTEM_PROMPT / TOOLS），
  不存在「命令行好用、界面效果不一样」的分裂；
* 新增一个能力 = 往 `TASKS` 里加一行 + 写一个 runner，路由和前端都不用改；
* runner 统一是 `Iterator[dict]`，HTTP 层只负责把 dict 编码成 SSE，不掺业务逻辑。

事件协议（与前端 app.js 一一对应）
----------------------------------
token       {"type":"token","text":"..."}                  模型逐字输出
tool_call   {"type":"tool_call","name":..,"args":..}       模型决定调用工具
tool_result {"type":"tool_result","name":..,"preview":..}  工具返回值（截断）
stage       {"type":"stage","name":..,"text":..}           链式流水线的某一阶段
final       {"type":"final","answer":..,"seconds":..,...}   本轮结束
error       {"type":"error","message":..}                  失败原因（不是 traceback）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from ..llm import get_llm
from ..tracing import StreamEvents, build_agent, stream_agent
from ..tools import CALC_TOOLS, NET_TOOLS, RESEARCH_TOOLS, TEXT_TOOLS, by_names


# ---------------------------------------------------------------- 通用对话
# 只挂 5 个「高频且不易误用」的工具：把 15 个工具全塞给 8B 模型，
# 它会开始胡乱组合（实测出现过「先 traceroute 再算百分比」这种荒谬序列）。
CHAT_TOOL_NAMES = ["web_search", "current_datetime", "calculator", "get_weather", "check_http"]

CHAT_SYSTEM_PROMPT = (
    "你是「多任务助手」，可以用工具查实时信息、做计算。\n"
    "规则：\n"
    "1. 需要事实、实时数据、算式时先调工具，不要凭记忆回答；\n"
    "2. 一次把要查的查完，不要反复调同一个工具；\n"
    "3. 中文回答，简短直接，先说结论再补依据；\n"
    "4. 工具报错就说明取数失败，不要编造结果。"
)


def chat_tools() -> list:
    return by_names(CHAT_TOOL_NAMES)


# 会话记忆：只存纯文本问答对（不存 ToolMessage），换后端也不会因为消息形状报错
_CHAT_HISTORY: dict[str, list[tuple[str, str]]] = {}
CHAT_MAX_TURNS = 8


def chat_history(session_id: str) -> list[tuple[str, str]]:
    return list(_CHAT_HISTORY.get(session_id, []))


def append_chat(session_id: str, question: str, answer: str) -> None:
    if not answer:
        return
    turns = _CHAT_HISTORY.setdefault(session_id, [])
    turns.append((question, answer))
    # 只留最近 N 轮：上下文越长越贵，也越容易让小模型跑题
    del turns[: max(0, len(turns) - CHAT_MAX_TURNS)]


def reset_chat(session_id: str) -> None:
    _CHAT_HISTORY.pop(session_id, None)


def run_chat(message: str, session_id: str = "default") -> Iterator[dict]:
    """通用对话：带工具 + 带记忆 + 逐字流式。"""
    agent = build_agent(get_llm(temperature=0.2), chat_tools(), CHAT_SYSTEM_PROMPT)
    history: list[tuple[str, str]] = []
    for question, answer in chat_history(session_id):
        history += [("user", question), ("assistant", answer)]
    answer_parts: list[str] = []
    for event in stream_agent(agent, message, history=history):
        if event["type"] == StreamEvents.TOKEN:
            answer_parts.append(event["text"])
        elif event["type"] == StreamEvents.FINAL:
            append_chat(session_id, message, event["answer"])
        yield event


# ---------------------------------------------------------------- 各任务
_agent_builders: dict[str, Callable[[], object]] = {}


def _register_agents() -> None:
    """延迟导入 agents 子模块：它们各自会 import 工具与提示词，放在模块顶层会让
    `import app.web.tasks` 顺带把整条 langchain 依赖链拉起来，拖慢冷启动。"""
    from ..agents import analyst, netdiag, research, toolkit

    _agent_builders.update(
        {
            "research": research.build,
            "analyst": analyst.build,
            "netdiag": netdiag.build,
            "toolkit": toolkit.build,
        }
    )


def _run_agent_task(task_id: str, text: str) -> Iterator[dict]:
    if not _agent_builders:
        _register_agents()
    agent = _agent_builders[task_id]()
    final: dict | None = None
    for event in stream_agent(agent, text):
        if event["type"] == StreamEvents.FINAL:
            final = event
        yield event

    # 诊断类任务顺手落盘：界面上「产物」一栏能直接回看，和 CLI 的行为保持一致
    if task_id == "netdiag" and final and final.get("answer"):
        path = save_diagnosis(text, final)
        yield {"type": "artifact", "name": path.name, "path": str(path)}


def save_diagnosis(question: str, final: dict) -> Path:
    from .. import config

    lines = [
        "# 网络诊断报告",
        "",
        f"- 问题：{question}",
        f"- 耗时：{final.get('seconds')}s",
        f"- 工具调用：{final.get('tool_calls')} 次",
        "",
        "## 排查轨迹",
        "",
    ]
    icon = {"tool_call": "↳", "tool_result": "←", "assistant": "✎"}
    for idx, step in enumerate(final.get("steps") or [], 1):
        lines.append(f"{idx}. {icon.get(step['kind'], '·')} **{step['name']}** {step['detail']}")
    lines += ["", "## 最终结论", "", final["answer"].strip(), ""]
    return config.save_report("network-diagnosis", "\n".join(lines))


def _run_pipeline(text: str) -> Iterator[dict]:
    """四种编排渐进输出：每算完一段就推一条 stage，界面能看到进度而不是干等。"""
    from ..chains.orchestrate import (
        SAMPLE_TEXT,
        analysis_branch,
        build_advice_chain,
        build_assign_chain,
        build_parallel_chain,
    )

    text = (text or "").strip() or SAMPLE_TEXT
    started = time.perf_counter()

    t0 = time.perf_counter()
    single = analysis_branch.invoke({"text": text})
    yield {"type": "stage", "name": "① 单步工具", "text": str(single), "seconds": round(time.perf_counter() - t0, 2)}

    t0 = time.perf_counter()
    assign = build_assign_chain().invoke({"text": text})
    yield {
        "type": "stage",
        "name": "② 串行 + assign",
        "text": f"{assign['analysis']} / {assign['lines']}",
        "seconds": round(time.perf_counter() - t0, 2),
    }

    t0 = time.perf_counter()
    parallel = build_parallel_chain().invoke({"text": text})
    yield {
        "type": "stage",
        "name": "③ 并行（RunnableParallel）",
        "text": f"{parallel['analysis']} / {parallel['lines']}",
        "seconds": round(time.perf_counter() - t0, 2),
    }

    t0 = time.perf_counter()
    advice = build_advice_chain(get_llm(temperature=0.2)).invoke({"text": text}).strip()
    yield {
        "type": "stage",
        "name": "④ 并行取数 → 模型给建议",
        "text": advice,
        "seconds": round(time.perf_counter() - t0, 2),
    }

    yield {
        "type": "final",
        "answer": f"改进建议：{advice}",
        "seconds": round(time.perf_counter() - started, 2),
        "tool_calls": 0,
        "steps": [],
    }


def _run_multistep(text: str) -> Iterator[dict]:
    """翻译 → 要点 → 回译：三段串行，逐段输出（每段都是一次完整推理）。"""
    from ..chains.multistep import DEFAULT_QUESTION, build_stages, to_output_dict

    question = (text or "").strip() or DEFAULT_QUESTION
    llm = get_llm(temperature=0.2)
    translate_to_en, analyze_en, translate_to_cn = build_stages(llm)
    started = time.perf_counter()
    previous = ""

    for name, chain, payload in (
        ("① 中文 → 英文", translate_to_en, {"input": question}),
        ("② 英文要点提炼", analyze_en, None),
        ("③ 英文 → 中文", translate_to_cn, None),
    ):
        if payload is None:
            payload = {"text": previous} if "要点" in name else to_output_dict(previous)
        t0 = time.perf_counter()
        previous = chain.invoke(payload).strip()
        yield {
            "type": "stage",
            "name": name,
            "text": previous,
            "seconds": round(time.perf_counter() - t0, 2),
        }

    yield {
        "type": "final",
        "answer": previous,
        "seconds": round(time.perf_counter() - started, 2),
        "tool_calls": 0,
        "steps": [],
    }


@dataclass(frozen=True)
class Task:
    id: str
    name: str
    desc: str
    placeholder: str
    sample: str
    tags: tuple[str, ...] = ()
    agent: bool = True  # True = 走 Agent（有工具轨迹），False = 走链式流水线


TASKS: tuple[Task, ...] = (
    Task(
        id="research",
        name="检索助理",
        desc="先取当天日期，再联网检索，结论逐条标注来源",
        placeholder="例如：今天是几月几号？历史上的今天有哪些重要事件？",
        sample="今天是几月几号？历史上的今天有哪些重要事件？",
        tags=("web_search", "current_datetime"),
    ),
    Task(
        id="analyst",
        name="换算助理",
        desc="实时天气取数，所有数字必须经计算器，不允许心算",
        placeholder="例如：北京现在气温多少华氏度？它的 1/4 是多少？",
        sample="北京现在气温是多少华氏度？这个华氏温度的 1/4 是多少？",
        tags=("get_weather", "calculator"),
    ),
    Task(
        id="netdiag",
        name="网络诊断",
        desc="分层取证：本机 → 链路 → DNS → 端口 → HTTP，产出带证据的结论",
        placeholder="例如：我打不开 https://github.com",
        sample="我打不开 https://github.com",
        tags=("ping", "dns", "port", "http"),
    ),
    Task(
        id="toolkit",
        name="文本工具箱",
        desc="5 个文本/数据工具交给模型自选，统计与转换不靠「感觉」",
        placeholder="贴一段文本，让助手统计字数、情感、行数，或做 CSV ↔ JSON 转换",
        sample=(
            "用 text_analysis 分析这段文本，然后告诉我它的情感倾向和行数："
            "「这个产品非常好用，我很喜欢它的设计，使用体验非常棒！\n"
            "价格也很合理，推荐大家购买。\n"
            "客服态度也很好，解答问题很及时。」"
        ),
        tags=("text_analysis", "data_conversion", "count_lines"),
    ),
    Task(
        id="pipeline",
        name="LCEL 编排",
        desc="同一段文本跑四种编排：单步 / 串行+assign / 并行 / 并行→模型",
        placeholder="留空则使用内置样本文案",
        sample="",
        tags=("RunnableLambda", "RunnableParallel", "assign"),
        agent=False,
    ),
    Task(
        id="multistep",
        name="多段流水线",
        desc="翻译 → 英文要点 → 回译，逐段输出并显示每段耗时",
        placeholder="例如：LangChain 是什么？",
        sample="LangChain 是什么？",
        tags=("LCEL", "stream"),
        agent=False,
    ),
)

TASK_MAP = {task.id: task for task in TASKS}


def task_catalog() -> list[dict]:
    """给前端渲染任务卡用。"""
    return [
        {
            "id": t.id,
            "name": t.name,
            "desc": t.desc,
            "placeholder": t.placeholder,
            "sample": t.sample,
            "tags": list(t.tags),
            "kind": "agent" if t.agent else "chain",
        }
        for t in TASKS
    ]


def run_task(task_id: str, text: str) -> Iterator[dict]:
    """执行一个任务并流式产出事件。未知任务直接抛 KeyError，由 HTTP 层转 400。"""
    if task_id not in TASK_MAP:
        raise KeyError(f"未知任务：{task_id}")
    if task_id == "pipeline":
        yield from _run_pipeline(text)
    elif task_id == "multistep":
        yield from _run_multistep(text)
    else:
        yield from _run_agent_task(task_id, (text or "").strip() or TASK_MAP[task_id].sample)


__all__ = [
    "TASKS",
    "TASK_MAP",
    "task_catalog",
    "run_task",
    "run_chat",
    "chat_history",
    "reset_chat",
    "CHAT_SYSTEM_PROMPT",
]
