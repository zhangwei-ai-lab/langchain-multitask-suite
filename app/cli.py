"""命令行入口：一条命令对应一个能力，所有能力共用同一套配置与模型工厂。

`python main.py doctor` 是第一条该跑的命令 —— 它把「环境到底哪里不通」一次性说清楚，
避免用户在全量运行时才看到一串 traceback。
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

from . import config
from .llm import LLMUnavailable, describe, get_llm, probe

ROOT = Path(__file__).resolve().parent.parent


def _banner(text: str) -> None:
    print(f"\n{'=' * 66}\n{text}\n{'=' * 66}")


def _display_width(text: str) -> int:
    """中文占两列，按列宽算才能真正对齐（否则 doctor 的输出会参差不齐）。"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def _llm():
    """统一入口：拿不到后端时给出可读提示并退出，而不是抛一屏 traceback。"""
    try:
        llm = get_llm()
    except LLMUnavailable as exc:
        print(exc)
        raise SystemExit(2)
    print(describe(llm))
    return llm


def _print_run(run) -> None:
    print("\n--- 最终回答 ---")
    print(run.answer.strip() or "（模型没有返回文本，检查工具是否可用、模型是否支持工具调用）")
    print(f"\n[统计] 工具调用 {len(run.tool_calls)} 次，耗时 {run.seconds:.1f}s")


# ---------------------------------------------------------------- doctor
def cmd_doctor(_args) -> int:
    _banner("环境体检")
    info = probe()
    checks: list[tuple[str, bool, str]] = []

    checks.append(("Python", sys.version_info >= (3, 10), platform.python_version()))
    checks.append(("配置文件", config.ENV_FILE.exists(), str(config.ENV_FILE)))
    checks.append(("输出目录可写", True, str(config.OUTPUT_DIR)))
    checks.append(
        (
            f"Ollama 服务（{info['ollama_url']}）",
            info["ollama_reachable"],
            f"已装 {len(info['local_models'])} 个模型" if info["ollama_reachable"] else "未响应",
        )
    )
    checks.append(
        (
            "对话模型可选",
            bool(info["picked"]),
            info["picked"] or "没有可用对话模型（向量模型会被自动排除）",
        )
    )
    checks.append(("DashScope 密钥", info["dashscope_key"], "已配置" if info["dashscope_key"] else "未配置（非必需）"))

    # 依赖与工具连通性
    try:
        import langchain  # noqa: F401
        from langchain.agents import create_agent  # noqa: F401
        from langchain_core.runnables import RunnableParallel  # noqa: F401

        checks.append(("LangChain 依赖", True, "已安装"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("LangChain 依赖", False, f"{type(exc).__name__}: {exc}"))

    try:
        from .tools.search import search

        started = time.perf_counter()
        hits = search("OpenAI", top_k=1)
        checks.append(
            ("搜索渠道", bool(hits), f"{len(hits)} 条，{(time.perf_counter() - started) * 1000:.0f}ms")
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(("搜索渠道", False, f"{type(exc).__name__}: {exc}"))

    try:
        from .tools.weather import current_weather

        started = time.perf_counter()
        w = current_weather("北京")
        checks.append(
            (
                "天气接口",
                True,
                f"{w['city']} {w['temperature_c']}°C，{(time.perf_counter() - started) * 1000:.0f}ms",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(("天气接口", False, f"{type(exc).__name__}: {exc}"))

    try:
        from .tools.netdiag import run_command

        code, _out = run_command(["ping", "-n" if platform.system() == "Windows" else "-c", "1", "127.0.0.1"], 10)
        checks.append(("ping 命令", code == 0, "可用" if code == 0 else f"退出码 {code}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("ping 命令", False, f"{type(exc).__name__}: {exc}"))

    width = max(_display_width(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  {'✓' if ok else '✗'} {_pad(name, width)}  {detail}")

    print(f"\n当前后端：{info['backend']}")
    hard_fail = [name for name, ok, _ in checks if not ok and name in {"对话模型可选", "LangChain 依赖"}]
    if hard_fail:
        print(f"\n阻塞项：{', '.join(hard_fail)} —— 修好之后才能跑调用模型的命令。")
        print("提示：`python -m pytest -q` 与 `python main.py tools` 不依赖模型，可以先跑。")
        return 1
    print("\n环境可用。建议顺序：python main.py basic → python main.py netdiag")
    return 0


# ---------------------------------------------------------------- 各能力
def cmd_tools(_args) -> int:
    from .tools import catalog

    _banner(f"已注册工具（{len(catalog())} 个）")
    for name, desc in catalog():
        print(f"  {name:22s} {desc}")
    return 0


def cmd_basic(_args) -> int:
    from .chains.basic import build_chain, run_batch, run_single

    llm = _llm()
    chain = build_chain(llm)
    for product in ("colorful socks", "广告设计"):
        _banner(f"单条调用：{product}")
        print(run_single(chain, product))

    _banner("批量调用（batch 并行）")
    products = ["智能水杯", "宠物殡葬服务", "乡村民宿"]
    started = time.perf_counter()
    for product, text in zip(products, run_batch(chain, products)):
        print(f"  {product}: {text.splitlines()[0][:70]}")
    print(f"\n[统计] {len(products)} 条并发耗时 {time.perf_counter() - started:.1f}s")
    return 0


def cmd_research(args) -> int:
    from .agents import research

    _llm()
    _print_run(research.run(args.question or research.DEFAULT_QUESTION, trace=args.trace))
    return 0


def cmd_analyst(args) -> int:
    from .agents import analyst

    _llm()
    _print_run(analyst.run(args.question or analyst.DEFAULT_QUESTION, trace=args.trace))
    return 0


def cmd_memory(args) -> int:
    from .memory import MemoryChat

    llm = _llm()
    chat = MemoryChat(llm, max_messages=args.trim)
    _banner("会话 A：三轮追问（后两轮必须依赖第一轮）")
    turns = [
        "我叫张伟，正在做一个 LangChain 多任务应用项目。",
        "我刚才提到的「多任务应用」指的是什么？一句话回答。",
        "我叫什么名字？在做什么项目？",
    ]
    for idx, text in enumerate(turns, 1):
        reply = chat.ask("session-A", text)
        print(f"\n[第{idx}轮] 我：{text}")
        print(f"        助手：{reply}")

    _banner("会话 B：同进程、不同 session_id")
    print("        助手：" + chat.ask("session-B", "你知道我叫什么名字吗？不知道就直说。"))

    _banner("上下文用量")
    for sid in ("session-A", "session-B"):
        s = chat.stats(sid)
        print(f"  {sid}: {s.messages} 条消息 / {s.chars} 字")

    if args.compare_api:
        from .memory import build_history_runnable

        _banner("对照：RunnableWithMessageHistory 等价实现")
        runnable, store = build_history_runnable(llm)
        cfg = {"configurable": {"session_id": "session-C"}}
        reply = runnable.invoke({"input": "只回答两个字：收到"}, config=cfg)
        print(f"  回答：{(reply.content or '').strip()}")
        print(f"  该会话消息数：{len(store['session-C'].messages)}")
    return 0


def cmd_product(args) -> int:
    from .agents.product import TOOLS, ProductAdvisor

    _llm()
    advisor = ProductAdvisor(memory=config.get_bool("MEMORY", False))

    if args.question:
        _print_run(advisor.ask(args.question, trace=args.trace))
        return 0

    print("输入问题开始；命令：/tools 看工具，/reset 清记忆，/exit 退出")
    while True:
        try:
            text = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0
        if not text:
            continue
        if text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("已退出。")
            return 0
        if text.lower() == "/reset":
            advisor.reset()
            print("记忆已清空。")
            continue
        if text.lower() == "/tools":
            for tool in TOOLS:
                print(f"  - {tool.name}: {(tool.description or '').splitlines()[0]}")
            continue
        _print_run(advisor.ask(text, trace=args.trace))


def cmd_toolkit(args) -> int:
    from .agents import toolkit

    _llm()
    task = args.task or toolkit.SAMPLES[args.sample]
    _print_run(toolkit.run(task, trace=args.trace))
    return 0


def cmd_pipeline(_args) -> int:
    from .chains.orchestrate import SAMPLE_TEXT, run_all

    llm = _llm()
    _banner("同一段文本，四种编排")
    result = run_all(llm, SAMPLE_TEXT)
    print(f"\n[单步工具]      {result['single']}")
    print(f"[串行 + assign] {result['assign']['analysis']} / {result['assign']['lines']}")
    print(f"[并行]          {result['parallel']['analysis']} / {result['parallel']['lines']}")
    print(f"[并行 → 模型]   {result['advice']}")

    print("\n编排语义速查：")
    print("  A | B                       串行，B 的入参 = A 的输出")
    print("  RunnableParallel           各分支吃同一份入参，并行执行")
    print("  RunnablePassthrough.assign 保留原始入参，同时注入新字段")
    print("  {'k': chain}               用 chain 的结果填充字典里的 k")
    return 0


def cmd_multistep(args) -> int:
    from .chains.multistep import DEFAULT_QUESTION, build_workflow, run_stepwise, run_stream

    llm = _llm()
    question = args.question or DEFAULT_QUESTION
    _banner(f"逐阶段执行：{question}")
    stages, total = run_stepwise(llm, question)
    for stage in stages:
        print(f"\n[阶段] {stage.name}（{stage.seconds:.1f}s）")
        print("  " + stage.text.replace("\n", " ")[:320])
    print(f"\n[统计] 三段串行总耗时 {total:.1f}s —— 链路每多一段，延迟就多一次完整推理")

    _banner("整体流式执行")
    workflow = build_workflow(llm)
    _text, first, total_stream = run_stream(workflow, question)
    print(f"[统计] 首字延迟 {first:.1f}s，总耗时 {total_stream:.1f}s")
    return 0


def cmd_netdiag(args) -> int:
    from .agents import netdiag

    if args.mock:
        # 工具层每次调用时现读该环境变量，所以这里设置即生效
        import os

        os.environ["NET_DIAG_MOCK"] = "1"

    _llm()
    issues = args.issue or list(netdiag.DEFAULT_ISSUES)
    reports = netdiag.diagnose_many(issues, trace=args.trace)
    for idx, report in enumerate(reports, 1):
        _banner(f"诊断结论 {idx}")
        print(report.run.answer.strip() or "（模型没有返回文本）")
    if not args.no_save:
        netdiag.save(reports)
    return 0


def cmd_verify(args) -> int:
    """转调验收脚本。

    这里不用 `argparse.REMAINDER` 透传：以 `--` 开头的参数在子命令后面会被
    顶层解析器抢走（argparse 的已知行为），逐个声明再显式拼装更可靠。
    """
    import runpy

    forwarded: list[str] = []
    if args.smoke:
        forwarded.append("--smoke")
    if args.list:
        forwarded.append("--list")
    if args.only:
        forwarded += ["--only", args.only]
    if args.timeout:
        forwarded += ["--timeout", str(args.timeout)]

    sys.argv = [str(ROOT / "scripts" / "run_all.py")] + forwarded
    runpy.run_path(str(ROOT / "scripts" / "run_all.py"), run_name="__main__")
    return 0


# ---------------------------------------------------------------- 参数
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="LangChain 多任务应用工程 —— 命令行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py doctor\n"
            "  python main.py basic\n"
            "  python main.py netdiag \"我打不开 https://github.com\"\n"
            "  python main.py verify\n"
        ),
    )
    parser.add_argument("--no-trace", dest="trace", action="store_false", help="不打印工具调用轨迹")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="环境体检").set_defaults(func=cmd_doctor)
    sub.add_parser("tools", help="列出已注册工具").set_defaults(func=cmd_tools)
    sub.add_parser("basic", help="最小链 + 批量调用").set_defaults(func=cmd_basic)
    sub.add_parser("pipeline", help="LCEL 串行/并行/assign 编排").set_defaults(func=cmd_pipeline)

    p = sub.add_parser("research", help="检索型 Agent（搜索 + 时间）")
    p.add_argument("question", nargs="?", help="问题，留空用内置示例")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("analyst", help="换算型 Agent（天气 + 计算器）")
    p.add_argument("question", nargs="?", help="问题，留空用内置示例")
    p.set_defaults(func=cmd_analyst)

    p = sub.add_parser("memory", help="多轮记忆 + 会话隔离")
    p.add_argument("--trim", type=int, default=None, help="只保留最近 N 条历史进上下文")
    p.add_argument("--compare-api", action="store_true", help="额外跑一遍框架等价实现做对照")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("product", help="产品顾问（无 -q 时进入交互模式）")
    p.add_argument("-q", "--question", help="只问一句后退出")
    p.set_defaults(func=cmd_product)

    p = sub.add_parser("toolkit", help="工具调用型 Agent（5 个文本/数据工具）")
    p.add_argument("task", nargs="?", help="任务描述，留空用内置示例")
    p.add_argument("--sample", choices=["analysis", "convert"], default="analysis", help="内置示例")
    p.set_defaults(func=cmd_toolkit)

    p = sub.add_parser("multistep", help="多段流水线（翻译→分析→回译）")
    p.add_argument("question", nargs="?", help="问题，留空用内置示例")
    p.set_defaults(func=cmd_multistep)

    p = sub.add_parser("netdiag", help="网络故障诊断 Agent")
    p.add_argument("issue", nargs="*", help="故障描述，可给多条")
    p.add_argument("--mock", action="store_true", help="用模拟数据演示调度（当前进程内生效）")
    p.add_argument("--no-save", action="store_true", help="不写报告文件")
    p.set_defaults(func=cmd_netdiag)

    p = sub.add_parser("verify", help="全量验收（等价于 scripts/run_all.py）")
    p.add_argument("--smoke", action="store_true", help="只做静态自检，不调模型")
    p.add_argument("--list", action="store_true", help="列出所有用例")
    p.add_argument("--only", default="", help="只跑名字含该关键字的用例")
    p.add_argument("--timeout", type=int, default=0, help="单个用例超时秒数")
    p.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0
