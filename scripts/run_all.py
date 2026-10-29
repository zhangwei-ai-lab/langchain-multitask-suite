"""全量验收：把 8 个能力依次跑在独立子进程里，产出可交付的验收报告。

为什么要子进程 + 报告
----------------------
* 每个能力都持有模型连接和进程内缓存，同一进程里连跑会互相污染状态；
* 一条用例抛异常不该影响其余用例 —— 验收要回答「哪几条挂了」，而不是「第一条挂了」；
* 报告落盘后，结果可以进版本库对比（这次 8/8、上次 7/8，差异一眼可见）。

用法：
    python scripts/run_all.py             # 全量（调模型，约 2~4 分钟）
    python scripts/run_all.py --smoke     # 只做静态自检（不调模型，秒级）
    python scripts/run_all.py --only 诊断  # 只跑名字含关键字的用例
    python scripts/run_all.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class Case:
    name: str
    args: list[str]
    note: str
    timeout: int = 600


CASES = [
    Case("最小链", ["basic"], "PromptTemplate | ChatModel | Parser，含 batch 并发", 300),
    Case("检索型Agent", ["research", "今天是几月几号？用一句话说明今天有什么值得关注的科技新闻"], "搜索 + 时间工具，验证取证与防编造", 300),
    Case("换算型Agent", ["analyst"], "天气取数 + 计算器换算，验证「不许心算」", 300),
    Case("多轮记忆", ["memory"], "会话隔离 + 上下文增长", 300),
    Case("产品顾问", ["product", "-q", "Model Y 多少钱？它和 Model 3 有什么区别"], "双工具 Agent，结构化查询 + 资料问答", 300),
    Case("工具调用Agent", ["toolkit"], "5 个文本/数据工具，验证工具选择", 300),
    Case("LCEL编排", ["pipeline"], "串行/并行/assign + 模型汇总", 300),
    Case("多段流水线", ["multistep"], "翻译→分析→回译，验证形状对齐", 300),
    Case("网络诊断Agent", ["netdiag", "我访问 https://www.baidu.com 很慢，帮我诊断", "--no-save"], "6 个真实探测工具，验证分层取证", 420),
]

SMOKE_IMPORTS = [
    ("配置层", "from app import config; config.get('LLM_PROVIDER')"),
    ("模型工厂", "from app.llm import get_llm, resolve_backend, pick_local_model"),
    ("工具注册表", "from app.tools import REGISTRY, NET_TOOLS, TEXT_TOOLS; len(REGISTRY)"),
    ("链", "from app.chains import basic, multistep, orchestrate"),
    ("Agent", "from app.agents import research, analyst, toolkit, product, netdiag"),
    ("轨迹", "from app.tracing import run_agent, AgentRun"),
    ("CLI", "from app.cli import build_parser; build_parser()"),
    ("create_agent", "from langchain.agents import create_agent"),
    ("LCEL", "from langchain_core.runnables import RunnableParallel, RunnablePassthrough"),
]


@dataclass
class Result:
    name: str
    ok: bool
    seconds: float
    tail: str = ""
    error: str = ""


def _child_env() -> dict:
    """子进程统一用 UTF-8 输出。

    Windows 默认控制台编码是 GBK，脚本里的 `→`、`✓` 这类符号不在 GBK 字符集里，
    直接 print 会抛 UnicodeEncodeError —— 表现为「单独跑没问题、放进验收就失败」。
    """
    import os

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(cmd: list[str], timeout: int) -> Result:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_child_env(),
        )
    except subprocess.TimeoutExpired:
        return Result(" ".join(cmd[-2:]), False, time.perf_counter() - started, error=f"超时（>{timeout}s）")

    output = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0 and "Traceback" not in output
    lines = [line for line in output.strip().splitlines() if line.strip()]
    return Result(
        " ".join(cmd[2:]),
        ok,
        time.perf_counter() - started,
        tail="\n".join(lines[-10:]),
        error="" if ok else output[-1200:],
    )


def smoke() -> list[Result]:
    print("=" * 70)
    print("[1/2] 模块导入自检")
    print("=" * 70)
    results: list[Result] = []
    for name, code in SMOKE_IMPORTS:
        started = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        ok = proc.returncode == 0
        print(f"  {'✓' if ok else '✗'} {name}")
        if not ok:
            print("      " + (proc.stderr or "").strip().splitlines()[-1][:150])
        results.append(Result(name, ok, time.perf_counter() - started, error=proc.stderr[-500:]))

    print("\n[2/2] 语法编译检查")
    targets = [ROOT / "main.py", *sorted((ROOT / "app").rglob("*.py")), *sorted((ROOT / "tests").rglob("*.py"))]
    for path in targets:
        started = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        ok = proc.returncode == 0
        rel = path.relative_to(ROOT).as_posix()
        if not ok:
            print(f"  ✗ {rel}\n      {(proc.stderr or '').strip().splitlines()[-1][:150]}")
        results.append(Result(rel, ok, time.perf_counter() - started, error=proc.stderr[-500:]))
    print(f"  ✓ 编译检查完成（{len(targets)} 个文件，仅列出失败项）")
    return results


def write_report(results: list[Result], mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"verify-{stamp}.md"
    passed = sum(1 for r in results if r.ok)

    lines = [
        f"# 验收报告（{mode}）",
        "",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 结果：{passed}/{len(results)} 通过",
        "",
        "| 用例 | 结果 | 耗时 |",
        "| --- | --- | --- |",
    ]
    for r in results:
        lines.append(f"| {r.name} | {'✅ 通过' if r.ok else '❌ 失败'} | {r.seconds:.1f}s |")

    for r in results:
        if not r.ok:
            lines += ["", f"## 失败详情：{r.name}", "", "```", (r.error or r.tail)[-1500:], "```"]

    lines += ["", "## 输出摘录", ""]
    for r in results:
        lines += [f"### {r.name}", "", "```text", r.tail or "(无输出)", "```", ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="LangChain 多任务应用工程 —— 全量验收")
    parser.add_argument("--smoke", action="store_true", help="只做静态自检，不调模型")
    parser.add_argument("--only", default="", help="只跑名字含该关键字的用例")
    parser.add_argument("--list", action="store_true", help="列出所有用例")
    parser.add_argument("--timeout", type=int, default=0, help="统一覆盖单用例超时秒数")
    args = parser.parse_args()

    if args.list:
        for case in CASES:
            print(f"{case.name:12s} python main.py {' '.join(case.args)}   # {case.note}")
        return 0

    if args.smoke:
        results = smoke()
        report = write_report(results, "静态自检")
        passed = sum(1 for r in results if r.ok)
        print(f"\n自检：{passed}/{len(results)} 通过\n报告：{report}")
        return 0 if passed == len(results) else 1

    selected = [c for c in CASES if not args.only or args.only in c.name]
    if not selected:
        print(f"没有匹配「{args.only}」的用例，用 --list 查看")
        return 1

    print("=" * 70)
    print(f"全量验收：{len(selected)} 个用例")
    print("=" * 70)
    results: list[Result] = []
    for idx, case in enumerate(selected, 1):
        print(f"\n[{idx}/{len(selected)}] {case.name} —— {case.note}")
        timeout = args.timeout or case.timeout
        result = _run([sys.executable, "main.py", *case.args], timeout)
        result.name = case.name
        results.append(result)
        print(f"  {'✅ 通过' if result.ok else '❌ 失败'}（{result.seconds:.1f}s）")
        for line in (result.tail or "").splitlines()[-3:]:
            print(f"     {line[:110]}")

    report = write_report(results, "全量运行")
    passed = sum(1 for r in results if r.ok)
    print("\n" + "=" * 70)
    print(f"验收结果：{passed}/{len(results)} 通过\n报告：{report}")
    print("=" * 70)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
