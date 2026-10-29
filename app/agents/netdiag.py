"""网络故障诊断 Agent：给一条自然语言故障描述，产出一份带证据的结论。

这是整套项目里「提示词工程」分量最重的一处，原因很现实：
第一版 system prompt 写的是「请分层排查：链路 → DNS → 端口 → 应用层 → 路由 → 本机」，
本地 8B 模型的表现是 —— 调一次 ping，看到「可达」，直接回一句「网络正常，建议进一步排查」。
抽象规则对它几乎无效。

改成「具体动作序列 + 反例 + 正例」之后，它才开始一次并行调四个工具。
结论写进提示词里的不是「要严谨」，而是「按这个顺序调这些工具」——
对 7B~8B 级模型，**可执行的步骤比正确的原则更管用**。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .. import config
from ..llm import get_llm
from ..tools import NET_TOOLS
from ..tracing import AgentRun, build_agent, run_agent

SYSTEM_PROMPT = (
    "你是一名网络运维工程师，根据用户的故障描述做只读排查并给出结论。\n\n"
    "【动作序列】默认按下面顺序取证，一次把能查的都查掉：\n"
    "1. local_network_info —— 本机有没有 IP、有没有网关（排除「本机根本没联网」）；\n"
    "2. ping_host —— 链路通不通、延迟多少；\n"
    "3. dns_lookup —— 域名解析到哪些 IP，还是解析失败；\n"
    "4. check_port(443 或 80) —— 端口有没有进程监听；\n"
    "5. check_http —— 应用层返回什么状态码；\n"
    "6. traceroute —— 只有前面出现「超时/绕路」时才需要，用来定位是哪一跳出问题。\n\n"
    "【硬性要求】\n"
    "- 至少完成 ping_host、dns_lookup、check_port、check_http 四项中的三项，不许只查一项就下结论；\n"
    "- 结论必须引用具体数字：延迟、丢包率、解析到的 IP、建连耗时、HTTP 状态码；\n"
    "- 工具返回错误时，按错误指向的方向继续查，不要停下来说「无法判断」；\n"
    "- 只读诊断，不得建议执行修改系统配置的命令来「试试看」。\n\n"
    "【输出格式】三段，中文：\n"
    "现象：用户看到的问题，一句话；\n"
    "证据：逐条列工具拿到的关键数据；\n"
    "结论与建议：故障定位在哪一层，下一步该查什么。\n\n"
    "【反例，不要这样】\n"
    "只调一次 ping_host，看到可达，就输出「网络连接正常，建议您进一步排查浏览器设置」然后结束。\n\n"
    "【正例，要这样】\n"
    "依次调用 ping_host → dns_lookup → check_port → check_http，拿到四份数据后，"
    "再一次性输出「现象/证据/结论」；若 ping 通但 HTTP 返回 404，就明确写成"
    "「网络层与应用层端口均正常，问题在应用层路由或资源路径」。"
)

DEFAULT_ISSUES = (
    "我访问 https://www.baidu.com 很慢而且偶尔打不开，帮我诊断。",
    "本机 127.0.0.1:8080 上的服务好像没起来，帮我确认一下。",
)


@dataclass
class DiagnosisReport:
    issue: str
    run: AgentRun

    def to_markdown(self, index: int | None = None) -> str:
        title = "网络故障诊断报告" if index is None else f"网络故障诊断报告（第 {index} 条）"
        return self.run.to_markdown(title)


_agent = None


def build():
    global _agent
    if _agent is None:
        _agent = build_agent(get_llm(temperature=0.0), NET_TOOLS, SYSTEM_PROMPT)
    return _agent


def diagnose(issue: str, trace: bool = True) -> DiagnosisReport:
    if config.get_bool("NET_DIAG_MOCK", False):
        print("（模拟模式：NET_DIAG_MOCK=1，工具返回带 [模拟] 前缀的假数据，仅用于演示调度）")
    return DiagnosisReport(issue, run_agent(build(), issue, trace=trace))


def diagnose_many(issues: list[str], trace: bool = True) -> list[DiagnosisReport]:
    return [diagnose(issue, trace=trace) for issue in issues]


def save(reports: list[DiagnosisReport]) -> str:
    """多条合并成一份报告；单条也走同一路径，保证文件名与结构一致。"""
    body = "\n\n---\n\n".join(
        r.to_markdown(None if len(reports) == 1 else idx) for idx, r in enumerate(reports, 1)
    )
    header = (
        f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}　"
        f"工具集：{len(NET_TOOLS)} 个真实探测　"
        f"模型反馈的每条结论均可在「排查轨迹」中回溯到具体工具输出。\n\n"
    )
    path = config.save_report("network-diagnosis", header + body)
    print(f"[report] 已保存 → {path}")
    return str(path)
