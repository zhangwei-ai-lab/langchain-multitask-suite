"""网络诊断工具集（6 个）：ping / DNS / TCP 端口 / HTTP / 路由跟踪 / 本机配置。

设计原则
--------
1. **只读**：全部是探测动作，不改任何系统配置，不发送任何业务数据；
2. **有超时**：目标不可达是常态，任何工具都不能把上层挂死；
3. **解析与执行分离**：命令执行（`run_command`）和输出解析（`parse_ping`）拆成两个函数，
   前者依赖环境、后者可以拿固定文本做单元测试 —— 这是这批工具能进 CI 的关键；
4. **跨平台**：Windows 用 `ping -n/-w`、`tracert`；POSIX 用 `ping -c/-W`、`traceroute`；
   Windows 中文控制台的 GBK 输出按 utf-8 → gbk → latin-1 依次回退解码。

无网络环境下要演示 Agent 调度（而不是演示网络），把 `NET_DIAG_MOCK` 设为 1，
返回的数据会带 `[模拟]` 前缀，避免把假数据当真结论。
"""

from __future__ import annotations

import platform
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request

from langchain_core.tools import tool

from .. import config

IS_WINDOWS = platform.system() == "Windows"


def is_mock() -> bool:
    """每次调用时现读开关，而不是在 import 时定值 —— 这样命令行 `--mock`、单元测试
    的 monkeypatch、脚本里的 os.environ 设置都能即时生效，不需要 reload 模块。"""
    return config.get_bool("NET_DIAG_MOCK", False)


# ---------------------------------------------------------------- 执行层
def decode_bytes(raw: bytes) -> str:
    """Windows 中文控制台输出 GBK，Linux/macOS 是 UTF-8，逐个尝试。"""
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def run_command(cmd: list[str], timeout: int) -> tuple[int, str]:
    """执行外部命令，返回 (returncode, 输出)。超时/命令缺失都转成可读文本，不抛异常。"""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        return proc.returncode, decode_bytes(proc.stdout or b"") + decode_bytes(proc.stderr or b"")
    except subprocess.TimeoutExpired:
        return 124, f"命令超时（>{timeout}s）：{' '.join(cmd)}"
    except FileNotFoundError:
        return 127, f"系统缺少命令：{cmd[0]}"
    except Exception as exc:  # noqa: BLE001
        return 1, f"执行失败：{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- 解析层（纯函数，可单测）
def parse_ping(output: str) -> dict:
    """从 ping 输出里解析丢包率与平均延迟。中英文、Windows/Linux 三种格式都覆盖。"""
    result = {"loss_percent": None, "avg_ms": None, "reachable": False}

    loss_match = re.search(r"(\d+)%\s*(?:loss|丢失|packet loss)", output)
    if loss_match:
        result["loss_percent"] = int(loss_match.group(1))

    # Windows 中文/英文："平均 = 37ms" / "Average = 37ms"
    avg_match = re.search(r"(?:Average|平均)\s*=\s*([\d.]+)\s*ms", output)
    if not avg_match:
        # Linux："rtt min/avg/max/mdev = 36.2/37.1/38.0/0.6 ms"
        avg_match = re.search(r"=\s*[\d.]+/([\d.]+)/[\d.]+/[\d.]+\s*ms", output)
    if not avg_match:
        # 兜底：Windows 单包 "时间=37ms" / "time=37ms"
        avg_match = re.search(r"(?:时间|time)\s*[=<]\s*(\d+)\s*ms", output)
    if avg_match:
        result["avg_ms"] = avg_match.group(1)

    timeout_hints = ("请求超时", "Request timed out", "无法访问目标主机", "Destination host unreachable")
    result["reachable"] = bool(result["avg_ms"]) and result["loss_percent"] != 100
    if not result["reachable"] and not any(h in output for h in timeout_hints):
        # 输出里没有任何超时关键字、也没有延迟：多半是命令本身失败（权限/DNS）
        result["error"] = output.strip()[:300]
    return result


def _mock(tool_name: str, detail: str) -> str:
    return f"[模拟 NET_DIAG_MOCK=1] {tool_name}: {detail}"


# ---------------------------------------------------------------- 工具 1：ping
@tool
def ping_host(target: str, count: int = 4) -> str:
    """检查到目标主机的 ICMP 连通性与往返延迟。

    参数:
        target: 主机名或 IP，例如 www.baidu.com / 8.8.8.8
        count: 发包数量，1~10，默认 4
    返回:
        是否可达、丢包率、平均延迟；不可达时附原始输出片段
    """
    target = (target or "").strip()
    if not target:
        return "请提供要探测的目标主机"
    if is_mock():
        return _mock("ping", f"到 {target} 的延迟 8~35ms，丢包率 0%")

    count = max(1, min(int(count), 10))
    cmd = (
        ["ping", "-n", str(count), "-w", "1500", target]
        if IS_WINDOWS
        else ["ping", "-c", str(count), "-W", "2", target]
    )
    code, output = run_command(cmd, timeout=count * 3 + 10)
    if code == 127:
        return output

    parsed = parse_ping(output)
    if not parsed["reachable"]:
        return (
            f"ping {target}：不可达（丢包率 {parsed['loss_percent']}%，未拿到有效延迟）\n"
            f"---- 原始输出（截断）----\n{output.strip()[-700:]}"
        )
    parts = [f"ping {target}：可达"]
    if parsed["loss_percent"] is not None:
        parts.append(f"丢包率 {parsed['loss_percent']}%")
    parts.append(f"平均延迟 {parsed['avg_ms']} ms")
    return "，".join(parts) + f"\n---- 原始输出（截断）----\n{output.strip()[-500:]}"


# ---------------------------------------------------------------- 工具 2：DNS
@tool
def dns_lookup(hostname: str) -> str:
    """把域名解析成 IP，返回全部 IPv4/IPv6 地址；输入已是 IP 时直接说明。

    参数:
        hostname: 要解析的域名，例如 www.baidu.com
    返回:
        解析结果，或解析失败的原因归类
    """
    hostname = (hostname or "").strip()
    if not hostname:
        return "请提供要解析的域名"
    if is_mock():
        return _mock("dns_lookup", f"{hostname} → 198.18.0.145")

    if re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", hostname):
        return f"{hostname} 本身就是 IPv4 地址，无需 DNS 解析"

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        return (
            f"DNS 解析 {hostname} 失败：{exc.strerror or exc}。\n"
            "常见原因：域名拼错、本机 DNS 配置错误、DNS 服务器不可达、域名被污染或封禁。"
        )
    except Exception as exc:  # noqa: BLE001
        return f"DNS 解析 {hostname} 异常：{type(exc).__name__}: {exc}"

    v4, v6 = [], []
    for family, _type, _proto, _canon, sockaddr in infos:
        ip = sockaddr[0]
        if family == socket.AF_INET and ip not in v4:
            v4.append(ip)
        elif family == socket.AF_INET6 and ip not in v6:
            v6.append(ip)
    return (
        f"DNS 解析 {hostname} 成功：\n"
        f"  IPv4: {', '.join(v4) or '无'}\n"
        f"  IPv6: {', '.join(v6) or '无'}"
    )


# ---------------------------------------------------------------- 工具 3：TCP 端口
@tool
def check_port(host: str, port: int, timeout: float = 3.0) -> str:
    """检测 host:port 的 TCP 端口能否建立连接（只握手，不发业务数据）。

    参数:
        host: 主机名或 IP
        port: 端口号，例如 80 / 443 / 3306
        timeout: 连接超时秒数，默认 3
    返回:
        是否可连接、建连耗时；失败时区分「超时」与「被拒绝」
    """
    if is_mock():
        return _mock("check_port", f"{host}:{port} 可连接，耗时 12ms")

    started = time.perf_counter()
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout)):
            cost = (time.perf_counter() - started) * 1000
        return f"TCP {host}:{port} 可连接，建连耗时 {cost:.0f} ms（端口开放，有进程在监听）"
    except socket.timeout:
        return (
            f"TCP {host}:{port} 连接超时（>{timeout}s）。\n"
            "常见原因：防火墙丢包、云安全组未放行、路由不可达。"
        )
    except ConnectionRefusedError:
        return (
            f"TCP {host}:{port} 连接被拒绝（Connection refused）。\n"
            "说明主机自身可达，但该端口没有进程监听（服务没起或监听了别的端口）。"
        )
    except socket.gaierror as exc:
        return f"TCP 检测失败：主机 {host} 无法解析（{exc.strerror or exc}）"
    except OSError as exc:
        return f"TCP {host}:{port} 检测失败：{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- 工具 4：HTTP
@tool
def check_http(url: str, timeout: float = 10.0) -> str:
    """检测 HTTP/HTTPS 服务是否正常响应，返回状态码、耗时与最终地址。

    参数:
        url: 完整地址，例如 https://www.baidu.com
        timeout: 超时秒数，默认 10
    返回:
        状态码与耗时；4xx/5xx 也算「服务可达」，会明确区分
    """
    if is_mock():
        return _mock("check_http", f"{url} → 200 OK，耗时 210ms")

    if not re.match(r"^https?://", url or ""):
        url = "http://" + (url or "")
    request = urllib.request.Request(url, headers={"User-Agent": "netdiag-agent/1.0"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as resp:
            cost = (time.perf_counter() - started) * 1000
            return (
                f"HTTP GET {url} → {resp.status} {resp.reason}，耗时 {cost:.0f} ms，"
                f"最终地址 {resp.geturl()}"
            )
    except urllib.error.HTTPError as exc:
        cost = (time.perf_counter() - started) * 1000
        return (
            f"HTTP GET {url} → {exc.code} {exc.reason}（服务可达，应用层返回错误状态），"
            f"耗时 {cost:.0f} ms"
        )
    except urllib.error.URLError as exc:
        return f"HTTP GET {url} 失败：{exc.reason}\nTCP 层可能就没通，建议先 check_port。"
    except Exception as exc:  # noqa: BLE001
        return f"HTTP GET {url} 异常：{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- 工具 5：路由
@tool
def traceroute(target: str, max_hops: int = 12) -> str:
    """跟踪到目标的分组路径，用于判断在哪一跳开始丢包、是否绕路。

    参数:
        target: 目标主机名或 IP
        max_hops: 最大跳数，1~20，默认 12（越大越慢）
    返回:
        逐跳摘要；命令缺失时返回安装提示
    """
    if is_mock():
        return _mock("traceroute", f"{target}: 第 1 跳 192.168.1.1，第 8 跳到达目标")

    max_hops = max(1, min(int(max_hops), 20))
    cmd = (
        ["tracert", "-d", "-h", str(max_hops), "-w", "800", target]
        if IS_WINDOWS
        else ["traceroute", "-n", "-m", str(max_hops), "-w", "2", target]
    )
    code, output = run_command(cmd, timeout=max_hops * 4 + 20)
    if code == 127:
        return output + "\n（Linux 需先安装：apt install traceroute / yum install traceroute）"

    hops = [line.strip() for line in output.splitlines() if re.match(r"^\s*\d+", line)]
    if not hops:
        return f"traceroute {target} 未取得有效路径：\n{output.strip()[-700:]}"
    return f"traceroute {target}（共 {len(hops)} 跳）：\n" + "\n".join(hops[:max_hops])


# ---------------------------------------------------------------- 工具 6：本机网络
@tool
def local_network_info() -> str:
    """查看本机网络配置：主机名、本机 IPv4、默认网关。排查「是本地配置问题还是远端问题」的第一步。"""
    if is_mock():
        return _mock("local_network_info", "主机名 PC-TEST，IPv4 192.168.1.50，网关 192.168.1.1")

    hostname = socket.gethostname()
    try:
        _, _, ips = socket.gethostbyname_ex(hostname)
        ips = [ip for ip in ips if not ip.startswith("127.")]
    except Exception:  # noqa: BLE001
        ips = []

    gateway = "未识别"
    if IS_WINDOWS:
        _code, output = run_command(["ipconfig"], timeout=15)
        match = re.search(r"(?:默认网关|Default Gateway)[^\d]*((?:\d{1,3}\.){3}\d{1,3})", output)
    else:
        _code, output = run_command(["ip", "route"], timeout=10)
        match = re.search(r"default via ([\d.]+)", output)
    if match:
        gateway = match.group(1)

    return (
        f"主机名：{hostname}\n"
        f"本机 IPv4：{', '.join(ips) or '未获取到'}\n"
        f"默认网关：{gateway}"
    )
