"""网络工具：解析层全部是纯函数，可以用固定文本喂进去测；执行层只测本机回环。

把「跑命令」和「解析输出」拆开，是这批工具能进单元测试的前提 ——
否则每条断言都要依赖真实网络，CI 一定不稳。
"""

from __future__ import annotations

import socket

from app.tools.netdiag import check_port, decode_bytes, is_mock, parse_ping

WINDOWS_CN_OK = """正在 Ping www.a.shifen.com [110.242.68.66] 具有 32 字节的数据:
来自 110.242.68.66 的回复: 字节=32 时间=37ms TTL=51
来自 110.242.68.66 的回复: 字节=32 时间=38ms TTL=51

110.242.68.66 的 Ping 统计信息:
    数据包: 已发送 = 4，已接收 = 4，丢失 = 0 (0% 丢失)，
往返行程的估计时间(以毫秒为单位):
    最短 = 36ms，最长 = 40ms，平均 = 37ms
"""

WINDOWS_CN_TIMEOUT = """正在 Ping 10.255.255.1 具有 32 字节的数据:
请求超时。
请求超时。

10.255.255.1 的 Ping 统计信息:
    数据包: 已发送 = 2，已接收 = 0，丢失 = 2 (100% 丢失)，
"""

LINUX_OK = """PING example.com (93.184.216.34) 56(84) bytes of data.
64 bytes from 93.184.216.34: icmp_seq=1 ttl=56 time=36.2 ms

--- example.com ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 3004ms
rtt min/avg/max/mdev = 36.162/37.100/38.001/0.600 ms
"""

WINDOWS_EN_TIMEOUT = """Pinging 10.255.255.1 with 32 bytes of data:
Request timed out.
Request timed out.

Packets: Sent = 4, Received = 0, Lost = 4 (100% loss),
"""


def test_parse_windows_chinese_success():
    parsed = parse_ping(WINDOWS_CN_OK)
    assert parsed["reachable"] is True
    assert parsed["loss_percent"] == 0
    assert parsed["avg_ms"] == "37"


def test_parse_windows_chinese_full_loss():
    parsed = parse_ping(WINDOWS_CN_TIMEOUT)
    assert parsed["reachable"] is False
    assert parsed["loss_percent"] == 100


def test_parse_linux_success_reads_rtt_line():
    parsed = parse_ping(LINUX_OK)
    assert parsed["reachable"] is True
    assert parsed["avg_ms"] == "37.100"


def test_parse_windows_english_timeout():
    parsed = parse_ping(WINDOWS_EN_TIMEOUT)
    assert parsed["reachable"] is False
    assert parsed["loss_percent"] == 100


def test_decode_gbk_bytes_from_chinese_console():
    raw = "请求超时。".encode("gbk")
    assert decode_bytes(raw) == "请求超时。"


def test_decode_utf8_bytes():
    assert decode_bytes("正常".encode("utf-8")) == "正常"


def test_check_port_detects_listening_socket():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        result = check_port.invoke({"host": "127.0.0.1", "port": port, "timeout": 2})
    finally:
        server.close()
    assert "可连接" in result


def test_mock_switch_reads_env_at_call_time(monkeypatch):
    monkeypatch.setenv("NET_DIAG_MOCK", "1")
    assert is_mock() is True
    monkeypatch.setenv("NET_DIAG_MOCK", "0")
    assert is_mock() is False
