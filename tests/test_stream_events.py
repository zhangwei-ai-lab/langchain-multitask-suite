"""流式事件采集的离线用例：工具调用分片、token 拼接、event 顺序。

为什么值得单独测
----------------
`stream_mode="messages"` 下工具调用是**分片**到达的，各家厂商的分片粒度还不一样。
采集层一旦把「同一轮并行的多个调用」合并成一条，界面上就会出现
「1 次工具调用、5 条工具返回」这种自相矛盾的轨迹 —— 模型其实是对的，
错的是采集。这里用一个假 Agent 把三种典型分片形态都钉住。
"""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk, ToolMessage

from app.tracing import StreamEvents, _accumulate_call, stream_agent


class FakeAgent:
    """只实现 `stream()` 的最小替身：按脚本逐条吐出 (chunk, metadata)。"""

    def __init__(self, chunks):
        self.chunks = chunks

    def stream(self, inputs, stream_mode=None):
        assert stream_mode == "messages"
        for chunk in self.chunks:
            yield chunk, {"langgraph_node": "model"}


def call_chunk(name: str, args: str, call_id: str | None = None, index: int | None = None):
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": name, "args": args, "id": call_id, "index": index, "type": "tool_call_chunk"}
        ],
    )


def text_chunk(text: str):
    return AIMessageChunk(content=text)


def tool_message(name: str, content: str):
    return ToolMessage(content=content, tool_call_id=f"call-{name}", name=name)


# ------------------------------------------------------------------ 分槽逻辑
def test_parallel_calls_without_id_are_not_merged():
    """Ollama 一次并行调 5 个工具、分片既无 id 也 index=None，必须是 5 条调用。"""
    pending: list[dict] = []
    payloads = [
        ("local_network_info", "{}"),
        ("ping_host", '{"target": "github.com", "count": 4}'),
        ("dns_lookup", '{"hostname": "github.com"}'),
        ("check_port", '{"host": "github.com", "port": 443}'),
        ("check_http", '{"url": "https://github.com"}'),
    ]
    for name, args in payloads:
        _accumulate_call(pending, {"id": None, "name": name, "args": args, "index": None})

    assert [item["name"] for item in pending] == [name for name, _ in payloads]
    # 每个槽的参数必须是它自己的，不能出现 {'{}{' 这种拼接
    assert pending[1]["args"] == '{"target": "github.com", "count": 4}'


def test_fragmented_args_with_same_id_are_merged():
    """OpenAI 兼容端点会带 id，同一个调用被切成多片时必须并回同一个槽。"""
    pending: list[dict] = []
    _accumulate_call(pending, {"id": "a1", "name": "ping_host", "args": '{"target": "gi'})
    _accumulate_call(pending, {"id": "a1", "name": None, "args": 'thub.com", "count": 4}'})

    assert len(pending) == 1
    assert pending[0]["args"] == '{"target": "github.com", "count": 4}'
    assert pending[0]["done"] is True


def test_same_tool_twice_in_a_row_opens_new_slot():
    """同名工具连续调用两次（例如先 ping 再看丢包率），不能并成一条。"""
    pending: list[dict] = []
    _accumulate_call(pending, {"id": None, "name": "ping_host", "args": '{"target": "a"}'})
    _accumulate_call(pending, {"id": None, "name": "ping_host", "args": '{"target": "b"}'})

    assert len(pending) == 2
    assert pending[0]["args"] == '{"target": "a"}'
    assert pending[1]["args"] == '{"target": "b"}'


# ------------------------------------------------------------------ 事件序列
def test_stream_agent_emits_call_result_token_final_in_order():
    agent = FakeAgent(
        [
            call_chunk("calculator", '{"expression": "1+1"}'),
            tool_message("calculator", "2"),
            text_chunk("答案是"),
            text_chunk(" 2"),
        ]
    )
    events = list(stream_agent(agent, "1+1 等于几"))

    assert [e["type"] for e in events] == [
        StreamEvents.TOOL_CALL,
        StreamEvents.TOOL_RESULT,
        StreamEvents.TOKEN,
        StreamEvents.TOKEN,
        StreamEvents.FINAL,
    ]
    assert events[0]["name"] == "calculator"
    assert events[1]["preview"] == "2"
    assert "".join(e["text"] for e in events if e["type"] == "token") == "答案是 2"
    final = events[-1]
    assert final["answer"] == "答案是 2"
    assert final["tool_calls"] == 1


def test_stream_agent_counts_each_parallel_call():
    """五个并行调用必须产生五条 tool_call 事件（前端轨迹栏按条数画）。"""
    chunks = [call_chunk("ping_host", '{"target": "a"}')]
    for name in ("dns_lookup", "check_port", "check_http", "local_network_info"):
        chunks.append(call_chunk(name, "{}"))
    chunks.append(tool_message("ping_host", "可达"))

    events = list(stream_agent(FakeAgent(chunks), "诊断"))
    calls = [e for e in events if e["type"] == StreamEvents.TOOL_CALL]
    assert [c["name"] for c in calls] == [
        "ping_host",
        "dns_lookup",
        "check_port",
        "check_http",
        "local_network_info",
    ]
    assert events[-1]["tool_calls"] == 5


def test_stream_agent_handles_list_content():
    """部分厂商把正文放在 list[dict] 分块里，采集层要能取到文本。"""
    class ListContentAgent(FakeAgent):
        pass

    agent = FakeAgent([AIMessageChunk(content=[{"type": "text", "text": "分块内容"}])])
    events = list(stream_agent(agent, "hi"))
    assert "".join(e["text"] for e in events if e["type"] == "token") == "分块内容"


def test_stream_agent_history_is_prepended():
    """多轮对话：历史必须以 (role, text) 元组形式拼在最新问题前面。"""
    seen = {}

    class SpyAgent:
        def stream(self, inputs, stream_mode=None):
            seen["messages"] = inputs["messages"]
            yield text_chunk("ok"), {}

    list(stream_agent(SpyAgent(), "第二问", history=[("user", "第一问"), ("assistant", "第一答")]))
    assert seen["messages"] == [("user", "第一问"), ("assistant", "第一答"), ("user", "第二问")]
