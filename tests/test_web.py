"""Web 层的离线用例：路由、SSE 事件协议、报告读取、后端切换。

原则：**不连任何模型**。
所有会真的去推理的入口（`tasks.run_chat` / `tasks.run_task` / `llm.get_llm`）
在用例里都被打桩，因此这套测试在没有 Ollama、没有密钥的机器上也能全绿。
真实链路的验证放在 `scripts/` 的手工验收与 README 的实测记录里，两者分工明确。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.llm import Backend, LLMUnavailable, set_runtime_provider
from app.web import server, tasks as tasklib


@pytest.fixture(autouse=True)
def _clean_runtime():
    """每个用例前后都把「运行时后端选择」清空，避免用例之间互相污染。"""
    set_runtime_provider(None, None)
    yield
    set_runtime_provider(None, None)


@pytest.fixture()
def client():
    return TestClient(server.app)


def parse_sse(text: str) -> list[dict]:
    """把接口返回的 SSE 文本还原成事件列表（与服务端帧格式一一对应）。"""
    events = []
    for frame in text.split("\n\n"):
        line = next((ln for ln in frame.split("\n") if ln.startswith("data:")), None)
        if line:
            events.append(json.loads(line[5:].strip()))
    return events


# ------------------------------------------------------------------ 静态资源
def test_index_page_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "多任务控制台" in res.text


@pytest.mark.parametrize("name", ["index.html", "app.js", "style.css"])
def test_static_assets_exist(name):
    """前端是纯静态文件，缺一个页面就是白屏 —— 用一条用例把它钉住。"""
    path = server.STATIC_DIR / name
    assert path.exists(), f"缺少静态资源 {name}"
    assert path.stat().st_size > 500, f"{name} 内容异常（过小）"


def test_js_has_no_unbalanced_template_literal():
    """前端只有一份 app.js，没有构建期语法检查；这里做一次最低成本的配平检查。"""
    code = (server.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert code.count("`") % 2 == 0, "app.js 里反引号数量为奇数，多半是模板字符串没闭合"


def test_js_translates_browser_network_errors():
    """断流时浏览器只给 `network error` / `Failed to fetch`，照搬给用户等于没说。

    这几条是实测抓到的真实原文（Chrome 在 SSE 流被掐断时抛的就是这两个），
    改动 app.js 时若把翻译丢掉，界面会退回「这一轮没跑通 network error」。
    """
    code = (server.STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "function explainNetworkError(" in code, "缺少断线文案翻译函数"
    # 三个真实原文都必须在匹配规则里
    for raw in ("network ?error", "failed to fetch", "load failed"):
        assert raw in code.lower(), f"未覆盖浏览器原文：{raw}"
    # 主流程的 catch 必须走翻译，不能退回裸 err.message
    assert "addError(wrap, explainNetworkError(err))" in code
    assert "addError(wrap, err.message)" not in code



# ------------------------------------------------------------------ 状态接口
def test_status_returns_backend_tasks_and_cloud_info(client, monkeypatch):
    monkeypatch.setattr(
        server,
        "probe",
        lambda cloud=False: {
            "ollama_url": "http://127.0.0.1:11434",
            "ollama_reachable": True,
            "local_models": ["qwen3-8b-local:latest"],
            "picked": "qwen3-8b-local:latest",
            "dashscope_key": False,
            "cloud": {"base_url": "http://fake/v1", "model": "qwen-turbo", "has_key": False},
            "provider": "auto",
            "runtime_provider": None,
            "runtime_model": None,
            "backend": "[LLM] 本地 Ollama | qwen3-8b-local:latest | http://127.0.0.1:11434",
            "backend_info": {
                "provider": "ollama",
                "kind": "本地",
                "model": "qwen3-8b-local:latest",
                "base_url": "http://127.0.0.1:11434",
                "banner": "x",
            },
        },
    )
    data = client.get("/api/status").json()
    assert data["backend_info"]["provider"] == "ollama"
    assert len(data["tasks"]) >= 6
    assert "cloud" in data
    assert data["providers"] == ["auto", "ollama", "cloud"]


def test_tools_endpoint_lists_registry(client):
    data = client.get("/api/tools").json()
    names = {item["name"] for item in data["tools"]}
    assert {"web_search", "calculator", "ping_host"} <= names


# ------------------------------------------------------------------ 对话流
def test_chat_stream_relays_token_and_final(client, monkeypatch):
    def fake_run_chat(message, session_id="default"):
        yield {"type": "tool_call", "name": "calculator", "args": '{"expression": "1+1"}'}
        yield {"type": "tool_result", "name": "calculator", "preview": "2"}
        yield {"type": "token", "text": "答案是"}
        yield {"type": "token", "text": " 2"}
        yield {"type": "final", "answer": "答案是 2", "seconds": 0.1, "tool_calls": 1, "steps": []}

    monkeypatch.setattr(tasklib, "run_chat", fake_run_chat)
    res = client.post("/api/chat", json={"message": "1+1", "session_id": "t1"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(res.text)
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "token", "token", "final"]
    assert events[-1]["answer"] == "答案是 2"


def test_chat_rejects_empty_message(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_llm_unavailable_becomes_error_event(client, monkeypatch):
    """后端没配好时，浏览器要收到一句人话，而不是 500 + traceback。"""

    def boom(message, session_id="default"):
        raise LLMUnavailable("没有可用的模型后端：\n  - 本地 Ollama 未响应")
        yield  # pragma: no cover  让它成为生成器

    monkeypatch.setattr(tasklib, "run_chat", boom)
    events = parse_sse(client.post("/api/chat", json={"message": "hi"}).text)
    assert events[-1]["type"] == "error"
    assert events[-1]["kind"] == "llm_unavailable"
    assert "Ollama" in events[-1]["message"]


def test_runtime_exception_becomes_error_event(client, monkeypatch):
    def boom(message, session_id="default"):
        raise RuntimeError("模型返回了空内容")
        yield  # pragma: no cover

    monkeypatch.setattr(tasklib, "run_chat", boom)
    events = parse_sse(client.post("/api/chat", json={"message": "hi"}).text)
    assert events[-1]["type"] == "error"
    assert "RuntimeError" in events[-1]["message"]


def test_quota_error_is_translated_for_user(client, monkeypatch):
    """厂商的 429 原文对用户没意义，界面必须显示「怎么办」。"""

    def boom(message, session_id="default"):
        raise RuntimeError(
            "RateLimitError: Error code: 429 - {'error': {'code': 'limit_requests'}}"
        )
        yield  # pragma: no cover

    monkeypatch.setattr(tasklib, "run_chat", boom)
    event = parse_sse(client.post("/api/chat", json={"message": "hi"}).text)[-1]
    assert event["type"] == "error"
    assert "额度或调用频率超限" in event["message"]
    # 提示里要带上可执行的下一步，而不是只有一句「失败了」
    assert "--provider ollama" in event["message"]


# ------------------------------------------------------------------ 任务流
def test_task_stream_relays_stage_events(client, monkeypatch):
    def fake_run_task(task_id, text):
        yield {"type": "stage", "name": "① 单步工具", "text": "字符数 72", "seconds": 0.01}
        yield {"type": "final", "answer": "改进建议：缩短第二段", "seconds": 0.02, "tool_calls": 0, "steps": []}

    monkeypatch.setattr(tasklib, "run_task", fake_run_task)
    events = parse_sse(client.post("/api/task", json={"task": "pipeline", "input": ""}).text)
    assert events[0]["type"] == "stage"
    assert events[-1]["type"] == "final"


def test_unknown_task_is_reported_as_error_event(client):
    """真实调用 run_task（不传未知 id 之外的参数），KeyError 应在流内变成 error 事件。"""
    events = parse_sse(client.post("/api/task", json={"task": "nope", "input": "x"}).text)
    assert events[-1]["type"] == "error"
    assert "未知任务" in events[-1]["message"]


# ------------------------------------------------------------------ 后端切换
def test_provider_switch_rejects_unknown_value(client):
    res = client.post("/api/provider", json={"provider": "skynet"})
    assert res.status_code == 400
    assert "未知的后端" in res.json()["detail"]


@pytest.mark.parametrize("payload", [{"provider": "cloud"}, {"provider": "cloud", "model": None}])
def test_provider_switch_accepts_blank_model(client, payload):
    """回归：请求体把 model 定义成非空 str 时，前端清空输入框发的 null 会 422，
    表现是「切后端 / 清空模型名点了没反应」。留空表示回到自动挑选，必须合法。"""
    res = client.post("/api/provider", json=payload)
    assert res.status_code == 200, res.text
    assert res.json()["runtime_model"] is None


def test_any_user_typed_model_name_is_accepted(client, monkeypatch):
    """模型名不是枚举：用户手敲的名字（包括候选列表里没有的）都要照收，
    能不能用由厂商回答，前端不做白名单。"""
    from app import llm

    # probe 反映的是真实运行时状态，不是写死的字典 —— 否则测的是假数据
    monkeypatch.setattr(
        server,
        "probe",
        lambda cloud=False: {
            "runtime_provider": llm.runtime_provider(),
            "runtime_model": llm.runtime_model(),
            "backend": "",
            "backend_info": None,
        },
    )
    res = client.post("/api/provider", json={"provider": "cloud", "model": "my-company-llm-v9"})
    assert res.status_code == 200
    assert res.json()["runtime_model"] == "my-company-llm-v9"
    assert llm.runtime_model() == "my-company-llm-v9"


def test_provider_switch_updates_runtime_state(client, monkeypatch):
    monkeypatch.setattr(
        server,
        "probe",
        lambda cloud=False: {
            "runtime_provider": "cloud",
            "runtime_model": "qwen-plus",
            "backend": "[LLM] 云端 API | qwen-plus | http://fake/v1",
            "backend_info": {"provider": "cloud"},
        },
    )
    data = client.post("/api/provider", json={"provider": "cloud", "model": "qwen-plus"}).json()
    assert data["ok"] is True
    assert data["runtime_provider"] == "cloud"
    assert data["runtime_model"] == "qwen-plus"


def test_provider_test_reports_success(client, monkeypatch):
    class FakeLLM:
        _backend = Backend("cloud", "qwen-turbo", "http://fake/v1")

        def invoke(self, _prompt):
            return type("Msg", (), {"content": "收到"})()

    monkeypatch.setattr(server, "get_llm", lambda **kwargs: FakeLLM())
    data = client.post("/api/provider/test", json={"provider": "cloud"}).json()
    assert data["ok"] is True
    assert data["reply"] == "收到"
    assert data["backend"]["provider"] == "cloud"


def test_provider_test_reports_missing_key(client, monkeypatch):
    def boom(**kwargs):
        raise LLMUnavailable("未配置 DASHSCOPE_API_KEY（或通用的 CLOUD_API_KEY）")

    monkeypatch.setattr(server, "get_llm", boom)
    data = client.post("/api/provider/test", json={"provider": "cloud"}).json()
    assert data["ok"] is False
    assert "CLOUD_API_KEY" in data["error"]


# ------------------------------------------------------------------ 报告
def test_reports_list_and_content(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server.config, "OUTPUT_DIR", tmp_path)
    (tmp_path / "verify-1.md").write_text("# 验收\n通过 9/9", encoding="utf-8")

    listing = client.get("/api/reports").json()
    assert [item["name"] for item in listing["reports"]] == ["verify-1.md"]

    res = client.get("/api/reports/verify-1.md")
    assert res.status_code == 200
    assert "通过 9/9" in res.text


@pytest.mark.parametrize("name", ["evil.txt", "report.json", ".env"])
def test_report_rejects_non_markdown(client, monkeypatch, tmp_path, name):
    monkeypatch.setattr(server.config, "OUTPUT_DIR", tmp_path)
    assert client.get(f"/api/reports/{name}").status_code == 400


@pytest.mark.parametrize("name", ["../.env", "sub/dir.md"])
def test_report_path_traversal_blocked(client, monkeypatch, tmp_path, name):
    """带路径分隔符的名字连路由都匹配不上（404），或进到处理器被拒（400）——
    两种都算安全，关键是绝不能把 outputs 之外的文件读出来。"""
    monkeypatch.setattr(server.config, "OUTPUT_DIR", tmp_path)
    (tmp_path.parent / ".env").write_text("SECRET=1", encoding="utf-8")
    res = client.get(f"/api/reports/{name}")
    assert res.status_code in {400, 404}
    assert "SECRET" not in res.text


def test_report_encoded_slash_is_not_served(client, monkeypatch, tmp_path):
    """编码后的 `../` 会被路由拆成多段，无论落到 400 还是 404，只要不吐出文件就是对的。"""
    monkeypatch.setattr(server.config, "OUTPUT_DIR", tmp_path)
    (tmp_path.parent / ".env").write_text("SECRET=1", encoding="utf-8")
    res = client.get("/api/reports/..%2F.env")
    assert res.status_code in {400, 404}
    assert "SECRET" not in res.text


def test_report_missing_returns_404(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server.config, "OUTPUT_DIR", tmp_path)
    assert client.get("/api/reports/none.md").status_code == 404


# ------------------------------------------------------------------ 任务注册表
def test_task_catalog_shape():
    catalog = tasklib.task_catalog()
    ids = [item["id"] for item in catalog]
    assert ids == ["research", "analyst", "netdiag", "toolkit", "pipeline", "multistep"]
    assert all(item["desc"] and item["placeholder"] for item in catalog)
    chain_kinds = {item["id"] for item in catalog if item["kind"] == "chain"}
    assert chain_kinds == {"pipeline", "multistep"}


def test_chat_tools_are_all_registered():
    """对话用的工具名必须都在注册表里，改名时这条会先炸。"""
    from app.tools import REGISTRY

    assert set(tasklib.CHAT_TOOL_NAMES) <= set(REGISTRY)


def test_chat_memory_trims_old_turns(monkeypatch):
    tasklib.reset_chat("mem")
    for i in range(tasklib.CHAT_MAX_TURNS + 3):
        tasklib.append_chat("mem", f"q{i}", f"a{i}")
    turns = tasklib.chat_history("mem")
    assert len(turns) == tasklib.CHAT_MAX_TURNS
    assert turns[-1] == (f"q{tasklib.CHAT_MAX_TURNS + 2}", f"a{tasklib.CHAT_MAX_TURNS + 2}")
    tasklib.reset_chat("mem")
