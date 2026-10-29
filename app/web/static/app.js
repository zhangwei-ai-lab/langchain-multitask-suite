/* ===========================================================================
   LangChain 多任务控制台 · 前端逻辑
   无框架、无构建：原生 DOM + fetch + 手写 SSE 解析。
   与后端的唯一契约是 `app/web/tasks.py` 顶部注释里的事件协议。
   =========================================================================== */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  provider: "auto",
  model: "",
  taskId: "chat",
  sessionId: `web-${Math.random().toString(36).slice(2, 8)}`,
  busy: false,
  status: null,
  chainMode: false,
};

/* ------------------------------------------------------------------ 工具 */
let toastTimer = null;
function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = String(message).slice(0, 400);
  el.classList.toggle("err", isError);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), isError ? 7000 : 2600);
}

async function requestJSON(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) { /* 非 JSON 错误体，保留状态码 */ }
    throw new Error(detail);
  }
  return res.json();
}

/* ------------------------------------------------------------------ 状态 */
function renderBackend(info) {
  const dot = $("#backendDot");
  const text = $("#backendText");
  dot.className = "dot";
  if (!info || !info.backend_info) {
    dot.classList.add("err");
    text.textContent = (info && info.backend) || "没有可用的模型后端";
    $("#tipBox").textContent =
      "本地 Ollama 未响应且云端未配密钥。修法：装 Ollama 后 ollama pull qwen3:8b，" +
      "或在 .env 填 CLOUD_API_KEY 后重启服务。";
    return;
  }
  const b = info.backend_info;
  dot.classList.add(b.provider === "ollama" ? "ok" : "ok");
  text.textContent = `${b.kind} · ${b.model}`;
  text.title = b.banner;
  $("#tipBox").textContent =
    b.provider === "ollama"
      ? `当前走本地 Ollama（${b.base_url}），零密钥零费用。切到「云端」需要 CLOUD_API_KEY。`
      : `当前走云端 API（${b.base_url}）。密钥只存在服务端环境变量里，不会下发到浏览器。`;
}

function fillModelOptions(info) {
  // 候选列表只是「提示」，不是白名单：用户手敲的名字一律照用，
  // 敲错了由厂商返回 404，界面再翻译成人话 —— 比前端偷偷拦掉更诚实。
  const list = $("#modelList");
  const input = $("#modelInput");
  const known = new Set();

  const local = (info.local_models || []).slice();
  const cloudModels = (info.cloud && info.cloud.models) || [];
  const preferred = cloudModels.filter((m) => /^(qwen|deepseek|kimi|glm|gpt|moonshot)/i.test(m));

  list.innerHTML = "";
  const addOption = (value, label) => {
    if (known.has(value)) return;
    known.add(value);
    const opt = document.createElement("option");
    opt.value = value;
    opt.label = label ? `${label}｜${value}` : value;
    list.appendChild(opt);
  };

  local.forEach((name) => addOption(name, "本地"));
  (preferred.length ? preferred : cloudModels).slice(0, 60).forEach((name) => addOption(name, "云端"));
  if (info.cloud && info.cloud.model) addOption(info.cloud.model, "云端默认");

  // 占位固定成短句：当前用哪个模型顶栏徽标已经写了，占位里再塞一遍
  // 只会在窄屏被截成「自动挑选（当前 qwen3-8b-loca」，反而更难读。
  if (!state.model) input.value = "";
}

function renderProviderSwitch(info) {
  const current = (info && info.runtime_provider) || info.provider || "auto";
  state.provider = ["auto", "ollama", "cloud"].includes(current) ? current : "auto";
  $$("#providerSwitch button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.provider === state.provider);
  });
}

async function loadStatus(withCloudModels = false) {
  try {
    const info = await requestJSON(`/api/status?cloud=${withCloudModels ? "true" : "false"}`);
    state.status = info;
    renderBackend(info);
    renderProviderSwitch(info);
    fillModelOptions(info);
    if (info.tasks) renderTasks(info.tasks);
    await refreshReportCount();
  } catch (err) {
    toast(`读取服务状态失败：${err.message}`, true);
  }
}

/* ------------------------------------------------------------------ 任务 */
const CHAT_TASK = {
  id: "chat",
  name: "通用对话",
  desc: "搜索 / 时间 / 计算 / 天气 / HTTP 探测",
  placeholder: "输入问题，Enter 发送，Shift+Enter 换行",
  sample: "帮我查一下今天北京天气，然后把气温换算成华氏度",
  kind: "agent",
};

function renderTasks(tasks) {
  const box = $("#taskList");
  box.innerHTML = "";
  [CHAT_TASK, ...tasks].forEach((task) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "task" + (task.id === state.taskId ? " active" : "");
    btn.dataset.id = task.id;
    btn.innerHTML = `<span class="t-name">${task.name}<span class="t-kind">${
      task.kind === "chain" ? "链" : "Agent"
    }</span></span><span class="t-desc">${task.desc}</span>`;
    btn.addEventListener("click", () => selectTask(task));
    box.appendChild(btn);
  });
}

function selectTask(task) {
  state.taskId = task.id;
  state.chainMode = task.kind === "chain";
  state.task = task;
  $$("#taskList .task").forEach((el) => el.classList.toggle("active", el.dataset.id === task.id));
  $("#convTitle").textContent = task.name;
  $("#convDesc").textContent = task.desc;
  $("#input").placeholder = task.placeholder || "输入内容";
  $("#btnSample").style.display = "inline-block";
  if (task.id === "chat") $("#btnReset").style.display = "";
  else $("#btnReset").style.display = "none";
  // 注意：这里不清理 #emptyState —— 启动时会先调一次 selectTask，
  // 顺手清掉欢迎卡片就会得到一个空白对话区（用户以为页面坏了）。
  // 清理由 addMessage() 负责。
}

function clearEmptyState() {
  const empty = $("#emptyState");
  if (empty) empty.remove();
}

/* ------------------------------------------------------------------ 消息 */
function addMessage(role, text) {
  clearEmptyState();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const head = document.createElement("div");
  head.className = "msg-head";
  const avatar = document.createElement("span");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "你" : "AI";
  const who = document.createElement("span");
  // 用户消息的头像已经写了「你」，再来一个「你」是重复噪音，只给助手署名
  who.textContent = role === "user" ? "" : state.task?.name || "助手";
  head.append(avatar, who);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (text) bubble.textContent = text;

  wrap.append(head, bubble);
  $("#messages").appendChild(wrap);
  scrollToBottom();
  return { wrap, bubble };
}

function addStage(container, name, text, seconds) {
  const card = document.createElement("div");
  card.className = "stage";
  const head = document.createElement("div");
  head.className = "s-name";
  const title = document.createElement("span");
  title.textContent = name;
  const time = document.createElement("span");
  time.className = "s-time";
  time.textContent = seconds != null ? `${seconds}s` : "";
  head.append(title, time);
  const body = document.createElement("div");
  body.className = "s-text";
  body.textContent = text || "";
  card.append(head, body);
  container.appendChild(card);
  scrollToBottom();
}

function addError(container, message) {
  const box = document.createElement("div");
  box.className = "error-box";
  const strong = document.createElement("strong");
  strong.textContent = "这一轮没跑通";
  box.append(strong, document.createTextNode(message));
  container.appendChild(box);
  scrollToBottom();
}

function scrollToBottom() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

/* ------------------------------------------------------------------ 轨迹 */
function addTraceCall(name, args) {
  const list = $("#traceList");
  const ph = $(".placeholder", list);
  if (ph) ph.remove();

  const item = document.createElement("div");
  item.className = "trace-item";
  const head = document.createElement("div");
  head.className = "tr-head";
  const spin = document.createElement("span");
  spin.className = "spinner";
  const label = document.createElement("span");
  label.textContent = name;
  head.append(spin, label);
  const body = document.createElement("div");
  body.className = "tr-body";
  body.textContent = args || "";
  item.append(head, body);
  list.appendChild(item);
  list.scrollTop = list.scrollHeight;
}

function addTraceResult(name, preview) {
  // 找到最近一个还在转圈的同名调用，把结果补在它下面
  const items = $$("#traceList .trace-item:not(.result)");
  const target = [...items].reverse().find((el) => $(".tr-head span:last-child", el)?.textContent === name);
  const list = $("#traceList");
  const item = document.createElement("div");
  item.className = "trace-item result";
  const head = document.createElement("div");
  head.className = "tr-head";
  const mark = document.createElement("span");
  mark.textContent = "←";
  const label = document.createElement("span");
  label.textContent = name;
  head.append(mark, label);
  const body = document.createElement("div");
  body.className = "tr-body";
  body.textContent = preview || "（空返回）";
  item.append(head, body);
  list.appendChild(item);
  if (target) {
    $(".spinner", target)?.remove();
    target.classList.add("result");
  }
  list.scrollTop = list.scrollHeight;
}

function resetTrace() {
  $("#traceList").innerHTML = '<p class="placeholder">还没有工具调用</p>';
  $("#statTime").textContent = "—";
  $("#statTools").textContent = "0";
  $("#statChars").textContent = "0";
}

/* ------------------------------------------------------------------ 发送 */
function setBusy(busy) {
  state.busy = busy;
  $("#btnSend").disabled = busy;
  $("#btnSample").disabled = busy;
  $("#input").disabled = busy;
  // 只改文字节点：直接写 button 的 lastChild 会把里面的 svg 图标覆盖掉，
  // 按钮就变成「发送 ✈ 发送」这种重复文案
  $("#btnSendLabel").textContent = busy ? "运行中" : "发送";
}

async function send() {
  if (state.busy) return;
  const input = $("#input");
  const text = input.value.trim();
  if (!text && !(state.chainMode && state.taskId === "pipeline")) {
    toast("先输入内容再发送");
    return;
  }

  addMessage("user", text || "（使用内置样本文本）");
  input.value = "";
  autoResize(input);
  resetTrace();
  setBusy(true);

  const { wrap, bubble } = addMessage("assistant", "");
  bubble.classList.add("streaming");
  const started = Date.now();
  const timer = setInterval(() => {
    $("#statTime").textContent = `${((Date.now() - started) / 1000).toFixed(1)}s`;
  }, 100);

  const timerStop = () => clearInterval(timer);
  const ui = { bubble, wrap, chars: 0, tools: 0, chainMode: state.chainMode };

  try {
    const path = state.taskId === "chat" ? "/api/chat" : "/api/task";
    const body =
      state.taskId === "chat"
        ? { message: text, session_id: state.sessionId }
        : { task: state.taskId, input: text };

    await streamEvents(path, body, (ev) => {
      switch (ev.type) {
        case "token":
          bubble.textContent += ev.text;
          ui.chars = bubble.textContent.length;
          $("#statChars").textContent = ui.chars;
          scrollToBottom();
          break;
        case "tool_call":
          ui.tools += 1;
          $("#statTools").textContent = ui.tools;
          addTraceCall(ev.name, ev.args);
          break;
        case "tool_result":
          addTraceResult(ev.name, ev.preview);
          break;
        case "stage":
          if (!ui.firstStage) {
            ui.firstStage = true;
            bubble.remove();
          }
          addStage(wrap, ev.name, ev.text, ev.seconds);
          break;
        case "final":
          $("#statTime").textContent = `${ev.seconds}s`;
          $("#statChars").textContent = (ev.answer || "").length;
          $("#statTools").textContent = ev.tool_calls ?? ui.tools;
          if (ev.answer && !ui.chainMode && !bubble.textContent) bubble.textContent = ev.answer;
          break;
        case "artifact":
          toast(`已生成报告：${ev.name}（可在左侧「运行报告」查看）`);
          refreshReportCount();
          break;
        case "error":
          addError(wrap, ev.message || "未知错误");
          break;
        default:
          break;
      }
    });
  } catch (err) {
    addError(wrap, err.message);
  } finally {
    timerStop();
    bubble.classList.remove("streaming");
    setBusy(false);
    if (!bubble.textContent && !ui.firstStage) bubble.remove();
    scrollToBottom();
  }
}

/**
 * 读 SSE 流。服务端每帧是 `data: {json}\n\n`，但网络分片不保证按帧边界到达，
 * 所以必须自己缓冲、按空行切帧，否则会把一条 JSON 切成两半解析失败。
 */
async function streamEvents(path, body, onEvent) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const errBody = await res.json();
      if (errBody && errBody.detail) detail = errBody.detail;
    } catch (_) { /* 保持状态码文本 */ }
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(5).trim()));
      } catch (err) {
        console.warn("无法解析的事件", frame, err);
      }
    }
  }
}

/* ------------------------------------------------------------------ 报告 */
async function refreshReportCount() {
  try {
    const data = await requestJSON("/api/reports");
    $("#reportCount").textContent = data.reports.length;
  } catch (_) { /* 报告接口失败不影响主流程 */ }
}

async function openReports() {
  $("#reportModal").classList.remove("hidden");
  const list = $("#reportList");
  list.innerHTML = '<p class="placeholder" style="padding:8px">加载中…</p>';
  try {
    const data = await requestJSON("/api/reports");
    if (!data.reports.length) {
      list.innerHTML =
        '<p class="placeholder" style="padding:8px">还没有报告。跑一次「网络诊断」就会生成。</p>';
      return;
    }
    list.innerHTML = "";
    data.reports.forEach((item, index) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "report-item" + (index === 0 ? " active" : "");
      btn.innerHTML = `<div class="r-name">${item.name}</div><div class="r-meta">${item.mtime} · ${(
        item.size / 1024
      ).toFixed(1)} KB</div>`;
      btn.addEventListener("click", () => {
        $$(".report-item", list).forEach((el) => el.classList.remove("active"));
        btn.classList.add("active");
        loadReport(item.name);
      });
      list.appendChild(btn);
    });
    loadReport(data.reports[0].name);
  } catch (err) {
    list.innerHTML = `<p class="placeholder" style="padding:8px">读取失败：${err.message}</p>`;
  }
}

async function loadReport(name) {
  const pre = $("#reportContent");
  pre.textContent = "读取中…";
  try {
    const res = await fetch(`/api/reports/${encodeURIComponent(name)}`);
    pre.textContent = res.ok ? await res.text() : `读取失败：${res.status}`;
  } catch (err) {
    pre.textContent = `读取失败：${err.message}`;
  }
}

/* ------------------------------------------------------------------ 事件绑定 */
function autoResize(el) {
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
}

function bind() {
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    send();
  });

  $("#input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  });
  $("#input").addEventListener("input", (e) => autoResize(e.target));

  $("#btnSample").addEventListener("click", () => {
    const sample = state.task?.sample ?? CHAT_TASK.sample;
    $("#input").value = sample;
    autoResize($("#input"));
    $("#input").focus();
  });

  $("#btnReset").addEventListener("click", async () => {
    try {
      await requestJSON(`/api/chat/reset?session_id=${encodeURIComponent(state.sessionId)}`, {
        method: "POST",
      });
      $("#messages").innerHTML = "";
      resetTrace();
      toast("会话已清空");
    } catch (err) {
      toast(`清空失败：${err.message}`, true);
    }
  });

  $$("#providerSwitch button").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const provider = btn.dataset.provider;
      const before = state.provider;
      state.provider = provider;
      // 换后端 = 换模型来源：本地的 `qwen3:8b` 拿到云端就是 404。
      // 与其让用户撞上「切完就报错」，不如直接回到自动挑选，想指定再自己敲。
      state.model = null;
      if ($("#modelInput")) $("#modelInput").value = "";
      $$("#providerSwitch button").forEach((b) => b.classList.toggle("active", b === btn));
      try {
        const res = await requestJSON("/api/provider", {
          method: "POST",
          body: JSON.stringify({ provider, model: state.model }),
        });
        toast(`已切到${provider === "cloud" ? "云端" : provider === "ollama" ? "本地" : "自动"}：${
          res.backend_banner || "（后端不可用）"
        }`);
        await loadStatus(true);
      } catch (err) {
        state.provider = before;
        renderProviderSwitch(state.status || {});
        toast(`切换失败：${err.message}`, true);
      }
    });
  });

  const modelInput = $("#modelInput");
  const applyModel = async () => {
    const name = modelInput.value.trim();
    if (name === (state.model || "")) return;
    state.model = name || null;
    try {
      const res = await requestJSON("/api/provider", {
        method: "POST",
        body: JSON.stringify({ provider: state.provider, model: state.model }),
      });
      toast(`模型已设为：${res.runtime_model || "自动挑选"}`);
      // /api/provider 的字段叫 backend，renderBackend 认的是 backend_info，转一手
      if (res.backend) renderBackend({ backend_info: res.backend });
    } catch (err) {
      toast(`设置模型失败：${err.message}`, true);
    }
  };

  // 手敲回车 / 从候选里点选 / 点「应用」，三个入口走同一条路
  modelInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      modelInput.blur();
      applyModel();
    }
  });
  modelInput.addEventListener("change", applyModel);
  $("#btnApplyModel").addEventListener("click", applyModel);

  $("#btnTest").addEventListener("click", async () => {
    const btn = $("#btnTest");
    btn.disabled = true;
    btn.textContent = "测试中…";
    try {
      const res = await requestJSON("/api/provider/test", {
        method: "POST",
        body: JSON.stringify({ provider: state.provider, model: state.model }),
      });
      if (res.ok) {
        toast(`连通正常 · ${res.latency_ms}ms · 回复「${res.reply}」`);
      } else {
        toast(`连接失败：${String(res.error).split("\n")[0]}`, true);
        $("#tipBox").textContent = res.error || "未知错误";
      }
    } catch (err) {
      toast(`测试失败：${err.message}`, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "测试连接";
    }
  });

  $("#btnReports").addEventListener("click", openReports);
  $("#btnCloseReport").addEventListener("click", () => $("#reportModal").classList.add("hidden"));
  $("#reportModal").addEventListener("click", (e) => {
    if (e.target.id === "reportModal") $("#reportModal").classList.add("hidden");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("#reportModal").classList.add("hidden");
  });
}

/* ------------------------------------------------------------------ 启动 */
bind();
selectTask(CHAT_TASK);
// 首次进来先拉一次云端模型清单：模型 id 变化快，写死下拉框很快就过时
loadStatus(true);
