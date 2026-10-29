"""会话记忆：按 session_id 隔离的短期上下文，带裁剪与用量统计。

记忆的本质
----------
模型不会「记住」任何东西。所谓多轮对话，是每次调用前把历史消息重新拼进 prompt。
所以三件事必须自己处理：

1. **隔离**：不同用户的对话不能串味，key 必须是 session_id；
2. **增长**：历史随轮次线性变长，token 成本随之上升，必须有裁剪或摘要；
3. **形状**：历史里混进 ToolMessage / 带 tool_calls 的 AIMessage 会让下一轮报错，
   回填上下文时只保留纯文本的 Human / AI 消息。

框架里的 `RunnableWithMessageHistory` 是把这三件事封进一个 Runnable，
本项目默认用显式实现（`MemoryChat`），因为它可读、可单测、没有版本弃用风险；
`build_history_runnable` 保留框架写法作为对照，两者行为一致。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = "你是一个中文助理。回答简洁；涉及上文信息时直接引用，不要重复提问。"


class MessageLog(BaseChatMessageHistory):
    """进程内消息日志。

    自己实现而不是用 `InMemoryChatMessageHistory`：后者在 LangChain 1.6.4 起被标记弃用，
    而它做的事就是一个 list 加两个方法 —— 30 行代码换掉一条永远消不掉的弃用告警，
    并且让「历史存在哪里、什么时候写、写什么」变成可读的显式逻辑。
    """

    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def clear(self) -> None:
        self._messages.clear()


@dataclass
class SessionStats:
    session_id: str
    messages: int
    chars: int


@dataclass
class MemoryChat:
    llm: object
    max_messages: int | None = None  # 只保留最近 N 条进入上下文（None = 不裁剪）
    store: dict[str, MessageLog] = field(default_factory=dict)

    def _prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

    def history(self, session_id: str) -> list[BaseMessage]:
        return list(self.get(session_id).messages)

    def get(self, session_id: str) -> MessageLog:
        if session_id not in self.store:
            self.store[session_id] = MessageLog()
        return self.store[session_id]

    def visible_history(self, session_id: str) -> list[BaseMessage]:
        """真正送进 prompt 的那部分历史（裁剪在此生效）。"""
        messages = self.history(session_id)
        if self.max_messages and len(messages) > self.max_messages:
            return messages[-self.max_messages :]
        return messages

    def ask(self, session_id: str, text: str) -> str:
        history = self.visible_history(session_id)
        messages = self._prompt().format_messages(history=history, input=text)
        reply = self.llm.invoke(messages)
        content = (reply.content or "").strip()

        # 写回完整历史（存全量，取裁剪），这样「历史多长」是可观测的
        session = self.get(session_id)
        session.add_message(HumanMessage(content=text))
        session.add_message(AIMessage(content=content))
        return content

    def reset(self, session_id: str) -> None:
        self.store.pop(session_id, None)

    def stats(self, session_id: str) -> SessionStats:
        messages = self.history(session_id)
        return SessionStats(session_id, len(messages), sum(len(m.content or "") for m in messages))


def build_history_runnable(llm, max_messages: int | None = None):
    """框架等价实现：`RunnableWithMessageHistory` 包一层。

    注意两点：
    * 它在 LangChain 1.x 已被标记弃用，官方推荐用 LangGraph 的 checkpointer；
      这里保留是为了能对照两种写法的行为是否一致；
    * 警告过滤器必须在 import 之后再设置，因为 langchain 导入时会重置
      `LangChainDeprecationWarning` 的过滤器，提前设置会被覆盖。
    """
    from langchain_core.runnables.history import RunnableWithMessageHistory

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        from langchain_core._api.deprecation import LangChainDeprecationWarning

        warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    except ImportError:
        pass

    store: dict[str, MessageLog] = {}

    def get_history(session_id: str) -> MessageLog:
        if session_id not in store:
            store[session_id] = MessageLog()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    runnable = RunnableWithMessageHistory(
        prompt | llm,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return runnable, store
