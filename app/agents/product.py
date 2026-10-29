"""产品顾问：一个「知识在本地、推理在模型」的最小业务闭环。

两个工具的分工体现了两种检索方式：
* `lookup_product` —— 结构化查询。型号是有限集合，直接查字典最准，不该让模型去「理解」文本；
* `ask_company_knowledge` —— 非结构化问答。公司、技术、自动驾驶这些描述性内容，
  用「把资料塞进 prompt 再问一次」的方式回答（最小号 RAG，没有向量库，但足够回答
  资料里写过的内容，且不会编资料外的内容）。

内层问答和外层 Agent 共用同一个模型实例：`get_llm()` 带缓存，这里也显式复用，
避免一次提问触发两次后端探测。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from ..llm import get_llm
from ..tracing import AgentRun, build_agent, run_agent

# ---------------- 业务数据：换成自己的产品库即可 ----------------
PRODUCT_DB: dict[str, str] = {
    "Model 3": "中型轿车，流线型车身、极简内饰。定价 23.19–33.19 万元。",
    "Model Y": "中型 SUV，与 Model 3 同平台，车身更高、后备箱更大。定价 26.39–36.39 万元。",
    "Model X": "大型 SUV，标志性鹰翼门，三排座椅。定价 89.89–105.89 万元。",
}

COMPANY_INFO = """
公司主营电动汽车，产品线包括 Model S、Model 3、Model X、Model Y。
核心技术方向为三电系统（电池、电机、电控）与整车电子电气架构，
并持续投入辅助驾驶能力的研发，已量产的功能包括自动紧急制动、自适应巡航、车道保持辅助。
服务网络由直营门店与超级充电站组成。
"""

CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=(
        # 只写「资料里没有就说没有提到」会让模型过度保守，把资料里明明写着的内容也判为「没有」。
        # 加上「有相关描述就整理作答」这一半，才是完整的约束。
        "请阅读下面的资料并回答问题。只要资料里有相关描述，就把它们整理成答案；"
        "只有资料完全没有相关内容时才回答「资料中没有提到」，不要补充资料之外的知识。\n"
        "资料：{context}\n\n问题：{query}"
    ),
)

SYSTEM_PROMPT = (
    "你是产品顾问，只回答资料范围内的问题。\n"
    "可用工具：lookup_product（查具体型号的价格、配置、特点）；"
    "ask_company_knowledge（查公司背景、技术路线、辅助驾驶）。\n"
    "规则：\n"
    "1. 问题里出现 Model 3 / Model Y / Model X，先用 lookup_product；"
    "价格与配置以该工具返回为准，不要用公司资料的内容代替。\n"
    "2. 问题不含具体型号、且属于公司或技术话题，用 ask_company_knowledge。\n"
    "3. 资料外的问题（股价、销量、竞品等）直接答「资料中没有相关信息，无法回答」。\n"
    "4. 中文，120 字以内，不复述指令、不承诺资料外的能力。"
)


@tool
def lookup_product(product_name: str) -> str:
    """按型号查询产品定位与价格。

    参数:
        product_name: 型号名，例如 Model 3 / Model Y / Model X
    """
    query = (product_name or "").strip().lower()
    for key, desc in PRODUCT_DB.items():
        if key.lower() in query:
            return f"{key}：{desc}"
    return f"没有「{product_name}」这个型号。当前支持：{', '.join(PRODUCT_DB)}"


@tool
def ask_company_knowledge(question: str) -> str:
    """回答公司背景、技术路线、辅助驾驶等描述性问题。**仅用于不涉及具体型号的问题**；
    凡涉及 Model 3 / Model Y / Model X 的配置与价格，请改用 lookup_product。

    参数:
        question: 用户的原始问题
    """
    prompt = CONTEXT_QA_PROMPT.format(query=question, context=COMPANY_INFO)
    answer = (get_llm(temperature=0.0).invoke(prompt).content or "").strip()
    if "资料中没有提到" in answer:
        # 把「走错工具」的后果说清楚，模型下一轮才知道该换一个
        return answer + "（提示：若问题涉及具体型号的配置或价格，请改用 lookup_product）"
    return answer


TOOLS = [lookup_product, ask_company_knowledge]


@dataclass
class ProductAdvisor:
    """带可选记忆的产品顾问。每次提问返回完整轨迹，便于观察工具选择是否合理。"""

    memory: bool = False
    agent: object = None
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.agent = build_agent(get_llm(temperature=0.2), TOOLS, SYSTEM_PROMPT)

    def ask(self, question: str, trace: bool = True) -> AgentRun:
        run = run_agent(self.agent, question, trace=trace, history=self.history)
        if self.memory:
            self.history = self.history + [("user", question), ("assistant", run.answer)]
        return run

    def reset(self) -> None:
        self.history = []


def run(question: str, trace: bool = True, memory: bool = False) -> AgentRun:
    return ProductAdvisor(memory=memory).ask(question, trace=trace)
