"""最小链：`PromptTemplate | ChatModel | StrOutputParser`。

这条链是整个项目所有复杂度的基线 —— 后面无论 Agent 还是多段流水线，
拆到底都是「模板填参 → 模型推理 → 取文本」。
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

NAME_PROMPT = PromptTemplate(
    input_variables=["product"],
    template=(
        "You are a brand naming consultant. Suggest one short, memorable company name "
        "for a company that makes {product}. "
        "Reply with the name and a one-sentence rationale. Keep it under 25 words."
    ),
)


@dataclass
class NamingResult:
    product: str
    name: str
    seconds: float


def build_chain(llm):
    """把模板、模型、解析器串成一条可复用的 Runnable。"""
    return NAME_PROMPT | llm | StrOutputParser()


def run_single(chain, product: str) -> str:
    return chain.invoke({"product": product}).strip()


def run_batch(chain, products: list[str]) -> list[str]:
    """LCEL 的 batch 会把多个输入并行送进链里，比 for 循环串行快很多。"""
    return [text.strip() for text in chain.batch([{"product": p} for p in products])]
