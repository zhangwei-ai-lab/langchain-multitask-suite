"""计算器工具：AST 白名单求值，替代裸 `eval`。

为什么不能用 eval + 字符白名单
------------------------------
「只允许数字和 +-*/()」这种正则白名单看着安全，实际上挡不住 `(1).__class__` 这类构造：
字符全在合法集合里，但对象属性链能一路走到 `__subclasses__()` 再拿到 `os`。
把计算接进 Agent 意味着「模型生成的表达式会真的被执行」，这里必须真隔离。

做法是把表达式解析成 AST，只允许「数字常量 + 四则/幂/取模 + 一元正负」，
其余节点（属性访问、下标、函数调用、推导式、lambda）一律拒绝，
再在 `{"__builtins__": {}}` 的空环境里求值。AST 层面拒绝了属性访问，
后面的 eval 就只剩纯算术。
"""

from __future__ import annotations

import ast

from langchain_core.tools import tool

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
    ast.Load,
)

# 表达式里允许出现的字符（先做一层廉价过滤，避免把超长垃圾串丢给 parser）
_SAFE_CHARS = r"[\d\s+\-*/.()%]+"


def safe_eval(expression: str) -> float:
    """只接受纯算术表达式，返回数值。非法表达式抛 ValueError。"""
    if not expression or not expression.strip():
        raise ValueError("表达式为空")

    tree = ast.parse(expression, mode="eval")

    def _check(node: ast.AST) -> None:
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"不支持的语法节点：{type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("表达式中只能出现数字常量")
        for child in ast.iter_child_nodes(node):
            _check(child)

    _check(tree)
    result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    return result  # type: ignore[return-value]


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，只接受数字、四则运算符、幂、取模和括号。

    涉及任何具体数字的换算或四则运算都应调用本工具，不要自己心算。

    参数:
        expression: 例如 "2+2"、"100/4"、"10.1*9/5+32"、"(32-32)*5/9"
    返回:
        计算结果；非法输入返回错误说明（不会抛异常打断 Agent）
    """
    import re

    if not re.fullmatch(_SAFE_CHARS, expression or ""):
        return f"错误：表达式 {expression!r} 含非法字符，只允许数字与 + - * / ( ) . % 和空格"
    try:
        value = safe_eval(expression)
    except Exception as exc:  # noqa: BLE001
        return f"错误：无法计算 {expression!r}（{exc}）"
    if isinstance(value, float):
        # 浮点表示会带出 60.980000000000004 这种尾巴；按 10 位有效数字输出，
        # 既不丢精度也不会把噪声塞给模型（模型有可能把它当"精确值"照抄进结论）
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return str(value)
