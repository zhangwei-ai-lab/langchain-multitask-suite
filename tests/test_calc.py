"""计算器安全边界：这里每一条断言，都是「模型生成的表达式会被真的执行」这个前提换来的。"""

from __future__ import annotations

import pytest

from app.tools.calc import calculator, safe_eval


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+2", 4),
        ("100/4", 25),
        ("10.1*9/5+32", 50.18),
        ("(10.1*9/5+32)/4", 12.545),
        ("2**10", 1024),
        ("10%3", 1),
        ("-3+1", -2),
    ],
)
def test_safe_eval_allows_arithmetic(expression, expected):
    assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "(1).__class__",  # 属性访问：白名单外的节点
        "__import__('os').system('echo hi')",  # 函数调用
        "[x for x in range(3)]",  # 推导式
        "lambda: 1",  # lambda
        "a+1",  # 名称引用
        "'abc'",  # 非数字常量
    ],
)
def test_safe_eval_rejects_escape_attempts(expression):
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_safe_eval_rejects_empty():
    with pytest.raises(ValueError):
        safe_eval("   ")


def test_calculator_returns_integer_when_whole():
    assert calculator.invoke({"expression": "8/4"}) == "2"


def test_calculator_reports_illegal_chars_instead_of_raising():
    result = calculator.invoke({"expression": "os.system('rm -rf /')"})
    assert result.startswith("错误") and "非法字符" in result
