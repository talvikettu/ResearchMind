from researchmind.presentation import rich_blocks


def test_formula_with_trailing_citation_does_not_swallow_following_sections():
    text = (
        "**Формула**\n\n$$\nf_w(x_i) = \\langle w,x_i \\rangle + w_0\n$$ [1]"
        "\n\n**Как обучается**\n\nИзменяем веса $w$. [1]"
        "\n\n**Ошибка модели**\n\nМинимизируем $\\frac{1}{N} \\sum_i(y_i-f_w(x_i))^2$."
    )
    blocks = list(rich_blocks(text))
    assert blocks[0] == ("markdown", "**Формула**")
    assert blocks[1] == ("latex", r"f_w(x_i) = \langle w,x_i \rangle + w_0")
    assert blocks[2][0] == "markdown"
    assert blocks[2][1].startswith("[1]\n\n**Как обучается**")
    assert "$\\frac{1}{N}" in blocks[2][1]


def test_bracket_math_and_inline_parentheses_keep_surrounding_prose():
    blocks = list(rich_blocks(r"Идея \(y=wx+b\). \[ L=\frac{1}{N}\sum_i e_i^2 \] [2]"))
    assert blocks == [
        ("markdown", "Идея $y=wx+b$."),
        ("latex", r"L=\frac{1}{N}\sum_i e_i^2"),
        ("markdown", "[2]"),
    ]


def test_math_delimiters_inside_code_stay_literal():
    text = 'Use `$$x$$` or:\n\n```python\nvalue = "$$y$$"\n```\n\n$$z^2$$'
    assert list(rich_blocks(text)) == [("markdown", text[: text.rfind("$$z")].strip()), ("latex", "z^2")]


def test_adjacent_equations_and_matrix_newlines_are_preserved():
    text = r"$$x=1$$$$\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}$$"
    assert list(rich_blocks(text)) == [
        ("latex", "x=1"),
        ("latex", r"\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}"),
    ]


def test_logistic_regression_without_math_delimiters_from_user_screenshot():
    formula = r"p = \frac{1}{1+e^{-\langle w,x_i \rangle}}"
    text = (
        "Формула логистической регрессии выглядит так: " + formula + " [1]\n\n**Как работает**\n\nОбучение."
    )
    assert list(rich_blocks(text)) == [
        ("markdown", "Формула логистической регрессии выглядит так:"),
        ("latex", formula),
        ("markdown", "[1]\n\n**Как работает**\n\nОбучение."),
    ]


def test_bare_math_stops_before_russian_or_english_explanation():
    for suffix in (" называется ошибкой.", " is the loss.", ". Далее текст."):
        blocks = list(rich_blocks(r"Loss: \frac{1}{N}\sum_{i=1}^{N}(y_i-w_i)^2" + suffix))
        assert blocks[1] == ("latex", r"\frac{1}{N}\sum_{i=1}^{N}(y_i-w_i)^2")
        assert blocks[2] == ("markdown", suffix.strip())


def test_existing_math_code_and_unbalanced_tex_are_unchanged():
    from researchmind.presentation import normalize_math

    text = r"Inline $\frac{1}{N}$, code `\frac{1}{N}` and broken \frac{1}{"
    assert normalize_math(text) == text
    assert list(rich_blocks(r"$$\frac{1}{N}$$ [1]"))[0] == ("latex", r"\frac{1}{N}")
