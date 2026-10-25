"""Render source math separately from Markdown, without changing stored evidence."""

import re

_TOKENS = re.compile(
    r"(?P<code>```[^\n]*\n.*?^\s*```[^\n]*$|~~~[^\n]*\n.*?^\s*~~~[^\n]*$|`[^`\n]+`)"
    r"|(?P<display>\$\$.*?\$\$|\\\[.*?\\\])"
    r"|(?P<inline>\\\([^\n]*?\\\))"
    r"|(?P<dollar_inline>(?<!\\)\$(?!\$)[^\n$]*?(?<!\\)\$)",
    re.DOTALL | re.MULTILINE,
)

_BARE_COMMAND = re.compile(
    r"\\(?:frac|dfrac|tfrac|sqrt|sum|prod|int|langle|sigma|theta|alpha|beta|lambda|mathbb|mathrm)\b"
)


def _bare_end(text, start):
    """Scan a balanced expression; stop before prose, citations and Markdown."""
    depth, end, position = 0, start, start
    while position < len(text):
        char = text[position]
        if char == "\n" or (depth == 0 and (char in "$`:;!?" or text.startswith("**", position))):
            break
        if depth == 0 and re.match(r"\s*\[\d+\]", text[position:]):
            break
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                break
        elif char == "\\":
            command = re.match(r"\\(?:[A-Za-z]+|[,;! ])", text[position:])
            if not command:
                break
            position += len(command.group())
            end = position
            continue
        elif depth == 0:
            if char.isalpha():
                word = re.match(r"[A-Za-z]+", text[position:])
                if not word or (
                    len(word.group()) > 1 and word.group() not in {"exp", "log", "sin", "cos", "max", "min"}
                ):
                    break
                position += len(word.group())
                end = position
                continue
            if char == "." and not (
                position > start
                and text[position - 1].isdigit()
                and position + 1 < len(text)
                and text[position + 1].isdigit()
            ):
                break
            if not (char.isspace() or char.isdigit() or char in "_=+-*/^(),.|[]<>"):
                break
        position += 1
        if not char.isspace():
            end = position
    return end if depth == 0 else start


def _wrap_bare_math(text):
    result, position = [], 0
    for command in _BARE_COMMAND.finditer(text):
        if command.start() < position:
            continue
        start = command.start()
        # Include a plain left-hand side, e.g. p =, without consuming preceding prose.
        prefix = re.search(r"[A-Za-z][A-Za-z0-9_{}()^|,+]*\s*=\s*$", text[position:start])
        if prefix:
            start = position + prefix.start()
        end = _bare_end(text, start)
        if end <= command.end():
            continue
        result.extend((text[position:start], "$$" + text[start:end] + "$$"))
        position = end
    result.append(text[position:])
    return "".join(result)


def normalize_math(text):
    """Recover unmarked TeX only outside existing math and literal code spans."""
    parts, position = [], 0
    for match in _TOKENS.finditer(text):
        parts.extend((_wrap_bare_math(text[position : match.start()]), match.group()))
        position = match.end()
    parts.append(_wrap_bare_math(text[position:]))
    return "".join(parts)


def rich_blocks(text):
    """Keep prose/citations out of display math, including the common '$$ [1]' case."""
    text = normalize_math(text)
    markdown = []
    position = 0
    for match in _TOKENS.finditer(text):
        markdown.append(text[position : match.start()])
        token = match.group()
        if match.lastgroup == "display":
            if "".join(markdown).strip():
                yield "markdown", "".join(markdown).strip()
            markdown = []
            formula = token[2:-2].strip()
            if formula:
                yield "latex", formula
        elif match.lastgroup == "inline":
            markdown.append("$" + token[2:-2] + "$")
        else:
            markdown.append(token)
        position = match.end()
    markdown.append(text[position:])
    if "".join(markdown).strip():
        yield "markdown", "".join(markdown).strip()


def render_rich_text(text, target, quote=False):
    for kind, content in rich_blocks(text):
        if kind == "latex":
            target.latex(content)
        else:
            target.markdown("> " + content.replace("\n", "\n> ") if quote else content)
