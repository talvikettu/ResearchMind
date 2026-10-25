import hashlib
from types import SimpleNamespace

from researchmind.config import Settings
from researchmind.engine import Engine
from researchmind.teaching import explain, explanation_topic, lookup, verify

CONTENT = r"""# Линейные модели

## Оптимизация гиперпараметров

Алгоритм выбирает classifier и linear kernel с помощью EI.

## Линейная регрессия и метод наименьших квадратов

Мы хотим моделировать зависимость $y$ от $x$ как линейную функцию со свободным членом.

Общий вид такой функции выглядит следующим образом:

$f(x) = wx + b.$

### Обучение

Мы минимизируем ошибку модели, меняя обучаемые параметры. В нашем примере обучаемые параметры — веса $w$.

Функция потерь $L = (y-f(x))^2$ называется среднеквадратической ошибкой (MSE). Следующее замечание не нужно в кратком ответе.

## Обобщённые линейные модели

Это другой раздел, в котором обсуждается достаточно широкий класс вероятностных моделей.
"""


def library(tmp_path, content=CONTENT):
    folder = tmp_path / "handbook"
    folder.mkdir()
    raw = content.encode()
    (folder / "chapter.md").write_bytes(raw)
    row = dict(
        id="chunk",
        source="chapter.md",
        kind="handbook",
        title="Линейные модели",
        document_id=hashlib.sha256(raw).hexdigest(),
        source_url="https://example.org/chapter",
        page=1,
    )
    return SimpleNamespace(settings=Settings(data_dir=tmp_path, _env_file=None), all_chunks=lambda: [row])


def test_original_user_query_returns_definition_formula_and_loss(tmp_path):
    store = library(tmp_path)
    context = lookup(store, "Опиши мне алгоритм линейной регрессии")
    result = explain(context)
    assert context["section_title"].startswith("Линейная регрессия")
    assert "f(x) = wx + b" in result["answer"]
    assert "минимизируем" in result["answer"] and "MSE" in result["answer"]
    assert "kernel" not in result["answer"] and "вероятностных" not in result["answer"]
    assert "Следующее замечание" not in result["answer"]
    assert verify(result["answer"], result, [context])
    assert not verify(result["answer"] + " Invented statement", result, [context])
    result["claims"][0]["text"] = "An unsupported new claim"
    from researchmind.teaching import render

    assert not verify(render(result["claims"]), result, [context])


def test_cached_but_not_indexed_content_is_not_used(tmp_path):
    store = library(tmp_path)
    (tmp_path / "handbook" / "chapter.md").write_text(CONTENT + " Changed", encoding="utf-8")
    assert lookup(store, "Опиши линейную регрессию") is None


def test_definition_uses_chapter_introduction_not_time_series_subsection(tmp_path):
    store = library(
        tmp_path,
        """# 3.3. Кросс-валидация

Автор: Тест

## Введение

Кросс-валидация — это процедура оценки качества модели. Она помогает сравнивать модели на различных частях данных.

## Применение

В этом разделе мы обсуждаем общие особенности применения процедуры на разных данных.

### Кросс-валидация для временных рядов

Это специальный способ разбиения данных во времени. Значения во временных рядах нельзя перемешивать.
""",
    )
    context = lookup(store, "Что такое кросс-валидация?")
    result = explain(context)
    assert context["section_title"] == "3.3. Кросс-валидация"
    assert "процедура оценки качества" in result["answer"]
    assert "нельзя перемешивать" not in result["answer"]


def test_explanation_keeps_source_cards_and_uses_guided_extracts(tmp_path):
    store = library(tmp_path)
    engine = Engine(store)
    result = engine.query("Опиши мне алгоритм линейной регрессии", detail="detailed")
    assert result["answer_kind"] == "guided_extracts"
    assert result["verification"]["passed"]
    assert result["verification"]["method"] == "structured_source_check"
    assert result["usage"]["llm_calls"] == 0
    assert len(result["articles"]) == 1
    assert result["articles"][0]["url"] == "https://example.org/chapter"


def test_comparisons_are_not_misrouted_and_formula_only_request_is_supported():
    assert explanation_topic("Сравни линейную и логистическую регрессии") is None
    assert explanation_topic("Формула линейной регрессии") == explanation_topic("Опиши линейную регрессию")


def test_base_method_wins_over_special_case_with_acronym(tmp_path):
    paragraph = "Мы обновляем параметры, чтобы уменьшить ошибку модели на обучающих данных. " * 6
    content = (
        "## Стохастический градиентный спуск\n\n"
        + paragraph
        + "\n\n## Градиентный спуск (GD)\n\n"
        + paragraph
    )
    store = library(tmp_path, content)
    assert lookup(store, "Объясни градиентный спуск")["section_title"] == "Градиентный спуск (GD)"
