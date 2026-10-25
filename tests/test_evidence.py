from researchmind.config import Settings
from researchmind.engine import Engine
from researchmind.evidence import (
    article_cards,
    excerpt,
    focused_contexts,
    reference_block,
    search_query,
    supported_extract,
)
from researchmind.store import Store


def test_article_grouping_preserves_citation_and_best_page():
    sources = [
        dict(
            id="a",
            document_id="doc",
            source="paper.pdf",
            page=1,
            text="An introduction to language models.",
            citation=1,
        ),
        dict(
            id="b",
            document_id="doc",
            source="paper.pdf",
            page=4,
            text="Quantization reduces memory storage for frozen weights.",
            citation=2,
        ),
        dict(
            id="c", document_id="other", source="unused.pdf", page=1, text="Unrelated evidence.", citation=3
        ),
    ]
    cards = article_cards(sources, "quantization memory", "Claims [1] [2]", {})
    assert len(cards) == 1
    assert cards[0]["citations"] == [1, 2]
    assert cards[0]["page"] == 4
    assert cards[0]["excerpt"] in sources[1]["text"]


def test_query_focused_excerpt_and_unknown_topic():
    text = "A general introduction to models. Quantization reduces memory requirements for weights."
    snippet, score = excerpt(text, "квантизация памяти")
    assert snippet.startswith("Quantization") and score > 0
    assert "memory" in search_query("память при обучении")
    assert focused_contexts([dict(id="a", source="a", text=text)], "volcano geology", True) == []


def test_duplicate_chunks_and_document_cap():
    chunks = [
        dict(id=str(i), document_id="doc", source="paper", text=f"Memory optimization {i}.") for i in range(5)
    ]
    selected = focused_contexts(chunks + chunks, "memory optimization", True)
    assert len(selected) == 2


def test_named_paper_does_not_include_incidental_mentions():
    chunks = [
        dict(id="a", source="turboquant-2025.pdf", text="TurboQuant uses random rotation."),
        dict(id="b", source="other.pdf", text="TurboQuant is mentioned in related work."),
    ]
    assert [c["id"] for c in focused_contexts(chunks, "Explain TurboQuant", True)] == ["a"]


def test_bibliography_is_not_an_explanatory_excerpt():
    text = "In International Conference on Machine Learning, pages 12-19. Generalization predicts unseen examples."
    snippet, score = excerpt(text, "What is generalization?")
    assert "Generalization predicts" in snippet
    assert not snippet.startswith("In International")
    assert score > 0


def test_reference_and_command_blocks_are_filtered():
    assert reference_block("Journal of ML 2001. In Proceedings 2002. Conference on trees 2003.")
    assert reference_block("$ python easy.py data $ ./svm-train data")
    assert not reference_block("Ridge regression adds a squared coefficient penalty.")
    assert not reference_block("The model is $y=w^Tx$ and the loss is $L=(y-t)^2$ with regularization.")


def test_russian_inflections_match_source_evidence():
    snippet, score = excerpt(
        "Линейная регрессия обучается минимизацией ошибки.", "обучение линейной регрессии"
    )
    assert score > 0 and "регрессия" in snippet
    from researchmind.evidence import lexical_tokens, query_terms

    assert query_terms("регрессии") & set(lexical_tokens("регрессия"))


def test_pdf_chart_noise_cannot_become_a_fallback_answer():
    from researchmind.evidence import passage

    chart = "Lasso Step Coefficients " + "● " * 40 + "0.02 0.04"
    assert excerpt(chart, "Lasso coefficients") == ("", 0)
    assert passage(chart, "Lasso coefficients") == ""
    explanation = "Ridge regression shrinks coefficients with a squared penalty."
    snippet, score = excerpt(chart + ". " + explanation, "Ridge regression")
    assert snippet == explanation and score > 0


def test_extract_gate_rejects_unrelated_text_and_invented_citations():
    context = [{"text": "Regularization reduces overfitting by penalizing complex models."}]
    assert not supported_extract("Once upon a time there was a happy dog.", context)
    assert not supported_extract(
        "Regularization reduces overfitting by penalizing complex models. [9]", context
    )
    assert supported_extract("Regularization reduces overfitting by penalizing complex models. [1]", context)
    cited_context = [{"text": "Regularization reduces overfitting according to Smith [99]."}]
    assert not supported_extract(
        "Regularization reduces overfitting according to Smith [99]. [1]", cited_context
    )


def test_source_bibliography_numbers_do_not_become_answer_citations(tmp_path):
    store = Store(Settings(data_dir=tmp_path, _env_file=None))
    store.ingest("paper.txt", b"Regularization reduces overfitting as shown by Smith [99].")
    result = Engine(store).query("regularization overfitting")
    assert "[99]" not in result["answer"]
    assert result["verification"]["citation_ids_valid"]
    store.client.close()
