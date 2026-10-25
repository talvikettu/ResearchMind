import json
from types import SimpleNamespace

import httpx
import numpy as np

from researchmind.composer import Composer, intent
from researchmind.config import Settings
from researchmind.engine import Engine
from researchmind.ingest import prepare_document
from researchmind.research_plan import select_evidence
from researchmind.store import Store


def test_hierarchy_and_math_survive_chunk_boundaries():
    equation = r"$y = w_0 + w_1 x_1 + w_2 x_2$"
    text = "# Models\n\n## Regression\n\n" + "word " * 18 + equation + " word" * 30
    text += "\n\n### Loss\n\n$$\nL = (y - f(x))^2\n$$\n\n```python\nx = 2\n```"
    result = prepare_document("paper.md", text.encode(), size=20, overlap=4)
    assert any(equation in c["text"] for c in result["chunks"])
    assert any(c["section_path"] == ["Models", "Regression", "Loss"] for c in result["chunks"])
    assert any("code" in c["block_types"] for c in result["chunks"])
    assert all(c["text"].count("$") % 2 == 0 for c in result["chunks"])
    parent_ids = {p["id"] for p in result["sections"]}
    assert all(c["parent_id"] in parent_ids for c in result["chunks"])


def test_e5_document_and_query_prefixes_are_different():
    store = Store.__new__(Store)
    store.settings = Settings(_env_file=None, embedding_model="intfloat/multilingual-e5-small")
    seen = []
    store.encoder = SimpleNamespace(encode=lambda texts, **kw: seen.extend(texts) or np.array([[1.0, 0.0]]))
    store.embed(["document"])
    store.embed(["question"], query=True)
    assert seen == ["passage: document", "query: question"]


def test_semantic_reranker_retains_cross_language_evidence():
    rows = [
        dict(id="en", source="english", text="Regression predicts a continuous target.", reranker_score=0.9),
        dict(id="ru", source="ru", text="Линейная регрессия упоминается случайно.", reranker_score=0.1),
    ]
    assert select_evidence(rows, "Как предсказать числовое значение?")[0]["id"] == "en"


def test_embedding_cache_reused_but_metadata_changes_invalidate(tmp_path):
    store = Store(Settings(data_dir=tmp_path, _env_file=None))
    calls = []
    original = store.embed
    store.embed = lambda texts, **kw: calls.append(texts) or original(texts, **kw)
    raw = b"Regression predicts a continuous numeric value using input features."
    store.ingest("regression.txt", raw, {"title": "Regression"})
    store.ingest("regression.txt", raw, {"title": "Regression"})
    assert len(calls) == 1
    store.ingest("regression.txt", raw, {"title": "New title"})
    assert len(calls) == 2
    chunk = store.all_chunks()[0]
    assert store.parent(chunk)["text"] == raw.decode()
    store.client.close()


def model(answers, tmp_path):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        item = answers[min(len(requests) - 1, len(answers) - 1)]
        if isinstance(item, int):
            return httpx.Response(item, text="Unavailable")
        return httpx.Response(
            200, json={"message": {"content": json.dumps(item)}, "prompt_eval_count": 10, "eval_count": 5}
        )

    return Composer(
        Settings(_env_file=None, data_dir=tmp_path), transport=httpx.MockTransport(handle)
    ), requests


CONTEXTS = [
    dict(
        source="paper", text="Regression predicts a continuous target. Coefficients minimize squared errors."
    )
]
CLAIM = dict(
    heading="Суть",
    text="Регрессия предсказывает числовое значение.",
    evidence=[dict(citation=1, quote="Regression predicts a continuous target.")],
)


def test_paraphrase_requires_both_exact_quote_and_verifier(tmp_path):
    composer, requests = model(
        [dict(claims=[CLAIM], missing=""), dict(verdicts=[dict(id=0, supported=True, reason="Supported")])],
        tmp_path,
    )
    result = composer.compose("Что такое регрессия?", CONTEXTS)
    assert result["passed"] and result["answer"].endswith("[1]")
    assert result["usage"]["llm_calls"] == 2 and len(requests) == 2
    assert "EVIDENCE" in json.loads(requests[1]["messages"][1]["content"])


def test_fabricated_quote_is_not_sent_to_semantic_verifier(tmp_path):
    bad = dict(CLAIM, evidence=[dict(citation=1, quote="Invented result with 99 percent accuracy")])
    composer, requests = model([dict(claims=[bad], missing="")], tmp_path)
    result = composer.compose("Что такое регрессия?", CONTEXTS)
    assert not result["passed"] and len(requests) == 1
    assert "99" not in result["answer"]


def test_missing_verdict_never_means_support(tmp_path):
    composer, _ = model([dict(claims=[CLAIM], missing=""), dict(verdicts=[])], tmp_path)
    result = composer.compose("Что такое регрессия?", CONTEXTS)
    assert not result["passed"] and result["rejected_claims"]


def test_wrong_source_id_is_rejected(tmp_path):
    claim = dict(CLAIM, evidence=[dict(citation=99, quote=CONTEXTS[0]["text"])])
    composer, _ = model([dict(claims=[claim], missing="")], tmp_path)
    assert not composer.compose("Что такое регрессия?", CONTEXTS)["passed"]


def test_unavailable_model_retains_explicit_extractive_fallback(tmp_path):
    settings = Settings(data_dir=tmp_path, _env_file=None, answer_backend="ollama")
    store = Store(settings)
    store.ingest("paper.txt", CONTEXTS[0]["text"].encode())
    engine = Engine(store)
    engine.composer, _ = model([503], tmp_path)
    result = engine.query("Regression continuous target")
    assert result["generation"]["available"] is False
    assert "Regression predicts" in result["answer"]
    assert result["answer_kind"] == "source_extracts"
    store.client.close()


def test_invalid_top_level_output_is_a_failure_not_a_crash(tmp_path):
    composer, _ = model([[]], tmp_path)
    assert not composer.compose("Explain regression", CONTEXTS)["available"]


def test_intents_and_explicit_format():
    assert intent("Сравни Lasso и Ridge") == "compare"
    assert intent("Формула MSE") == "formula"
    assert intent("Explain a classifier") == "explain"
    assert intent("What is regression?", "research") == "research"


def test_rechunk_same_content_removes_obsolete_points(tmp_path):
    store = Store(Settings(data_dir=tmp_path, _env_file=None, chunk_words=50))
    raw = ("Regression predicts continuous targets from features. " * 100).encode()
    first = store.ingest("paper.txt", raw)
    store.settings.chunk_words = 120
    second = store.ingest("paper.txt", raw)
    assert first["chunks"] > second["chunks"]
    assert len(store.all_chunks()) == second["chunks"]
    store.client.close()


def test_array_indices_in_formula_are_not_removed_as_citations(tmp_path):
    claim = dict(CLAIM, text="$$f(x[0]) = w x[0] + b$$")
    composer, _ = model(
        [dict(claims=[claim], missing=""), dict(verdicts=[dict(id=0, supported=True, reason="Supported")])],
        tmp_path,
    )
    result = composer.compose("Формула регрессии", CONTEXTS)
    assert "x[0]" in result["answer"] and result["answer"].endswith("[1]")


def test_preselected_unit_supplies_exact_quote_and_invalid_unit_fails(tmp_path):
    claim = dict(CLAIM, evidence=[dict(citation=1, unit="S1.1")])
    composer, requests = model(
        [dict(claims=[claim], missing=""), dict(verdicts=[dict(id=0, supported=True, reason="Supported")])],
        tmp_path,
    )
    result = composer.compose("Что такое регрессия?", CONTEXTS)
    assert result["passed"]
    assert result["accepted_claims"][0]["evidence"][0]["quote"] == CONTEXTS[0]["text"]
    assert requests[0]["format"]["properties"]["claims"]["items"]["properties"]["evidence"]["items"][
        "properties"
    ]["unit"]["enum"] == ["S1.1"]
    claim = dict(CLAIM, evidence=[dict(citation=1, unit="S99.99")])
    composer, _ = model([dict(claims=[claim], missing="")], tmp_path)
    assert not composer.compose("Что такое регрессия?", CONTEXTS)["passed"]


def test_handbook_formula_is_not_lost_when_model_omits_it(tmp_path):
    text = "Мы хотим моделировать зависимость переменной от признаков как линейную функцию со свободным членом.\n\n$f(x) = wx + b$\n\nМы минимизируем ошибку модели, изменяя обучаемые параметры и веса на данных."
    contexts = [
        dict(
            source="book",
            text=text,
            introduction=text,
            teaching_section=True,
            section_title="Линейная регрессия",
        )
    ]
    composer, _ = model([dict(claims=[], missing="Not enough evidence")], tmp_path)
    result = composer.compose("Опиши линейную регрессию", contexts)
    assert result["passed"] and "$$" in result["answer"] and "минимизируем" in result["answer"]
    assert result["model_claims"] == 0 and result["source_fallback_claims"] >= 3
    assert result["method"] == "exact_source_spans"


def test_legacy_font_repair_is_reversible_and_does_not_touch_english():
    from researchmind.ingest import repair_pdf_text

    original = "Лекция Соколов. Методы обучения для данных. Введение в методы классификации. " * 3
    broken = original.encode("cp1251").decode("latin1")
    assert repair_pdf_text(broken) == original
    english = "A survey of regression and classification models." * 3
    assert repair_pdf_text(english) == english


def test_identical_accepted_claims_are_rendered_once(tmp_path):
    composer, _ = model(
        [
            dict(claims=[CLAIM, CLAIM], missing=""),
            dict(verdicts=[dict(id=0, supported=True), dict(id=1, supported=True)]),
        ],
        tmp_path,
    )
    result = composer.compose("Explain regression", CONTEXTS)
    assert len(result["accepted_claims"]) == 1
    assert result["model_claims"] == 1


def test_structured_verifier_cannot_approve_a_claim_from_another_batch(tmp_path):
    claims = [
        dict(CLAIM, text=text, evidence=[dict(citation=1, unit="S1.1")])
        for text in (
            "Regression predicts a continuous target.",
            "Regression uses model coefficients.",
            "Regression has a loss function.",
        )
    ]
    composer, requests = model(
        [
            dict(claims=claims, missing=""),
            dict(
                verdicts=[
                    dict(id=0, supported=True, relevant=True),
                    dict(id=1, supported=True, relevant=True),
                ]
            ),
            dict(verdicts=[dict(id=0, supported=True, relevant=True)]),
        ],
        tmp_path,
    )
    composer.settings.answer_profile = "structured"
    result = composer.compose("Explain regression", CONTEXTS)
    assert len(result["accepted_claims"]) == 2
    assert any(row["id"] == 2 for row in result["rejected_claims"])
    assert requests[-1]["format"]["properties"]["verdicts"]["items"]["properties"]["id"]["enum"] == [2]


def test_a_supported_but_irrelevant_fact_is_rejected_in_structured_mode(tmp_path):
    composer, _ = model(
        [
            dict(claims=[dict(CLAIM, evidence=[dict(citation=1, unit="S1.1")])], missing=""),
            dict(
                verdicts=[dict(id=0, supported=True, relevant=False, reason="Does not answer the question")]
            ),
        ],
        tmp_path,
    )
    composer.settings.answer_profile = "structured"
    result = composer.compose("Explain regression", CONTEXTS)
    assert not result["passed"]
    assert not result["accepted_claims"]


def test_source_first_never_renders_the_models_invented_paraphrase(tmp_path):
    composer, _ = model(
        [
            dict(
                claims=[
                    dict(
                        CLAIM,
                        text="A fabricated statement about independent boosting trees.",
                        evidence=[dict(citation=1, unit="S1.1")],
                    )
                ],
                missing="",
            ),
            dict(verdicts=[dict(id=0, supported=True, relevant=True)]),
        ],
        tmp_path,
    )
    composer.settings.answer_profile = "structured"
    composer.settings.answer_source_first = True
    result = composer.compose("Explain regression", CONTEXTS)
    assert "fabricated" not in result["answer"]
    assert CONTEXTS[0]["text"] in result["answer"]
    assert result["model_claims"] == 0 and result["source_fallback_claims"] == 1


def test_grounded_selection_ignores_invented_text_and_unknown_units(tmp_path):
    composer, _ = model(
        [dict(selections=[dict(aspect=0, units=["S99.1", "S1.1"])], missing="", text="Invented false facts")],
        tmp_path,
    )
    composer.settings.answer_profile = "grounded"
    result = composer.compose("Explain regression", CONTEXTS)
    assert result["passed"] and result["usage"]["llm_calls"] == 1
    assert CONTEXTS[0]["text"] in result["answer"]
    assert "Invented" not in result["answer"] and "S99" not in result["answer"]
    assert result["model_claims"] == 0


def test_final_verification_rechecks_quote_content_not_only_citation_id():
    engine = Engine.__new__(Engine)
    result = engine.verify(
        dict(
            contexts=[dict(text="The original source.")],
            trace=[],
            generation=dict(
                available=True,
                passed=True,
                partial=False,
                method="exact_source_spans",
                accepted_claims=[dict(evidence=[dict(citation=1, quote="A fabricated source quote.")])],
            ),
        )
    )
    assert result["verification"]["citation_ids_valid"]
    assert not result["verification"]["passed"]


def test_grounded_bibliography_numbers_are_not_app_citations(tmp_path):
    contexts = [
        dict(source="paper", text="A regression model predicts continuous targets [8] and uses $x[0]$.")
    ]
    composer, _ = model([dict(selections=[dict(aspect=0, units=["S1.1"])], missing="")], tmp_path)
    composer.settings.answer_profile = "grounded"
    result = composer.compose("Explain regression", contexts)
    assert "[8]" not in result["answer"] and "x[0]" in result["answer"]
    assert result["answer"].endswith("[1]")
