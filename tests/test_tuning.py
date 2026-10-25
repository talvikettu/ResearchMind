import json

import pytest
from pydantic import ValidationError

from researchmind.answer_context import evidence_units, output_plan
from researchmind.config import Settings
from researchmind.store import Store
from researchmind.tuning_capture import score


def test_evidence_windows_are_exact_original_paragraphs():
    irrelevant = "The paper also describes image classification benchmarks with unrelated experiments."
    relevant = "Regularization adds a penalty to the objective and controls the coefficients of a model."
    context = dict(text=irrelevant + "\n\n" + relevant, retrieved_text=relevant)
    units = evidence_units(context, "How does regularization control coefficients?", max_chars=110)
    assert [text for _, text in units] == [relevant]
    assert all(text in context["text"] for _, text in units)


def test_aspects_describe_user_request_without_inventing_answers():
    plan = output_plan("Сравни методы по штрафу, отбору признаков и переобучению.", "compare")
    assert plan == ["штрафу", "отбору признаков", "переобучению"]
    assert "Formula" in output_plan("Explain a model", "explain")[1]


def test_short_latex_equation_is_not_discarded_as_a_short_paragraph():
    formula = r"$$L=\sum_i(y_i-wx_i)^2$$"
    text = "This is the squared error loss used for regression.\n\n" + formula
    units = evidence_units(dict(text=text, retrieved_text=text), "Formula of squared error")
    assert formula in [quote for _, quote in units]


def test_tuning_split_is_disjoint_and_contains_no_empty_rubrics():
    rows = json.load(open("examples/tuning_questions.json", encoding="utf-8"))
    assert len({r["id"] for r in rows}) == len(rows) == 24
    assert len({r["question"] for r in rows}) == 24
    assert {r["split"] for r in rows} == {"dev", "test"}
    assert all(r["rubric"] for r in rows)


def test_coverage_does_not_inspect_source_cards_and_exposes_missing_aspects():
    sample = dict(rubric={"first": ["L1"], "second": ["L2"]})
    assert score("L1 penalty", sample)["coverage"] == 0.5
    assert score("No evidence available", sample)["coverage"] == 0


def test_invalid_search_parameters_fail_early():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, dense_weight=1.5)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_k=0)


def test_explicit_entity_routing_excludes_unrelated_method_documents(tmp_path):
    store = Store(Settings(data_dir=tmp_path, entity_routing=True, _env_file=None))
    try:
        store.ingest("pca.txt", b"Principal component analysis finds directions of maximum variance.")
        store.ingest("svm.txt", b"Support vector machines learn a classification boundary.")
        rows = store.search("Explain PCA and its objective", "hybrid", 5)
        assert rows and all(row["source"] == "pca.txt" for row in rows)
    finally:
        store.client.close()


def test_rubric_numbers_do_not_match_citation_ids():
    assert score("A generic explanation [2]", dict(rubric={"squared": ["2"]}))["coverage"] == 0


def test_strict_entities_use_body_not_just_filenames_and_refresh_after_ingest(tmp_path):
    store = Store(
        Settings(data_dir=tmp_path, entity_routing=True, strict_entity_evidence=True, _env_file=None)
    )
    try:
        store.ingest("notes.txt", b"An L1 penalty promotes sparse model coefficients.")
        store.ingest("other.txt", b"A general discussion of unrelated machine learning models.")
        assert {r["source"] for r in store.search("Explain Lasso", "hybrid")} == {"notes.txt"}
        store.ingest("notes.txt", b"This document now covers unrelated datasets.")
        store.ingest("new.txt", b"The L1 penalty is an absolute norm of the coefficients.")
        assert {r["source"] for r in store.search("Explain Lasso", "hybrid")} == {"new.txt"}
    finally:
        store.client.close()


def test_quiz_choices_are_not_offered_as_explanatory_evidence():
    text = "Choose an option.\n\n(а) The first deliberately wrong choice.\n\n(б) Another deliberately wrong choice.\n\nRegularization adds a penalty to the model objective."
    units = evidence_units(dict(text=text), "Explain regularization", exclude_exercises=True)
    assert units and all("deliberately wrong" not in quote for _, quote in units)


def test_precise_teaching_definition_ignores_previous_method_background():
    from researchmind.teaching import explain

    text = "В предыдущих разделах мы строили случайный лес, где деревья обучаются независимо друг от друга.\n\nБустинг строит последовательность алгоритмов, каждый из которых исправляет ошибки предыдущей композиции."
    result = explain(
        dict(text=text, introduction=text, section_title="Градиентный бустинг", teaching_precise=True)
    )
    assert "последовательность" in result["claims"][0]["text"]
    assert "независимо" not in result["claims"][0]["text"]


def test_pdf_chart_glyphs_do_not_become_answer_units():
    chart = "●" * 100 + " 2 4 6 8 10 Step Coefficients Lasso"
    context = dict(text=chart, retrieved_text=chart)
    assert evidence_units(context, "Explain Lasso", exclude_exercises=True) == []


def test_article_card_shows_the_verified_answer_passage():
    from researchmind.evidence import proof_excerpts

    cards = [dict(source="book", excerpt="An irrelevant chart.")]
    sources = [dict(source="book", text="Introduction. The relevant definition is here.")]
    claims = [dict(evidence=[dict(citation=1, quote="The relevant definition is here.")])]
    assert proof_excerpts(cards, sources, claims)[0]["excerpt"] == "The relevant definition is here."
    assert proof_excerpts(cards, sources, [dict(evidence=[dict(citation=1, quote="Invented.")])]) == cards
