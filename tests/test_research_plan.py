from researchmind.handbook import chapter_text
from researchmind.research_plan import coverage, facets, select_evidence


def test_comparison_plan_and_gaps():
    plan = facets("Compare Lasso and Ridge regularization", "detailed")
    assert {p["label"] for p in plan} == {"Lasso", "Ridge"}
    assert "Ridge" not in plan[0]["query"] and "Lasso" not in plan[1]["query"]
    result = coverage([{"label": "missing", "query": "unseen volcano geology"}], [])
    assert not result[0]["covered"]


def test_near_duplicates_do_not_fill_evidence_budget():
    contexts = [
        dict(
            id=str(i), source="a", document_id="a", text="Ridge regression applies squared weight penalties."
        )
        for i in range(5)
    ]
    contexts.append(
        dict(
            id="other",
            source="b",
            document_id="b",
            text="Lasso regression uses an absolute coefficient penalty.",
        )
    )
    result = select_evidence(contexts, "regression penalty")
    assert len(result) == 2


def test_handbook_extract_preserves_math_and_removes_navigation():
    html = '<nav>Wrong navigation</nav><main class="dc-doc-page__content"><h1>Linear models</h1><p>A linear model estimates a weighted sum of input features and an intercept. This is educational content. The regression coefficients are estimated using a loss function and can be regularized to reduce overfitting.</p><p><span class="katex"><span>duplicate</span><annotation encoding="application/x-tex">y=w^Tx+b</annotation></span></p></main>'
    title, text = chapter_text(html)
    assert title == "Linear models"
    assert "$y=w^Tx+b$" in text
    assert "Wrong navigation" not in text and "duplicate" not in text
    _, native = chapter_text(
        html.replace(
            '<p><span class="katex"><span>duplicate</span><annotation encoding="application/x-tex">y=w^Tx+b</annotation></span></p>',
            '<p class="yfm-latex" data-content="y%3Dw%5ETx%2Bb"></p>',
        )
    )
    assert "$y=w^Tx+b$" in native


def test_detailed_question_prefers_topic_chapter_over_incidental_mention():
    contexts = [
        dict(
            id="a",
            source="ensemble",
            title="Ансамбли",
            text="В ансамблях иногда применяется кросс-валидация.",
        ),
        dict(
            id="b",
            source="cv",
            title="Кросс-валидация",
            text="Кросс-валидация оценивает качество модели на разных частях выборки.",
        ),
    ]
    result = select_evidence(contexts, "Что такое кросс-валидация?", prefer_titles=True)
    assert result[0]["source"] == "cv"
