import os
from urllib.parse import quote

import httpx
import streamlit as st

from researchmind.presentation import render_rich_text

st.set_page_config(page_title="ResearchMind", page_icon="📚", layout="wide")
st.markdown(
    """<style>
.block-container {max-width:1100px;padding-top:2.5rem;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px;}
div[data-testid="stMetricValue"] {font-size:1.5rem;}
blockquote {border-left:3px solid #7c83fd!important;padding:0.5rem 1rem!important;}
.katex-display {overflow-x:auto;overflow-y:hidden;padding:0.3rem 0;}
</style>""",
    unsafe_allow_html=True,
)
st.title("📚 ResearchMind")
st.caption("Исследуйте документы, проверяйте источники и сравнивайте три режима RAG")
api = os.getenv("RESEARCH_API_URL", "http://127.0.0.1:8000")


def request(method, path, **kwargs):
    try:
        response = httpx.request(method, api + path, timeout=600, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as error:
        st.error(f"Ошибка API: {error}")
        return None


with st.sidebar:
    st.header("Библиотека")
    health = request("GET", "/health")
    if health:
        if health.get("answer_backend") == "ollama":
            st.success(f"Локальная модель ответов · {health['answer_model']}")
        st.caption(f"Поиск: {health['embeddings']} · reranker: {health.get('reranker', 'lexical')}")
        if not health["llm"] and health.get("answer_backend") != "ollama":
            st.info("Деморежим: извлечение цитат. Для синтеза ответа настройте LLM в .env.")
    files = st.file_uploader("PDF, TXT, Markdown", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Добавить документы", disabled=not files):
        for file in files:
            result = request("POST", "/documents", files={"file": (file.name, file.getvalue())})
            if result:
                st.success(f"{file.name}: {result['chunks']} фрагментов")
    docs = request("GET", "/documents")
    if docs:
        st.caption(f"В библиотеке: {len(docs)} документов")
        with st.expander("Все документы"):
            st.dataframe(
                [
                    {
                        "Документ": d.get("title", d["source"]),
                        "Язык": d.get("language", "—"),
                        "Тип": d.get("kind", "document"),
                        "Фрагментов": d["chunks"],
                    }
                    for d in docs
                ],
                hide_index=True,
            )

question = st.text_area("Вопрос к документам", placeholder="Опиши алгоритм линейной регрессии")
mode = st.radio("Режим", ["naive", "hybrid", "agentic"], index=2, horizontal=True)
format_label = st.selectbox("Формат ответа", ["Авто", "Объяснение", "Формула", "Сравнение", "Исследование"])
answer_mode = {
    "Авто": "auto",
    "Объяснение": "explain",
    "Формула": "formula",
    "Сравнение": "compare",
    "Исследование": "research",
}[format_label]
detail_label = st.select_slider(
    "Подробность ответа", options=["Кратко", "Обычно", "Развёрнуто"], value="Обычно"
)
detail = {"Кратко": "short", "Обычно": "standard", "Развёрнуто": "detailed"}[detail_label]
web = st.checkbox("Разрешить внешний поиск (нужен Tavily API key)")
if st.button("Исследовать", type="primary", disabled=len(question.strip()) < 3):
    with st.spinner("Ищу и проверяю источники…"):
        result = request(
            "POST",
            "/query",
            json=dict(question=question, mode=mode, allow_web=web, detail=detail, answer_mode=answer_mode),
        )
    if result:
        st.session_state.result = result
if "result" in st.session_state:
    result = st.session_state.result
    st.divider()
    st.caption("РЕЗУЛЬТАТ ИССЛЕДОВАНИЯ")
    st.subheader(result["question"])
    if result.get("generation", {}).get("available"):
        if not result["generation"].get("model_claims"):
            st.caption("Ответ собран из исходных фрагментов; их принадлежность источникам проверена.")
            st.caption("Цитаты сохраняют язык источника.")
        else:
            st.caption(f"Ответ локальной модели {result['generation']['model']} · с проверкой источников")
            st.caption(
                "Пересказ и его проверку выполняет одна модель; возможны неточности. Сверяйте важные выводы с источниками."
            )
        if result["generation"].get("source_fallback_claims"):
            st.caption("Некоторые блоки приведены дословно из источника.")
        if result["generation"].get("partial"):
            st.info("Для части запроса не хватило подтверждённого контекста; ответ может быть неполным.")
    elif result.get("generation"):
        st.warning("Модель ответов недоступна. Показаны выдержки из источников.")
    elif result.get("explanation"):
        st.caption("Краткое объяснение по разделу учебника · формулы и выдержки сверены с источником")
    elif result.get("demo"):
        st.info(
            "Найдены выдержки из статей. Это поиск по источникам; связный ответ и проверка утверждений доступны после подключения LLM."
        )
    with st.container(border=True):
        render_rich_text(result["answer"], st)
    if not result["verification"]["passed"]:
        st.warning("Проверка не пройдена: ответ требует проверки по источникам.")
    articles = result.get("articles", [])
    left, middle, right = st.columns(3)
    left.metric("Статей использовано", len(articles))
    middle.metric("Время поиска", f"{result['latency_ms'] / 1000:.2f} с")
    right.metric("Проверка", "Ссылки проверены" if result["verification"]["passed"] else "Нужна проверка")
    st.subheader("Статьи и фрагменты по вашему вопросу")
    if not articles:
        st.caption("Подходящих подтверждённых выдержек нет. Попробуйте уточнить тему или добавить документы.")
    for article in articles:
        with st.container(border=True):
            refs = " ".join(f"[{n}]" for n in article["citations"])
            st.markdown(f"**{refs} {article['title']}**")
            st.caption(f"{article['source']} · страница {article['page'] or '—'}")
            render_rich_text(article["excerpt"], st, quote=True)
            if article.get("url"):
                st.link_button("Страница публикации ↗", article["url"])
            if article["source"].lower().endswith(".pdf"):
                # Download through the API also works when UI and API run in different Docker services.
                if st.button("Подготовить PDF", key="pdf_" + article["source"]):
                    try:
                        pdf = httpx.get(api + "/papers/" + quote(article["source"], safe=""), timeout=30)
                        pdf.raise_for_status()
                        st.download_button(
                            "Скачать статью", pdf.content, article["source"], "application/pdf"
                        )
                    except httpx.HTTPError:
                        st.warning("Локальный PDF недоступен; используйте страницу публикации.")
            with st.expander("Полный найденный фрагмент"):
                render_rich_text(article["text"], st)
    with st.expander("Как получен ответ · технические детали"):
        st.json({"План": result.get("research_plan", []), "Лексическое покрытие": result.get("coverage", [])})
        if result.get("generation"):
            st.caption(
                "Проверяется принадлежность фрагментов источникам; это не независимая оценка полноты и правильности."
                if result["generation"].get("method") == "exact_source_spans"
                else "Проверка пересказа выполняется той же локальной моделью и может ошибаться."
            )
            st.json(result["generation"])
        st.json(result["trace"])
        st.json(result["usage"])
