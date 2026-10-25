"""Query-focused excerpts and article cards; preserve original citation IDs."""

import re
import threading
from functools import lru_cache

import snowballstemmer

from .ingest import tokens

STOP = set(
    "what how why does do is are the a an of to in and for with on by from this that explain compare "
    "как что почему чем это для при по на из или также мне расскажи объясни сравни про какие какой "
    "статьи статья найти найди метод методы используется работает about using used".split()
)
STOP.update(
    "can could should would will before after purpose important importance affect used use "
    "fail imply contribute number main help".split()
)
STOP.update("опиши описать алгоритм небольшое небольшое объяснение пожалуйста дай".split())
ALIASES = {
    "регуляризац": "regularization penalty",
    "переобуч": "overfitting",
    "признак": "features",
    "линейн": "linear",
    "вероятн": "probability",
    "кросс": "cross validation",
    "лассо": "lasso",
    "ридж": "ridge",
    "памят": "memory storage",
    "квантизац": "quantization quantizer",
    "обучен": "training learning",
    "скорост": "speed throughput",
    "качеств": "quality accuracy",
    "случайн": "random",
    "лес": "forest forests",
    "бустинг": "boosting xgboost",
    "вниман": "attention",
    "диффуз": "diffusion denoising",
    "трансформер": "transformer",
    "адаптер": "adapters",
    "сжат": "compression quantization",
    "искажен": "distortion",
    "вращен": "rotation",
}

_RUSSIAN = snowballstemmer.stemmer("russian")
_STEM_LOCK = threading.Lock()


def proof_excerpts(cards, sources, claims):
    """Show the passage actually used in the answer, not a fresh lexical excerpt."""
    quotes = {}
    for claim in claims:
        for proof in claim.get("evidence", []):
            citation = proof.get("citation")
            if type(citation) is not int or not 1 <= citation <= len(sources):
                continue
            source = sources[citation - 1]
            quote = proof.get("quote", "")
            if quote and " ".join(quote.split()) in " ".join(source["text"].split()):
                quotes.setdefault(source["source"], quote)
    return [
        dict(card, excerpt=quotes[card["source"]]) if card["source"] in quotes else card for card in cards
    ]


@lru_cache(maxsize=65536)
def lexical_word(word):
    if not re.search("[а-яё]", word):
        return word
    with _STEM_LOCK:
        return _RUSSIAN.stemWord(word)


@lru_cache(maxsize=8192)
def lexical_tokens(text):
    return tuple(lexical_word(word) for word in tokens(text))


def passage(text, question, max_words=180):
    """Return a contiguous multi-sentence source window, without invented transitions."""
    clean = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZА-Я0-9])", clean)
    terms = query_terms(question)
    windows = []
    for index in range(len(sentences)):
        window = " ".join(sentences[index : index + 3])
        window = " ".join(window.split()[:max_words])
        score = len(terms & set(lexical_tokens(window))) / max(len(terms), 1)
        if not reference_block(window):
            windows.append((score, window))
    return max(windows, key=lambda x: x[0], default=(0, ""))[1]


def layout_noise(text: str) -> bool:
    """PDF chart markers and repeated drawing symbols cannot explain a method."""
    return len(re.findall(r"[●■◆▪◦]", text)) > 8 or bool(re.search(r"([^\w\s])\1{15,}", text))


def reference_block(text: str) -> bool:
    years = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", text))
    publications = len(re.findall(r"journal|proceedings|conference|press|\bvol\b|\bpages\b", text, re.I))
    commands = len(re.findall(r"\$\s+(?:python\b|\./|pip\b|uv\b|cd\b)", text))
    return (
        layout_noise(text)
        or (years >= 3 and publications >= 2)
        or commands >= 2
        or text.count("Accuracy =") >= 2
    )


def raw_query_terms(question: str) -> set[str]:
    terms = set(tokens(question)) - STOP
    for term in list(terms):
        for prefix, english in ALIASES.items():
            if term.startswith(prefix):
                terms.update(english.split())
    return terms


def query_terms(question: str) -> set[str]:
    return {lexical_word(word) for word in raw_query_terms(question)}


def search_query(question: str) -> str:
    return " ".join(sorted(raw_query_terms(question))) or question


def excerpt(text: str, question: str) -> tuple[str, float]:
    clean = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZА-Я0-9])", clean)
    terms = query_terms(question)
    candidates = []
    for index, sentence in enumerate(sentences):
        # Bibliography and command-line logs are not explanatory evidence.
        if reference_block(sentence) or re.search(
            r"(?:Journal of|In (?:Proceedings|International)|editors,|pages \d|\$ ./|Accuracy =)", sentence
        ):
            continue
        words = sentence.split()
        # Long PDF blocks: score short contiguous windows instead of truncating away the answer.
        for start in range(0, max(1, len(words) - 25), 40):
            window = " ".join(words[start : start + 85])
            if len(words) < 45 and index + 1 < len(sentences):
                if not reference_block(sentences[index + 1]):
                    window += " " + sentences[index + 1]
            if reference_block(window):
                continue
            overlap = terms & set(lexical_tokens(window))
            score = len(overlap) / max(len(terms), 1) / max(len(window.split()), 12) ** 0.25
            if len(words) < 6:
                score *= 0.25
            candidates.append((score, window))
    score, selected = max(candidates, key=lambda pair: pair[0], default=(0, ""))
    # Excerpt always consists of contiguous source text, never generated paraphrases.
    return selected, score


def focused_contexts(contexts: list[dict], question: str, demo: bool) -> list[dict]:
    # A named-paper request should not return incidental mentions in other papers.
    named = set(tokens(question)) & {"turboquant", "qlora", "lora", "xgboost", "ddpm"}
    if named:
        matching = [c for c in contexts if named & set(tokens(c["source"]))]
        if matching:
            contexts = matching
    scored = [(excerpt(c["text"], question)[1], c) for c in contexts]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    threshold = scored[0][0] * 0.4 if scored else 0
    selected, counts, seen = [], {}, set()
    for score, context in scored:
        if demo and (score == 0 or score < threshold):
            continue
        key = context.get("document_id", context["source"])
        normalized = " ".join(tokens(context["text"]))
        if counts.get(key, 0) >= 2 or normalized in seen:
            continue
        counts[key] = counts.get(key, 0) + 1
        seen.add(normalized)
        selected.append(context)
    return selected[:10]


def article_cards(sources: list[dict], question: str, answer: str, metadata: dict) -> list[dict]:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    groups = {}
    for source in sources:
        if source["citation"] not in cited:
            continue
        key = source.get("document_id", source["source"])
        groups.setdefault(key, []).append(source)
    cards = []
    for group in groups.values():
        best = max(group, key=lambda s: excerpt(s["text"], question)[1])
        snippet, _ = excerpt(best["text"], question)
        info = metadata.get(best["source"], {})
        cards.append(
            dict(
                source=best["source"],
                title=best.get("title") or info.get("title", best["source"]),
                url=best.get("source_url") or info.get("page_url", ""),
                page=best["page"],
                excerpt=snippet,
                citations=[s["citation"] for s in group],
                citation=best["citation"],
                text=best["text"],
            )
        )
    return cards


def supported_extract(draft: str, contexts: list[dict]) -> bool:
    """Accept only verbatim cited extracts; token overlap is not factual verification."""
    claims = [line.strip() for line in draft.splitlines() if line.strip()]
    if not claims:
        return False
    for claim in claims:
        match = re.fullmatch(r"(.+?)\s*\[(\d+)\][.!]?", claim)
        if not match:
            return False
        text, ref = match.group(1).strip(), int(match.group(2))
        # Numeric citations copied from the paper's bibliography are not our IDs.
        # Fall back rather than presenting these as references to unrelated chunks.
        if re.search(r"\[\d+(?:\s*[,–—-]\s*\d+)*\]", text):
            return False
        if len(text.split()) < 5 or not 1 <= ref <= len(contexts):
            return False

        def normalize(s):
            return " ".join(s.lower().split())

        if normalize(text) not in normalize(contexts[ref - 1]["text"]):
            return False
    return True
