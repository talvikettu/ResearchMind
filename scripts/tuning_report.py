"""Build a reviewable report from completed runs, never from planned measurements."""

import csv
import hashlib
import json
from pathlib import Path

from researchmind.tuning_capture import score


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percent(value):
    return f"{value * 100:.1f}%"


root = Path("reports/tuning")
rag = read(root / "answers/comparison.json")
samples = {s["id"]: s for s in read("examples/tuning_questions.json")}
before = {
    r["id"]: r for r in map(json.loads, (root / "baseline.jsonl").read_text(encoding="utf-8").splitlines())
}
after = {
    r["id"]: r
    for r in map(json.loads, (root / "answers/selected-test.jsonl").read_text(encoding="utf-8").splitlines())
}
for key, row in before.items():
    row["scores"] = score(row["result"]["answer"], samples[key])


def source_metrics(rows):
    ranks = [
        next(
            (
                i
                for i, c in enumerate(r["result"]["sources"][:5], 1)
                if c["source"] in samples[r["id"]]["sources"]
            ),
            None,
        )
        for r in rows
    ]
    return dict(
        reference_hit_at_5=sum(v is not None for v in ranks) / len(ranks),
        mrr_at_5=sum(1 / v if v else 0 for v in ranks) / len(ranks),
    )


retrieval = dict(
    baseline=source_metrics([r for r in before.values() if r["split"] == "test"]),
    selected=source_metrics(list(after.values())),
)

r0, r1 = rag["baseline_test"], rag["selected_test"]
document = [
    "# Подбор параметров RAG",
    "",
    "Эксперименты: 24–25 октября 2026.",
    "",
    "Это ограниченный внутренний эксперимент на одной машине, а не независимый benchmark. Настройки выбирались на dev/validation; итоговые результаты записаны отдельно. Увеличение длины не считалось самостоятельным улучшением.",
    "",
    "## Протокол",
    "",
    "- RAG: 24 RU/EN вопроса, 12 dev и 12 test. Проверены восемь исходных профилей retrieval и дополнительный профиль focused после анализа dev-ошибок с аббревиатурами.",
    "- Затем проверены шесть профилей ответа: прежний формат, структурированный 6K, структурированный 8K, структурированный 8K с глоссарием, source_first и grounded. Последний включает исправления выбора учебных разделов и отбора контекста; поэтому результат нельзя приписывать только изменению гиперпараметров. Профили добавлялись по найденным dev-ошибкам: это итеративная разработка, не заранее зарегистрированный слепой эксперимент.",
    "- Размер chunk 140 слов, embeddings, reranker и Qwen остались прежними. Корпус не переиндексировался, новые модели не скачивались, контейнеры не создавались.",
    "",
    "## RAG: итоговые 12 вопросов — кандидат отклонён",
    "",
    "| Диагностика | До | Экспериментальный grounded |",
    "|---|---:|---:|",
    f"| Предпочтительный размеченный источник в top-5 | {percent(retrieval['baseline']['reference_hit_at_5'])} | {percent(retrieval['selected']['reference_hit_at_5'])} |",
    f"| MRR@5 по этим источникам | {retrieval['baseline']['mrr_at_5']:.3f} | {retrieval['selected']['mrr_at_5']:.3f} |",
    f"| Покрытие аспектов ответа по regex-рубрике | {percent(r0['coverage'])} | {percent(r1['coverage'])} |",
    f"| Среднее число слов | {r0['words']:.1f} | {r1['words']:.1f} |",
    f"| Средняя доля повторённых 5-грамм ↓ | {percent(r0['repetition'])} | {percent(r1['repetition'])} |",
    f"| Наблюдаемая медиана задержки, с | {r0['median_seconds']:.1f} | {r1['median_seconds']:.1f} |",
    "",
    "Рубрика считает наличие необходимых понятий в самом ответе, а не в карточках источников; номера цитат исключены из проверки. **Это proxy полноты, не точность и не faithfulness.** Размеченные источники не исчерпывают все релевантные документы. RAG оценивается на маленькой внутренней выборке, а latency зависит от прогрева и повторного использования кэшей — это не изолированный speed benchmark.",
    "",
    "**Grounded не принят по умолчанию:** покрытие аспектов снизилось с 72,2% до 52,8%, хотя ответы стали длиннее. На вопрос без достаточных данных о TurboQuant и RTX 5090 режим выдал цитаты вместо корректного отказа. Проверка принадлежности цитат документу не гарантирует смысловую релевантность. В рабочем конфиге сохранены прежние параметры генерации и поиска; кандидат доступен для экспериментов.",
    "",
    "Отдельная проверка retrieval focused до генерации: reference Hit@5 вырос с 8/12 до 9/12, MRR@5 — с 0,569 до 0,653. Это улучшение поиска по ограниченной разметке, а не доказательство улучшения итоговых ответов; оно не переносится автоматически на всю систему.",
    "",
    "## Параметры отклонённого кандидата grounded",
    "",
    "```json",
    json.dumps(rag["selection"]["config"], ensure_ascii=False, indent=2),
    "```",
    "",
    "Распознавание явных методов и глоссарий расширяют поисковые термины, в том числе обозначения L1/L2, и помогают reranker связать названия с текстом. В глоссарии нет готовых ответов на контрольные вопросы. Содержательные утверждения по-прежнему требуют фрагментов источника.",
    "",
    "В экспериментальном режиме grounded Qwen выбирает ID фрагментов для аспектов вопроса, а содержательная часть ответа собирается программой из исходного текста. Формулы сохраняются. Из доказательств исключаются задания и варианты ответов учебных тестов. Это ограничивает добавления при свободном пересказе, но не доказывает релевантность каждого выбранного фрагмента. Для иностранных источников цитаты могут оставаться на исходном языке.",
    "При выборе учитывался просмотр dev-ответов: профили с высоким формальным покрытием отклонялись, если путали независимое и последовательное обучение, AdamW с другими оптимизаторами или случайные леса со случайными признаками. Проверка той же Qwen пропускала такие ошибки; её положительный verdict не считался достаточным основанием для внедрения.",
    "",
    "## Воспроизведение",
    "",
    "Перед экспериментами остановите нативный API, удерживающий локальный Qdrant. Готовые каталоги экспериментов защищены от перезаписи; используйте новые пути.",
    "",
    "```powershell",
    r".\stop.ps1",
    r".venv\Scripts\python.exe -m researchmind.tune_retrieval --output reports/tuning-repro/retrieval",
    r".venv\Scripts\python.exe -m researchmind.tune_answers --retrieval reports/tuning-repro/retrieval/selection.json --output reports/tuning-repro/answers --dev-only",
    "# После просмотра dev-ответов убрать --dev-only и указать --preferred-profile / --selection-note.",
    "```",
    "",
    "Для исходного сравнения также нужен baseline.jsonl, сохранённый работающим API до изменения конфигурации через researchmind.tuning_capture. После обновления новый baseline будет описывать уже текущую систему. При повторении указывайте соответствующий --baseline и фиксируйте версии кода, данных и .env.",
    "",
    "Сырые данные: reports/tuning/. Полные ответы до/после: reports/tuning/answers-comparison.md. Они остаются локальными; сводка в этом документе доступна в Git.",
    "",
    "## Ограничения вывода",
    "",
    "Небольшие настроечные выборки, адаптивное добавление профилей и участие разработчика в чтении исходных ответов ограничивают обобщаемость. Финальный test не использовался скриптами для выбора победителя, но это внутренний development study. Нельзя утверждать, что агент стал безошибочным или превзошёл другую систему на независимом benchmark.",
    "",
    "## Что осталось в рабочей программе",
    "",
    "Добавлены воспроизводимые скрипты HPO и сравнения ответов, настройки retrieval/генерации и экспериментальные профили. Проверка цитат повторно сопоставляет текст с источником, карточки используют подтверждённый фрагмент, когда он доступен. После Mistral smoke исправлен отдельный дефект extractive fallback: графические маркеры PDF больше не выдаются как объяснение. Эта последняя правка проверена регрессионным тестом; приведённые выше сохранённые benchmark-ответы после неё не пересчитывались. Ruff check прошёл. CPU/GPU Compose-конфигурации валидны; контейнеры и образы в этом эксперименте не создавались.",
    "",
]
mistral_path = root / "mistral/mistral_existing-dev.jsonl"
if mistral_path.exists():
    mistral = [json.loads(line) for line in mistral_path.read_text(encoding="utf-8").splitlines()]
    document += [
        "## Дополнительная проба Mistral 7B",
        "",
        "Повторно использована уже установленная mistral:instruct Q4_0: manifest скопирован, blobs подключены жёсткими ссылками без повторного скачивания 4,11 GB весов. На этой машине RTX 3050 4 GB / RAM 16 GB при загруженном retrieval свободная RAM снизилась примерно до 0,47 GB. Модель заняла около 5,10 GB суммарно, из них 2,32 GB VRAM при окне 6144.",
        "",
        "| Dev-вопрос | Покрытие рубрики | Время запроса, с | Модель доступна |",
        "|---|---:|---:|---|",
    ]
    for row in mistral:
        document.append(
            f"| {row['id']} | {percent(row['scores']['coverage'])} | {row['result']['latency_ms'] / 1000:.1f} | {row['result'].get('generation', {}).get('available', False)} |"
        )
    document += [
        "",
        "После двух завершённых вопросов проба остановлена: второй запрос превысил таймаут модели, fallback показал графический шум PDF. Третий начатый запрос прерван, итоговый test для Mistral не запускался. Первый ответ покрыл рубрику, но содержал повреждённое TeX-обозначение; формальное покрытие не равно качеству. Это smoke-проверка ресурсов и доступности, не сравнение качества моделей. Mistral не включена по умолчанию.",
        "",
    ]
Path("docs").mkdir(exist_ok=True)
Path("docs/TUNING.md").write_text("\n".join(document), encoding="utf-8")

full = [
    "# Ответы RAG: исходный режим и отклонённый кандидат grounded",
    "",
    "Сравниваются одинаковые 12 test-вопросов. Процент — автоматическое покрытие аспектов, не оценка правильности.",
    "",
]
with (root / "per-question.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["id", "question", "coverage_before", "coverage_after", "words_before", "words_after"],
    )
    writer.writeheader()
    for key, row in after.items():
        old = before[key]
        writer.writerow(
            dict(
                id=key,
                question=row["question"],
                coverage_before=old["scores"]["coverage"],
                coverage_after=row["scores"]["coverage"],
                words_before=old["scores"]["words"],
                words_after=row["scores"]["words"],
            )
        )
        full += [
            "## " + row["question"],
            "",
            "### До",
            "",
            old["result"]["answer"],
            "",
            "### Эксперимент grounded (не внедрён)",
            "",
            row["result"]["answer"],
            "",
            "Источники после настройки: " + "; ".join(a["title"] for a in row["result"]["articles"]),
            "",
        ]
(root / "answers-comparison.md").write_text("\n".join(full), encoding="utf-8")
manifest = dict(
    dataset_sha256=hashlib.sha256(Path("examples/tuning_questions.json").read_bytes()).hexdigest(),
    models=read("data/model-revisions.json"),
    corpus=read("data/index-status.json")["collection"],
    experimental_candidate_parameters=rag["selection"]["config"],
    deployed_parameters=read("examples/tuned_settings.json")["parameters"],
    rag_deployment="baseline retained; grounded candidate rejected after test",
)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Created docs/TUNING.md and reports/tuning/answers-comparison.md")
