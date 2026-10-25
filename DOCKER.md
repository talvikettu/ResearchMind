# Docker

## Ограничение пользователя

**Не больше трёх контейнеров суммарно в Docker, включая уже существующие и остановленные.** Использовать существующие контейнеры, не создавать параллельные тестовые проекты. Постоянные инструкции для агента находятся в [AGENTS.md](AGENTS.md).

Compose теперь содержит ровно три сервиса: **app** (подготовка моделей/индекса, API и Streamlit UI), **qdrant**, **ollama**. Отдельные bootstrap/api/ui контейнеры больше не создаются.

## Запуск с проверкой лимита

В PowerShell из папки проекта:

```powershell
.\stop.ps1
.\start-docker.ps1
# Или NVIDIA GPU:
.\start-docker.ps1 -Gpu
```

Скрипт считает все существующие контейнеры через docker ps -a, включая остановленные. Если запуск превысит три контейнера, он прекращается до создания новых. Используется постоянное имя Compose-проекта researchmind, без дополнительных тестовых копий.

Сохранён один актуальный образ приложения **researchmind:local**. Для полноценного Docker-запуска также нужны два официальных служебных образа Qdrant и Ollama: после очистки они скачаются снова при запуске. Это зависимости, а не дополнительные версии приложения. Не обещается запуск всего стека из одного образа.

Для намеренной пересборки после изменения кода:

```powershell
.\start-docker.ps1 -Build
# С GPU:
.\start-docker.ps1 -Gpu -Build
```

Скрипт не удаляет чужие контейнеры и не очищает данные автоматически. Удаляйте только подтверждённые старые сборки этого проекта; не используйте глобальную очистку образов/томов.

Используется --no-recreate: существующие контейнеры переиспользуются без временного четвёртого контейнера при замене. Для применения новой сборки к уже созданным контейнерам сначала выполните docker compose -p researchmind down, затем запуск с -Build.

## Готовность и данные

```powershell
docker compose -p researchmind logs -f app
docker compose -p researchmind ps -a
```

В app сначала выполняется подготовка, затем API и UI. Интерфейс: http://127.0.0.1:8501, Swagger: http://127.0.0.1:8000/docs. Первый запуск с пустым кэшем требует интернета и может занять много времени.

- ./data:/app/data — документы, метаданные, HF-модели и кэш векторов.
- ./data/models/ollama:/root/.ollama/models — веса Qwen, общие с нативной Ollama проекта.
- Именованный том qdrant-data — серверный индекс, отдельный от нативного data/qdrant-semantic.

Поиск и reranking работают на CPU. GPU-вариант требует NVIDIA GPU/WSL2 в Docker Desktop. Проверка GPU: docker compose -p researchmind exec ollama ollama ps. На GPU с 4 ГБ перед генерацией выгрузите модель нативной Ollama, чтобы не держать две копии:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:11435/api/generate -ContentType application/json -Body '{"model":"qwen2.5:3b","keep_alive":0}'
```

Не запускайте одновременно нативную подготовку индекса и Docker-подготовку с записью в общий кэш. API/UI публикуются только на localhost; Qdrant и Ollama не имеют внешних портов. Для изменения портов до запуска:

```powershell
$env:RESEARCH_API_PORT='18000'
$env:RESEARCH_UI_PORT='18501'
.\start-docker.ps1
```

## Остановка

```powershell
docker compose -p researchmind down
# Если запускали с GPU:
docker compose -p researchmind -f compose.yaml -f compose.gpu.yaml down
```

down сохраняет данные. Не добавляйте -v, если не намерены удалять серверный индекс. Модели, статьи и тома при очистке образов не удалялись.

Python-зависимости закреплены в requirements.runtime.txt из uv.lock; CPU torch устанавливается отдельно. При изменении зависимостей обновите export uv с extra models, исключив torch/nvidia/triton, которые относятся к отдельной установке torch.

