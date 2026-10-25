# Как отправить ResearchMind на GitHub

В Git отправляются исходники, тесты, README, Docker-конфигурация, скрипты запуска, lockfile и каталог источников. Статьи, модели, индексы, окружение Python, логи и личный `.env` остаются на компьютере.

## 1. Создать пустой репозиторий

Откройте [github.com/new](https://github.com/new), задайте имя, например `researchmind`, и выберите видимость Public или Private. Оставьте выключенными создание README, `.gitignore` и лицензии: файлы проекта будут отправлены с компьютера. Скопируйте HTTPS URL созданного репозитория.

## 2. Открыть PowerShell в папке проекта

Для текущего расположения:

```powershell
Set-Location -LiteralPath 'C:\Users\Denchik\OneDrive\Рабочий стол\agentic-rag-research-assistant'
git init -b main
```

Эта инструкция рассчитана на первую публикацию: сейчас в этой папке Git ещё не инициализирован. В другом расположении замените путь на свой.

Если при создании коммита Git попросит имя и email, настройте их только для этого проекта:

```powershell
git config user.name "ВАШЕ ИМЯ"
git config user.email "ВАШ EMAIL ИЛИ GITHUB NOREPLY EMAIL"
```

## 3. Добавить только файлы проекта

Выполните одну команду. Здесь явно перечислены разрешённые файлы и каталоги; `git add .` не требуется.

```powershell
git add -- .gitignore .dockerignore .env.example AGENTS.md README.md GITHUB.md DOCKER.md Dockerfile compose.yaml compose.gpu.yaml pyproject.toml uv.lock requirements.runtime.txt ui.py start.ps1 stop.ps1 start-docker.ps1 sources.txt researchmind tests scripts docs examples
```

Каталоги `researchmind`, `tests`, `scripts`, `docs` и `examples` добавляются рекурсивно с учётом `.gitignore`. В частности, вложенные `__pycache__` не попадут в коммит. `.env.example` включается как шаблон с пустыми ключами, настоящий `.env` — исключён.

Перед коммитом посмотрите подготовленный список и краткую сводку:

```powershell
git diff --cached --name-only
git diff --cached --stat
git status --short
```

В списке должны быть код и вспомогательные файлы. Там не должно быть `data/`, `reports/`, `.venv/`, `.idea/`, `.env`, весов моделей или скачанных PDF. Также исключены локальные `requirements.export.txt` и `qdrant-image-manifest.json`; необходимые `uv.lock` и `requirements.runtime.txt` включены.

## 4. Создать коммит и отправить

В следующей команде замените `YOUR_USERNAME` и `YOUR_REPOSITORY` на значения из URL своего репозитория:

```powershell
git commit -m "Initial commit: ResearchMind RAG assistant"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git remote -v
git push -u origin main
```

При запросе авторизации выполните вход в GitHub выбранным способом. Не вставляйте токен в исходники или URL remote. После успешного `push` обновите страницу репозитория: GitHub покажет README.

Если `origin` уже существует, сначала проверьте `git remote -v`; для исправления адреса используйте `git remote set-url origin URL_ВАШЕГО_РЕПОЗИТОРИЯ`.

## Следующие обновления

Повторите команду `git add -- ...` из шага 3 — она также учитывает изменения и удаления внутри перечисленных каталогов. Затем:

```powershell
git diff --cached --stat
git commit -m "Update ResearchMind"
git push
```

Если какой-то лишний файл уже был добавлен в индекс, `.gitignore` сам его не уберёт. До первого коммита можно снять конкретный файл командой `git rm --cached -- ПУТЬ_К_ФАЙЛУ` (для папки — `git rm -r --cached -- ПУТЬ_К_ПАПКЕ`): локальные данные сохранятся. Уже опубликованные секреты требуют отдельного удаления из истории и замены ключа.

## Как другой человек получит документы

Модели и библиотека документов не хранятся в Git. После клонирования можно выполнить команды из README:

```powershell
uv sync --frozen
uv run python scripts/download_sources.py
```

Список материалов опубликован в `sources.txt`. Для полноценного запуска с подготовкой готовых моделей и индекса используется `start.ps1` или Docker по README.

Основа инструкции: [публикация локального проекта — GitHub Docs](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github), [команда git add](https://git-scm.com/docs/git-add/).
