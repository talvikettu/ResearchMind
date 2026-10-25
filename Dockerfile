FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY requirements.runtime.txt ./
RUN pip install --no-cache-dir -r requirements.runtime.txt
COPY researchmind ./researchmind
RUN pip install --no-cache-dir --no-deps .
COPY examples ./examples
COPY sources.txt ./
COPY scripts/download_sources.py ./scripts/download_sources.py
COPY ui.py ./
RUN useradd --create-home --uid 10001 app && mkdir -p /app/data && chown -R app:app /app
USER app
EXPOSE 8000 8501
CMD ["python", "-m", "researchmind.container"]
