FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-chi-sim curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-rag.txt constraints-rag.txt ./
RUN pip install --no-cache-dir -r requirements-rag.txt && pip check
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/var && chown appuser:appuser /app/var
COPY studylens ./studylens
COPY app.py ./
COPY evaluate_rag.py ./
COPY evaluation/rag_cases.jsonl ./evaluation/rag_cases.jsonl
COPY data/sample_notes ./data/sample_notes
COPY .streamlit ./.streamlit
USER appuser
CMD ["python", "-m", "uvicorn", "studylens.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--limit-concurrency", "16"]
