FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY reme ./reme
COPY reme_ai ./reme_ai
COPY memory_api.py ./
COPY memory_manager_agent.py ./
COPY agent_memory_client.py ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "memory_api:app", "--host", "0.0.0.0", "--port", "8000"]
