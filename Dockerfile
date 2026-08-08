FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flask gunicorn tokenizers huggingface_hub numpy python-dotenv

COPY . .

ENV PORT=10000

CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --preload
