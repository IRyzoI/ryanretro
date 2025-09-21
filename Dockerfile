FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your whole project (static/, data/, app.py, etc.)
COPY . .

ENV PYTHONUNBUFFERED=1
# Railway provides $PORT at runtime; fall back to 8000 locally
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]