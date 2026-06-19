FROM python:3.11-slim

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY app.py .
COPY artifacts/ ./artifacts/

EXPOSE 5001

CMD ["python3", "app.py"]