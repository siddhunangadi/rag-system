FROM python:3.11-slim
WORKDIR /app
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu \
    && pip install --no-cache-dir -r requirements-prod.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY app.py ingest.py retrieval.py storage.py ./
COPY templates ./templates
COPY static ./static
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
