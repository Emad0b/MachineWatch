FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c requirements-lock.txt
COPY machinewatch ./machinewatch
COPY web ./web
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "machinewatch.service:app", "--host", "0.0.0.0", "--port", "8000"]
