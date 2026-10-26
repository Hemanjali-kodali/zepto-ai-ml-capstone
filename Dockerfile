FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY support_assistant ./support_assistant

ENV MOCK_LLM=1

EXPOSE 8000

CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]