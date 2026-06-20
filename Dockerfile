FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libmagic1 libmagic-dev file git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install all Python dependencies from the canonical requirements file.
# Previously the Dockerfile had a manually-maintained pip list that diverged
# from backend/requirements.txt on three packages:
#   diffusers 0.25.0 (Dockerfile) vs 0.32.0 (requirements) — CVE-flagged
#   python-multipart 0.0.22 vs 0.0.29
#   Pillow 12.1.1 vs 12.2.0
# This single-source approach also adds pillow-heif (HEIC/HEIF support)
# which was present in requirements.txt but missing from the Dockerfile.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

ENV PYTHONPATH=/app
ENV HF_HOME=/tmp/huggingface
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "120"]
