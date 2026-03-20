FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libmagic1 libmagic-dev file git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install lightweight packages first, then heavy ones
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.27.0" \
    pydantic==2.5.3 \
    pydantic-settings==2.1.0 \
    python-dotenv==1.0.0 \
    python-magic==0.4.27 \
    python-multipart==0.0.18 \
    Pillow==10.4.0 \
    imagehash==4.3.1 \
    numpy==1.26.3 \
    scipy==1.11.4 \
    scikit-learn==1.4.0 \
    opencv-python==4.9.0.80 \
    slowapi==0.1.9 \
    PyWavelets==1.4.1 \
    scikit-image==0.22.0 \
    cryptography==41.0.7 \
    psutil==5.9.8 && \
    pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    transformers==4.36.0 \
    diffusers==0.25.0 \
    accelerate==0.25.0 \
    huggingface_hub==0.20.0 \
    safetensors==0.4.1 \
    ftfy==6.1.1 \
    regex==2023.12.25 \
    tqdm==4.66.1 \
    "clip @ git+https://github.com/openai/CLIP.git"

ENV PYTHONPATH=/app
ENV HF_HOME=/tmp/huggingface
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "120"]
