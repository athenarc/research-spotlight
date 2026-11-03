FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    GENRE_MODEL_PATH=/models/genre-linking-blink \
    MODEL_ROOT=/models \
    OCR_MODEL_PATH=/models/LightOnOCR-3-0.8B.Q6_K.gguf \
    OCR_MMPROJ_PATH=/models/LightOnOCR-3-0.8B.mmproj-q8_0.gguf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 \
    && pip install -r requirements.txt \
    && pip install \
        https://huggingface.co/NikosKprl/en_deberta_v3_base_ner_method/resolve/main/en_deberta_v3_base_ner_method-1.0.0-py3-none-any.whl \
        https://huggingface.co/NikosKprl/en_deberta_v3_base_ner_activity/resolve/main/en_deberta_v3_base_ner_activity-1.0.0-py3-none-any.whl \
        https://huggingface.co/NikosKprl/en_deberta_v3_base_ner_goal/resolve/main/en_deberta_v3_base_ner_goal-1.0.0-py3-none-any.whl

COPY app ./app

RUN python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download(
    repo_id="facebook/genre-linking-blink",
    local_dir="/models/genre-linking-blink",
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "merges.txt",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
)

repo = "prithivMLmods/LightOnOCR-3-0.8B-GGUF"
hf_hub_download(
    repo_id=repo,
    filename="LightOnOCR-3-0.8B.Q6_K.gguf",
    local_dir="/models",
)
hf_hub_download(
    repo_id=repo,
    filename="LightOnOCR-3-0.8B.mmproj-q8_0.gguf",
    local_dir="/models",
)
PY

RUN mkdir -p /app/data/runs /app/dependencies /cache/huggingface /root/.cache \
    && curl -fL \
        https://huggingface.co/facebook/genre-linking-blink/resolve/main/kilt_titles_trie_dict.pkl \
        -o /app/dependencies/kilt_titles_trie_dict.pkl \
    && curl -fL \
        https://huggingface.co/facebook/genre-linking-blink/raw/main/trie.py \
        -o /app/dependencies/trie.py

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/pipeline >/dev/null || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
