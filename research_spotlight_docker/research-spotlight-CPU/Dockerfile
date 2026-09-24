# ------------------------------------------------------------
# Stage 1: obtain prebuilt llama-server
# ------------------------------------------------------------
FROM ghcr.io/ggml-org/llama.cpp:server AS llama


# ------------------------------------------------------------
# Stage 2: Research Spotlight application
# ------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    GENRE_MODEL_PATH=/models/genre-linking-blink \
    HOME=/root \
    MODEL_ROOT=/models \
    RS_DEPENDENCIES_DIR=/app/dependencies \
    GLMOCR_CONFIG_PATH=/app/dependencies/config.yaml \
    LLAMA_HF_MODEL=ggml-org/GLM-OCR-GGUF:F16 \
    LLAMA_HOST=127.0.0.1 \
    LLAMA_PORT=8080 \
    PATH=/opt/llama:${PATH} \
    LD_LIBRARY_PATH=/opt/llama:${LD_LIBRARY_PATH}

WORKDIR /app


# ------------------------------------------------------------
# System packages
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------
# Copy PREBUILT llama-server and its libraries
# ------------------------------------------------------------
COPY --from=llama /app/ /opt/llama/


# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
        torchvision \
    && pip install -r requirements.txt \
    && pip install \
        https://huggingface.co/NikosKprl/en_deberta_v3_base_ner_method/resolve/main/en_deberta_v3_base_ner_method-1.0.0-py3-none-any.whl \
        https://huggingface.co/NikosKprl/en_deberta_v3_base_ner_activity/resolve/main/en_deberta_v3_base_ner_activity-1.0.0-py3-none-any.whl \
        https://huggingface.co/NikosKprl/en_deberta_v3_base_ner_goal/resolve/main/en_deberta_v3_base_ner_goal-1.0.0-py3-none-any.whl


# ------------------------------------------------------------
# Application + dependencies
# ------------------------------------------------------------
COPY app ./app
COPY dependencies ./dependencies
COPY docker/entrypoint.sh /usr/local/bin/research-spotlight-entrypoint


# ------------------------------------------------------------
# Download GENRE model during IMAGE BUILD
# ------------------------------------------------------------
RUN python - <<'PY'
from huggingface_hub import snapshot_download

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
PY


# ------------------------------------------------------------
# Download original notebook trie implementation + trie data
# ------------------------------------------------------------
RUN chmod +x /usr/local/bin/research-spotlight-entrypoint \
    && mkdir -p \
        /app/data/runs \
        /models \
        /cache/huggingface \
        /root/.cache \
    && curl -fL \
        https://huggingface.co/facebook/genre-linking-blink/resolve/main/kilt_titles_trie_dict.pkl \
        -o /app/dependencies/kilt_titles_trie_dict.pkl \
    && curl -fL \
        https://huggingface.co/facebook/genre-linking-blink/raw/main/trie.py \
        -o /app/dependencies/trie.py


# ------------------------------------------------------------
# Make GLM-OCR talk to llama-server inside this SAME container
# ------------------------------------------------------------
RUN python - <<'PY'
from pathlib import Path

p = Path("/app/dependencies/config.yaml")
s = p.read_text()

s = s.replace(
    "api_host: llama",
    "api_host: 127.0.0.1"
)

p.write_text(s)
PY


EXPOSE 8000


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD curl -fsS \
        http://127.0.0.1:8000/api/pipeline \
        >/dev/null || exit 1


# ------------------------------------------------------------
# Start both FastAPI + llama-server
# ------------------------------------------------------------
ENTRYPOINT ["/usr/local/bin/research-spotlight-entrypoint"]