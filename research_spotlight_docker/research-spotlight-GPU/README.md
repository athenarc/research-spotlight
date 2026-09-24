# Research Spotlight Web

A custom FastAPI + HTML/CSS/JS web application around the Research Spotlight pipeline, packaged as **one Docker image**.

## What the user runs

The runtime is intentionally a single container. Inside that container:

- FastAPI serves the Research Spotlight web UI on port `8000`.
- `llama-server` runs locally on port `8080` for GLM-OCR.
- The three Research Spotlight spaCy NER models are installed into the image during the Docker build.
- The GENRE trie is bundled into the image.
- The GENRE transformer weights and GLM-OCR GGUF are cached automatically when first needed.

There is no second application container and no separate llama.cpp service to start.

## Pipeline

The web UI follows the notebook's sequential workflow:

1. Text Extraction
2. Entity Extraction
3. Entity Disambiguation
4. Entity Linking
5. Relation Extraction
6. RDF Generation

Every stage writes an artifact before the **Next** button becomes available. Each uploaded dataset gets its own isolated directory under `data/runs/<run_id>/`.

## Build the one image

```bash
docker compose build
```

The build compiles `llama-server` into the same image and installs the three Research Spotlight NER model packages.

## Run the one image

```bash
docker run --rm \
  -p 8000:8000 \
  -v research-spotlight-data:/app/data \
  -v research-spotlight-cache:/root/.cache \
  -v research-spotlight-hf:/cache/huggingface \
  research-spotlight:latest
```

Then open:

```text
http://localhost:8000
```

On first startup, `llama-server` downloads the `ggml-org/GLM-OCR-GGUF:F16` model into its cache. The GENRE model is downloaded by Hugging Face Transformers the first time Entity Disambiguation is used. The named Docker volumes keep those downloads between container recreations.

### NVIDIA GPU

The image is CPU-compatible, but llama.cpp can use a GPU when the image is built with a CUDA-enabled llama.cpp build. The current Dockerfile intentionally uses the portable CPU build so the same image does not depend on NVIDIA Container Toolkit being installed.

## Optional Compose shortcut

`docker compose up --build` now starts only the **same single application container**. Compose is not required; the `docker build` + `docker run` commands above are the canonical one-image deployment.

## Models and assets

The Research Spotlight NER model packages are installed directly from their Hugging Face-hosted wheels:

- `NikosKprl/en_deberta_v3_base_ner_method`
- `NikosKprl/en_deberta_v3_base_ner_activity`
- `NikosKprl/en_deberta_v3_base_ner_goal`

The GENRE disambiguation model remains a Hugging Face download because it is several gigabytes by itself. The GLM-OCR GGUF is also downloaded by `llama-server` on first startup. This avoids producing an unnecessarily enormous source image while still keeping runtime to one container.

## Input

Upload one or more PDFs. You can also upload the notebook's `metadata.jsonl`.

Without metadata, the web app generates minimal metadata entries from the uploaded filenames so the pipeline can still run.

## Runtime persistence

Artifacts:

```text
/app/data/runs/<run_id>/
├── PDF/
├── Text_Extraction/
├── Entity_Extraction/
├── Entity_Disambiguation/
├── Entity_Linking/
├── Relation_Extraction/
└── RDF_Generation/
```

This means users can stop and restart the container without losing their persisted runs when `/app/data` is mounted as a Docker volume.