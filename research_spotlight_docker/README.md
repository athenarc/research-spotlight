# Research Spotlight Docker version

### ⚠️ Disclaimer

Research Spotlight is provided primarily for **educational, research, and demonstrator purposes**.

This demonstrator was **not designed for bulk or large-scale production processing**. Processing large collections of papers may require substantially more memory, storage, GPU resources, and additional engineering.


### Requirements

You need:

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/)

To use the GPU you additionally need:

- An NVIDIA CUDA GPU (and, on Linux, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html); on Windows, Docker Desktop with the WSL2 backend)

There is **one image for everything**. It uses the NVIDIA GPU when you start it with GPU access, and the CPU otherwise.

---

# 1. Install Docker

## macOS

Install [**Docker Desktop for Mac**](https://docs.docker.com/desktop/setup/install/mac-install/) from the official Docker website.

After installation, start Docker Desktop and verify it from a terminal:

```bash
docker --version
docker compose version
```

You should see version information for both commands.

## Windows

Install [**Docker Desktop for Windows**](https://docs.docker.com/desktop/setup/install/windows-install/) from the official Docker website.

After installation, start Docker Desktop and verify:

```powershell
docker --version
docker compose version
```
---

## Linux

You can install Docker Engine using the [official Docker instructions](https://docs.docker.com/engine/install/) for your Linux distribution.

Verify the installation:

```bash
docker --version
docker compose version
```

---

# 2. Get the code

Clone the repository and enter the `research_spotlight_docker_container` folder (the one that contains the `Dockerfile`):

```bash
git clone https://github.com/athenarc/research-spotlight.git
cd research-spotlight/research_spotlight_docker/research_spotlight_docker_container
```
---

# 3. Choose how to run it

| Your computer | Command |
|---|---|
| macOS (Intel or Apple Silicon) | CPU command |
| Windows / Linux without NVIDIA GPU | CPU command |
| Windows / Linux with NVIDIA GPU | GPU command |

Macs always use the CPU: Docker cannot give containers access to the Mac GPU. Apple Silicon (M1–M4) builds a native arm64 image automatically.

---

# 4. Run on the CPU (also macOS)

Build and start (the first build takes a while and downloads several GB of models):

```bash
docker compose up --build
```

Next time you can skip the build:

```bash
docker compose up
```

Go to <http://localhost:8080>. Keep this terminal open while the application is running.

To stop the application press `Ctrl+C`, then remove the containers:

```bash
docker compose down
```

---

# 5. Run on the NVIDIA GPU

Name the GPU service, otherwise Compose also starts the CPU one and they clash:

```bash
docker compose --profile gpu up --build research-spotlight-gpu
```

Go to <http://localhost:8080>.

To stop it:

```bash
docker compose --profile gpu down
```

Compose uses all GPUs. To choose one GPU, use `docker run` instead (see below).

---

# 6. Run with plain `docker run` (optional)

Build once:

```bash
docker build -t research-spotlight .
```

Then run. CPU:

```bash
docker run -p 127.0.0.1:8080:8000 -v "${PWD}/data:/app/data" research-spotlight
```

GPU, all GPUs / one specific GPU:

```bash
docker run --gpus all -p 127.0.0.1:8080:8000 -v "${PWD}/data:/app/data" research-spotlight
docker run --gpus device=1 -p 127.0.0.1:8080:8000 -v "${PWD}/data:/app/data" research-spotlight
```

`${PWD}` works in PowerShell, macOS and Linux. In Windows `cmd` use `%cd%` instead.

When the container starts, the log says either `Running on the NVIDIA GPU.` or `Running on the CPU`. If you expected the GPU and see CPU, you forgot `--gpus`/the GPU service.

---

# 7. Open Research Spotlight

Once the log says `All models are loaded`, open:

```text
http://localhost:8080
```

The results of a run are inside the `data` folder you can also download them by pressing `download run` on the end of the workflow.

---

# 8. Additional information

### Input

The input should be PDF(s) renamed as `1.pdf`, `2.pdf`, `3.pdf` alongside the required metadata.
Check the PDF folder for examples.

### Configuration (optional)

To use your own NER models, copy `.env.example` to `.env` and edit it. Compose picks it up automatically; with `docker run` add `--env-file .env`.

### Rebuilding from scratch

If you need to rebuild the containers and images after making substantial changes:

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

`--no-cache` forces Docker to rebuild the image layers instead of reusing the existing build cache.

### Disk space

The image is large (about 26 GB) because it contains both the CUDA libraries and the models.

### Out-of-memory errors

Research Spotlight uses machine-learning models and can require substantial RAM/VRAM.

Large PDFs and especially multiple PDFs can increase memory usage considerably. On Mac or Windows, raise the memory limit in Docker Desktop (Settings → Resources; 8 GB or more).
