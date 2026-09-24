# Research Spotlight Docker version

This repository contains two Docker variants:

- `research-spotlight-CPU` — CPU-based version. Use this on macOS and on machines without a supported NVIDIA GPU.
- `research-spotlight-GPU` — NVIDIA CUDA GPU version. Use this when you have a supported NVIDIA GPU.

> **⚠️ Important:** Use **only one variant at a time**.

### ⚠️ Disclaimer

Research Spotlight is provided primarily for **educational, research, and demonstrator purposes**.

This demonstrator was **not designed for bulk or large-scale production processing**. Processing large collections of papers may require substantially more memory, storage, GPU resources, and additional engineering.


### Requirements

You need:

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/)

For the GPU version you additionally need:

- An NVIDIA CUDA GPU

---

# 1. Install Docker

## macOS

Install [**Docker Desktop for Mac**](https://docs.docker.com/desktop/setup/install/mac-install/) from the official Docker website:


After installation, start Docker Desktop and verify it from a terminal:

```bash
docker --version
docker compose version
```

You should see version information for both commands.

### Apple Silicon Macs

Apple Silicon Macs (for example M1, M2, M3, M4) should use:

```text
research-spotlight-CPU
```
🚧 We plan to add support for Apple Silicon Macs in the future.

---

## Windows

Install [**Docker Desktop for Windows**](https://docs.docker.com/desktop/setup/install/windows-install/) from the official Docker website:

After installation, start Docker Desktop and verify:

```powershell
docker --version
docker compose version
```

If you have an NVIDIA GPU and want to use the GPU version, make sure your NVIDIA driver and Docker Desktop/WSL2 GPU support are working before starting the project.

If you do not have a supported NVIDIA GPU, use:

```text
research-spotlight-CPU
```

---

## Linux

You can install Docker Engine using the [official Docker instructions](https://docs.docker.com/engine/install/) for your Linux distribution:

Verify the installation:

```bash
docker --version
docker compose version
```

---

# 2. Clone the repository

Clone the repository using:
```bash
git clone https://github.com/athenarc/research-spotlight.git
```

Then enter the repository:

```bash
cd research_spotlight_code
```

You should see folders similar to:

```text
research-spotlight-CPU/
research-spotlight-GPU/
```

---

# 3. Choose the correct version

Use the following rule:

| Your computer | Folder to use |
|---|---|
| macOS Intel | `research-spotlight-CPU` |
| macOS Apple Silicon | `research-spotlight-CPU` |
| Windows without NVIDIA GPU | `research-spotlight-CPU` |
| Linux without NVIDIA GPU | `research-spotlight-CPU` |
| Windows with supported NVIDIA GPU | `research-spotlight-GPU` |
| Linux with supported NVIDIA GPU | `research-spotlight-GPU` |

---

# 4. Run the CPU version

Use this version on macOS and on computers without a supported NVIDIA GPU.

From the repository root:

```bash
cd research-spotlight-CPU
```

Build the Docker image:

```bash
docker compose build
```

Start the application:

```bash
docker compose up
```

Keep this terminal open while the application is running.

To stop the application:

```text
Ctrl+C
```

To stop the background containers:

```bash
docker compose down
```

---

# 5. Run the NVIDIA GPU version

Use this version only when your NVIDIA GPU is correctly configured for Docker.

From the repository root:

```bash
cd research-spotlight-GPU
```

Then build the image:

```bash
docker compose build
```

Start the application:

```bash
docker compose up
```

To stop it:

```bash
docker compose down
```

---

# 6. Open Research Spotlight

Once the container is running, Docker Compose will expose the application's configured port.

Use the the following URL to access the Web UI:

```text
http://localhost:8000
```
The results of a run are inside the `data` folder.

---

# 7. Additional information

### 🔄 Rebuilding from scratch

If you need to rebuild the containers and images after making substantial changes:

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

`--no-cache` forces Docker to rebuild the image layers instead of reusing the existing build cache.

### 🧠 Out-of-memory errors

Research Spotlight uses machine-learning models and can require substantial RAM/VRAM.

Large PDFs and especially multiple PDFs can increase memory usage considerably.

This project is not designed as a bulk-processing system. For experiments, start with a small number of papers and monitor available RAM/VRAM.
