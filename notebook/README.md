# Research Spotlight

## Requirements

Before running the project, make sure you have:

* **Python 3.13**
* **Jupyter / JupyterLab**
* **[llama.cpp](https://llama.app)**

Clone the repository and enter the project directory:

```bash
git clone https://github.com/athenarc/research-spotlight.git
```

## Set Up the Python Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

## Run the GLM-OCR Server

GLM-OCR is run locally through `llama.cpp`.

Start the server with:

```bash
llama-server -hf ggml-org/GLM-OCR-GGUF:F16 --port 8080
```

The first time you run this command, `llama.cpp` will download the GLM-OCR model from Hugging Face.
Keep this terminal running while using the notebook.

## Run the Notebook

With the virtual environment activated, start Jupyter:

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

Open the project notebook and run the cells.
