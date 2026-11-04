import sys
import types

import pytest

from app.pipeline import modules


def _fake_stack(monkeypatch, cuda, ort_providers):
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda))
    ort = types.SimpleNamespace(get_available_providers=lambda: ort_providers)
    spacy = types.SimpleNamespace(require_gpu=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "spacy", spacy)


def test_require_gpu_fails_without_cuda(monkeypatch):
    _fake_stack(monkeypatch, cuda=False, ort_providers=["CUDAExecutionProvider"])
    with pytest.raises(RuntimeError, match="NVIDIA GPU"):
        modules.require_gpu()


def test_require_gpu_fails_without_onnx_cuda(monkeypatch):
    _fake_stack(monkeypatch, cuda=True, ort_providers=["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        modules.require_gpu()


def test_require_gpu_passes_with_gpu(monkeypatch):
    _fake_stack(monkeypatch, cuda=True, ort_providers=["CUDAExecutionProvider"])
    modules.require_gpu()
