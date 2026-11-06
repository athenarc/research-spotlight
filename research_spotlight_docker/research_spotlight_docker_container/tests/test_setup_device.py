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


def test_cpu_when_no_gpu_is_visible(monkeypatch):
    _fake_stack(monkeypatch, cuda=False, ort_providers=["CPUExecutionProvider"])
    assert modules.device() == "cpu"
    assert modules.setup_device() == "cpu"


def test_gpu_when_visible(monkeypatch):
    _fake_stack(monkeypatch, cuda=True, ort_providers=["CUDAExecutionProvider"])
    assert modules.device() == "cuda"
    assert modules.setup_device() == "cuda"


def test_gpu_visible_but_onnx_has_no_cuda_fails(monkeypatch):
    _fake_stack(monkeypatch, cuda=True, ort_providers=["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        modules.setup_device()
