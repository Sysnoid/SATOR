from sator_os_engine.core.optimizer.device import resolve_torch_device


def test_resolve_cpu():
    d, idx = resolve_torch_device("cpu", 0)
    assert str(d) == "cpu"
    assert idx is None


def test_resolve_cpu_when_cuda_requested_but_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d, idx = resolve_torch_device("cuda", 3)
    assert str(d) == "cpu"
    assert idx is None
