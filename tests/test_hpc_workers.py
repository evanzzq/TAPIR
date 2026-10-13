
import pytest

from vespainv.runner import _available_cpu_count


def test_explicit_worker_limit(monkeypatch):
    monkeypatch.setenv("TAPIR_MAX_WORKERS", "4")
    assert _available_cpu_count() == 4


def test_single_worker(monkeypatch):
    monkeypatch.setenv("TAPIR_MAX_WORKERS", "1")
    assert _available_cpu_count() == 1


@pytest.mark.parametrize("value", ["0", "-1", "abc", "2.5", ""])
def test_invalid_worker_limit(monkeypatch, value):
    monkeypatch.setenv("TAPIR_MAX_WORKERS", value)

    with pytest.raises(ValueError, match="TAPIR_MAX_WORKERS"):
        _available_cpu_count()


def test_default_worker_count(monkeypatch):
    monkeypatch.delenv("TAPIR_MAX_WORKERS", raising=False)

    workers = _available_cpu_count()

    assert isinstance(workers, int)
    assert workers >= 1