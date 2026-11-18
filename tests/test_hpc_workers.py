
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

@pytest.mark.parametrize(("env_limit", "yaml_limit", "expected"), [(8, 3, 3), (2, 8, 2), (8, 1, 1)])
def test_yaml_cap_composes_with_hpc_limit(monkeypatch, env_limit, yaml_limit, expected):
    monkeypatch.setenv("TAPIR_MAX_WORKERS", str(env_limit))
    assert _available_cpu_count(yaml_limit) == expected


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "4"])
def test_invalid_yaml_worker_limit(value):
    with pytest.raises(ValueError, match="max_procs"):
        _available_cpu_count(value)
