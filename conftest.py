import pytest


@pytest.fixture(autouse=True)
def tmp_traces(tmp_path, monkeypatch):
    """Send all traces from tests to a temp dir."""
    from agentcore import config, trace

    s = config.settings()
    patched = type(s)(**{**s.__dict__, "trace_dir": tmp_path})
    monkeypatch.setattr(trace, "settings", lambda: patched)
    return tmp_path
