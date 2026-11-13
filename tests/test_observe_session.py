from datetime import date
import json

import pandas as pd
import pytest

from api import observe_session


def test_non_session_rejected_before_reading_data(tmp_path):
    with pytest.raises(ValueError, match="completed NYSE session"):
        observe_session.publish(date(2026, 11, 8), tmp_path, tmp_path / "observation.json")


def test_missing_symbol_publishes_incomplete_snapshot(tmp_path, monkeypatch):
    bars = pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "session_date": [date(2026, 11, 12), date(2026, 11, 11)],
        "open": [10.0, 10.0], "high": [11.0, 11.0], "low": [9.0, 9.0],
        "close": [10.0, 10.0], "volume": [100, 100],
    })
    monkeypatch.setattr(observe_session, "read_bars", lambda path: bars)

    def unexpected_scoring(*args):
        pytest.fail("An incomplete session must not be scored")

    monkeypatch.setattr(observe_session.server, "build_signals", unexpected_scoring)
    output = tmp_path / "observation.json"
    assert observe_session.publish(date(2026, 11, 12), tmp_path, output) == "Incomplete"
    dashboard = json.loads(output.read_text())["dashboard"]
    assert dashboard["signals"] == []
    assert "BBB" in dashboard["blockedMlEntries"][0]["reason"]
