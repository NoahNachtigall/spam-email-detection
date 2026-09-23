from pathlib import Path

from main import load_training_data


def test_load_training_data_uses_fallback_when_spam_csv_is_placeholder(tmp_path, monkeypatch):
    broken_file = tmp_path / "spam.csv"
    broken_file.write_text("404: Not Found", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    data = load_training_data()

    assert not data.empty
    assert {"label", "text"}.issubset(data.columns)
    assert set(data["label"].unique()).issubset({0, 1})
    assert len(data) >= 4
