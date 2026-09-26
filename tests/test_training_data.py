from pathlib import Path

from main import load_training_data
import main


def test_load_training_data_uses_fallback_when_spam_csv_is_placeholder(tmp_path, monkeypatch):
    broken_file = tmp_path / "spam.csv"
    broken_file.write_text("404: Not Found", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    data = load_training_data()

    assert not data.empty
    assert {"label", "text"}.issubset(data.columns)
    assert set(data["label"].unique()).issubset({0, 1})
    assert len(data) >= 4


def test_credentials_are_encrypted_and_can_be_decrypted(tmp_path, monkeypatch):
    stored_values = {}

    def get_password(service, username):
        return stored_values.get((service, username))

    def set_password(service, username, password):
        stored_values[(service, username)] = password

    monkeypatch.setattr(main.keyring, "get_password", get_password)
    monkeypatch.setattr(main.keyring, "set_password", set_password)

    path = tmp_path / "credentials.enc"
    credentials = {"email": "person@gmail.com", "app_password": "secret-app-password"}
    main.save_credentials(credentials, path)

    assert b"secret-app-password" not in path.read_bytes()
    assert main.decrypt_credentials(path) == credentials
