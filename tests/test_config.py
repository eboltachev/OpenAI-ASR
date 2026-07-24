from app.core.config import Settings


def test_alignment_models_are_loaded_from_language_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm:8000/v1/")
    monkeypatch.setenv("ALIGNMENT_MODEL_RU", "ru-model")
    monkeypatch.setenv("ALIGNMENT_MODEL_UZ", "/app/models/uz")
    settings = Settings.from_env()
    assert settings.openai_base_url == "http://vllm:8000/v1"
    assert settings.alignment_models == {"ru": "ru-model", "uz": "/app/models/uz"}
