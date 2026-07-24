from app.domain.languages import normalize_language


def test_normalizes_language_names_and_regional_codes() -> None:
    assert normalize_language("Russian") == "ru"
    assert normalize_language("Uzbek") == "uz"
    assert normalize_language("ru-RU") == "ru"
    assert normalize_language("uz_UZ") == "uz"
    assert normalize_language(None) is None
