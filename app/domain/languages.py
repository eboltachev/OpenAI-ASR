from __future__ import annotations

_LANGUAGE_ALIASES = {
    "english": "en",
    "russian": "ru",
    "uzbek": "uz",
    "русский": "ru",
    "русский язык": "ru",
    "узбекский": "uz",
    "узбекский язык": "uz",
    "o'zbek": "uz",
}


def normalize_language(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = (
        value.strip()
        .lower()
        .replace("_", "-")
        .replace("\u02bb", "'")
        .replace("\u2018", "'")
    )
    if not normalized:
        return None
    if normalized in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[normalized]
    primary = normalized.split("-", maxsplit=1)[0]
    if 2 <= len(primary) <= 3 and primary.isalpha():
        return primary
    return normalized
