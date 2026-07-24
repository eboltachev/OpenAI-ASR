from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import Settings


class BearerAuth:
    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.service_api_key

    async def __call__(self, authorization: str | None = Header(default=None)) -> None:
        if not self._api_key:
            return
        expected = f"Bearer {self._api_key}"
        if authorization != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": {"message": "Invalid API key", "type": "authentication_error"}},
                headers={"WWW-Authenticate": "Bearer"},
            )
