"""Single-user token auth for reaching the API from your phone.

Off when PS_AUTH_TOKEN is unset (local development). When set, every /api/
route except health and login needs either

  * `Authorization: Bearer <token>` (scripts, the CLI), or
  * the `ps_session` cookie, which the web app gets by POSTing the token to
    /api/auth/login once. The cookie is what lets <img> and <video> tags load
    thumbnails and originals, since they can't send headers.

The cookie holds an HMAC derived from the token, not the token itself, and is
HttpOnly + SameSite=Strict. Rotating PS_AUTH_TOKEN logs every device out.
This is a lock on a door that is already private (Tailscale, no public ports),
not the only line of defence.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from api.config import get_settings

COOKIE = "ps_session"
OPEN_PATHS = {"/api/health", "/api/auth/login", "/api/auth/status", "/api/auth/logout"}
MAX_FAILURES_PER_MIN = 10

router = APIRouter(prefix="/auth", tags=["auth"])
_failures: dict[str, deque[float]] = defaultdict(deque)


def session_value(token: str) -> str:
    return hmac.new(token.encode(), b"photo-search-session-v1", hashlib.sha256).hexdigest()


def is_authorized(request: Request) -> bool:
    token = get_settings().auth_token
    if not token:
        return True
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer ") and hmac.compare_digest(header[7:].strip(), token):
        return True
    cookie = request.cookies.get(COOKIE, "")
    return bool(cookie) and hmac.compare_digest(cookie, session_value(token))


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path not in OPEN_PATHS and not is_authorized(request):
            return JSONResponse({"detail": "not authenticated"}, status_code=401)
        return await call_next(request)


class LoginIn(BaseModel):
    token: str


def _too_many(ip: str) -> bool:
    q = _failures[ip]
    now = time.monotonic()
    while q and now - q[0] > 60:
        q.popleft()
    return len(q) >= MAX_FAILURES_PER_MIN


@router.get("/status")
def status(request: Request) -> dict[str, bool]:
    return {"required": bool(get_settings().auth_token), "authenticated": is_authorized(request)}


@router.post("/login")
def login(body: LoginIn, request: Request, response: Response) -> dict[str, bool]:
    token = get_settings().auth_token
    if not token:
        return {"ok": True}
    ip = request.client.host if request.client else "?"
    if _too_many(ip):
        raise HTTPException(429, "too many attempts, wait a minute")
    if not hmac.compare_digest(body.token.strip(), token):
        _failures[ip].append(time.monotonic())
        raise HTTPException(401, "wrong token")
    response.set_cookie(
        COOKIE,
        session_value(token),
        max_age=365 * 24 * 3600,
        httponly=True,
        samesite="strict",
        # Tailscale serve terminates HTTPS; plain http only on localhost / the LAN
        secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
        path="/",
    )
    return {"ok": True}


@router.post("/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}
