"""Signing in to the F1 TV account that Formula 1 now asks for its timing data.

livetiming.formula1.com answers 401 unless a request carries an F1 TV
subscription token. FastF1 3.9 gets that token by signing in, with the
credentials in F1_USERNAME / F1_PASSWORD or a terminal prompt; neither reaches
a graphical interface. This module signs in with credentials handed to it
directly and gives the token to FastF1's own token store, which keeps it on
disk (readable only by this user) until it expires and attaches it to every
download. The password itself is used for the sign-in and never kept.

It relies on `fastf1.internals.f1account`, which FastF1 does not promise to
keep stable: every entry point here fails with a plain message if it moves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Status:
    signed_in: bool
    expires_at: float | None = None  # unix time
    problem: str | None = None  # why the status could not be read

    def expires_text(self) -> str:
        if self.expires_at is None:
            return ""
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.expires_at))


def _account():
    try:
        from fastf1.internals import f1account
        from fastf1.internals.f1account.login import login
    except ImportError as exc:
        raise RuntimeError(
            "This FastF1 cannot sign in to an F1 TV account. Update it with "
            "`pip install -U fastf1` (3.9 or newer)."
        ) from exc
    return f1account, login


def status() -> Status:
    """Whether a valid token is stored, without signing in."""
    try:
        f1account, _ = _account()
    except RuntimeError as exc:
        return Status(False, problem=str(exc))
    token = f1account.token_manager.get_stored()
    if token is None or not token.is_valid():
        return Status(False)
    return Status(True, token.expires_at)


def sign_in(username: str, password: str) -> Status:
    """Sign in and store the token. Raises RuntimeError with the reason on failure."""
    f1account, login = _account()
    username = username.strip()
    if not username or not password:
        raise RuntimeError("Write the email and the password of the F1 TV account.")
    try:
        token = f1account.token_manager.get(
            force_renew=True, auth_func=lambda: login(username, password)
        )
    except Exception as exc:  # F1AuthError, or a network error during the sign-in
        import requests

        cause = exc if isinstance(exc, requests.ConnectionError) else exc.__cause__
        if isinstance(cause, requests.ConnectionError):
            raise RuntimeError(
                "Could not reach account.formula1.com: check the internet connection."
            ) from exc
        raise RuntimeError(str(exc) or type(exc).__name__) from exc
    return Status(True, token.expires_at)


def sign_out() -> None:
    """Forget the stored token."""
    f1account, _ = _account()
    f1account.clear_token()
