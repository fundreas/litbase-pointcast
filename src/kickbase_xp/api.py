"""Thin client for the unofficial Kickbase v4 API.

Only read endpoints are used. Requests are throttled and retried; this is an
unofficial API and the pipeline is a guest on it.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests

from .config import API_BASE, COMPETITION_BUNDESLIGA, Credentials

log = logging.getLogger(__name__)

USER_AGENT = "kickbase-xp/0.1 (+https://github.com/fundreas/litbase-pointcast)"


class KickbaseError(RuntimeError):
    pass


class KickbaseClient:
    """Authenticated session against api.kickbase.com.

    Parameters
    ----------
    delay:
        Base pause between requests, in seconds. Jittered by +/-50%.
    """

    def __init__(
        self,
        credentials: Credentials,
        *,
        delay: float = 0.25,
        max_retries: int = 4,
        timeout: float = 30.0,
        competition_id: str = COMPETITION_BUNDESLIGA,
    ) -> None:
        self.credentials = credentials
        self.delay = delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.competition_id = competition_id
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            }
        )
        self._token: str | None = None
        self._last_request = 0.0

    # ----------------------------------------------------------------- auth

    def login(self) -> dict[str, Any]:
        payload = {
            "em": self.credentials.email,
            "pass": self.credentials.password,
            "loy": False,
            "rep": {},
        }
        resp = self._session.post(
            f"{API_BASE}/v4/user/login", json=payload, timeout=self.timeout
        )
        if resp.status_code != 200:
            raise KickbaseError(f"Login failed ({resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        token = data.get("tkn")
        if not token:
            raise KickbaseError("Login response contained no token")
        self._token = token
        self._session.headers["Authorization"] = f"Bearer {token}"
        log.info("Logged in as %s (token valid until %s)", data["u"]["name"], data.get("tknex"))
        return data

    # ------------------------------------------------------------- plumbing

    def _throttle(self) -> None:
        if self.delay <= 0:
            return
        wait = self.delay * random.uniform(0.5, 1.5)
        elapsed = time.monotonic() - self._last_request
        if elapsed < wait:
            time.sleep(wait - elapsed)

    def get(self, path: str, **params: Any) -> Any:
        if self._token is None:
            self.login()
        url = f"{API_BASE}{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self._session.get(url, params=params or None, timeout=self.timeout)
            except requests.RequestException as exc:  # network flake
                last_error = exc
                log.warning("GET %s failed (%s), retrying", path, exc)
            else:
                self._last_request = time.monotonic()
                if resp.status_code == 401:
                    log.info("Token rejected, re-authenticating")
                    self.login()
                    continue
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = KickbaseError(f"{resp.status_code} for {path}")
                    backoff = 2**attempt + random.random()
                    log.warning("GET %s -> %s, backing off %.1fs", path, resp.status_code, backoff)
                    time.sleep(backoff)
                    continue
                if not resp.ok:
                    raise KickbaseError(f"GET {path} -> {resp.status_code}: {resp.text[:200]}")
                if not resp.content:
                    return None
                return resp.json()
            time.sleep(2**attempt)
        raise KickbaseError(f"GET {path} failed after {self.max_retries} attempts: {last_error}")

    # ------------------------------------------------------------ endpoints

    def table(self) -> list[dict[str, Any]]:
        """League table -- also the authoritative list of teams in the competition."""
        data = self.get(f"/v4/competitions/{self.competition_id}/table") or {}
        return data.get("it", [])

    def matchdays(self) -> dict[str, Any]:
        """All matchdays of the running season, with fixtures and their status."""
        return self.get(f"/v4/competitions/{self.competition_id}/matchdays") or {}

    def team_profile(self, team_id: str) -> dict[str, Any]:
        """Squad of one team: current market value, position, status per player."""
        return self.get(f"/v4/competitions/{self.competition_id}/teams/{team_id}/teamprofile") or {}

    def player(self, player_id: str) -> dict[str, Any]:
        """Player detail: names, status list, market value, upcoming fixtures."""
        return self.get(f"/v4/competitions/{self.competition_id}/players/{player_id}") or {}

    def player_performance(self, player_id: str) -> list[dict[str, Any]]:
        """Full per-matchday history, all seasons the player appeared in."""
        data = self.get(
            f"/v4/competitions/{self.competition_id}/players/{player_id}/performance"
        ) or {}
        return data.get("it", [])

    def player_market_value(self, player_id: str, timeframe: int = 365) -> list[dict[str, Any]]:
        """Market-value history. `dt` is days since the Unix epoch."""
        data = self.get(
            f"/v4/competitions/{self.competition_id}/players/{player_id}/marketvalue/{timeframe}"
        ) or {}
        return data.get("it", [])
