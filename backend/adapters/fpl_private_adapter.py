#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

# ---------- paths ----------
REPO = Path(__file__).resolve().parents[1]
AIRSENAL_HOME = REPO / ".airsenal_home"
OUTDIR = REPO / "data" / "api"
TOKENS_PATH = AIRSENAL_HOME / "FPL_TOKENS.json"

# ---------- endpoints ----------
CLIENT_ID = "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"
STANDARD_CONNECTION_ID = "867ed4363b2bc21c860085ad2baa817d"

URLS = {
    "auth": "https://account.premierleague.com/as/authorize",
    "start": "https://account.premierleague.com/davinci/policy/262ce4b01d19dd9d385d26bddb4297b6/start",
    "login": "https://account.premierleague.com/davinci/connections/{}/capabilities/customHTMLTemplate",
    "resume": "https://account.premierleague.com/as/resume",
    "token": "https://account.premierleague.com/as/token",
    "me": "https://fantasy.premierleague.com/api/me/",
    "my_team": "https://fantasy.premierleague.com/api/my-team/{}/",
    "transfers_latest": "https://fantasy.premierleague.com/api/entry/{}/transfers-latest/",
}

# ---------- helpers ----------
def read_secret_file(path: Path) -> str:
    s = path.read_text(encoding="utf-8").strip()
    if not s:
        raise RuntimeError(f"Empty secret file: {path}")
    return s

def generate_code_verifier() -> str:
    return secrets.token_urlsafe(64)[:128]

def generate_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")

def login(session: requests.Session, email: str, password: str) -> Tuple[str, str]:
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    initial_state = uuid.uuid4().hex

    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": "https://fantasy.premierleague.com/",
        "response_type": "code",
        "scope": "openid profile email offline_access",
        "state": initial_state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    auth_response = session.get(URLS["auth"], params=params, timeout=20)
    login_html = auth_response.text

    access_token = re.search(r'"accessToken":"([^"]+)"', login_html).group(1)
    new_state = re.search(r'<input[^>]+name="state"[^>]+value="([^"]+)"', login_html).group(1)

    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    start = session.post(URLS["start"], headers=headers, timeout=20).json()
    interaction_id = start["interactionId"]

    step1 = session.post(
        URLS["login"].format(STANDARD_CONNECTION_ID),
        headers={"interactionId": interaction_id},
        json={
            "id": start["id"],
            "eventName": "continue",
            "parameters": {"eventType": "polling"},
            "pollProps": {"status": "continue", "delayInMs": 10, "retriesAllowed": 1, "pollChallengeStatus": False},
        },
        timeout=20,
    )

    step2 = session.post(
        URLS["login"].format(STANDARD_CONNECTION_ID),
        headers={"interactionId": interaction_id},
        json={
            "id": step1.json()["id"],
            "nextEvent": {
                "constructType": "skEvent",
                "eventName": "continue",
                "params": [],
                "eventType": "post",
                "postProcess": {},
            },
            "parameters": {
                "buttonType": "form-submit",
                "buttonValue": "SIGNON",
                "username": email,
                "password": password,
            },
            "eventName": "continue",
        },
        timeout=20,
    ).json()

    step3 = session.post(
        URLS["login"].format(step2["connectionId"]),
        headers=headers,
        json={
            "id": step2["id"],
            "nextEvent": {
                "constructType": "skEvent",
                "eventName": "continue",
                "params": [],
                "eventType": "post",
                "postProcess": {},
            },
            "parameters": {"buttonType": "form-submit", "buttonValue": "SIGNON"},
            "eventName": "continue",
        },
        timeout=20,
    )

    resume = session.post(
        URLS["resume"],
        data={"dvResponse": step3.json()["dvResponse"], "state": new_state},
        allow_redirects=False,
        timeout=20,
    )

    location = resume.headers["Location"]
    auth_code = re.search(r"[?&]code=([^&]+)", location).group(1)

    token = session.post(
        URLS["token"],
        data={
            "grant_type": "authorization_code",
            "redirect_uri": "https://fantasy.premierleague.com/",
            "code": auth_code,
            "code_verifier": code_verifier,
            "client_id": CLIENT_ID,
        },
        timeout=20,
    ).json()

    return token["access_token"], token["refresh_token"]

def refresh_tokens(refresh_token: str) -> Tuple[str, str]:
    r = requests.post(
        URLS["token"],
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
            "scope": "openid profile email",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    j = r.json()
    return j["access_token"], j["refresh_token"]

def load_cached_tokens() -> Optional[Dict[str, Any]]:
    if not TOKENS_PATH.exists():
        return None
    return json.loads(TOKENS_PATH.read_text(encoding="utf-8"))

def save_cached_tokens(access_token: str, refresh_token: str) -> None:
    TOKENS_PATH.write_text(
        json.dumps({"access_token": access_token, "refresh_token": refresh_token, "saved_at": time.time()}, indent=2),
        encoding="utf-8",
    )

def authed_get(session: requests.Session, url: str, access_token: str) -> requests.Response:
    return session.get(url, headers={"X-API-Authorization": f"Bearer {access_token}"}, timeout=20)

def get_me(session: requests.Session, access_token: str) -> Dict[str, Any]:
    r = authed_get(session, URLS["me"], access_token)
    r.raise_for_status()
    return r.json()

def extract_entry_id(me: Dict[str, Any]) -> Optional[int]:
    # common shapes: me["player"]["entry"] or me["entry"]
    if isinstance(me.get("player"), dict) and isinstance(me["player"].get("entry"), int):
        return me["player"]["entry"]
    if isinstance(me.get("entry"), int):
        return me["entry"]
    return None

def main() -> None:
    email = read_secret_file(AIRSENAL_HOME / "FPL_LOGIN")
    password = read_secret_file(AIRSENAL_HOME / "FPL_PASSWORD")

    tid: Optional[int] = None
    tid_path = AIRSENAL_HOME / "FPL_TEAM_ID"
    if tid_path.exists():
        tid = int(read_secret_file(tid_path))

    OUTDIR.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        cached = load_cached_tokens()

        # 1) Get a valid access token (try cache -> refresh -> login)
        access_token = None
        refresh_token = None

        if cached:
            access_token = cached.get("access_token")
            refresh_token = cached.get("refresh_token")

        # Try cached token quickly
        if access_token:
            test = authed_get(session, URLS["me"], access_token)
            if test.status_code in (401, 403):
                access_token = None

        # Refresh if needed
        if not access_token and refresh_token:
            try:
                access_token, refresh_token = refresh_tokens(refresh_token)
                save_cached_tokens(access_token, refresh_token)
            except Exception:
                access_token = None

        # Full login if still needed
        if not access_token:
            access_token, refresh_token = login(session, email=email, password=password)
            save_cached_tokens(access_token, refresh_token)

        # 2) Fetch /api/me (also gives entry id if you didn’t store it)
        me = get_me(session, access_token)
        (OUTDIR / "me.json").write_text(json.dumps(me, indent=2), encoding="utf-8")

        if tid is None:
            tid = extract_entry_id(me)
        if tid is None:
            raise RuntimeError("Could not determine TEAM_ID (entry). Put it in .airsenal_home/FPL_TEAM_ID")

        # 3) Fetch private “next deadline” squad (THIS is your next GW picked team)
        my_team_url = URLS["my_team"].format(tid)
        r = authed_get(session, my_team_url, access_token)
        if r.status_code in (401, 403):
            # token expired between calls, refresh once
            access_token, refresh_token = refresh_tokens(refresh_token)
            save_cached_tokens(access_token, refresh_token)
            r = authed_get(session, my_team_url, access_token)
        r.raise_for_status()
        my_team = r.json()
        (OUTDIR / "my_team.json").write_text(json.dumps(my_team, indent=2), encoding="utf-8")

        # 4) transfers-latest
        tl_url = URLS["transfers_latest"].format(tid)
        r2 = authed_get(session, tl_url, access_token)
        if r2.status_code in (401, 403):
            access_token, refresh_token = refresh_tokens(refresh_token)
            save_cached_tokens(access_token, refresh_token)
            r2 = authed_get(session, tl_url, access_token)
        r2.raise_for_status()
        transfers_latest = r2.json()
        (OUTDIR / "transfers_latest.json").write_text(json.dumps(transfers_latest, indent=2), encoding="utf-8")

    print(f"[fpl_private_adapter] wrote {OUTDIR/'my_team.json'} (+ me.json, transfers_latest.json)")

if __name__ == "__main__":
    main()

