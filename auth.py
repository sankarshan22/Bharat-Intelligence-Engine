import json
import uuid
import hashlib
import os
import threading
from datetime import datetime, timedelta

USERS_FILE = "users.json"
TOKEN_EXPIRY_DAYS = 30

_file_lock = threading.Lock()
def _load():
    with _file_lock:
        if not os.path.exists(USERS_FILE) or os.path.getsize(USERS_FILE) == 0:
            return {"users": {}, "tokens": {}}
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"users": {}, "tokens": {}}


def _save(data):
    with _file_lock:
        with open(USERS_FILE, "w") as f:
            json.dump(data, f, indent=2)


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _make_token() -> str:
    return str(uuid.uuid4())


def signup(username: str, password: str) -> dict:
    """Register a new user. Returns {success, token, user_id, error}."""
    data = _load()

    # Check username taken
    for uid, u in data["users"].items():
        if u["username"].lower() == username.lower():
            return {"success": False, "error": "Username already taken."}

    user_id = str(uuid.uuid4())
    token = _make_token()

    data["users"][user_id] = {
        "username": username,
        "password": _hash(password),
        "created_at": datetime.utcnow().isoformat(),
    }
    expiry = (datetime.utcnow() + timedelta(days=TOKEN_EXPIRY_DAYS)).isoformat()
    data["tokens"][token] = {"user_id": user_id, "expiry": expiry}
    _save(data)

    return {"success": True, "token": token, "user_id": user_id, "username": username}


def login(username: str, password: str) -> dict:
    """Authenticate user. Returns {success, token, user_id, error}."""
    data = _load()

    for user_id, u in data["users"].items():
        if u["username"].lower() == username.lower():
            if u["password"] == _hash(password):
                token = _make_token()
                expiry = (datetime.utcnow() + timedelta(days=TOKEN_EXPIRY_DAYS)).isoformat()
                data["tokens"][token] = {"user_id": user_id, "expiry": expiry}
                _save(data)
                return {"success": True, "token": token, "user_id": user_id, "username": u["username"]}
            else:
                return {"success": False, "error": "Incorrect password."}

    return {"success": False, "error": "User not found."}


def logout(token: str) -> dict:
    """Invalidate a token."""
    data = _load()
    if token in data["tokens"]:
        del data["tokens"][token]
        _save(data)
    return {"success": True}


def verify_token(token: str) -> dict | None:
    """Return {user_id, username} if token is valid and not expired, else None."""
    data = _load()
    entry = data["tokens"].get(token)
    if not entry:
        return None

    # Support old format (plain user_id string) for backward compatibility
    if isinstance(entry, str):
        user_id = entry
    else:
        user_id = entry["user_id"]
        expiry = datetime.fromisoformat(entry["expiry"])
        if datetime.utcnow() > expiry:
            # Token expired — clean it up
            del data["tokens"][token]
            _save(data)
            return None

    user = data["users"].get(user_id)
    if not user:
        return None

    # Clean up other expired tokens while we're here
    expired = [
        t for t, v in data["tokens"].items()
        if isinstance(v, dict) and "expiry" in v and datetime.utcnow() > datetime.fromisoformat(v["expiry"])
    ]
    if expired:
        for t in expired:
            del data["tokens"][t]
        _save(data)

    return {"user_id": user_id, "username": user["username"]}