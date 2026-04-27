import hashlib
import json
import os

USERS_FILE = 'users.json'


def _hash(password: str) -> str:
    return 'sha256:' + hashlib.sha256(password.encode('utf-8')).hexdigest()


def _load() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def register(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (True, '') or (False, error_message)."""
    username = username.strip()
    if not username or not password:
        return False, 'Username and password are required.'
    users = _load()
    if username in users:
        return False, f'Username "{username}" already exists.'
    users[username] = _hash(password)
    _save(users)
    return True, ''


def login(username: str, password: str) -> bool:
    """Return True if credentials match, False otherwise."""
    users = _load()
    stored = users.get(username.strip())
    if stored is None:
        return False
    return stored == _hash(password)
