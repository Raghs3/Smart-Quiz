import os
import json
import pytest
import auth


@pytest.fixture(autouse=True)
def tmp_users(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, 'USERS_FILE', str(tmp_path / 'users.json'))


def test_register_success():
    ok, msg = auth.register('alice', 'secret')
    assert ok is True
    assert msg == ''


def test_register_creates_file():
    auth.register('alice', 'secret')
    assert os.path.exists(auth.USERS_FILE)


def test_register_hashes_password():
    auth.register('alice', 'secret')
    with open(auth.USERS_FILE) as f:
        data = json.load(f)
    assert 'secret' not in data['alice']
    assert data['alice'].startswith('sha256:')


def test_register_duplicate_fails():
    auth.register('alice', 'secret')
    ok, msg = auth.register('alice', 'other')
    assert ok is False
    assert 'already exists' in msg


def test_register_empty_username_fails():
    ok, msg = auth.register('', 'secret')
    assert ok is False


def test_login_correct():
    auth.register('bob', 'pass123')
    assert auth.login('bob', 'pass123') is True


def test_login_wrong_password():
    auth.register('bob', 'pass123')
    assert auth.login('bob', 'wrong') is False


def test_login_nonexistent_user():
    assert auth.login('nobody', 'anything') is False
