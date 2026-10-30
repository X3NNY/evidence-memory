"""Process-local network guard for deterministic offline regressions."""

from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def deny_network():
    audit = {"blocked_attempts": 0, "policy": "socket connections and DNS disabled in this process"}

    def blocked(*args, **kwargs):
        audit["blocked_attempts"] += 1
        raise RuntimeError("This experiment is strictly offline; network access is disabled")

    with patch("socket.socket.connect", blocked), patch("socket.socket.connect_ex", blocked), \
            patch("socket.create_connection", blocked), patch("socket.getaddrinfo", blocked):
        yield audit

