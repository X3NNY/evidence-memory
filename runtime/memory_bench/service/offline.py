"""Allow loopback HTTP and OS event-loop sockets, deny external connections.

Windows asyncio uses a local socket pair internally, so the retrieval replay's
all-socket guard cannot wrap an HTTP server or ASGI test client unchanged.
"""

from contextlib import contextmanager
import ipaddress
import socket
from unittest.mock import patch


@contextmanager
def loopback_only():
    audit = {"blocked_attempts": 0, "policy": "only numeric loopback connections; external connections and DNS denied"}
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex
    create_connection, getaddrinfo = socket.create_connection, socket.getaddrinfo

    def blocked():
        audit["blocked_attempts"] += 1
        raise RuntimeError("External network access is disabled in the BM25 text service")

    def check(host):
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = False
        if not allowed:
            blocked()

    def checked_connect(sock, address):
        if sock.family != getattr(socket, "AF_UNIX", object()):
            check(address[0])
        return connect(sock, address)

    def checked_connect_ex(sock, address):
        if sock.family != getattr(socket, "AF_UNIX", object()):
            check(address[0])
        return connect_ex(sock, address)

    def checked_create_connection(address, *args, **kwargs):
        check(address[0])
        return create_connection(address, *args, **kwargs)

    def checked_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if not flags & socket.AI_PASSIVE:
            check(host)
        else:
            # Passive resolution is for listener binding, not outbound access.
            if host is not None:
                try:
                    ipaddress.ip_address(host)
                except ValueError:
                    blocked()
        return getaddrinfo(host, port, family, type, proto, flags)

    with patch("socket.socket.connect", checked_connect), patch("socket.socket.connect_ex", checked_connect_ex), \
         patch("socket.create_connection", checked_create_connection), patch("socket.getaddrinfo", checked_getaddrinfo):
        yield audit
