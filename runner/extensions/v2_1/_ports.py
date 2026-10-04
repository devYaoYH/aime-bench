"""Service-port probe compatible with recently closed HTTP connections.

An active listener remains an error. Reuse the address as the HTTP servers do,
so completed cells' TIME_WAIT connections do not look like running services.
This helper does not change the frozen baseline runtime or any solving policy.
"""

import socket


def ensure_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
