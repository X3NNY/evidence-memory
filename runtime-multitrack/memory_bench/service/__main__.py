"""Run the text API; remote v4 embeddings require explicit environment opt-in."""

import argparse
from contextlib import nullcontext
from pathlib import Path

from .app import Settings, create_app
from .offline import loopback_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    settings = Settings.from_environment()
    if args.database is not None:
        from dataclasses import replace
        settings = replace(settings, database=args.database)
    import uvicorn

    network = (nullcontext() if settings.retriever == "hybrid" and settings.embedding_mode == "remote"
               else loopback_only())
    with network:
        uvicorn.run(create_app(settings), host=args.host, port=args.port,
                    access_log=False, limit_concurrency=16, timeout_keep_alive=15)


if __name__ == "__main__":
    main()
