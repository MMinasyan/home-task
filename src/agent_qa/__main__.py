"""Process entry: run the chat server from explicit launch arguments."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from agent_qa.app import create_app
from agent_qa.model import ModelRef
from agent_qa.providers.config import load_config, resolve_model


def main(argv: Sequence[str] | None = None) -> None:
    """Launch one server process; every launch input is explicit."""
    parser = argparse.ArgumentParser(prog="agent_qa")
    parser.add_argument("--config", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--db", default="agent_qa.sqlite3")
    arguments = parser.parse_args(argv)

    model = ModelRef(provider=arguments.provider, model=arguments.model)
    try:
        config = load_config(Path(arguments.config))
        resolve_model(config, model)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    try:
        system_prompt = Path(arguments.prompt).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise SystemExit("prompt file cannot be read as UTF-8") from None

    uvicorn.run(
        create_app(config, model, Path(arguments.db), system_prompt),
        timeout_graceful_shutdown=0,
    )


if __name__ == "__main__":
    main()
