"""CLI entry point for the `liteparse` command."""

import sys

from liteparse._liteparse import run_cli


def main() -> None:
    try:
        run_cli(sys.argv)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
