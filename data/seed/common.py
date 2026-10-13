"""Small helpers shared by the seed scripts."""

from __future__ import annotations

import sys

from pydantic import SecretStr


def resolve_url(cli_url: str | None, seed_url: SecretStr | None, app_url: SecretStr | None, label: str) -> str:
    """Pick the connection URL to seed with: --url flag, then the *_SEED_URL, then the app URL.

    The app URL is normally a read-only user, so seeding with it will fail with a
    permission error. That fallback is only for setups that use a single user.
    """
    if cli_url:
        return cli_url
    if seed_url is not None:
        return seed_url.get_secret_value()
    if app_url is not None:
        print(f"[{label}] warning: no {label.upper()}_SEED_URL set; trying the app URL "
              "(this fails if it is a read-only user)", file=sys.stderr)
        return app_url.get_secret_value()
    raise SystemExit(f"[{label}] no connection URL: set {label.upper()}_SEED_URL in .env or pass --url")


def print_counts(label: str, counts: dict[str, int]) -> None:
    """Print row/document counts as an aligned list."""
    width = max(len(name) for name in counts)
    print(f"[{label}] row counts:")
    for name, count in counts.items():
        print(f"  {name:<{width}}  {count:>7,}")
