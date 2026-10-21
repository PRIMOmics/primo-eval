"""Board versions: the Hub commits that each version of the leaderboard reads.

A board version is a minor release of primo-eval: version ``0.1`` is scored by the
``v0.1.x`` releases. It pins the commit of every Hub repo that a score reads, so a
push to one of these repos changes no score until a release opens a new version.

``versions.yaml`` lists every version. ``current`` is the version this code scores
and shows by default. Each version names its release tag, the day it opened, and
three commits: ``data`` of the public data repo, ``labels`` of the private labels
repo, and ``reference`` of the results repo, which holds the HVG features.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

VERSIONS_FILE = Path(__file__).parent / "versions.yaml"


@dataclass(frozen=True)
class Version:
    """One board version and the Hub commits its scores read."""

    number: str
    release: str
    opened: str
    data: str
    labels: str
    reference: str


def _quoted(entry: dict, path: Path) -> dict:
    """Refuse the values that YAML reads as numbers or dates, such as ``0.10``."""
    unquoted = sorted(key for key, value in entry.items() if not isinstance(value, str))
    if unquoted:
        raise ValueError(f"quote the values of {unquoted} in {path}")
    return entry


def load_versions(path: str | Path) -> tuple[Version, list[Version]]:
    """The current version and every version listed in ``path``."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    current = _quoted({"current": raw["current"]}, path)["current"]
    versions = [Version(**_quoted(entry, path)) for entry in raw["versions"]]
    numbers = [version.number for version in versions]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"a version is listed twice in {path}: {numbers}")
    if current not in numbers:
        raise ValueError(f"the current version {current} is not listed in {path}")
    return versions[numbers.index(current)], versions


CURRENT, VERSIONS = load_versions(VERSIONS_FILE)
