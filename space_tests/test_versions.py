"""Unit tests for the board versions (no network)."""

import re
import tomllib
from pathlib import Path

import pytest
from versions import CURRENT, VERSIONS, load_versions

COMMIT = re.compile(r"[0-9a-f]{40}")
PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def _entry(number: str, commit: str = "a" * 40) -> str:
    return (
        f'  - number: "{number}"\n'
        f'    release: "v{number}.0"\n'
        '    opened: "2026-01-01"\n'
        f'    data: "{commit}"\n'
        f'    labels: "{commit}"\n'
        f'    reference: "{commit}"\n'
    )


def _write(tmp_path: Path, current: str, *entries: str) -> Path:
    path = tmp_path / "versions.yaml"
    path.write_text(f"current: {current}\nversions:\n" + "".join(entries))
    return path


def test_the_current_version_is_listed():
    assert CURRENT in VERSIONS


def test_every_version_pins_full_commits():
    for version in VERSIONS:
        for commit in (version.data, version.labels, version.reference):
            assert COMMIT.fullmatch(commit)


def test_each_release_tag_belongs_to_its_version():
    for version in VERSIONS:
        assert version.release.startswith(f"v{version.number}.")


def test_the_current_version_is_the_minor_version_of_the_package():
    package = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    assert CURRENT.number == ".".join(package.split(".")[:2])


def test_the_current_version_is_read_from_the_list(tmp_path):
    path = _write(tmp_path, '"0.2"', _entry("0.2", "b" * 40), _entry("0.1"))

    current, versions = load_versions(path)

    assert [version.number for version in versions] == ["0.2", "0.1"]
    assert current.number == "0.2"
    assert current.labels == "b" * 40


def test_an_unlisted_current_version_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not listed"):
        load_versions(_write(tmp_path, '"0.2"', _entry("0.1")))


def test_a_version_listed_twice_is_refused(tmp_path):
    with pytest.raises(ValueError, match="listed twice"):
        load_versions(_write(tmp_path, '"0.1"', _entry("0.1"), _entry("0.1")))


def test_an_unquoted_number_is_refused(tmp_path):
    entry = _entry("0.10").replace('"0.10"', "0.10", 1)

    with pytest.raises(ValueError, match="quote"):
        load_versions(_write(tmp_path, '"0.10"', entry))
