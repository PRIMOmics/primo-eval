import tomllib
from pathlib import Path

import primomics

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def test_installed_version_matches_pyproject():
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    assert primomics.__version__ == project["version"]
