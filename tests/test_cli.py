import logging
import subprocess
import sys

import pytest
import yaml

from primomics import cli


@pytest.fixture(autouse=True)
def _info_logs(caplog):
    caplog.set_level(logging.INFO)


def test_download_lists_each_dataset_with_its_public_tasks(
    fake_hub, public_tasks, tmp_path, caplog
):
    code = cli.main(["download", "--modality", "bulk-rna", "--data-dir", str(tmp_path)])

    assert code == 0
    assert (tmp_path / "d901" / "expression.h5ad").exists()
    assert "up to date in" in caplog.text
    lines = caplog.text.splitlines()
    assert any("d901" in line and "t901" in line for line in lines)
    assert any("d902" in line and line.rstrip().endswith("-") for line in lines)


def test_download_parses_comma_separated_datasets(fake_hub, public_tasks, tmp_path):
    code = cli.main(
        [
            "download",
            "--modality",
            "bulk-rna",
            "--datasets",
            "d902, d901",
            "--data-dir",
            str(tmp_path),
        ]
    )

    assert code == 0
    assert (tmp_path / "d901").exists() and (tmp_path / "d902").exists()


def test_an_input_error_exits_with_1(fake_hub, public_tasks, tmp_path, caplog):
    code = cli.main(
        [
            "download",
            "--modality",
            "bulk-rna",
            "--datasets",
            "d999",
            "--data-dir",
            str(tmp_path),
        ]
    )

    assert code == 1
    assert "unknown dataset id" in caplog.text


def test_a_failure_on_our_side_exits_with_2(
    fake_hub, public_tasks, tmp_path, monkeypatch, caplog
):
    def offline(repo_id):
        raise ConnectionError("no network")

    monkeypatch.setattr(fake_hub, "dataset_info", offline)

    code = cli.main(["download", "--modality", "bulk-rna", "--data-dir", str(tmp_path)])

    assert code == 2
    assert "on our side" in caplog.text


def test_an_unexpected_error_exits_with_2(fake_hub, public_tasks, tmp_path, caplog):
    manifest = {"datasets": [{"id": "d901", "modality": "bulk RNA"}]}
    fake_hub.files["datasets.yaml"] = yaml.safe_dump(manifest).encode()

    code = cli.main(["download", "--modality", "bulk-rna", "--data-dir", str(tmp_path)])

    assert code == 2
    assert "KeyError: 'path'" in caplog.text


def test_errors_go_to_stderr(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    script = "import sys; from primomics.cli import main; sys.exit(main(sys.argv[1:]))"
    args = ["download", "--modality", "bulk-rna", "--data-dir", str(tmp_path)]

    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "on our side" in result.stderr
    assert result.stdout == ""


def test_a_missing_modality_is_an_argparse_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download"])

    assert exit_info.value.code == 1
    assert "--modality" in capsys.readouterr().err
