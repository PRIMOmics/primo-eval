import hashlib
import json
import logging
import subprocess
import sys

import pytest
import yaml
from conftest import PUBLIC_TASKS, signal_blocks, write_submission

from primomics import cli, data
from primomics.evaluate import score_all


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


def _evaluate(submission, data_dir, *extra):
    return cli.main(
        [
            "evaluate",
            str(submission),
            "--modality",
            "bulk-rna",
            "--data-dir",
            str(data_dir),
            *extra,
        ]
    )


def test_evaluate_scores_public_tasks_and_validates_the_rest(
    public_tasks, local_data, tmp_path, caplog
):
    submission = write_submission(tmp_path / "sub.parquet", signal_blocks())
    output = tmp_path / "results.json"

    code = _evaluate(submission, local_data, "-o", str(output))

    assert code == 0
    report = json.loads(output.read_text())
    assert report["checks"] == {"d901": "ok", "d902": "ok"}
    assert [score["task_id"] for score in report["per_task"]] == ["t901"]
    assert report["per_task"][0]["score"] > 0.9
    assert len(report["per_task"][0]["repeat_scores"]) == 2
    assert report["n_scored"] == report["n_total"] == 1
    assert report["unlabelled"] == ["t905"]
    assert (
        report["submission"]["sha256"]
        == hashlib.sha256(submission.read_bytes()).hexdigest()
    )
    assert sorted(report["data"]["files"]) == [
        "d901/expression.h5ad",
        "d902/expression.h5ad",
        "public_tasks/t901/labels.csv",
    ]
    assert "Scored 1 of 1 labelled public task(s)." in caplog.text
    assert "no local labels, not scored: ['t905']" in caplog.text
    assert "Validated, scored on submit: d902" in caplog.text


def test_evaluate_matches_score_all_on_the_same_files(
    public_tasks, local_data, tmp_path
):
    submission = write_submission(tmp_path / "sub.parquet", signal_blocks())
    output = tmp_path / "results.json"
    fetch_labels, _ = data.local_fetchers(local_data)
    expected = score_all(
        submission,
        datasets=data.local_manifest(local_data),
        tasks=[PUBLIC_TASKS[0]],
        fetch_labels=fetch_labels,
        modality="bulk-rna",
    )

    _evaluate(submission, local_data, "-o", str(output))

    report = json.loads(output.read_text())
    assert report["per_task"][0]["score"] == expected["per_task"][0].score


def test_evaluate_stops_before_scoring_when_a_dataset_fails(
    public_tasks, local_data, tmp_path, caplog
):
    blocks = signal_blocks()
    ids, matrix = blocks["d902"]
    blocks["d902"] = (ids[:5], matrix[:5])
    submission = write_submission(tmp_path / "sub.parquet", blocks)
    output = tmp_path / "results.json"

    code = _evaluate(submission, local_data, "-o", str(output))

    assert code == 1
    assert not output.exists()
    assert "5/10 samples missing" in caplog.text
    assert "fix these datasets" in caplog.text


def test_evaluate_without_data_asks_for_a_download(public_tasks, tmp_path, caplog):
    submission = write_submission(tmp_path / "sub.parquet", signal_blocks())

    code = _evaluate(submission, tmp_path / "nothing")

    assert code == 1
    assert "run primo download first" in caplog.text
