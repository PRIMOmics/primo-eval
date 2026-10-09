import io
import pickle

import numpy as np
import pandas as pd
import pytest

from primomics.errors import SubmissionError
from primomics.submission import (
    DATASET_ID,
    SAMPLE_ID,
    _split_by_dataset,
    l2_normalize_rows,
    load_submission,
    norm_id,
    to_embedding_frame,
)


def _frame(blocks):
    """blocks = {dataset_id: (ids, matrix)} -> one NaN-padded submission frame."""
    frames = []
    for dataset_id, (ids, mat) in blocks.items():
        df = pd.DataFrame(mat, columns=[f"e{j}" for j in range(mat.shape[1])])
        df.insert(0, SAMPLE_ID, ids)
        df.insert(0, DATASET_ID, dataset_id)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_loads_a_multi_dataset_csv(tmp_path):
    rng = np.random.default_rng(0)
    blocks = {
        "d901": (["a", "b"], rng.normal(size=(2, 4))),
        "d902": (["0", "1", "2"], rng.normal(size=(3, 4))),
    }
    _frame(blocks).to_csv(tmp_path / "sub.csv", index=False)

    out = load_submission(tmp_path / "sub.csv")

    assert set(out) == {"d901", "d902"}
    assert to_embedding_frame(out["d901"]).shape == (2, 4)
    assert list(to_embedding_frame(out["d902"]).index) == ["0", "1", "2"]


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".txt", ".parquet", ".pq"])
def test_every_table_format_reads_the_same(tmp_path, suffix):
    frame = _frame({"d901": (["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]]))})
    path = tmp_path / f"sub{suffix}"
    if suffix in (".parquet", ".pq"):
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, sep="," if suffix == ".csv" else "\t", index=False)

    block = to_embedding_frame(load_submission(path)["d901"])

    assert block.loc["b"].tolist() == [3.0, 4.0]


def test_loads_a_multi_dataset_npz(tmp_path):
    rng = np.random.default_rng(1)
    np.savez(
        tmp_path / "sub.npz",
        dataset_ids=np.array(["d901", "d901", "d902", "d902"]),
        sample_ids=np.array(["a", "b", "c", "d"]),
        embeddings=rng.normal(size=(4, 5)),
    )

    out = load_submission(tmp_path / "sub.npz")

    assert set(out) == {"d901", "d902"}
    assert to_embedding_frame(out["d901"]).shape == (2, 5)


def test_each_dataset_keeps_its_own_width(tmp_path):
    blocks = {
        "d901": (["a", "b"], np.ones((2, 4))),
        "d902": (["a", "b"], np.ones((2, 6))),
    }
    _frame(blocks).to_csv(tmp_path / "sub.csv", index=False)

    out = load_submission(tmp_path / "sub.csv")

    assert to_embedding_frame(out["d901"]).shape[1] == 4
    assert to_embedding_frame(out["d902"]).shape[1] == 6


def test_sample_id_reused_across_datasets_ok(tmp_path):
    blocks = {
        "d901": (["x", "y"], np.ones((2, 3))),
        "d902": (["x", "y"], np.ones((2, 3))),
    }
    _frame(blocks).to_csv(tmp_path / "sub.csv", index=False)

    out = load_submission(tmp_path / "sub.csv")

    assert to_embedding_frame(out["d901"]).shape == (2, 3)
    assert to_embedding_frame(out["d902"]).shape == (2, 3)


def test_int_str_and_float_spellings_of_an_id_join():
    assert norm_id(1) == norm_id("1") == norm_id(1.0) == norm_id(" 1 ") == "1"
    assert norm_id(1.5) == "1.5"


def test_zero_vector_survives_normalisation():
    """An all-zero sample has no direction; it must not become NaN."""
    frame = l2_normalize_rows(
        pd.DataFrame([[0.0, 0.0], [3.0, 4.0]], index=["s0", "s1"])
    )

    assert frame.loc["s0"].tolist() == [0.0, 0.0]
    assert frame.loc["s1"].tolist() == [0.6, 0.8]


def test_unsupported_file_type_raises(tmp_path):
    path = tmp_path / "sub.xlsx"
    path.write_text("")
    with pytest.raises(SubmissionError, match="unsupported"):
        load_submission(path)


def test_missing_dataset_id_column_raises(tmp_path):
    frame = pd.DataFrame({SAMPLE_ID: ["a", "b"], "e0": [0.1, 0.2]})
    frame.to_csv(tmp_path / "sub.csv", index=False)
    with pytest.raises(SubmissionError, match="dataset_id"):
        load_submission(tmp_path / "sub.csv")


def test_empty_submission_raises(tmp_path):
    frame = pd.DataFrame(columns=[DATASET_ID, SAMPLE_ID, "e0"])
    frame.to_csv(tmp_path / "sub.csv", index=False)
    with pytest.raises(SubmissionError, match="empty"):
        load_submission(tmp_path / "sub.csv")


def test_npz_without_the_expected_arrays_raises(tmp_path):
    np.savez(tmp_path / "sub.npz", ids=np.array(["a"]), X=np.ones((1, 2)))
    with pytest.raises(SubmissionError, match="dataset-id array"):
        load_submission(tmp_path / "sub.npz")


def test_npz_with_1d_embeddings_raises(tmp_path):
    np.savez(
        tmp_path / "sub.npz",
        dataset_ids=np.array(["d901"]),
        sample_ids=np.array(["a"]),
        embeddings=np.ones(3),
    )
    with pytest.raises(SubmissionError, match="2D"):
        load_submission(tmp_path / "sub.npz")


def test_npz_with_object_arrays_is_refused(tmp_path):
    """Object arrays need pickle, which can run code hidden in an uploaded file."""
    np.savez(
        tmp_path / "sub.npz",
        dataset_ids=np.array(["d901"], dtype=object),
        sample_ids=np.array(["a"], dtype=object),
        embeddings=np.ones((1, 2)),
    )
    with pytest.raises(SubmissionError, match="Save ids as text"):
        load_submission(tmp_path / "sub.npz")


def _saved(save, array):
    buffer = io.BytesIO()
    save(buffer, array)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "content",
    [
        b"dataset_id,sample_id,e0\nd901,a,1.0\n",
        pickle.dumps({"sample_ids": ["a"]}),
        _saved(np.save, np.ones((2, 2))),
        _saved(np.savez, np.ones(3))[:60],
        b"",
    ],
    ids=["csv", "pickle", "npy", "truncated-npz", "empty"],
)
def test_a_file_that_is_not_an_npz_archive_is_refused(tmp_path, content):
    path = tmp_path / "sub.npz"
    path.write_bytes(content)
    with pytest.raises(SubmissionError, match="not a .npz archive"):
        load_submission(path)


def test_one_dataset_under_two_spellings_raises():
    frame = pd.DataFrame(
        {DATASET_ID: [1, "1"], SAMPLE_ID: ["a", "b"], "e0": [0.1, 0.2]}
    )
    with pytest.raises(SubmissionError, match="multiple spellings"):
        _split_by_dataset(frame)


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (pd.DataFrame({"e0": [1.0]}), "sample_id"),
        (pd.DataFrame({SAMPLE_ID: ["a"], "e0": ["x"]}), "numeric"),
        (pd.DataFrame({SAMPLE_ID: ["a"]}), "no embedding columns"),
        (pd.DataFrame({SAMPLE_ID: ["a", "a"], "e0": [1.0, 2.0]}), "duplicate"),
    ],
)
def test_a_bad_dataset_block_raises(block, message):
    with pytest.raises(SubmissionError, match=message):
        to_embedding_frame(block)
