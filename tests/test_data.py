"""
Unit tests for sentiment_bias.data.

Covers MentionTensor's invariants and every way of building one: wide and
long DataFrames, records (dicts or objects), raw arrays, and CSV/JSON files.

The sentiment axis is ordered (negative, neutral, positive); outlets and
subjects are sorted alphabetically, which fixes the index order used
everywhere downstream.
"""

import json

import numpy as np
import pandas as pd
import pytest

from sentiment_bias.data import (
    SENTIMENTS,
    MentionTensor,
    from_array,
    from_long,
    from_records,
    from_wide,
    read_csv,
    read_json,
    to_tensor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def record(outlet: str, subject: str, mention_type: str, n: int) -> dict:
    return {
        "outlet": outlet,
        "subject": subject,
        "mention_type": mention_type,
        "amount_of_mentions": n,
    }


class MentionLike:
    """Duck-typed stand-in for a Pydantic Mention (attributes, not keys)."""

    def __init__(self, outlet, subject, mention_type, amount_of_mentions):
        self.outlet = outlet
        self.subject = subject
        self.mention_type = mention_type
        self.amount_of_mentions = amount_of_mentions


SIMPLE_RECORDS = [
    record("A", "X", "positive", 10),
    record("A", "X", "negative", 3),
    record("B", "X", "neutral", 6),
    record("B", "Y", "positive", 5),
]


# ---------------------------------------------------------------------------
# MentionTensor
# ---------------------------------------------------------------------------

class TestMentionTensor:
    def test_shape_property(self):
        tensor = from_array(np.zeros((3, 2, 3)))
        assert tensor.shape == (3, 2, 3)

    def test_n_outlets_and_n_subjects(self):
        tensor = from_array(np.zeros((4, 2, 3)))
        assert tensor.n_outlets == 4
        assert tensor.n_subjects == 2

    def test_n_mentions_sums_every_cell(self):
        counts = np.arange(2 * 2 * 3).reshape(2, 2, 3)
        assert from_array(counts).n_mentions == int(counts.sum())

    def test_n_mentions_is_python_int(self):
        assert isinstance(from_array(np.ones((1, 1, 3))).n_mentions, int)

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError, match=r"shape \(m, k, 3\)"):
            MentionTensor(counts=np.zeros((2, 2)), outlets=["A", "B"], subjects=["X", "Y"])

    def test_rejects_wrong_last_axis(self):
        with pytest.raises(ValueError, match=r"shape \(m, k, 3\)"):
            MentionTensor(counts=np.zeros((2, 2, 4)), outlets=["A", "B"], subjects=["X", "Y"])

    def test_rejects_outlet_name_count_mismatch(self):
        with pytest.raises(ValueError, match="outlets"):
            MentionTensor(counts=np.zeros((2, 1, 3)), outlets=["A"], subjects=["X"])

    def test_rejects_subject_name_count_mismatch(self):
        with pytest.raises(ValueError, match="subjects"):
            MentionTensor(counts=np.zeros((1, 2, 3)), outlets=["A"], subjects=["X"])

    def test_is_frozen(self):
        tensor = from_array(np.zeros((1, 1, 3)))
        with pytest.raises(Exception):
            tensor.counts = np.zeros((1, 1, 3))


# ---------------------------------------------------------------------------
# to_frame
# ---------------------------------------------------------------------------

class TestToFrame:
    @pytest.fixture
    def tensor(self):
        return from_records(SIMPLE_RECORDS)

    def test_wide_columns(self, tensor):
        frame = tensor.to_frame("wide")
        assert list(frame.columns) == ["outlet", "subject", *SENTIMENTS]

    def test_wide_row_count_is_pairs(self, tensor):
        assert len(tensor.to_frame("wide")) == tensor.n_outlets * tensor.n_subjects

    def test_long_columns(self, tensor):
        frame = tensor.to_frame("long")
        assert set(frame.columns) == {"outlet", "subject", "mention_type", "amount_of_mentions"}

    def test_long_row_count_is_pairs_times_three(self, tensor):
        assert len(tensor.to_frame("long")) == tensor.n_outlets * tensor.n_subjects * 3

    def test_wide_is_default(self, tensor):
        pd.testing.assert_frame_equal(tensor.to_frame(), tensor.to_frame("wide"))

    def test_long_total_matches_n_mentions(self, tensor):
        assert tensor.to_frame("long")["amount_of_mentions"].sum() == tensor.n_mentions

    def test_invalid_layout_raises(self, tensor):
        with pytest.raises(ValueError, match="layout"):
            tensor.to_frame("tall")

    def test_wide_round_trip(self, tensor):
        np.testing.assert_array_equal(from_wide(tensor.to_frame("wide")).counts, tensor.counts)

    def test_long_round_trip(self, tensor):
        np.testing.assert_array_equal(from_long(tensor.to_frame("long")).counts, tensor.counts)

    def test_wide_to_long_to_tensor_round_trip(self, tensor):
        via_wide = from_wide(tensor.to_frame("wide"))
        assert from_long(via_wide.to_frame("long")).counts.tolist() == tensor.counts.tolist()

    def test_names_survive_round_trip(self, tensor):
        rebuilt = from_wide(tensor.to_frame("wide"))
        assert rebuilt.outlets == tensor.outlets
        assert rebuilt.subjects == tensor.subjects


# ---------------------------------------------------------------------------
# from_long — the shape on disk and on the wire
# ---------------------------------------------------------------------------

class TestFromLong:
    def test_output_shape(self):
        tensor = from_records(SIMPLE_RECORDS)
        assert tensor.shape == (tensor.n_outlets, tensor.n_subjects, 3)

    def test_outlets_are_sorted(self):
        rows = [record(o, "X", "positive", 1) for o in ["C", "A", "B"]]
        assert from_long(pd.DataFrame(rows)).outlets == ["A", "B", "C"]

    def test_subjects_are_sorted(self):
        rows = [record("A", s, "positive", 1) for s in ["Z", "M", "A"]]
        assert from_long(pd.DataFrame(rows)).subjects == ["A", "M", "Z"]

    def test_negative_count_placed_at_index_0(self):
        tensor = from_long(pd.DataFrame([record("A", "X", "negative", 4)]))
        assert tensor.counts[0, 0, 0] == 4

    def test_neutral_count_placed_at_index_1(self):
        tensor = from_long(pd.DataFrame([record("A", "X", "neutral", 9)]))
        assert tensor.counts[0, 0, 1] == 9

    def test_positive_count_placed_at_index_2(self):
        tensor = from_long(pd.DataFrame([record("A", "X", "positive", 7)]))
        assert tensor.counts[0, 0, 2] == 7
        assert tensor.counts[0, 0, 0] == 0
        assert tensor.counts[0, 0, 1] == 0

    def test_duplicate_rows_are_summed(self):
        """Preserves the old build_tensor behaviour: repeated triples accumulate."""
        rows = [record("A", "X", "positive", 5), record("A", "X", "positive", 3)]
        assert from_long(pd.DataFrame(rows)).counts[0, 0, 2] == 8

    def test_duplicates_summed_across_three_rows(self):
        rows = [record("A", "X", "negative", 1)] * 3
        assert from_long(pd.DataFrame(rows)).counts[0, 0, 0] == 3

    def test_unobserved_pair_is_zero(self):
        tensor = from_records(SIMPLE_RECORDS)
        a_idx, y_idx = tensor.outlets.index("A"), tensor.subjects.index("Y")
        assert tensor.counts[a_idx, y_idx].sum() == 0

    def test_multiple_outlets_and_subjects(self):
        tensor = from_records(SIMPLE_RECORDS)
        assert tensor.outlets == ["A", "B"]
        assert tensor.subjects == ["X", "Y"]
        a_idx, x_idx = tensor.outlets.index("A"), tensor.subjects.index("X")
        assert tensor.counts[a_idx, x_idx, 2] == 10
        assert tensor.counts[a_idx, x_idx, 0] == 3
        b_idx, y_idx = tensor.outlets.index("B"), tensor.subjects.index("Y")
        assert tensor.counts[b_idx, y_idx, 2] == 5

    def test_single_outlet_single_subject(self):
        tensor = from_long(pd.DataFrame([record("Solo", "Topic", "neutral", 1)]))
        assert tensor.shape == (1, 1, 3)
        assert tensor.outlets == ["Solo"]
        assert tensor.subjects == ["Topic"]

    def test_counts_dtype_is_integer(self):
        assert from_records(SIMPLE_RECORDS).counts.dtype == np.int64

    def test_missing_column_raises(self):
        df = pd.DataFrame([{"outlet": "A", "subject": "X", "mention_type": "positive"}])
        with pytest.raises(ValueError, match="missing columns"):
            from_long(df)

    def test_unknown_mention_type_raises(self):
        df = pd.DataFrame([record("A", "X", "mixed", 1)])
        with pytest.raises(ValueError, match="unknown mention_type"):
            from_long(df)

    def test_custom_column_names(self):
        df = pd.DataFrame([{"src": "A", "topic": "X", "pol": "positive", "n": 6}])
        tensor = from_long(df, outlet="src", subject="topic", mention_type="pol", count="n")
        assert tensor.outlets == ["A"]
        assert tensor.counts[0, 0, 2] == 6


# ---------------------------------------------------------------------------
# from_wide
# ---------------------------------------------------------------------------

class TestFromWide:
    def test_basic(self):
        df = pd.DataFrame([
            {"outlet": "A", "subject": "X", "negative": 1, "neutral": 2, "positive": 3},
            {"outlet": "B", "subject": "X", "negative": 4, "neutral": 5, "positive": 6},
        ])
        tensor = from_wide(df)
        assert tensor.outlets == ["A", "B"]
        np.testing.assert_array_equal(tensor.counts[0, 0], [1, 2, 3])
        np.testing.assert_array_equal(tensor.counts[1, 0], [4, 5, 6])

    def test_missing_pair_becomes_zero(self):
        df = pd.DataFrame([
            {"outlet": "A", "subject": "X", "negative": 1, "neutral": 0, "positive": 0},
            {"outlet": "B", "subject": "Y", "negative": 0, "neutral": 0, "positive": 2},
        ])
        tensor = from_wide(df)
        assert tensor.shape == (2, 2, 3)
        assert tensor.counts[0, 1].sum() == 0   # A/Y never observed
        assert tensor.counts[1, 0].sum() == 0   # B/X never observed

    def test_duplicate_pairs_are_summed(self):
        df = pd.DataFrame([
            {"outlet": "A", "subject": "X", "negative": 1, "neutral": 0, "positive": 2},
            {"outlet": "A", "subject": "X", "negative": 3, "neutral": 0, "positive": 4},
        ])
        np.testing.assert_array_equal(from_wide(df).counts[0, 0], [4, 0, 6])

    def test_missing_column_raises(self):
        df = pd.DataFrame([{"outlet": "A", "subject": "X", "negative": 1, "neutral": 2}])
        with pytest.raises(ValueError, match="missing columns"):
            from_wide(df)

    def test_custom_sentiment_columns(self):
        df = pd.DataFrame([{"outlet": "A", "subject": "X", "neg": 1, "neu": 2, "pos": 3}])
        tensor = from_wide(df, sentiment_columns=("neg", "neu", "pos"))
        np.testing.assert_array_equal(tensor.counts[0, 0], [1, 2, 3])

    def test_custom_key_columns(self):
        df = pd.DataFrame([
            {"src": "A", "topic": "X", "negative": 1, "neutral": 2, "positive": 3}
        ])
        tensor = from_wide(df, outlet="src", subject="topic")
        assert tensor.outlets == ["A"]
        assert tensor.subjects == ["X"]


# ---------------------------------------------------------------------------
# from_records
# ---------------------------------------------------------------------------

class TestFromRecords:
    def test_from_dicts(self):
        tensor = from_records(SIMPLE_RECORDS)
        assert tensor.n_mentions == 24

    def test_from_objects_with_attributes(self):
        records = [MentionLike("A", "X", "positive", 4), MentionLike("B", "X", "negative", 6)]
        tensor = from_records(records)
        assert tensor.outlets == ["A", "B"]
        assert tensor.counts[0, 0, 2] == 4
        assert tensor.counts[1, 0, 0] == 6

    def test_mixed_dicts_and_objects(self):
        records = [record("A", "X", "positive", 2), MentionLike("A", "X", "negative", 3)]
        tensor = from_records(records)
        assert tensor.counts[0, 0, 2] == 2
        assert tensor.counts[0, 0, 0] == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty record list"):
            from_records([])

    def test_missing_key_raises(self):
        with pytest.raises((KeyError, AttributeError)):
            from_records([{"outlet": "A", "subject": "X"}])


# ---------------------------------------------------------------------------
# from_array
# ---------------------------------------------------------------------------

class TestFromArray:
    def test_generated_outlet_names(self):
        assert from_array(np.zeros((3, 1, 3))).outlets == ["outlet_00", "outlet_01", "outlet_02"]

    def test_generated_subject_names(self):
        assert from_array(np.zeros((1, 2, 3))).subjects == ["subject_00", "subject_01"]

    def test_explicit_names(self):
        tensor = from_array(np.zeros((2, 1, 3)), outlets=["CNN", "Fox"], subjects=["Economy"])
        assert tensor.outlets == ["CNN", "Fox"]
        assert tensor.subjects == ["Economy"]

    def test_accepts_nested_lists(self):
        tensor = from_array([[[1, 2, 3]]])
        assert tensor.shape == (1, 1, 3)
        assert tensor.n_mentions == 6

    def test_casts_to_int64(self):
        assert from_array(np.ones((1, 1, 3), dtype=float)).counts.dtype == np.int64

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match=r"shape \(m, k, 3\)"):
            from_array(np.zeros((2, 2)))

    def test_wrong_last_axis_raises(self):
        with pytest.raises(ValueError, match=r"shape \(m, k, 3\)"):
            from_array(np.zeros((2, 2, 5)))

    def test_names_are_not_sorted(self):
        """from_array trusts the caller's order — the array already fixes it."""
        tensor = from_array(np.zeros((2, 1, 3)), outlets=["Z", "A"], subjects=["X"])
        assert tensor.outlets == ["Z", "A"]


# ---------------------------------------------------------------------------
# to_tensor dispatch
# ---------------------------------------------------------------------------

class TestToTensor:
    def test_mention_tensor_passes_through(self):
        tensor = from_records(SIMPLE_RECORDS)
        assert to_tensor(tensor) is tensor

    def test_wide_dataframe(self):
        tensor = from_records(SIMPLE_RECORDS)
        np.testing.assert_array_equal(to_tensor(tensor.to_frame("wide")).counts, tensor.counts)

    def test_long_dataframe(self):
        tensor = from_records(SIMPLE_RECORDS)
        np.testing.assert_array_equal(to_tensor(tensor.to_frame("long")).counts, tensor.counts)

    def test_ndarray(self):
        counts = np.arange(2 * 1 * 3).reshape(2, 1, 3)
        result = to_tensor(counts)
        np.testing.assert_array_equal(result.counts, counts)
        assert result.outlets == ["outlet_00", "outlet_01"]

    def test_list_of_dicts(self):
        result = to_tensor([record("A", "X", "positive", 3)])
        assert result.counts.tolist() == [[[0, 0, 3]]]

    def test_list_of_objects(self):
        result = to_tensor([MentionLike("A", "X", "negative", 4)])
        assert result.counts.tolist() == [[[4, 0, 0]]]

    def test_unrecognised_dataframe_raises_type_error(self):
        with pytest.raises(TypeError, match="matches neither layout"):
            to_tensor(pd.DataFrame({"foo": [1], "bar": [2]}))

    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError):
            to_tensor(42)

    def test_wide_wins_over_leftover_mention_type_column(self):
        """A frame with the wide columns is read as wide, extra columns ignored."""
        df = pd.DataFrame([{
            "outlet": "A", "subject": "X",
            "negative": 1, "neutral": 2, "positive": 3,
            "mention_type": "positive",
        }])
        np.testing.assert_array_equal(to_tensor(df).counts[0, 0], [1, 2, 3])


# ---------------------------------------------------------------------------
# read_csv / read_json
# ---------------------------------------------------------------------------

class TestReaders:
    @pytest.fixture
    def long_csv(self, tmp_path):
        path = tmp_path / "long.csv"
        pd.DataFrame(SIMPLE_RECORDS).to_csv(path, index=False)
        return path

    @pytest.fixture
    def wide_csv(self, tmp_path):
        path = tmp_path / "wide.csv"
        from_records(SIMPLE_RECORDS).to_frame("wide").to_csv(path, index=False)
        return path

    def test_read_csv_long_auto(self, long_csv):
        np.testing.assert_array_equal(
            read_csv(long_csv).counts, from_records(SIMPLE_RECORDS).counts
        )

    def test_read_csv_wide_auto(self, wide_csv):
        np.testing.assert_array_equal(
            read_csv(wide_csv).counts, from_records(SIMPLE_RECORDS).counts
        )

    def test_read_csv_explicit_long(self, long_csv):
        assert read_csv(long_csv, layout="long").n_mentions == 24

    def test_read_csv_explicit_wide(self, wide_csv):
        assert read_csv(wide_csv, layout="wide").n_mentions == 24

    def test_read_csv_bad_layout_raises(self, long_csv):
        with pytest.raises(ValueError, match="layout"):
            read_csv(long_csv, layout="tall")

    def test_read_json_envelope(self, tmp_path):
        path = tmp_path / "input.json"
        path.write_text(json.dumps({"data": SIMPLE_RECORDS}))
        np.testing.assert_array_equal(
            read_json(path).counts, from_records(SIMPLE_RECORDS).counts
        )

    def test_read_json_bare_list(self, tmp_path):
        path = tmp_path / "bare.json"
        path.write_text(json.dumps(SIMPLE_RECORDS))
        np.testing.assert_array_equal(
            read_json(path).counts, from_records(SIMPLE_RECORDS).counts
        )

    def test_csv_and_json_agree(self, long_csv, tmp_path):
        json_path = tmp_path / "same.json"
        json_path.write_text(json.dumps({"data": SIMPLE_RECORDS}))
        np.testing.assert_array_equal(read_csv(long_csv).counts, read_json(json_path).counts)
