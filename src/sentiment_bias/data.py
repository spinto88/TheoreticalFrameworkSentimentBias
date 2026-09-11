"""
Mention counts: the canonical container and every way to build one.

The canonical form is :class:`MentionTensor`, an ``(m, k, 3)`` integer array
indexed by (outlet, subject, sentiment) where the sentiment axis is ordered
``(negative, neutral, positive)``.

Two DataFrame layouts are supported:

- **wide** — one row per (outlet, subject), with ``negative``/``neutral``/
  ``positive`` columns. This is the tensor with its last axis unrolled, so it
  round-trips losslessly.
- **long** — one row per (outlet, subject, sentiment), with ``mention_type``
  and ``amount_of_mentions`` columns. This is what ``data/input.csv`` holds
  and what the HTTP API receives.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "MentionTensor",
    "SENTIMENTS",
    "to_tensor",
    "from_wide",
    "from_long",
    "from_records",
    "from_array",
    "read_csv",
    "read_json",
]

SENTIMENTS: tuple[str, str, str] = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class MentionTensor:
    """Mention counts for every (outlet, subject) pair.

    Attributes:
        counts: Integer array of shape ``(m, k, 3)``; the last axis is
            ordered ``(negative, neutral, positive)``.
        outlets: Outlet names, length *m*, matching ``counts`` axis 0.
        subjects: Subject names, length *k*, matching ``counts`` axis 1.
    """

    counts: NDArray[np.int64]
    outlets: list[str]
    subjects: list[str]

    def __post_init__(self) -> None:
        if self.counts.ndim != 3 or self.counts.shape[2] != 3:
            raise ValueError(f"counts must have shape (m, k, 3), got {self.counts.shape}")
        if not np.issubdtype(self.counts.dtype, np.integer):
            # Float counts would be silently truncated by every consumer.
            raise TypeError(f"counts must have an integer dtype, got {self.counts.dtype}")
        if self.counts.shape[0] != len(self.outlets):
            raise ValueError(
                f"counts has {self.counts.shape[0]} outlets but {len(self.outlets)} names were given"
            )
        if self.counts.shape[1] != len(self.subjects):
            raise ValueError(
                f"counts has {self.counts.shape[1]} subjects but {len(self.subjects)} names were given"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the count array, ``(m, k, 3)``."""
        return self.counts.shape

    @property
    def n_outlets(self) -> int:
        """Number of outlets *m*."""
        return self.counts.shape[0]

    @property
    def n_subjects(self) -> int:
        """Number of subjects *k*."""
        return self.counts.shape[1]

    @property
    def n_mentions(self) -> int:
        """Total number of mentions across every cell."""
        return int(self.counts.sum())

    def to_frame(self, layout: Literal["wide", "long"] = "wide") -> pd.DataFrame:
        """Render the counts as a DataFrame.

        Args:
            layout: ``"wide"`` gives one row per (outlet, subject) with a
                column per sentiment; ``"long"`` gives one row per
                (outlet, subject, sentiment).

        Returns:
            A DataFrame with ``outlet``/``subject`` columns plus either the
            three sentiment columns or ``mention_type``/``amount_of_mentions``.
        """
        index = pd.MultiIndex.from_product(
            [self.outlets, self.subjects], names=["outlet", "subject"]
        )
        wide = pd.DataFrame(
            self.counts.reshape(-1, 3), index=index, columns=list(SENTIMENTS)
        ).reset_index()
        if layout == "wide":
            return wide
        if layout == "long":
            return wide.melt(
                id_vars=["outlet", "subject"],
                value_vars=list(SENTIMENTS),
                var_name="mention_type",
                value_name="amount_of_mentions",
            ).sort_values(["outlet", "subject"], kind="stable", ignore_index=True)
        raise ValueError(f"layout must be 'wide' or 'long', got {layout!r}")


def from_array(
    counts: ArrayLike,
    outlets: Sequence[str] | None = None,
    subjects: Sequence[str] | None = None,
) -> MentionTensor:
    """Wrap a raw ``(m, k, 3)`` count array, generating names if needed.

    Args:
        counts: Array-like of shape ``(m, k, 3)``.
        outlets: Outlet names; defaults to ``outlet_00``, ``outlet_01``, …
        subjects: Subject names; defaults to ``subject_00``, ``subject_01``, …

    Returns:
        The corresponding :class:`MentionTensor`.
    """
    array = np.asarray(counts, dtype=np.int64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"counts must have shape (m, k, 3), got {array.shape}")
    m, k = array.shape[0], array.shape[1]
    return MentionTensor(
        counts=array,
        outlets=list(outlets) if outlets is not None else [f"outlet_{i:02d}" for i in range(m)],
        subjects=list(subjects) if subjects is not None else [f"subject_{j:02d}" for j in range(k)],
    )


def from_wide(
    df: pd.DataFrame,
    *,
    outlet: str = "outlet",
    subject: str = "subject",
    sentiment_columns: Sequence[str] = SENTIMENTS,
) -> MentionTensor:
    """Build a tensor from a wide DataFrame (one row per outlet-subject pair).

    Args:
        df: Frame with outlet/subject key columns and one column per sentiment.
        outlet: Name of the outlet column.
        subject: Name of the subject column.
        sentiment_columns: Column names for negative, neutral and positive,
            **in that order**.

    Returns:
        The corresponding :class:`MentionTensor`. Duplicate pairs are summed.
    """
    missing = {outlet, subject}.union(sentiment_columns) - set(df.columns)
    if missing:
        raise ValueError(f"wide frame is missing columns: {sorted(missing)}")

    # Melt only the columns we need: any other column the caller happens to
    # carry (an 'amount_of_mentions' left over from a long frame, say) would
    # collide with melt's value_name and fail deep inside pandas.
    long = df[[outlet, subject, *sentiment_columns]].melt(
        id_vars=[outlet, subject],
        value_vars=list(sentiment_columns),
        var_name="mention_type",
        value_name="amount_of_mentions",
    )
    long["mention_type"] = long["mention_type"].map(
        dict(zip(sentiment_columns, SENTIMENTS))
    )
    return from_long(long, outlet=outlet, subject=subject)


def from_long(
    df: pd.DataFrame,
    *,
    outlet: str = "outlet",
    subject: str = "subject",
    mention_type: str = "mention_type",
    count: str = "amount_of_mentions",
) -> MentionTensor:
    """Build a tensor from a long/tidy DataFrame (one row per sentiment).

    Outlets and subjects are sorted alphabetically; this ordering determines
    the tensor's index order everywhere downstream. Rows repeating the same
    ``(outlet, subject, mention_type)`` triple are summed, and absent
    combinations become zeros.

    Args:
        df: Frame with outlet, subject, sentiment and count columns.
        outlet: Name of the outlet column.
        subject: Name of the subject column.
        mention_type: Name of the sentiment column; its values must be
            ``negative``, ``neutral`` or ``positive``.
        count: Name of the mention-count column.

    Returns:
        The corresponding :class:`MentionTensor`.
    """
    missing = {outlet, subject, mention_type, count} - set(df.columns)
    if missing:
        raise ValueError(f"long frame is missing columns: {sorted(missing)}")

    unknown = set(df[mention_type].unique()) - set(SENTIMENTS)
    if unknown:
        raise ValueError(f"unknown mention_type values: {sorted(unknown)}")

    outlets = sorted(df[outlet].unique())
    subjects = sorted(df[subject].unique())

    pivot = df.pivot_table(
        index=[outlet, subject],
        columns=mention_type,
        values=count,
        aggfunc="sum",
        fill_value=0,
    ).reindex(
        index=pd.MultiIndex.from_product([outlets, subjects]),
        columns=list(SENTIMENTS),
        fill_value=0,
    )

    counts = pivot.to_numpy(dtype=np.int64).reshape(len(outlets), len(subjects), 3)
    return MentionTensor(counts=counts, outlets=outlets, subjects=subjects)


def from_records(records: Iterable[Mapping[str, Any] | Any]) -> MentionTensor:
    """Build a tensor from mention records (dicts or objects with attributes).

    Each record must expose ``outlet``, ``subject``, ``mention_type`` and
    ``amount_of_mentions``, either as mapping keys or as attributes. This is
    what lets the HTTP layer pass Pydantic ``Mention`` objects straight in
    without the core ever importing Pydantic.

    Args:
        records: Iterable of mention records.

    Returns:
        The corresponding :class:`MentionTensor`.
    """
    fields = ("outlet", "subject", "mention_type", "amount_of_mentions")
    rows = [
        {name: _field(record, name) for name in fields}
        for record in records
    ]
    if not rows:
        raise ValueError("cannot build a tensor from an empty record list")
    return from_long(pd.DataFrame(rows))


def _field(record: Mapping[str, Any] | Any, name: str) -> Any:
    """Read ``name`` off a record, whether it is a mapping or an object."""
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def to_tensor(data: Any) -> MentionTensor:
    """Coerce any supported input into a :class:`MentionTensor`.

    Dispatches on type: a :class:`MentionTensor` passes through, a DataFrame
    is sniffed for the wide or long layout, a 3-D array goes through
    :func:`from_array`, and any other iterable is treated as records.

    Args:
        data: A MentionTensor, DataFrame, ``(m, k, 3)`` array, or iterable of
            mention records.

    Returns:
        The corresponding :class:`MentionTensor`.
    """
    if isinstance(data, MentionTensor):
        return data
    if isinstance(data, pd.DataFrame):
        columns = set(data.columns)
        if set(SENTIMENTS) <= columns:
            return from_wide(data)
        if "mention_type" in columns:
            return from_long(data)
        raise TypeError(
            "DataFrame matches neither layout: wide needs columns "
            f"{sorted(set(SENTIMENTS))} and long needs 'mention_type'; got {sorted(columns)}"
        )
    if isinstance(data, np.ndarray):
        return from_array(data)
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        return from_records(data)
    raise TypeError(f"cannot build a MentionTensor from {type(data).__name__}")


def read_csv(
    path: str | Path,
    *,
    layout: Literal["auto", "wide", "long"] = "auto",
    **kwargs: Any,
) -> MentionTensor:
    """Read mention counts from a CSV file.

    Args:
        path: Path to the CSV file.
        layout: ``"auto"`` sniffs the columns; force ``"wide"`` or ``"long"``
            to skip the sniffing.
        **kwargs: Forwarded to :func:`pandas.read_csv`.

    Returns:
        The corresponding :class:`MentionTensor`.
    """
    df = pd.read_csv(path, **kwargs)
    if layout == "auto":
        return to_tensor(df)
    if layout == "wide":
        return from_wide(df)
    if layout == "long":
        return from_long(df)
    raise ValueError(f"layout must be 'auto', 'wide' or 'long', got {layout!r}")


def read_json(path: str | Path) -> MentionTensor:
    """Read mention counts from a JSON file.

    Accepts either the API's ``{"data": [...]}`` envelope or a bare list of
    mention records.

    Args:
        path: Path to the JSON file.

    Returns:
        The corresponding :class:`MentionTensor`.
    """
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["data"] if isinstance(payload, Mapping) else payload
    return from_records(records)
