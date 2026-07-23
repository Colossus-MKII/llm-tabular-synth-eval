from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TSTRResult:
    synthetic_auc: float
    real_auc: float
    n_train_synthetic: int
    n_train_real: int
    n_test_real: int


@dataclass(frozen=True)
class RevisionReport:
    n_rows_before: int
    n_rows_after: int
    n_columns_compared: int
    compared_cells: int
    revised_cells: int
    revised_rows: int
    cell_revision_rate: float
    row_revision_rate: float
    dropped_row_rate: float
    added_row_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_rows_before": self.n_rows_before,
            "n_rows_after": self.n_rows_after,
            "n_columns_compared": self.n_columns_compared,
            "compared_cells": self.compared_cells,
            "revised_cells": self.revised_cells,
            "revised_rows": self.revised_rows,
            "cell_revision_rate": self.cell_revision_rate,
            "row_revision_rate": self.row_revision_rate,
            "dropped_row_rate": self.dropped_row_rate,
            "added_row_rate": self.added_row_rate,
        }


def revision_rate(
    before: pd.DataFrame,
    after: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> RevisionReport:
    """Measure how much a correction/post-processing step changed a table.

    Rows are aligned by position. This is the right default for synthetic-data
    post-processing because generated rows usually do not have stable IDs.
    """

    compared_columns = list(columns or [col for col in before.columns if col in after.columns])
    n_before = len(before)
    n_after = len(after)
    n_aligned = min(n_before, n_after)
    n_columns = len(compared_columns)

    if n_aligned == 0 or n_columns == 0:
        compared_cells = n_aligned * n_columns
        return RevisionReport(
            n_rows_before=n_before,
            n_rows_after=n_after,
            n_columns_compared=n_columns,
            compared_cells=compared_cells,
            revised_cells=0,
            revised_rows=0,
            cell_revision_rate=0.0,
            row_revision_rate=0.0,
            dropped_row_rate=(max(n_before - n_after, 0) / n_before) if n_before else 0.0,
            added_row_rate=(max(n_after - n_before, 0) / n_after) if n_after else 0.0,
        )

    lhs = before.iloc[:n_aligned].loc[:, compared_columns].reset_index(drop=True)
    rhs = after.iloc[:n_aligned].loc[:, compared_columns].reset_index(drop=True)
    changes = pd.DataFrame(False, index=lhs.index, columns=lhs.columns)

    for column in compared_columns:
        left = lhs[column]
        right = rhs[column]
        both_missing = left.isna() & right.isna()
        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        both_numeric = left_num.notna() & right_num.notna()
        numeric_equal = pd.Series(
            np.isclose(left_num, right_num), index=left.index
        ) & both_numeric
        string_equal = left.astype(str).eq(right.astype(str))
        changes[column] = ~(numeric_equal | string_equal | both_missing)

    revised_cells = int(changes.to_numpy().sum())
    revised_rows = int(changes.any(axis=1).sum())
    compared_cells = n_aligned * n_columns
    return RevisionReport(
        n_rows_before=n_before,
        n_rows_after=n_after,
        n_columns_compared=n_columns,
        compared_cells=compared_cells,
        revised_cells=revised_cells,
        revised_rows=revised_rows,
        cell_revision_rate=float(revised_cells / compared_cells) if compared_cells else 0.0,
        row_revision_rate=float(revised_rows / n_aligned) if n_aligned else 0.0,
        dropped_row_rate=(max(n_before - n_after, 0) / n_before) if n_before else 0.0,
        added_row_rate=(max(n_after - n_before, 0) / n_after) if n_after else 0.0,
    )


def roc_auc_score_binary(y_true: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def tstr_auc(
    real_train: pd.DataFrame,
    synthetic_train: pd.DataFrame,
    real_test: pd.DataFrame,
    target_col: str,
    steps: int = 700,
    lr: float = 0.08,
    l2: float = 1e-4,
) -> TSTRResult:
    """Train-on-synthetic/test-on-real AUC with a tiny numpy logistic model."""

    synth_x, synth_y, test_x, test_y = _prepare_xy(synthetic_train, real_test, target_col)
    real_x, real_y, real_test_x, real_test_y = _prepare_xy(real_train, real_test, target_col)
    synth_coef = _fit_logistic(synth_x, synth_y, steps=steps, lr=lr, l2=l2)
    real_coef = _fit_logistic(real_x, real_y, steps=steps, lr=lr, l2=l2)
    synth_scores = _predict_logistic(test_x, synth_coef)
    real_scores = _predict_logistic(real_test_x, real_coef)
    return TSTRResult(
        synthetic_auc=roc_auc_score_binary(test_y, synth_scores),
        real_auc=roc_auc_score_binary(real_test_y, real_scores),
        n_train_synthetic=len(synthetic_train),
        n_train_real=len(real_train),
        n_test_real=len(real_test),
    )


def column_similarity(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    categorical_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    categorical = set(categorical_columns or [])
    rows = []
    for column in real.columns:
        if column not in synthetic.columns:
            rows.append({"column": column, "type": "missing", "distance": 1.0, "similarity": 0.0})
            continue

        if column in categorical or real[column].dtype == "object":
            distance = _total_variation_distance(real[column], synthetic[column])
            kind = "categorical_tvd"
        else:
            distance = _ks_distance(real[column], synthetic[column])
            kind = "numeric_ks"
        rows.append(
            {
                "column": column,
                "type": kind,
                "distance": distance,
                "similarity": max(0.0, 1.0 - distance),
            }
        )
    return pd.DataFrame(rows)


def distance_to_closest_record(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    categorical_columns: Sequence[str] | None = None,
    chunk_size: int = 256,
    max_real_records: int | None = None,
    max_synthetic_records: int | None = None,
    seed: int = 42,
) -> pd.Series:
    """Mixed-type Distance to Closest Record.

    Numeric columns are min-max scaled using real data. Categorical columns are
    compared by exact match with a 0/1 penalty.
    """

    categorical = set(categorical_columns or [])
    if max_real_records is not None and len(real) > max_real_records:
        real = real.sample(n=max_real_records, random_state=seed).reset_index(drop=True)
    if max_synthetic_records is not None and len(synthetic) > max_synthetic_records:
        synthetic = synthetic.sample(
            n=max_synthetic_records, random_state=seed
        ).reset_index(drop=True)

    common = [column for column in real.columns if column in synthetic.columns]
    if not common or synthetic.empty or real.empty:
        return pd.Series(dtype=float, name="dcr")

    real_blocks = []
    synth_blocks = []
    for column in common:
        if column in categorical or real[column].dtype == "object":
            categories = sorted(
                set(real[column].dropna().astype(str))
                | set(synthetic[column].dropna().astype(str))
            )
            mapping = {value: i for i, value in enumerate(categories)}
            real_blocks.append(real[column].astype(str).map(mapping).fillna(-1).to_numpy()[:, None])
            synth_blocks.append(synthetic[column].astype(str).map(mapping).fillna(-1).to_numpy()[:, None])
        else:
            r = pd.to_numeric(real[column], errors="coerce")
            s = pd.to_numeric(synthetic[column], errors="coerce")
            lo = float(np.nanmin(r.to_numpy()))
            hi = float(np.nanmax(r.to_numpy()))
            scale = hi - lo if hi > lo else 1.0
            real_blocks.append(((r.fillna(lo) - lo) / scale).to_numpy()[:, None])
            synth_blocks.append(((s.fillna(lo) - lo) / scale).to_numpy()[:, None])

    real_matrix = np.hstack(real_blocks).astype(float)
    synth_matrix = np.hstack(synth_blocks).astype(float)
    categorical_indices = [
        idx for idx, column in enumerate(common) if column in categorical or real[column].dtype == "object"
    ]

    distances = []
    for start in range(0, len(synth_matrix), chunk_size):
        chunk = synth_matrix[start : start + chunk_size]
        diff = chunk[:, None, :] - real_matrix[None, :, :]
        if categorical_indices:
            diff[:, :, categorical_indices] = diff[:, :, categorical_indices] != 0
        dist = np.sqrt(np.mean(diff * diff, axis=2))
        distances.extend(np.min(dist, axis=1).tolist())
    return pd.Series(distances, name="dcr")


def summarize_column_similarity(similarity: pd.DataFrame) -> dict[str, float]:
    if similarity.empty:
        return {"column_similarity_mean": float("nan")}
    return {"column_similarity_mean": float(similarity["similarity"].mean())}


def _ks_distance(real: pd.Series, synthetic: pd.Series) -> float:
    r = pd.to_numeric(real, errors="coerce").dropna().to_numpy()
    s = pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy()
    if len(r) == 0 or len(s) == 0:
        return 1.0
    values = np.sort(np.unique(np.concatenate([r, s])))
    r_cdf = np.searchsorted(np.sort(r), values, side="right") / len(r)
    s_cdf = np.searchsorted(np.sort(s), values, side="right") / len(s)
    return float(np.max(np.abs(r_cdf - s_cdf)))


def _total_variation_distance(real: pd.Series, synthetic: pd.Series) -> float:
    r = real.astype(str).value_counts(normalize=True)
    s = synthetic.astype(str).value_counts(normalize=True)
    keys = sorted(set(r.index) | set(s.index))
    return float(0.5 * sum(abs(r.get(key, 0.0) - s.get(key, 0.0)) for key in keys))


def _prepare_xy(
    train: pd.DataFrame, test: pd.DataFrame, target_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = train.dropna(subset=[target_col]).copy()
    test = test.dropna(subset=[target_col]).copy()
    classes = sorted(set(train[target_col].unique()) | set(test[target_col].unique()), key=lambda value: str(value))
    if len(classes) != 2:
        raise ValueError(f"Expected binary target, got classes={classes}")
    class_to_int = {classes[0]: 0, classes[1]: 1}
    y_train = train[target_col].map(class_to_int).to_numpy(dtype=float)
    y_test = test[target_col].map(class_to_int).to_numpy(dtype=float)
    x_train, x_test = _design_matrices(
        train.drop(columns=[target_col]),
        test.drop(columns=[target_col]),
    )
    return x_train, y_train, x_test, y_test


def _design_matrices(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_blocks = [np.ones((len(train_x), 1))]
    test_blocks = [np.ones((len(test_x), 1))]

    for column in train_x.columns:
        train_col = train_x[column]
        test_col = test_x[column]
        is_categorical = train_col.dtype == "object" or train_col.nunique(dropna=True) <= 16
        if is_categorical:
            categories = sorted(train_col.dropna().astype(str).unique())
            for category in categories:
                train_blocks.append((train_col.astype(str) == category).to_numpy(dtype=float)[:, None])
                test_blocks.append((test_col.astype(str) == category).to_numpy(dtype=float)[:, None])
        else:
            train_num = pd.to_numeric(train_col, errors="coerce")
            test_num = pd.to_numeric(test_col, errors="coerce")
            mean = float(train_num.mean()) if train_num.notna().any() else 0.0
            std = float(train_num.std()) if train_num.notna().any() else 1.0
            std = std if std > 0 else 1.0
            train_blocks.append(((train_num.fillna(mean) - mean) / std).to_numpy()[:, None])
            test_blocks.append(((test_num.fillna(mean) - mean) / std).to_numpy()[:, None])

    return np.hstack(train_blocks), np.hstack(test_blocks)


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    steps: int = 700,
    lr: float = 0.08,
    l2: float = 1e-4,
) -> np.ndarray:
    coef = np.zeros(x.shape[1], dtype=float)
    for _ in range(steps):
        scores = np.clip(x @ coef, -35, 35)
        pred = 1 / (1 + np.exp(-scores))
        grad = x.T @ (pred - y) / len(y)
        grad[1:] += l2 * coef[1:]
        coef -= lr * grad
    return coef


def _predict_logistic(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    scores = np.clip(x @ coef, -35, 35)
    return 1 / (1 + np.exp(-scores))
