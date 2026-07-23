from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


Predicate = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Constraint:
    """A row-wise hard constraint.

    The predicate returns True for rows that satisfy the constraint. Missing
    required columns or invalid predicate output are treated as violations.
    """

    name: str
    columns: tuple[str, ...]
    description: str
    predicate: Predicate


@dataclass(frozen=True)
class ConstraintReport:
    n_rows: int
    n_constraints: int
    cvr: float
    cvc: float
    scvc: float
    row_violation_coverage: float
    per_constraint: pd.DataFrame
    violations: pd.DataFrame

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_rows": self.n_rows,
            "n_constraints": self.n_constraints,
            "cvr": self.cvr,
            "cvc": self.cvc,
            "scvc": self.scvc,
            "row_violation_coverage": self.row_violation_coverage,
        }


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _integerish(series: pd.Series) -> pd.Series:
    values = _numeric(series)
    return values.notna() & np.isclose(values, np.round(values))


def _predicate_to_valid_series(
    df: pd.DataFrame, constraint: Constraint
) -> pd.Series:
    missing = [col for col in constraint.columns if col not in df.columns]
    if missing:
        return pd.Series(False, index=df.index, name=constraint.name)

    try:
        result = constraint.predicate(df)
    except Exception:
        return pd.Series(False, index=df.index, name=constraint.name)

    if isinstance(result, pd.Series):
        valid = result.reindex(df.index)
    elif np.isscalar(result):
        valid = pd.Series(bool(result), index=df.index)
    else:
        valid = pd.Series(result, index=df.index)

    return valid.fillna(False).astype(bool).rename(constraint.name)


def evaluate_constraints(
    df: pd.DataFrame, constraints: Sequence[Constraint]
) -> ConstraintReport:
    """Evaluate hard constraints and compute CVR/CVC/sCVC-style summaries.

    CVR is the fraction of constraint checks that are violated.
    CVC is the fraction of constraints that are violated at least once.
    sCVC is the mean row-wise fraction of violated constraints.
    row_violation_coverage is the fraction of rows with at least one violation.
    """

    if not constraints:
        empty = pd.DataFrame(index=df.index)
        return ConstraintReport(
            n_rows=len(df),
            n_constraints=0,
            cvr=0.0,
            cvc=0.0,
            scvc=0.0,
            row_violation_coverage=0.0,
            per_constraint=pd.DataFrame(
                columns=["constraint", "violations", "violation_rate", "description"]
            ),
            violations=empty,
        )

    valid = pd.concat(
        [_predicate_to_valid_series(df, constraint) for constraint in constraints],
        axis=1,
    )
    violations = ~valid
    total_checks = max(len(df) * len(constraints), 1)
    per_constraint = pd.DataFrame(
        {
            "constraint": [constraint.name for constraint in constraints],
            "violations": violations.sum(axis=0).astype(int).to_numpy(),
            "violation_rate": violations.mean(axis=0).to_numpy(),
            "description": [constraint.description for constraint in constraints],
        }
    )
    row_violation_fraction = violations.mean(axis=1) if len(df) else pd.Series(dtype=float)
    return ConstraintReport(
        n_rows=len(df),
        n_constraints=len(constraints),
        cvr=float(violations.to_numpy().sum() / total_checks),
        cvc=float((violations.sum(axis=0) > 0).mean()),
        scvc=float(row_violation_fraction.mean()) if len(df) else 0.0,
        row_violation_coverage=float(violations.any(axis=1).mean()) if len(df) else 0.0,
        per_constraint=per_constraint,
        violations=violations,
    )


def in_numeric_set(column: str, allowed: Iterable[int]) -> Constraint:
    allowed_values = tuple(allowed)

    return Constraint(
        name=f"{column}_domain",
        columns=(column,),
        description=f"{column} must be an integer in {allowed_values}.",
        predicate=lambda df: _integerish(df[column])
        & _numeric(df[column]).isin(allowed_values),
    )


def numeric_between(column: str, low: float, high: float) -> Constraint:
    return Constraint(
        name=f"{column}_range",
        columns=(column,),
        description=f"{column} must be numeric and lie in [{low}, {high}].",
        predicate=lambda df: _numeric(df[column]).between(low, high, inclusive="both"),
    )


def numeric_at_least(column: str, low: float) -> Constraint:
    return Constraint(
        name=f"{column}_min",
        columns=(column,),
        description=f"{column} must be numeric and >= {low}.",
        predicate=lambda df: _numeric(df[column]) >= low,
    )


def default_credit_constraints(
    label_col: str = "default.payment.next.month",
) -> list[Constraint]:
    """Hard constraints for the UCI Default of Credit Card Clients schema."""

    constraints: list[Constraint] = [
        numeric_at_least("LIMIT_BAL", 1),
        in_numeric_set("SEX", [1, 2]),
        in_numeric_set("EDUCATION", [0, 1, 2, 3, 4, 5, 6]),
        in_numeric_set("MARRIAGE", [0, 1, 2, 3]),
        numeric_between("AGE", 18, 100),
    ]

    for column in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        constraints.append(
            Constraint(
                name=f"{column}_status_domain",
                columns=(column,),
                description=f"{column} must be an integer repayment status from -2 to 9.",
                predicate=lambda df, col=column: _integerish(df[col])
                & _numeric(df[col]).between(-2, 9, inclusive="both"),
            )
        )

    for column in [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]:
        constraints.append(numeric_at_least(column, 0))

    if label_col:
        constraints.append(in_numeric_set(label_col, [0, 1]))

    return constraints


def repair_default_credit(
    df: pd.DataFrame,
    label_col: str = "default.payment.next.month",
) -> pd.DataFrame:
    """Apply simple deterministic repairs for Default Credit hard constraints.

    This is deliberately conservative and auditable: it rounds integer-coded
    fields, clips bounded ranges, and floors non-negative payment amounts.
    """

    repaired = df.copy()
    if "LIMIT_BAL" in repaired:
        repaired["LIMIT_BAL"] = _round_clip_int(repaired["LIMIT_BAL"], low=1)

    domain_columns = {
        "SEX": [1, 2],
        "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
        "MARRIAGE": [0, 1, 2, 3],
        label_col: [0, 1],
    }
    for column, allowed in domain_columns.items():
        if column in repaired:
            repaired[column] = _nearest_allowed_integer(repaired[column], allowed)

    if "AGE" in repaired:
        repaired["AGE"] = _round_clip_int(repaired["AGE"], low=18, high=100)

    for column in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        if column in repaired:
            repaired[column] = _round_clip_int(repaired[column], low=-2, high=9)

    for column in [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]:
        if column in repaired:
            repaired[column] = _round_clip_int(repaired[column], low=0)

    return repaired


def _round_clip_int(
    series: pd.Series,
    low: int,
    high: int | None = None,
) -> pd.Series:
    values = np.rint(_numeric(series).fillna(low)).clip(lower=low, upper=high)
    return values.astype("Int64")


def _nearest_allowed_integer(series: pd.Series, allowed: Sequence[int]) -> pd.Series:
    values = np.rint(_numeric(series).fillna(allowed[0])).to_numpy()
    allowed_array = np.asarray(allowed)
    nearest = allowed_array[np.abs(values[:, None] - allowed_array[None, :]).argmin(axis=1)]
    return pd.Series(nearest, index=series.index, dtype="Int64")


ADULT_EDUCATION_NUM: Mapping[str, int] = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}


def adult_constraints(label_col: str = "income") -> list[Constraint]:
    """Hard-ish constraints for the Adult Census Income schema.

    The education/education-num consistency is a true deterministic constraint
    in the UCI encoding and is therefore especially useful for alignment checks.
    """

    constraints = [
        numeric_between("age", 16, 100),
        numeric_between("education-num", 1, 16),
        numeric_between("hours-per-week", 1, 99),
        numeric_at_least("capital-gain", 0),
        numeric_at_least("capital-loss", 0),
        Constraint(
            name="education_num_consistency",
            columns=("education", "education-num"),
            description="education-num must match the deterministic Adult education mapping.",
            predicate=lambda df: df["education"].map(ADULT_EDUCATION_NUM).astype(float)
            == _numeric(df["education-num"]),
        ),
    ]

    if label_col:
        constraints.append(
            Constraint(
                name=f"{label_col}_domain",
                columns=(label_col,),
                description=f"{label_col} must be one of <=50K, >50K.",
                predicate=lambda df: df[label_col].astype(str).str.strip().isin(
                    ["<=50K", ">50K", "<=50K.", ">50K."]
                ),
            )
        )

    return constraints


def repair_adult(df: pd.DataFrame, label_col: str = "income") -> pd.DataFrame:
    """Conservative repairs for Adult hard constraints."""

    repaired = df.copy()
    integer_ranges = {
        "age": (16, 100),
        "education-num": (1, 16),
        "hours-per-week": (1, 99),
        "capital-gain": (0, None),
        "capital-loss": (0, None),
    }
    for column, (low, high) in integer_ranges.items():
        if column in repaired:
            repaired[column] = _round_clip_int(repaired[column], low=low, high=high)

    if "education" in repaired and "education-num" in repaired:
        mapped = repaired["education"].map(ADULT_EDUCATION_NUM)
        repaired.loc[mapped.notna(), "education-num"] = mapped[mapped.notna()].astype("Int64")

    if label_col in repaired:
        normalized = repaired[label_col].astype(str).str.strip().str.rstrip(".")
        repaired[label_col] = normalized.where(normalized.isin(["<=50K", ">50K"]), "<=50K")

    return repaired


HELOC_LABEL = "RiskPerformance"
HELOC_SPECIAL_CODES = (-9, -8, -7)
HELOC_NUMERIC_COLUMNS = (
    "ExternalRiskEstimate",
    "MSinceOldestTradeOpen",
    "MSinceMostRecentTradeOpen",
    "AverageMInFile",
    "NumSatisfactoryTrades",
    "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec",
    "PercentTradesNeverDelq",
    "MSinceMostRecentDelq",
    "MaxDelq2PublicRecLast12M",
    "MaxDelqEver",
    "NumTotalTrades",
    "NumTradesOpeninLast12M",
    "PercentInstallTrades",
    "MSinceMostRecentInqexcl7days",
    "NumInqLast6M",
    "NumInqLast6Mexcl7days",
    "NetFractionRevolvingBurden",
    "NetFractionInstallBurden",
    "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance",
    "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance",
)
HELOC_PERCENT_COLUMNS = (
    "PercentTradesNeverDelq",
    "PercentInstallTrades",
    "PercentTradesWBalance",
)
HELOC_NET_FRACTION_COLUMNS = (
    "NetFractionRevolvingBurden",
    "NetFractionInstallBurden",
)


def heloc_constraints(label_col: str = HELOC_LABEL) -> list[Constraint]:
    """Hard constraints for the FICO HELOC schema.

    FICO HELOC uses special integer codes -9, -8, and -7. These are accepted
    as valid special values; ordinary non-special values must obey type/range
    and basic trade-count consistency constraints.
    """

    constraints: list[Constraint] = []
    if label_col:
        constraints.append(
            Constraint(
                name=f"{label_col}_domain",
                columns=(label_col,),
                description=f"{label_col} must be Good or Bad.",
                predicate=lambda df, col=label_col: df[col]
                .astype(str)
                .str.strip()
                .isin(["Good", "Bad"]),
            )
        )

    constraints.append(_heloc_integer_or_special("ExternalRiskEstimate", low=0, high=100))

    for column in HELOC_PERCENT_COLUMNS:
        constraints.append(_heloc_integer_or_special(column, low=0, high=100))

    for column in HELOC_NET_FRACTION_COLUMNS:
        constraints.append(_heloc_integer_or_special(column, low=0, high=300))

    for column in ("MaxDelq2PublicRecLast12M", "MaxDelqEver"):
        constraints.append(_heloc_integer_or_special(column, low=0, high=9))

    bounded = {
        "NumTrades60Ever2DerogPubRec",
        "NumTrades90Ever2DerogPubRec",
        "NumSatisfactoryTrades",
        "NumTotalTrades",
        "NumTradesOpeninLast12M",
        "NumInqLast6M",
        "NumInqLast6Mexcl7days",
        "NumRevolvingTradesWBalance",
        "NumInstallTradesWBalance",
        "NumBank2NatlTradesWHighUtilization",
    }
    for column in HELOC_NUMERIC_COLUMNS:
        if column in HELOC_PERCENT_COLUMNS or column in HELOC_NET_FRACTION_COLUMNS:
            continue
        if column in {"ExternalRiskEstimate", "MaxDelq2PublicRecLast12M", "MaxDelqEver"}:
            continue
        constraints.append(
            _heloc_integer_or_special(column, low=0, high=500 if column in bounded else 1000)
        )

    constraints.extend(
        [
            _heloc_less_equal(
                "NumTrades90Ever2DerogPubRec",
                "NumTrades60Ever2DerogPubRec",
            ),
            _heloc_less_equal("NumTrades60Ever2DerogPubRec", "NumTotalTrades"),
            _heloc_less_equal("NumTrades90Ever2DerogPubRec", "NumTotalTrades"),
            _heloc_less_equal("NumTradesOpeninLast12M", "NumTotalTrades"),
            _heloc_less_equal("NumSatisfactoryTrades", "NumTotalTrades"),
            _heloc_less_equal("NumRevolvingTradesWBalance", "NumTotalTrades"),
            _heloc_less_equal("NumInstallTradesWBalance", "NumTotalTrades"),
        ]
    )
    return constraints


def repair_heloc(df: pd.DataFrame, label_col: str = HELOC_LABEL) -> pd.DataFrame:
    """Conservative repair pass for HELOC hard constraints."""

    repaired = df.copy()
    if label_col in repaired:
        normalized = repaired[label_col].astype(str).str.strip().str.rstrip(".")
        lower = normalized.str.lower()
        repaired[label_col] = np.where(
            lower.isin(["bad", "1", "true"]),
            "Bad",
            np.where(lower.isin(["good", "0", "false"]), "Good", normalized),
        )
        repaired[label_col] = repaired[label_col].where(
            repaired[label_col].isin(["Good", "Bad"]),
            "Bad",
        )

    bounds: dict[str, tuple[int, int | None]] = {
        "ExternalRiskEstimate": (0, 100),
        "MaxDelq2PublicRecLast12M": (0, 9),
        "MaxDelqEver": (0, 9),
    }
    for column in HELOC_PERCENT_COLUMNS:
        bounds[column] = (0, 100)
    for column in HELOC_NET_FRACTION_COLUMNS:
        bounds[column] = (0, 300)

    for column in HELOC_NUMERIC_COLUMNS:
        if column in repaired:
            low, high = bounds.get(column, (0, 1000))
            repaired[column] = _round_clip_int_preserve_special(
                repaired[column],
                low=low,
                high=high,
                special_codes=HELOC_SPECIAL_CODES,
            )

    _clip_heloc_left_to_right(
        repaired,
        [
            ("NumTrades90Ever2DerogPubRec", "NumTrades60Ever2DerogPubRec"),
            ("NumTrades60Ever2DerogPubRec", "NumTotalTrades"),
            ("NumTrades90Ever2DerogPubRec", "NumTotalTrades"),
            ("NumTradesOpeninLast12M", "NumTotalTrades"),
            ("NumSatisfactoryTrades", "NumTotalTrades"),
            ("NumRevolvingTradesWBalance", "NumTotalTrades"),
            ("NumInstallTradesWBalance", "NumTotalTrades"),
        ],
    )
    return repaired


def _heloc_integer_or_special(
    column: str,
    low: int,
    high: int | None = None,
) -> Constraint:
    if high is None:
        description = (
            f"{column} must be an integer special code in {HELOC_SPECIAL_CODES} "
            f"or a value >= {low}."
        )
    else:
        description = (
            f"{column} must be an integer special code in {HELOC_SPECIAL_CODES} "
            f"or a value in [{low}, {high}]."
        )

    def predicate(df: pd.DataFrame, col: str = column) -> pd.Series:
        values = _numeric(df[col])
        special = values.isin(HELOC_SPECIAL_CODES)
        in_range = values >= low if high is None else values.between(low, high)
        return _integerish(df[col]) & (special | in_range)

    return Constraint(
        name=f"{column}_heloc_domain",
        columns=(column,),
        description=description,
        predicate=predicate,
    )


def _heloc_less_equal(left: str, right: str) -> Constraint:
    def predicate(df: pd.DataFrame, lhs: str = left, rhs: str = right) -> pd.Series:
        left_values = _numeric(df[lhs])
        right_values = _numeric(df[rhs])
        unknown = left_values.isin(HELOC_SPECIAL_CODES) | right_values.isin(HELOC_SPECIAL_CODES)
        return unknown | (left_values <= right_values)

    return Constraint(
        name=f"{left}_le_{right}",
        columns=(left, right),
        description=(
            f"{left} must be <= {right} whenever neither field is a HELOC special code."
        ),
        predicate=predicate,
    )


def _round_clip_int_preserve_special(
    series: pd.Series,
    low: int,
    high: int | None,
    special_codes: Sequence[int],
) -> pd.Series:
    values = _numeric(series)
    rounded = np.rint(values.fillna(low))
    special = rounded.isin(special_codes)
    clipped = rounded.clip(lower=low, upper=high)
    repaired = clipped.where(~special, rounded)
    return repaired.astype("Int64")


def _clip_heloc_left_to_right(df: pd.DataFrame, pairs: Sequence[tuple[str, str]]) -> None:
    for left, right in pairs:
        if left not in df or right not in df:
            continue
        left_values = _numeric(df[left])
        right_values = _numeric(df[right])
        comparable = ~left_values.isin(HELOC_SPECIAL_CODES) & ~right_values.isin(HELOC_SPECIAL_CODES)
        over = comparable & (left_values > right_values)
        df.loc[over, left] = right_values[over].astype("Int64")
