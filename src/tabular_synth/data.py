from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_CREDIT_LABEL = "default.payment.next.month"
DEFAULT_CREDIT_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    DEFAULT_CREDIT_LABEL,
]

DEFAULT_CREDIT_CATEGORICAL = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    DEFAULT_CREDIT_LABEL,
]

ADULT_LABEL = "income"
ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    ADULT_LABEL,
]

ADULT_CATEGORICAL = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ADULT_LABEL,
]

HELOC_LABEL = "RiskPerformance"
HELOC_SPECIAL_CODES = [-9, -8, -7]
HELOC_COLUMNS = [
    HELOC_LABEL,
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
]
HELOC_NUMERIC_COLUMNS = [column for column in HELOC_COLUMNS if column != HELOC_LABEL]
HELOC_CATEGORICAL = [HELOC_LABEL]

HELOC_HF_TO_CANONICAL = {
    "estimate_of_risk": "ExternalRiskEstimate",
    "months_since_first_trade": "MSinceOldestTradeOpen",
    "months_since_last_trade": "MSinceMostRecentTradeOpen",
    "average_duration_of_resolution": "AverageMInFile",
    "number_of_satisfactory_trades": "NumSatisfactoryTrades",
    "nr_trades_insolvent_for_over_60_days": "NumTrades60Ever2DerogPubRec",
    "nr_trades_insolvent_for_over_90_days": "NumTrades90Ever2DerogPubRec",
    "percentage_of_legal_trades": "PercentTradesNeverDelq",
    "months_since_last_illegal_trade": "MSinceMostRecentDelq",
    "maximum_illegal_trades_over_last_year": "MaxDelq2PublicRecLast12M",
    "maximum_illegal_trades": "MaxDelqEver",
    "nr_total_trades": "NumTotalTrades",
    "nr_trades_initiated_in_last_year": "NumTradesOpeninLast12M",
    "percentage_of_installment_trades": "PercentInstallTrades",
    "months_since_last_inquiry_not_recent": "MSinceMostRecentInqexcl7days",
    "nr_inquiries_in_last_6_months": "NumInqLast6M",
    "nr_inquiries_in_last_6_months_not_recent": "NumInqLast6Mexcl7days",
    "net_fraction_of_revolving_burden": "NetFractionRevolvingBurden",
    "net_fraction_of_installment_burden": "NetFractionInstallBurden",
    "nr_revolving_trades_with_balance": "NumRevolvingTradesWBalance",
    "nr_installment_trades_with_balance": "NumInstallTradesWBalance",
    "nr_banks_with_high_ratio": "NumBank2NatlTradesWHighUtilization",
    "percentage_trades_with_balance": "PercentTradesWBalance",
    "is_at_risk": HELOC_LABEL,
}


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_default_credit(path: str | Path) -> pd.DataFrame:
    """Load the real UCI Default of Credit Card Clients data from CSV/XLS(X)."""

    path = Path(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        try:
            raw = pd.read_excel(path, header=1)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel files requires an Excel engine such as openpyxl/xlrd. "
                "Convert the dataset to CSV or install the required engine."
            ) from exc
    else:
        raw = pd.read_csv(path)
    return normalize_default_credit(raw)


def normalize_default_credit(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(col).strip() for col in df.columns]

    x_column_map = {
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_0",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
        "Y": DEFAULT_CREDIT_LABEL,
        "default payment next month": DEFAULT_CREDIT_LABEL,
    }
    df = df.rename(columns=x_column_map)
    df = df.rename(columns={col: col.replace(" ", ".") for col in df.columns})
    df = df.rename(columns={"default.payment.next.month": DEFAULT_CREDIT_LABEL})

    for column in ["ID", "Unnamed: 0"]:
        if column in df.columns:
            df = df.drop(columns=[column])

    missing = [column for column in DEFAULT_CREDIT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Default Credit data is missing columns: {missing}")

    df = df.loc[:, DEFAULT_CREDIT_COLUMNS].copy()
    return coerce_to_reference_schema(df, pd.DataFrame(columns=DEFAULT_CREDIT_COLUMNS).astype({
        column: "int64" for column in DEFAULT_CREDIT_COLUMNS
    }))


def load_adult(path: str | Path) -> pd.DataFrame:
    """Load UCI Adult data from adult.data/adult.test-style CSV files."""

    df = pd.read_csv(
        path,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        comment="|",
        na_values=["?", " ?"],
    )
    if len(df) and str(df.loc[df.index[0], "age"]).strip().lower() == "age":
        df = df.iloc[1:].reset_index(drop=True)

    for column in ADULT_CATEGORICAL:
        df[column] = df[column].astype(str).str.strip()
    df[ADULT_LABEL] = df[ADULT_LABEL].str.rstrip(".")

    numeric_columns = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    return df.dropna().reset_index(drop=True)


def load_heloc(path: str | Path) -> pd.DataFrame:
    """Load the FICO HELOC dataset.

    Supports the common `heloc_dataset_v1.csv` layout with a header, and simple
    headerless CSVs containing the 24 canonical columns.
    """

    raw = pd.read_csv(path)
    if HELOC_LABEL not in raw.columns and raw.shape[1] == len(HELOC_COLUMNS):
        raw = pd.read_csv(path, header=None, names=HELOC_COLUMNS)
    return normalize_heloc(raw)


def normalize_heloc(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df = df.rename(columns=HELOC_HF_TO_CANONICAL)
    for column in ["ID", "Unnamed: 0"]:
        if column in df.columns:
            df = df.drop(columns=[column])

    missing = [column for column in HELOC_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"HELOC data is missing columns: {missing}")

    df = df.loc[:, HELOC_COLUMNS].copy()
    target = df[HELOC_LABEL].astype(str).str.strip().str.rstrip(".")
    target_lower = target.str.lower()
    df[HELOC_LABEL] = np.where(
        target_lower.isin(["bad", "1", "true"]),
        "Bad",
        np.where(target_lower.isin(["good", "0", "false"]), "Good", target),
    )
    df = df[df[HELOC_LABEL].isin(["Bad", "Good"])].reset_index(drop=True)

    for column in HELOC_NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").round().astype("Int64")
    return df.dropna().reset_index(drop=True)


def make_heloc_toy(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Create a HELOC-shaped toy dataset for offline smoke tests."""

    rng = np.random.default_rng(seed)
    risk_score = np.clip(np.rint(rng.normal(68, 12, size=n)), 20, 100).astype(int)
    num_total = rng.integers(3, 80, size=n)
    num_satisfactory = np.minimum(num_total, rng.binomial(num_total, 0.78))
    trades_60 = np.minimum(num_total - num_satisfactory, rng.poisson(0.8, size=n))
    trades_90 = np.minimum(trades_60, rng.poisson(0.35, size=n))
    open_12 = np.minimum(num_total, rng.poisson(2.0, size=n))
    revolving_balance = np.minimum(num_total, rng.poisson(4.0, size=n))
    install_balance = np.minimum(num_total, rng.poisson(2.0, size=n))

    data = {
        HELOC_LABEL: np.where(
            rng.random(n)
            < 1
            / (
                1
                + np.exp(
                    -(
                        2.4
                        - 0.055 * risk_score
                        + 0.35 * trades_60
                        + 0.25 * trades_90
                    )
                )
            ),
            "Bad",
            "Good",
        ),
        "ExternalRiskEstimate": risk_score,
        "MSinceOldestTradeOpen": rng.integers(1, 520, size=n),
        "MSinceMostRecentTradeOpen": rng.integers(0, 80, size=n),
        "AverageMInFile": rng.integers(4, 250, size=n),
        "NumSatisfactoryTrades": num_satisfactory,
        "NumTrades60Ever2DerogPubRec": trades_60,
        "NumTrades90Ever2DerogPubRec": trades_90,
        "PercentTradesNeverDelq": np.clip(
            np.rint(100 * num_satisfactory / np.maximum(num_total, 1)),
            0,
            100,
        ).astype(int),
        "MSinceMostRecentDelq": rng.choice(
            [-7, *range(0, 85)],
            size=n,
            p=[0.35] + [0.65 / 85] * 85,
        ),
        "MaxDelq2PublicRecLast12M": rng.integers(0, 9, size=n),
        "MaxDelqEver": rng.integers(0, 9, size=n),
        "NumTotalTrades": num_total,
        "NumTradesOpeninLast12M": open_12,
        "PercentInstallTrades": rng.integers(0, 101, size=n),
        "MSinceMostRecentInqexcl7days": rng.choice(
            [-7, *range(0, 25)],
            size=n,
            p=[0.25] + [0.75 / 25] * 25,
        ),
        "NumInqLast6M": rng.poisson(1.5, size=n),
        "NumInqLast6Mexcl7days": rng.poisson(1.2, size=n),
        "NetFractionRevolvingBurden": rng.integers(0, 201, size=n),
        "NetFractionInstallBurden": rng.choice(
            [-8, *range(0, 201)],
            size=n,
            p=[0.2] + [0.8 / 201] * 201,
        ),
        "NumRevolvingTradesWBalance": revolving_balance,
        "NumInstallTradesWBalance": install_balance,
        "NumBank2NatlTradesWHighUtilization": np.minimum(
            revolving_balance,
            rng.poisson(1.0, size=n),
        ),
        "PercentTradesWBalance": rng.integers(0, 101, size=n),
    }
    df = pd.DataFrame(data, columns=HELOC_COLUMNS)

    # Add a few all-feature special-code rows, mirroring a common HELOC pattern.
    special_rows = rng.choice(df.index.to_numpy(), size=max(1, n // 50), replace=False)
    for column in HELOC_NUMERIC_COLUMNS:
        df.loc[special_rows, column] = -9
    return df


def coerce_to_reference_schema(
    df: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Coerce generated data to the column order and numeric dtypes of a reference."""

    coerced = df.copy()
    columns = [column for column in reference.columns if column in coerced.columns]
    coerced = coerced.loc[:, columns]
    for column in columns:
        dtype = reference[column].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
            if pd.api.types.is_integer_dtype(dtype):
                coerced[column] = np.rint(coerced[column]).astype("Int64")
    return coerced


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    test_n = int(round(len(df) * test_size))
    test_idx = indices[:test_n]
    train_idx = indices[test_n:]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def infer_categorical_columns(
    df: pd.DataFrame, max_unique_for_int: int = 20
) -> list[str]:
    categorical: list[str] = []
    for column in df.columns:
        series = df[column]
        if series.dtype == "object" or str(series.dtype).startswith("category"):
            categorical.append(column)
        elif pd.api.types.is_bool_dtype(series):
            categorical.append(column)
        elif pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= max_unique_for_int:
            categorical.append(column)
    return categorical


def make_default_credit_toy(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Create a local Default-Credit-shaped dataset with hard constraints.

    This is not the UCI data. It is a deterministic, privacy-safe stand-in that
    mirrors the public schema closely enough to exercise synthesis and alignment.
    """

    rng = np.random.default_rng(seed)
    limit_values = np.array(
        [10000, 20000, 30000, 50000, 80000, 100000, 150000, 200000, 300000, 500000]
    )
    limit_probs = np.array([0.06, 0.08, 0.08, 0.16, 0.08, 0.18, 0.12, 0.1, 0.1, 0.04])
    limit_bal = rng.choice(limit_values, size=n, p=limit_probs)
    sex = rng.choice([1, 2], size=n, p=[0.42, 0.58])
    education = rng.choice([1, 2, 3, 4], size=n, p=[0.34, 0.46, 0.17, 0.03])
    marriage = rng.choice([1, 2, 3], size=n, p=[0.45, 0.48, 0.07])

    base_age = rng.normal(38 + 3 * (marriage == 1) + 2 * (education == 1), 10, size=n)
    age = np.clip(np.rint(base_age), 21, 79).astype(int)

    risk = (
        rng.normal(0, 1, size=n)
        + 0.25 * (education == 3)
        + 0.2 * (marriage == 3)
        - 0.15 * (limit_bal >= 200000)
    )
    pay_columns = {}
    previous = np.rint(risk + rng.normal(0, 1, size=n)).astype(int)
    for column in ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]:
        previous = np.rint(0.65 * previous + 0.8 * risk + rng.normal(0, 1.1, size=n)).astype(int)
        previous = np.clip(previous, -2, 8)
        pay_columns[column] = previous

    utilization = np.clip(rng.beta(2.2, 4.5, size=n) + 0.08 * risk, 0.0, 1.35)
    bill_columns = {}
    current_bill = limit_bal * utilization
    for i in range(6, 0, -1):
        current_bill = current_bill + rng.normal(0, 0.05, size=n) * limit_bal
        current_bill = np.clip(current_bill, -0.12 * limit_bal, 1.6 * limit_bal)
        bill_columns[f"BILL_AMT{i}"] = np.rint(current_bill).astype(int)

    pay_amt_columns = {}
    for i in range(1, 7):
        bill = np.maximum(bill_columns[f"BILL_AMT{i}"], 0)
        willingness = np.clip(0.18 - 0.025 * risk + rng.normal(0, 0.05, size=n), 0.01, 0.45)
        payment = np.where(
            bill > 0,
            bill * willingness + rng.gamma(shape=2.0, scale=350.0, size=n),
            rng.gamma(shape=1.1, scale=80.0, size=n),
        )
        pay_amt_columns[f"PAY_AMT{i}"] = np.rint(np.maximum(payment, 0)).astype(int)

    delinquency = sum(np.maximum(pay_columns[col], 0) for col in pay_columns)
    utilization_now = np.maximum(bill_columns["BILL_AMT1"], 0) / limit_bal
    logits = -2.15 + 0.23 * delinquency + 1.0 * utilization_now + 0.25 * (education == 3)
    prob_default = 1 / (1 + np.exp(-logits))
    default = rng.binomial(1, np.clip(prob_default, 0.02, 0.92))

    data = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_columns["PAY_0"],
        "PAY_2": pay_columns["PAY_2"],
        "PAY_3": pay_columns["PAY_3"],
        "PAY_4": pay_columns["PAY_4"],
        "PAY_5": pay_columns["PAY_5"],
        "PAY_6": pay_columns["PAY_6"],
        "BILL_AMT1": bill_columns["BILL_AMT1"],
        "BILL_AMT2": bill_columns["BILL_AMT2"],
        "BILL_AMT3": bill_columns["BILL_AMT3"],
        "BILL_AMT4": bill_columns["BILL_AMT4"],
        "BILL_AMT5": bill_columns["BILL_AMT5"],
        "BILL_AMT6": bill_columns["BILL_AMT6"],
        "PAY_AMT1": pay_amt_columns["PAY_AMT1"],
        "PAY_AMT2": pay_amt_columns["PAY_AMT2"],
        "PAY_AMT3": pay_amt_columns["PAY_AMT3"],
        "PAY_AMT4": pay_amt_columns["PAY_AMT4"],
        "PAY_AMT5": pay_amt_columns["PAY_AMT5"],
        "PAY_AMT6": pay_amt_columns["PAY_AMT6"],
        DEFAULT_CREDIT_LABEL: default,
    }
    return pd.DataFrame(data, columns=DEFAULT_CREDIT_COLUMNS)


def bootstrap_default_credit_with_jitter(
    df: pd.DataFrame, n: int, seed: int = 123
) -> pd.DataFrame:
    """A deterministic local stand-in for an LLM sampler.

    It is useful for testing the EPIC/GReaT plumbing offline. It should not be
    reported as an LLM result.
    """

    rng = np.random.default_rng(seed)
    sampled = df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True).copy()

    if "AGE" in sampled:
        sampled["AGE"] = np.clip(
            sampled["AGE"].to_numpy() + rng.integers(-2, 3, size=n), 18, 100
        ).astype(int)

    bill_cols = [col for col in sampled.columns if col.startswith("BILL_AMT")]
    pay_amt_cols = [col for col in sampled.columns if col.startswith("PAY_AMT")]
    for column in bill_cols:
        noise = rng.normal(0, 0.035, size=n) * sampled["LIMIT_BAL"].to_numpy()
        lower = -0.15 * sampled["LIMIT_BAL"].to_numpy()
        upper = 1.75 * sampled["LIMIT_BAL"].to_numpy()
        sampled[column] = np.rint(
            np.clip(sampled[column].to_numpy() + noise, lower, upper)
        ).astype(int)

    for column in pay_amt_cols:
        noise = rng.normal(0, 0.02, size=n) * sampled["LIMIT_BAL"].to_numpy()
        sampled[column] = np.rint(
            np.maximum(sampled[column].to_numpy() + noise, 0)
        ).astype(int)

    return sampled[DEFAULT_CREDIT_COLUMNS].copy()


def write_dataframe_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
