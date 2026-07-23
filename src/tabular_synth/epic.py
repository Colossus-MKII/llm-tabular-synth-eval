from __future__ import annotations

import csv
import io
import random
import string
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .constraints import Constraint, evaluate_constraints


LLMCallable = Callable[[str], str]


def dataframe_to_csv_text(
    df: pd.DataFrame, columns: Sequence[str], include_header: bool = True
) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if include_header:
        writer.writerow(list(columns))
    for _, row in df.loc[:, list(columns)].iterrows():
        writer.writerow(["" if pd.isna(row[col]) else row[col] for col in columns])
    return buffer.getvalue().rstrip("\n")


class UniqueValueMapper:
    """Map categorical values to unique alphanumeric tokens and back.

    EPIC uses this trick to make repeated category labels less ambiguous to an
    LLM. A value such as 1 can mean SEX=male in one column and EDUCATION=graduate
    in another; unique tokens make those supports column-specific.
    """

    def __init__(self, code_length: int = 4, seed: int = 42):
        self.code_length = code_length
        self.seed = seed
        self.forward_: dict[str, dict[object, str]] = {}
        self.backward_: dict[str, dict[str, object]] = {}

    def fit(self, df: pd.DataFrame, categorical_columns: Sequence[str]) -> "UniqueValueMapper":
        rng = random.Random(self.seed)
        used: set[str] = set()
        self.forward_.clear()
        self.backward_.clear()

        for column in categorical_columns:
            values = sorted(df[column].dropna().unique(), key=lambda value: str(value))
            self.forward_[column] = {}
            self.backward_[column] = {}
            for value in values:
                code = self._new_code(rng, used)
                self.forward_[column][value] = code
                self.backward_[column][code] = value
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        mapped = df.copy()
        for column, col_map in self.forward_.items():
            if column in mapped.columns:
                mapped[column] = mapped[column].map(
                    lambda value, mapping=col_map: mapping.get(value, value)
                )
        return mapped

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        restored = df.copy()
        for column, col_map in self.backward_.items():
            if column in restored.columns:
                restored[column] = restored[column].map(
                    lambda value, mapping=col_map: mapping.get(value, value)
                )
        return restored

    def support(self) -> dict[str, set[str]]:
        return {
            column: set(mapping.values())
            for column, mapping in self.forward_.items()
        }

    def _new_code(self, rng: random.Random, used: set[str]) -> str:
        alphabet = string.ascii_uppercase + string.digits
        while True:
            first = rng.choice(string.ascii_uppercase)
            rest = "".join(rng.choice(alphabet) for _ in range(self.code_length - 1))
            code = first + rest
            if code not in used:
                used.add(code)
                return code


@dataclass(frozen=True)
class EPICPromptConfig:
    target_col: str
    n_samples_per_class: int = 8
    n_sets: int = 3
    generated_rows_per_prompt: int = 32
    use_unique_value_mapping: bool = True
    seed: int = 42


class EPICPromptBuilder:
    """Build, parse, and filter EPIC-style prompts.

    The original EPIC repository generates prompts for a specific dataset. This
    class keeps the same mechanism general: target-first CSV rows, class-balanced
    groups, repeated sets, optional descriptions, and a final CSV header trigger.
    """

    def __init__(
        self,
        config: EPICPromptConfig,
        categorical_columns: Sequence[str],
        descriptions: Mapping[str, str] | None = None,
    ):
        self.config = config
        self.categorical_columns = list(dict.fromkeys(categorical_columns))
        if config.target_col not in self.categorical_columns:
            self.categorical_columns.append(config.target_col)
        self.descriptions = dict(descriptions or {})
        self.mapper = UniqueValueMapper(seed=config.seed)
        self.columns_: list[str] = []
        self.reference_dtypes_: dict[str, object] = {}
        self.category_support_: dict[str, set[object]] = {}
        self.mapped_category_support_: dict[str, set[str]] = {}

    @property
    def columns(self) -> list[str]:
        return self.columns_.copy()

    def fit(self, df: pd.DataFrame) -> "EPICPromptBuilder":
        target_col = self.config.target_col
        if target_col not in df.columns:
            raise ValueError(f"target_col {target_col!r} is not in the DataFrame")

        self.columns_ = [target_col] + [col for col in df.columns if col != target_col]
        self.reference_dtypes_ = {col: df[col].dtype for col in self.columns_}
        self.category_support_ = {
            column: set(df[column].dropna().unique())
            for column in self.categorical_columns
            if column in df.columns
        }
        if self.config.use_unique_value_mapping:
            self.mapper.fit(df, self.categorical_columns)
            self.mapped_category_support_ = self.mapper.support()
        else:
            self.mapped_category_support_ = {
                column: {str(value) for value in values}
                for column, values in self.category_support_.items()
            }
        return self

    def build_prompt(self, df: pd.DataFrame, seed: int | None = None) -> str:
        if not self.columns_:
            self.fit(df)

        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        target_col = self.config.target_col
        prompt_df = self._prompt_dataframe(df)
        class_values = sorted(prompt_df[target_col].dropna().unique(), key=lambda value: str(value))
        if not class_values:
            raise ValueError("No target classes found for EPIC prompt construction")

        lines: list[str] = []
        if self.descriptions:
            lines.append("Column descriptions:")
            for column in self.columns_:
                if column in self.descriptions:
                    lines.append(f"{column}: {self.descriptions[column]}")
            lines.append("")

        lines.extend(
            [
                "Generate synthetic tabular rows in CSV format.",
                "Use the same columns, value ranges, categorical supports, and class-wise patterns as the examples.",
                "Return only CSV rows after the final header. Do not add prose.",
                "",
            ]
        )

        set_names = list(string.ascii_uppercase)
        for set_idx in range(self.config.n_sets):
            lines.append(f"Set {set_names[set_idx]}:")
            lines.append(dataframe_to_csv_text(prompt_df.head(0), self.columns_, include_header=True))
            for class_value in class_values:
                class_rows = prompt_df[prompt_df[target_col] == class_value]
                replace = len(class_rows) < self.config.n_samples_per_class
                sampled_idx = rng.choice(
                    class_rows.index.to_numpy(),
                    size=self.config.n_samples_per_class,
                    replace=replace,
                )
                sampled = class_rows.loc[sampled_idx, self.columns_]
                lines.append(dataframe_to_csv_text(sampled, self.columns_, include_header=False))
            lines.append("")

        lines.append(
            f"Generate {self.config.generated_rows_per_prompt} new rows, balanced across {target_col} classes."
        )
        lines.append(dataframe_to_csv_text(prompt_df.head(0), self.columns_, include_header=True))
        return "\n".join(lines)

    def parse_response(self, text: str) -> pd.DataFrame:
        if not self.columns_:
            raise ValueError("Call fit() before parsing responses")

        cleaned_lines = self._extract_csv_like_lines(text)
        if not cleaned_lines:
            return pd.DataFrame(columns=self.columns_)

        rows: list[list[str]] = []
        for row in csv.reader(cleaned_lines):
            if len(row) != len(self.columns_):
                continue
            if [item.strip() for item in row] == self.columns_:
                continue
            rows.append([item.strip() for item in row])

        return pd.DataFrame(rows, columns=self.columns_)

    def postprocess(
        self,
        generated: pd.DataFrame,
        constraints: Sequence[Constraint] | None = None,
        drop_invalid_categories: bool = True,
    ) -> pd.DataFrame:
        if generated.empty:
            return pd.DataFrame(columns=self.columns_)

        restored = self.mapper.inverse_transform(generated) if self.config.use_unique_value_mapping else generated.copy()
        restored = restored.loc[:, self.columns_].copy()
        for column, dtype in self.reference_dtypes_.items():
            if column not in restored:
                continue
            if pd.api.types.is_numeric_dtype(dtype):
                restored[column] = pd.to_numeric(restored[column], errors="coerce")
                if pd.api.types.is_integer_dtype(dtype):
                    restored[column] = np.rint(restored[column]).astype("Int64")

        keep = pd.Series(True, index=restored.index)
        if drop_invalid_categories:
            for column, support in self.category_support_.items():
                if column in restored:
                    keep &= restored[column].isin(support)

        restored = restored[keep].reset_index(drop=True)
        if constraints:
            report = evaluate_constraints(restored, constraints)
            restored = restored.loc[~report.violations.any(axis=1)].reset_index(drop=True)
        return restored

    def generate(
        self,
        df: pd.DataFrame,
        llm: LLMCallable,
        n_samples: int,
        constraints: Sequence[Constraint] | None = None,
        drop_invalid_categories: bool = True,
        max_rounds: int = 50,
    ) -> pd.DataFrame:
        """Generate rows by repeatedly prompting an LLM callable."""

        if not self.columns_:
            self.fit(df)

        chunks: list[pd.DataFrame] = []
        generated_n = 0
        rounds = 0
        while generated_n < n_samples and rounds < max_rounds:
            prompt = self.build_prompt(df, seed=self.config.seed + rounds)
            raw = llm(prompt)
            parsed = self.parse_response(raw)
            cleaned = self.postprocess(
                parsed,
                constraints=constraints,
                drop_invalid_categories=drop_invalid_categories,
            )
            if not cleaned.empty:
                chunks.append(cleaned)
                generated_n += len(cleaned)
            rounds += 1

        if not chunks:
            return pd.DataFrame(columns=self.columns_)
        return pd.concat(chunks, ignore_index=True).head(n_samples)

    def _prompt_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        prompt_df = df.loc[:, self.columns_].copy()
        if self.config.use_unique_value_mapping:
            prompt_df = self.mapper.transform(prompt_df)
        return prompt_df

    def _extract_csv_like_lines(self, text: str) -> list[str]:
        lines = []
        in_fence = False
        expected_commas = len(self.columns_) - 1
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if line.lower().startswith(("here", "sure", "note:", "rows:")) and not in_fence:
                continue
            if line.count(",") == expected_commas:
                lines.append(line)
        return lines


def make_local_dataframe_llm(
    df: pd.DataFrame,
    builder: EPICPromptBuilder,
    n_rows: int = 32,
    seed: int = 123,
    row_mutator: Callable[[pd.DataFrame, int], pd.DataFrame] | None = None,
) -> LLMCallable:
    """Return a deterministic CSV-emitting stand-in for offline tests.

    This is intentionally simple: it bootstraps examples from a DataFrame and
    serializes them as if an LLM returned CSV rows. It validates the EPIC parser
    and post-processing without spending tokens or requiring network access.
    """

    rng = np.random.default_rng(seed)

    def llm(_: str) -> str:
        source = df.loc[:, builder.columns]
        target_col = builder.config.target_col
        class_values = sorted(source[target_col].dropna().unique(), key=lambda value: str(value))
        rows = []
        rows_per_class = max(1, n_rows // max(len(class_values), 1))
        for class_value in class_values:
            class_rows = source[source[target_col] == class_value]
            sampled_idx = rng.choice(
                class_rows.index.to_numpy(),
                size=rows_per_class,
                replace=len(class_rows) < rows_per_class,
            )
            rows.append(class_rows.loc[sampled_idx])
        sampled = pd.concat(rows, ignore_index=True).head(n_rows)
        if row_mutator is not None:
            sampled = row_mutator(sampled, int(rng.integers(0, 2**31 - 1)))
            sampled = sampled.loc[:, builder.columns]
        if builder.config.use_unique_value_mapping:
            sampled = builder.mapper.transform(sampled)
        return dataframe_to_csv_text(sampled, builder.columns, include_header=False)

    return llm
