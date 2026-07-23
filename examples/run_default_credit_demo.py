from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabular_synth.constraints import (
    default_credit_constraints,
    evaluate_constraints,
    repair_default_credit,
)
from tabular_synth.data import (
    DEFAULT_CREDIT_CATEGORICAL,
    DEFAULT_CREDIT_LABEL,
    bootstrap_default_credit_with_jitter,
    make_default_credit_toy,
    split_train_test,
    write_dataframe_csv,
)
from tabular_synth.epic import EPICPromptBuilder, EPICPromptConfig, make_local_dataframe_llm
from tabular_synth.great import GReaTTextCodec, GReaTTextConfig
from tabular_synth.metrics import (
    column_similarity,
    distance_to_closest_record,
    revision_rate,
    summarize_column_similarity,
    tstr_auc,
)


def main() -> None:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)

    real = make_default_credit_toy(n=1200, seed=7)
    train, test = split_train_test(real, test_size=0.2, seed=7)
    constraints = default_credit_constraints(DEFAULT_CREDIT_LABEL)

    descriptions = {
        "LIMIT_BAL": "amount of given credit in NT dollars",
        "PAY_0": "repayment status in the most recent month",
        "BILL_AMT1": "bill statement amount in the most recent month",
        "PAY_AMT1": "payment amount in the most recent month",
        DEFAULT_CREDIT_LABEL: "whether the client defaults next month",
    }
    epic = EPICPromptBuilder(
        EPICPromptConfig(
            target_col=DEFAULT_CREDIT_LABEL,
            n_samples_per_class=5,
            n_sets=3,
            generated_rows_per_prompt=40,
            seed=7,
        ),
        categorical_columns=DEFAULT_CREDIT_CATEGORICAL,
        descriptions=descriptions,
    ).fit(train)
    epic_prompt = epic.build_prompt(train)
    (artifacts / "epic_default_credit_prompt.txt").write_text(epic_prompt)

    def default_credit_mutator(rows, seed):
        return bootstrap_default_credit_with_jitter(rows, n=len(rows), seed=seed)

    local_llm = make_local_dataframe_llm(
        train,
        epic,
        n_rows=40,
        seed=11,
        row_mutator=default_credit_mutator,
    )
    epic_raw_synth = epic.generate(
        train,
        llm=local_llm,
        n_samples=len(train),
        constraints=None,
        max_rounds=40,
    )
    epic_synth = repair_default_credit(epic_raw_synth, DEFAULT_CREDIT_LABEL)

    great_codec = GReaTTextCodec(
        config=GReaTTextConfig(random_feature_order=True, float_precision=0)
    ).fit(train)
    great_texts = great_codec.encode_frame(train.head(80), seed=7)
    (artifacts / "great_default_credit_training_texts.txt").write_text(
        "\n".join(great_texts)
    )

    # Offline proxy for quick metric validation. Swap this with
    # HuggingFaceGReaTFineTuner.sample(...) after optional dependencies are installed.
    great_proxy_raw_synth = bootstrap_default_credit_with_jitter(train, n=len(train), seed=19)
    great_proxy_synth = repair_default_credit(great_proxy_raw_synth, DEFAULT_CREDIT_LABEL)

    write_dataframe_csv(train, artifacts / "default_credit_train.csv")
    write_dataframe_csv(test, artifacts / "default_credit_test.csv")
    write_dataframe_csv(epic_raw_synth, artifacts / "epic_raw_synthetic_default_credit.csv")
    write_dataframe_csv(epic_synth, artifacts / "epic_synthetic_default_credit.csv")
    write_dataframe_csv(
        great_proxy_raw_synth,
        artifacts / "great_proxy_raw_synthetic_default_credit.csv",
    )
    write_dataframe_csv(great_proxy_synth, artifacts / "great_proxy_synthetic_default_credit.csv")

    reports = {
        "real_train_constraints": evaluate_constraints(train, constraints).as_dict(),
        "epic_raw_constraints": evaluate_constraints(epic_raw_synth, constraints).as_dict(),
        "epic_constraints": evaluate_constraints(epic_synth, constraints).as_dict(),
        "epic_revision": revision_rate(epic_raw_synth, epic_synth).as_dict(),
        "great_proxy_raw_constraints": evaluate_constraints(great_proxy_raw_synth, constraints).as_dict(),
        "great_proxy_constraints": evaluate_constraints(great_proxy_synth, constraints).as_dict(),
        "great_proxy_revision": revision_rate(
            great_proxy_raw_synth, great_proxy_synth
        ).as_dict(),
        "epic_column_similarity": summarize_column_similarity(
            column_similarity(train, epic_synth, DEFAULT_CREDIT_CATEGORICAL)
        ),
        "great_proxy_column_similarity": summarize_column_similarity(
            column_similarity(train, great_proxy_synth, DEFAULT_CREDIT_CATEGORICAL)
        ),
        "epic_tstr_auc": tstr_auc(train, epic_synth, test, DEFAULT_CREDIT_LABEL).__dict__,
        "great_proxy_tstr_auc": tstr_auc(
            train, great_proxy_synth, test, DEFAULT_CREDIT_LABEL
        ).__dict__,
        "epic_dcr": distance_to_closest_record(
            train.drop(columns=[DEFAULT_CREDIT_LABEL]),
            epic_synth.drop(columns=[DEFAULT_CREDIT_LABEL]),
            DEFAULT_CREDIT_CATEGORICAL,
        ).describe().to_dict(),
        "great_proxy_dcr": distance_to_closest_record(
            train.drop(columns=[DEFAULT_CREDIT_LABEL]),
            great_proxy_synth.drop(columns=[DEFAULT_CREDIT_LABEL]),
            DEFAULT_CREDIT_CATEGORICAL,
        ).describe().to_dict(),
    }
    (artifacts / "default_credit_eval_summary.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True)
    )

    print(json.dumps(reports, indent=2, sort_keys=True))
    print(f"\nWrote artifacts to {artifacts}")


if __name__ == "__main__":
    main()
