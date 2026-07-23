from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabular_synth.constraints import (  # noqa: E402
    adult_constraints,
    default_credit_constraints,
    evaluate_constraints,
    heloc_constraints,
    repair_adult,
    repair_default_credit,
    repair_heloc,
)
from tabular_synth.data import (  # noqa: E402
    ADULT_CATEGORICAL,
    ADULT_LABEL,
    DEFAULT_CREDIT_CATEGORICAL,
    DEFAULT_CREDIT_LABEL,
    HELOC_CATEGORICAL,
    HELOC_LABEL,
    bootstrap_default_credit_with_jitter,
    coerce_to_reference_schema,
    load_adult,
    load_default_credit,
    load_heloc,
    make_default_credit_toy,
    make_heloc_toy,
    split_train_test,
    write_dataframe_csv,
)
from tabular_synth.epic import (  # noqa: E402
    EPICPromptBuilder,
    EPICPromptConfig,
    make_local_dataframe_llm,
)
from tabular_synth.great import HuggingFaceGReaTFineTuner  # noqa: E402
from tabular_synth.llm_backends import (  # noqa: E402
    OpenAICompatibleChatConfig,
    OpenAICompatibleChatLLM,
)
from tabular_synth.metrics import (  # noqa: E402
    column_similarity,
    distance_to_closest_record,
    revision_rate,
    summarize_column_similarity,
    tstr_auc,
)


RepairFn = Callable[[pd.DataFrame], pd.DataFrame]


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, target_col, categorical_columns, constraints, repair_fn = load_experiment_data(args)
    run_summaries = []

    for seed in args.seeds:
        seed_dir = outdir / f"{args.dataset}_{args.method}_seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train, test = split_train_test(df, test_size=args.test_size, seed=seed)
        n_synthetic = args.n_synthetic or len(train)

        raw_synth = generate_synthetic(
            args=args,
            train=train,
            target_col=target_col,
            categorical_columns=categorical_columns,
            constraints=constraints,
            seed=seed,
            n_synthetic=n_synthetic,
            seed_dir=seed_dir,
        )
        raw_synth = coerce_to_reference_schema(raw_synth, train)
        fixed_synth = repair_fn(raw_synth)
        fixed_synth = coerce_to_reference_schema(fixed_synth, train)

        write_dataframe_csv(train, seed_dir / "real_train.csv")
        write_dataframe_csv(test, seed_dir / "real_test.csv")
        write_dataframe_csv(raw_synth, seed_dir / "synthetic_raw.csv")
        write_dataframe_csv(fixed_synth, seed_dir / "synthetic_repaired.csv")

        summary = evaluate_run(
            train=train,
            test=test,
            raw_synth=raw_synth,
            fixed_synth=fixed_synth,
            target_col=target_col,
            categorical_columns=categorical_columns,
            constraints=constraints,
            seed=seed,
            method=args.method,
            dcr_max_real=None if args.dcr_max_real <= 0 else args.dcr_max_real,
            dcr_max_synthetic=None
            if args.dcr_max_synthetic <= 0
            else args.dcr_max_synthetic,
            seed_dir=seed_dir,
        )
        (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        run_summaries.append(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))

    (outdir / "summary_all.json").write_text(
        json.dumps(run_summaries, indent=2, sort_keys=True)
    )
    pd.DataFrame([flatten_dict(summary) for summary in run_summaries]).to_csv(
        outdir / "summary_by_seed.csv",
        index=False,
    )
    print(f"\nWrote real-scale experiment artifacts to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-scale EPIC/GReaT tabular synthesis experiments."
    )
    parser.add_argument("--dataset", choices=["default-credit", "adult", "heloc"], required=True)
    parser.add_argument("--data-path", help="Path to real dataset CSV/XLS(X).")
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Use a local toy dataset for smoke testing only.",
    )
    parser.add_argument(
        "--method",
        choices=["epic", "great", "great-proxy"],
        required=True,
        help="great-proxy is an offline smoke-test baseline, not a real fine-tuned LM.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-synthetic", type=int, default=None)
    parser.add_argument("--outdir", default="artifacts/real_scale")

    parser.add_argument("--epic-samples-per-class", type=int, default=8)
    parser.add_argument("--epic-sets", type=int, default=3)
    parser.add_argument("--epic-rows-per-prompt", type=int, default=40)
    parser.add_argument("--epic-max-rounds", type=int, default=None)
    parser.add_argument(
        "--offline-proxy",
        action="store_true",
        help="Use a deterministic local EPIC proxy instead of an LLM. Not a real EPIC result.",
    )

    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-api-key-env", default="LLM_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-max-tokens", type=int, default=2048)
    parser.add_argument("--llm-timeout-seconds", type=int, default=120)

    parser.add_argument("--great-model", default="distilgpt2")
    parser.add_argument("--great-epochs", type=int, default=1)
    parser.add_argument("--great-batch-size", type=int, default=4)
    parser.add_argument("--great-learning-rate", type=float, default=5e-5)
    parser.add_argument("--great-max-length", type=int, default=256)
    parser.add_argument("--great-max-new-tokens", type=int, default=128)
    parser.add_argument("--great-sample-batch-size", type=int, default=8)
    parser.add_argument("--great-gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--great-bf16", action="store_true")
    parser.add_argument("--great-fp16", action="store_true")
    parser.add_argument("--great-gradient-checkpointing", action="store_true")
    parser.add_argument("--great-use-lora", action="store_true")
    parser.add_argument("--great-lora-r", type=int, default=16)
    parser.add_argument("--great-lora-alpha", type=int, default=32)
    parser.add_argument("--great-lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--great-condition-on-target",
        action="store_true",
        help="Sample each target class in the same proportion as the real train split.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--wandb-log-model",
        default=None,
        choices=["false", "checkpoint", "end"],
    )

    parser.add_argument("--dcr-max-real", type=int, default=5000)
    parser.add_argument("--dcr-max-synthetic", type=int, default=5000)
    args = parser.parse_args()
    if args.great_bf16 and args.great_fp16:
        parser.error("--great-bf16 and --great-fp16 are mutually exclusive")
    return args


def load_experiment_data(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, str, list[str], list, RepairFn]:
    if args.dataset == "default-credit":
        if args.toy:
            df = make_default_credit_toy(n=1200, seed=7)
        else:
            if not args.data_path:
                raise ValueError("--data-path is required unless --toy is set")
            df = load_default_credit(args.data_path)
        return (
            df,
            DEFAULT_CREDIT_LABEL,
            DEFAULT_CREDIT_CATEGORICAL,
            default_credit_constraints(DEFAULT_CREDIT_LABEL),
            lambda frame: repair_default_credit(frame, DEFAULT_CREDIT_LABEL),
        )

    if args.dataset == "heloc":
        if args.toy:
            df = make_heloc_toy(n=1200, seed=7)
        else:
            if not args.data_path:
                raise ValueError("--data-path is required unless --toy is set")
            df = load_heloc(args.data_path)
        return (
            df,
            HELOC_LABEL,
            HELOC_CATEGORICAL,
            heloc_constraints(HELOC_LABEL),
            lambda frame: repair_heloc(frame, HELOC_LABEL),
        )

    if not args.data_path:
        raise ValueError("--data-path is required for Adult")
    df = load_adult(args.data_path)
    return (
        df,
        ADULT_LABEL,
        ADULT_CATEGORICAL,
        adult_constraints(ADULT_LABEL),
        lambda frame: repair_adult(frame, ADULT_LABEL),
    )


def generate_synthetic(
    args: argparse.Namespace,
    train: pd.DataFrame,
    target_col: str,
    categorical_columns: list[str],
    constraints: list,
    seed: int,
    n_synthetic: int,
    seed_dir: Path,
) -> pd.DataFrame:
    if args.method == "epic":
        return run_epic(
            args,
            train,
            target_col,
            categorical_columns,
            constraints,
            seed,
            n_synthetic,
            seed_dir,
        )
    if args.method == "great":
        return run_great(args, train, target_col, seed, n_synthetic, seed_dir)
    return run_great_proxy(args, train, seed, n_synthetic)


def run_epic(
    args: argparse.Namespace,
    train: pd.DataFrame,
    target_col: str,
    categorical_columns: list[str],
    constraints: list,
    seed: int,
    n_synthetic: int,
    seed_dir: Path,
) -> pd.DataFrame:
    builder = EPICPromptBuilder(
        EPICPromptConfig(
            target_col=target_col,
            n_samples_per_class=args.epic_samples_per_class,
            n_sets=args.epic_sets,
            generated_rows_per_prompt=args.epic_rows_per_prompt,
            seed=seed,
        ),
        categorical_columns=categorical_columns,
    ).fit(train)
    (seed_dir / "epic_prompt_preview.txt").write_text(builder.build_prompt(train, seed=seed))

    if args.offline_proxy:
        llm = make_local_dataframe_llm(
            train,
            builder,
            n_rows=args.epic_rows_per_prompt,
            seed=seed,
            row_mutator=offline_mutator_for(args.dataset),
        )
    else:
        if not args.llm_model:
            raise ValueError("--llm-model is required for real EPIC generation")
        llm = OpenAICompatibleChatLLM(
            OpenAICompatibleChatConfig(
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key_env=args.llm_api_key_env,
                temperature=args.llm_temperature,
                max_tokens=args.llm_max_tokens,
                timeout_seconds=args.llm_timeout_seconds,
            )
        )

    max_rounds = args.epic_max_rounds or math.ceil(
        n_synthetic / max(args.epic_rows_per_prompt, 1) * 2
    )
    return builder.generate(
        train,
        llm=llm,
        n_samples=n_synthetic,
        constraints=None,
        drop_invalid_categories=True,
        max_rounds=max_rounds,
    )


def run_great(
    args: argparse.Namespace,
    train: pd.DataFrame,
    target_col: str,
    seed: int,
    n_synthetic: int,
    seed_dir: Path,
) -> pd.DataFrame:
    if args.wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_log_model:
            os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    tuner = HuggingFaceGReaTFineTuner(
        model_name=args.great_model,
        output_dir=seed_dir / "great_model",
        epochs=args.great_epochs,
        batch_size=args.great_batch_size,
        max_length=args.great_max_length,
        learning_rate=args.great_learning_rate,
        float_precision=0,
        report_to=["wandb"] if args.wandb_project else [],
        run_name=f"great-{args.great_model.replace('/', '_')}-seed{seed}",
        gradient_accumulation_steps=args.great_gradient_accumulation_steps,
        bf16=args.great_bf16,
        fp16=args.great_fp16,
        gradient_checkpointing=args.great_gradient_checkpointing,
        use_lora=args.great_use_lora,
        lora_r=args.great_lora_r,
        lora_alpha=args.great_lora_alpha,
        lora_dropout=args.great_lora_dropout,
    )
    tuner.fit(train)
    if not args.great_condition_on_target:
        return tuner.sample(
            n_synthetic,
            max_new_tokens=args.great_max_new_tokens,
            batch_size=args.great_sample_batch_size,
        )

    chunks = []
    proportions = train[target_col].value_counts(normalize=True).sort_index()
    remaining = n_synthetic
    for index, (target_value, proportion) in enumerate(proportions.items()):
        count = int(round(n_synthetic * proportion))
        if index == len(proportions) - 1:
            count = remaining
        remaining -= count
        chunks.append(
            tuner.sample(
                count,
                conditions={target_col: target_value},
                max_new_tokens=args.great_max_new_tokens,
                batch_size=args.great_sample_batch_size,
            )
        )
    return pd.concat(chunks, ignore_index=True).head(n_synthetic)


def run_great_proxy(
    args: argparse.Namespace,
    train: pd.DataFrame,
    seed: int,
    n_synthetic: int,
) -> pd.DataFrame:
    if args.dataset == "default-credit":
        return bootstrap_default_credit_with_jitter(train, n=n_synthetic, seed=seed)
    return generic_bootstrap_with_jitter(train, n=n_synthetic, seed=seed)


def offline_mutator_for(dataset: str):
    if dataset == "default-credit":
        return lambda rows, seed: bootstrap_default_credit_with_jitter(
            rows,
            n=len(rows),
            seed=seed,
        )
    return lambda rows, seed: generic_bootstrap_with_jitter(rows, n=len(rows), seed=seed)


def generic_bootstrap_with_jitter(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled = df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True).copy()
    for column in sampled.columns:
        if pd.api.types.is_numeric_dtype(sampled[column]) and sampled[column].nunique(dropna=True) > 16:
            std = pd.to_numeric(sampled[column], errors="coerce").std()
            if pd.notna(std) and std > 0:
                noise = pd.Series(rng.normal(0, 0.02 * std, size=n))
                sampled[column] = pd.to_numeric(sampled[column], errors="coerce") + noise
    return sampled


def evaluate_run(
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_synth: pd.DataFrame,
    fixed_synth: pd.DataFrame,
    target_col: str,
    categorical_columns: list[str],
    constraints: list,
    seed: int,
    method: str,
    dcr_max_real: int | None,
    dcr_max_synthetic: int | None,
    seed_dir: Path,
) -> dict:
    raw_constraints = evaluate_constraints(raw_synth, constraints)
    fixed_constraints = evaluate_constraints(fixed_synth, constraints)
    similarity = column_similarity(train, fixed_synth, categorical_columns)
    similarity.to_csv(seed_dir / "column_similarity.csv", index=False)

    dcr = distance_to_closest_record(
        train.drop(columns=[target_col]),
        fixed_synth.drop(columns=[target_col]),
        categorical_columns,
        max_real_records=dcr_max_real,
        max_synthetic_records=dcr_max_synthetic,
        seed=seed,
    )
    try:
        tstr = tstr_auc(train, fixed_synth, test, target_col).__dict__
    except Exception as exc:
        tstr = {"error": str(exc)}
    return {
        "seed": seed,
        "method": method,
        "n_real_train": len(train),
        "n_real_test": len(test),
        "n_synthetic_raw": len(raw_synth),
        "n_synthetic_repaired": len(fixed_synth),
        "raw_constraints": raw_constraints.as_dict(),
        "repaired_constraints": fixed_constraints.as_dict(),
        "revision": revision_rate(raw_synth, fixed_synth).as_dict(),
        "column_similarity": summarize_column_similarity(similarity),
        "tstr_auc": tstr,
        "dcr": dcr.describe().to_dict(),
    }


def flatten_dict(value: dict, prefix: str = "") -> dict:
    flat = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flat.update(flatten_dict(item, name))
        else:
            flat[name] = item
    return flat


if __name__ == "__main__":
    main()
