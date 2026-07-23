from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REAL_SCALE_SCRIPT = ROOT / "scripts" / "run_real_scale_experiment.py"


def main() -> None:
    args = parse_args()
    model_slug = args.base_model.replace("/", "__")
    pair_outdir = Path(args.outdir) / model_slug
    pair_outdir.mkdir(parents=True, exist_ok=True)

    common = [
        "--dataset",
        args.dataset,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--test-size",
        str(args.test_size),
        "--dcr-max-real",
        str(args.dcr_max_real),
        "--dcr-max-synthetic",
        str(args.dcr_max_synthetic),
    ]
    if args.toy:
        common.append("--toy")
    else:
        common.extend(["--data-path", args.data_path])
    if args.n_synthetic is not None:
        common.extend(["--n-synthetic", str(args.n_synthetic)])

    if not args.skip_epic:
        epic_cmd = [
            sys.executable,
            str(REAL_SCALE_SCRIPT),
            *common,
            "--method",
            "epic",
            "--llm-model",
            args.base_model,
            "--llm-base-url",
            args.llm_base_url,
            "--llm-api-key-env",
            args.llm_api_key_env,
            "--llm-temperature",
            str(args.llm_temperature),
            "--llm-max-tokens",
            str(args.llm_max_tokens),
            "--epic-samples-per-class",
            str(args.epic_samples_per_class),
            "--epic-sets",
            str(args.epic_sets),
            "--epic-rows-per-prompt",
            str(args.epic_rows_per_prompt),
            "--outdir",
            str(pair_outdir / "epic"),
        ]
        if args.offline_proxy:
            epic_cmd.append("--offline-proxy")
        run(epic_cmd)

    if not args.skip_great:
        great_cmd = [
            sys.executable,
            str(REAL_SCALE_SCRIPT),
            *common,
            "--method",
            "great",
            "--great-model",
            args.base_model,
            "--great-epochs",
            str(args.great_epochs),
            "--great-batch-size",
            str(args.great_batch_size),
            "--great-gradient-accumulation-steps",
            str(args.great_gradient_accumulation_steps),
            "--great-learning-rate",
            str(args.great_learning_rate),
            "--great-max-length",
            str(args.great_max_length),
            "--great-max-new-tokens",
            str(args.great_max_new_tokens),
            "--great-sample-batch-size",
            str(args.great_sample_batch_size),
            "--outdir",
            str(pair_outdir / "great"),
        ]
        if args.great_bf16:
            great_cmd.append("--great-bf16")
        if args.great_fp16:
            great_cmd.append("--great-fp16")
        if args.great_gradient_checkpointing:
            great_cmd.append("--great-gradient-checkpointing")
        if args.great_use_lora:
            great_cmd.extend(
                [
                    "--great-use-lora",
                    "--great-lora-r",
                    str(args.great_lora_r),
                    "--great-lora-alpha",
                    str(args.great_lora_alpha),
                    "--great-lora-dropout",
                    str(args.great_lora_dropout),
                ]
            )
        if args.great_condition_on_target:
            great_cmd.append("--great-condition-on-target")
        if args.wandb_project:
            great_cmd.extend(["--wandb-project", args.wandb_project])
        if args.wandb_log_model:
            great_cmd.extend(["--wandb-log-model", args.wandb_log_model])
        run(great_cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired EPIC and GReaT experiments with the same base model."
    )
    parser.add_argument("--dataset", choices=["default-credit", "adult", "heloc"], required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-synthetic", type=int, default=None)
    parser.add_argument("--outdir", default="artifacts/paired_models")
    parser.add_argument("--skip-epic", action="store_true")
    parser.add_argument("--skip-great", action="store_true")

    parser.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-api-key-env", default="LLM_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-max-tokens", type=int, default=2048)
    parser.add_argument("--offline-proxy", action="store_true")
    parser.add_argument("--epic-samples-per-class", type=int, default=8)
    parser.add_argument("--epic-sets", type=int, default=3)
    parser.add_argument("--epic-rows-per-prompt", type=int, default=40)

    parser.add_argument("--great-epochs", type=int, default=1)
    parser.add_argument("--great-batch-size", type=int, default=1)
    parser.add_argument("--great-gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--great-learning-rate", type=float, default=5e-5)
    parser.add_argument("--great-max-length", type=int, default=256)
    parser.add_argument("--great-max-new-tokens", type=int, default=128)
    parser.add_argument("--great-sample-batch-size", type=int, default=8)
    parser.add_argument("--great-bf16", action="store_true")
    parser.add_argument("--great-fp16", action="store_true")
    parser.add_argument("--great-gradient-checkpointing", action="store_true")
    parser.add_argument("--great-use-lora", action="store_true")
    parser.add_argument("--great-lora-r", type=int, default=16)
    parser.add_argument("--great-lora-alpha", type=int, default=32)
    parser.add_argument("--great-lora-dropout", type=float, default=0.05)
    parser.add_argument("--great-condition-on-target", action="store_true")

    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-log-model", default=None, choices=["false", "checkpoint", "end"])
    parser.add_argument("--dcr-max-real", type=int, default=5000)
    parser.add_argument("--dcr-max-synthetic", type=int, default=5000)
    args = parser.parse_args()
    if not args.toy and not args.data_path:
        parser.error("--data-path is required unless --toy is set")
    if args.skip_epic and args.skip_great:
        parser.error("At least one of EPIC or GReaT must run")
    if args.great_bf16 and args.great_fp16:
        parser.error("--great-bf16 and --great-fp16 are mutually exclusive")
    return args


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
