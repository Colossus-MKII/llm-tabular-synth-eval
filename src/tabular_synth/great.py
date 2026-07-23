from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GReaTTextConfig:
    random_feature_order: bool = True
    float_precision: int | None = None
    delimiter: str = ", "


class GReaTTextCodec:
    """Textual encoder/decoder for the GReaT method.

    GReaT turns each row into an auto-regressive language-model sequence:
    "Age is 39, Education is Bachelors, ...". Training with shuffled feature
    orders lets the model later accept arbitrary feature-value prefixes.
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        config: GReaTTextConfig | None = None,
    ):
        self.columns = list(columns or [])
        self.config = config or GReaTTextConfig()

    def fit(self, df: pd.DataFrame) -> "GReaTTextCodec":
        self.columns = list(df.columns)
        return self

    def encode_row(
        self,
        row: pd.Series | Mapping[str, object],
        rng: np.random.Generator | None = None,
        random_feature_order: bool | None = None,
    ) -> str:
        if not self.columns:
            self.columns = list(row.keys())

        use_shuffle = (
            self.config.random_feature_order
            if random_feature_order is None
            else random_feature_order
        )
        ordered_columns = self.columns.copy()
        if use_shuffle:
            rng = rng or np.random.default_rng()
            ordered_columns = list(rng.permutation(ordered_columns))

        parts = []
        for column in ordered_columns:
            value = row[column]
            if pd.isna(value):
                continue
            parts.append(f"{column} is {self._format_value(value)}")
        return self.config.delimiter.join(parts)

    def encode_frame(
        self,
        df: pd.DataFrame,
        seed: int = 42,
        random_feature_order: bool | None = None,
    ) -> list[str]:
        if not self.columns:
            self.fit(df)
        rng = np.random.default_rng(seed)
        return [
            self.encode_row(row, rng=rng, random_feature_order=random_feature_order)
            for _, row in df.loc[:, self.columns].iterrows()
        ]

    def prompt_from_conditions(self, conditions: Mapping[str, object]) -> str:
        unknown = [column for column in conditions if column not in self.columns]
        if unknown:
            raise ValueError(f"Unknown condition columns: {unknown}")
        return self.config.delimiter.join(
            f"{column} is {self._format_value(value)}"
            for column, value in conditions.items()
        )

    def prompt_from_partial_row(
        self,
        row: pd.Series,
        missing_col: str | None = None,
        seed: int = 42,
    ) -> str:
        if not self.columns:
            self.columns = list(row.index)
        known = row.dropna()
        if missing_col is None:
            missing = [column for column in self.columns if column not in known.index]
            missing_col = missing[0] if missing else self.columns[0]
        encoded = self.encode_row(known, rng=np.random.default_rng(seed))
        prefix = f"{encoded}{self.config.delimiter}" if encoded else ""
        return f"{prefix}{missing_col} is"

    def decode_texts(self, texts: Sequence[str]) -> pd.DataFrame:
        rows = [self.decode_text(text) for text in texts]
        return pd.DataFrame(rows, columns=self.columns)

    def decode_text(self, text: str) -> dict[str, object]:
        row: dict[str, object] = {column: np.nan for column in self.columns}
        normalized = text.replace(";", ",").replace("\n", ",")
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
        for part in parts:
            match = re.match(r"^(?P<name>.+?)\s+is\s+(?P<value>.*)$", part)
            if not match:
                continue
            column = match.group("name").strip()
            value = match.group("value").strip().strip("\"'")
            if column in row and pd.isna(row[column]):
                row[column] = value
        return row

    def _format_value(self, value: object) -> object:
        if isinstance(value, (float, np.floating)) and self.config.float_precision is not None:
            formatted = f"{value:.{self.config.float_precision}f}"
            return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted
        return value


class HuggingFaceGReaTFineTuner:
    """Optional low-compute GReaT fine-tuner.

    This intentionally does not pre-train. It fine-tunes an existing causal LM
    checkpoint on GReaTTextCodec outputs. Install the optional dependencies only
    when you have local compute and model weights available.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        output_dir: str | Path = "artifacts/great_model",
        epochs: int = 1,
        batch_size: int = 4,
        max_length: int = 256,
        learning_rate: float = 5e-5,
        float_precision: int | None = None,
        report_to: Sequence[str] | None = None,
        run_name: str | None = None,
        gradient_accumulation_steps: int = 1,
        bf16: bool = False,
        fp16: bool = False,
        gradient_checkpointing: bool = False,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.report_to = list(report_to or [])
        self.run_name = run_name
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.bf16 = bf16
        self.fp16 = fp16
        self.gradient_checkpointing = gradient_checkpointing
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = list(
            lora_target_modules
            or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
        self.codec = GReaTTextCodec(
            config=GReaTTextConfig(float_precision=float_precision)
        )
        self.tokenizer = None
        self.model = None

    def fit(self, df: pd.DataFrame):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise ImportError(
                "HuggingFace GReaT fine-tuning requires torch and transformers. "
                "Install with: pip install torch transformers"
            ) from exc

        self.codec.fit(df)
        texts = self.codec.encode_frame(df, random_feature_order=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        torch_dtype = torch.bfloat16 if self.bf16 else (torch.float16 if self.fp16 else None)
        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        if self.use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise ImportError(
                    "LoRA fine-tuning requires peft. Install with: pip install peft"
                ) from exc
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, rows: list[str]):
                self.rows = rows

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> str:
                return self.rows[index]

        def collate(batch: list[str]):
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            labels = encoded["input_ids"].clone()
            labels[encoded["attention_mask"] == 0] = -100
            encoded["labels"] = labels
            return encoded

        args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            bf16=self.bf16,
            fp16=self.fp16,
            report_to=self.report_to,
            run_name=self.run_name,
            save_strategy="epoch",
            logging_steps=25,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=TextDataset(texts),
            data_collator=collate,
            processing_class=tokenizer,
        )
        trainer.train()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        self.tokenizer = tokenizer
        self.model = model
        return trainer

    def sample(
        self,
        n_samples: int,
        conditions: Mapping[str, object] | None = None,
        max_new_tokens: int = 96,
        temperature: float = 0.7,
        batch_size: int = 8,
    ) -> pd.DataFrame:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Call fit() or load a saved model before sampling.")

        try:
            import torch
        except ImportError as exc:
            raise ImportError("Sampling requires torch.") from exc

        prompt = (
            self.codec.prompt_from_conditions(conditions)
            if conditions
            else f"{self.codec.columns[0]} is"
        )
        texts = []
        self.model.eval()
        for start in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - start)
            prompts = [prompt] * current_batch_size
            encoded = self.tokenizer(prompts, padding=True, return_tensors="pt")
            with torch.no_grad():
                output = self.model.generate(
                    encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            texts.extend(self.tokenizer.batch_decode(output, skip_special_tokens=True))
        return self.codec.decode_texts(texts)
