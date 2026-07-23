import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabular_synth.constraints import (
    default_credit_constraints,
    evaluate_constraints,
    heloc_constraints,
    repair_default_credit,
    repair_heloc,
)
from tabular_synth.data import (
    DEFAULT_CREDIT_CATEGORICAL,
    DEFAULT_CREDIT_LABEL,
    HELOC_LABEL,
    HELOC_COLUMNS,
    make_default_credit_toy,
    make_heloc_toy,
    load_heloc,
    split_train_test,
)
from tabular_synth.epic import EPICPromptBuilder, EPICPromptConfig, make_local_dataframe_llm
from tabular_synth.great import GReaTTextCodec
from tabular_synth.metrics import revision_rate, tstr_auc


class CoreTests(unittest.TestCase):
    def test_default_credit_constraints_catch_invalid_rows(self):
        df = make_default_credit_toy(n=20, seed=1)
        report_ok = evaluate_constraints(df, default_credit_constraints())
        self.assertEqual(report_ok.cvr, 0.0)

        bad = df.copy()
        bad.loc[0, "AGE"] = -3
        bad.loc[1, "PAY_AMT1"] = -10
        report_bad = evaluate_constraints(bad, default_credit_constraints())
        self.assertGreater(report_bad.cvr, 0.0)
        self.assertGreaterEqual(report_bad.row_violation_coverage, 2 / len(bad))

    def test_epic_mapping_generation_roundtrip(self):
        df = make_default_credit_toy(n=120, seed=2)
        constraints = default_credit_constraints()
        builder = EPICPromptBuilder(
            EPICPromptConfig(
                target_col=DEFAULT_CREDIT_LABEL,
                n_samples_per_class=2,
                n_sets=2,
                generated_rows_per_prompt=10,
                seed=2,
            ),
            categorical_columns=DEFAULT_CREDIT_CATEGORICAL,
        ).fit(df)
        prompt = builder.build_prompt(df)
        self.assertIn(DEFAULT_CREDIT_LABEL, prompt)
        self.assertIn("Generate", prompt)

        llm = make_local_dataframe_llm(df, builder, n_rows=10, seed=3)
        synth = builder.generate(df, llm, n_samples=25, constraints=constraints)
        self.assertEqual(list(synth.columns), builder.columns)
        self.assertEqual(len(synth), 25)
        self.assertEqual(evaluate_constraints(synth, constraints).cvr, 0.0)

    def test_great_codec_roundtrip(self):
        df = pd.DataFrame(
            {
                "AGE": [39],
                "EDUCATION": [2],
                DEFAULT_CREDIT_LABEL: [1],
            }
        )
        codec = GReaTTextCodec().fit(df)
        text = codec.encode_frame(df, random_feature_order=False)[0]
        self.assertEqual(
            text,
            f"AGE is 39, EDUCATION is 2, {DEFAULT_CREDIT_LABEL} is 1",
        )
        decoded = codec.decode_texts([text])
        self.assertEqual(decoded.loc[0, "AGE"], "39")
        self.assertEqual(decoded.loc[0, DEFAULT_CREDIT_LABEL], "1")

    def test_tstr_auc_runs(self):
        df = make_default_credit_toy(n=300, seed=4)
        train, test = split_train_test(df, test_size=0.25, seed=4)
        result = tstr_auc(train, train, test, DEFAULT_CREDIT_LABEL, steps=50)
        self.assertFalse(math.isnan(result.synthetic_auc))
        self.assertGreaterEqual(result.synthetic_auc, 0.0)
        self.assertLessEqual(result.synthetic_auc, 1.0)

    def test_revision_rate_after_default_credit_repair(self):
        raw = make_default_credit_toy(n=10, seed=5)
        raw.loc[0, "AGE"] = -3
        raw.loc[0, "PAY_AMT1"] = -20
        raw.loc[1, "SEX"] = 9

        repaired = repair_default_credit(raw)
        constraints_report = evaluate_constraints(repaired, default_credit_constraints())
        revision = revision_rate(raw, repaired)

        self.assertEqual(constraints_report.cvr, 0.0)
        self.assertEqual(revision.revised_cells, 3)
        self.assertEqual(revision.revised_rows, 2)
        self.assertGreater(revision.cell_revision_rate, 0.0)
        self.assertGreater(revision.row_revision_rate, 0.0)

    def test_heloc_constraints_and_repair(self):
        df = make_heloc_toy(n=80, seed=6)
        report_ok = evaluate_constraints(df, heloc_constraints(HELOC_LABEL))
        self.assertEqual(report_ok.cvr, 0.0)

        bad = df.copy()
        bad.loc[0, "RiskPerformance"] = "Maybe"
        bad.loc[0, "PercentTradesNeverDelq"] = 130
        bad.loc[1, "NumTrades90Ever2DerogPubRec"] = 10
        bad.loc[1, "NumTrades60Ever2DerogPubRec"] = 2
        bad.loc[1, "NumTotalTrades"] = 3

        report_bad = evaluate_constraints(bad, heloc_constraints(HELOC_LABEL))
        repaired = repair_heloc(bad, HELOC_LABEL)
        report_repaired = evaluate_constraints(repaired, heloc_constraints(HELOC_LABEL))

        self.assertGreater(report_bad.cvr, 0.0)
        self.assertEqual(report_repaired.cvr, 0.0)

    def test_heloc_normalized_csv_roundtrip(self):
        df = make_heloc_toy(n=20, seed=8)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "heloc_dataset_v1.csv"
            df.to_csv(path, index=False)
            loaded = load_heloc(path)
        self.assertEqual(list(loaded.columns), HELOC_COLUMNS)
        self.assertEqual(len(loaded), 20)

    def test_heloc_huggingface_yes_no_labels(self):
        df = pd.DataFrame(
            {
                "estimate_of_risk": [55, 82],
                "months_since_first_trade": [144, 96],
                "months_since_last_trade": [4, 5],
                "average_duration_of_resolution": [84, 47],
                "number_of_satisfactory_trades": [20, 16],
                "nr_trades_insolvent_for_over_60_days": [3, 0],
                "nr_trades_insolvent_for_over_90_days": [0, 0],
                "percentage_of_legal_trades": [83, 100],
                "months_since_last_illegal_trade": [2, -7],
                "maximum_illegal_trades_over_last_year": [3, 7],
                "maximum_illegal_trades": [5, 8],
                "nr_total_trades": [23, 16],
                "nr_trades_initiated_in_last_year": [1, 3],
                "percentage_of_installment_trades": [43, 31],
                "months_since_last_inquiry_not_recent": [0, 0],
                "nr_inquiries_in_last_6_months": [0, 0],
                "nr_inquiries_in_last_6_months_not_recent": [0, 0],
                "net_fraction_of_revolving_burden": [33, 15],
                "net_fraction_of_installment_burden": [-8, 88],
                "nr_revolving_trades_with_balance": [8, 4],
                "nr_installment_trades_with_balance": [1, 2],
                "nr_banks_with_high_ratio": [1, 0],
                "percentage_trades_with_balance": [69, 67],
                "is_at_risk": ["yes", "no"],
            }
        )
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hf_heloc.csv"
            df.to_csv(path, index=False)
            loaded = load_heloc(path)
        self.assertEqual(list(loaded.columns), HELOC_COLUMNS)
        self.assertEqual(loaded[HELOC_LABEL].tolist(), ["Bad", "Good"])


if __name__ == "__main__":
    unittest.main()
