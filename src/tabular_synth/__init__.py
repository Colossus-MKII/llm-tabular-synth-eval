"""Lightweight reproductions of LLM tabular synthesis ideas.

The package keeps the core mechanics small and inspectable:

- EPIC-style class-balanced CSV prompts with unique categorical value mapping.
- GReaT-style row-to-text serialization and optional HuggingFace fine-tuning.
- Constraint, fidelity, utility, and privacy-oriented evaluation helpers.
"""

from .constraints import (
    Constraint,
    ConstraintReport,
    adult_constraints,
    default_credit_constraints,
    evaluate_constraints,
    heloc_constraints,
    repair_adult,
    repair_default_credit,
    repair_heloc,
)
from .data import load_adult, load_default_credit, load_heloc
from .epic import EPICPromptBuilder, EPICPromptConfig, UniqueValueMapper
from .great import GReaTTextCodec, HuggingFaceGReaTFineTuner
from .llm_backends import OpenAICompatibleChatConfig, OpenAICompatibleChatLLM
from .metrics import RevisionReport, revision_rate

__all__ = [
    "Constraint",
    "ConstraintReport",
    "EPICPromptBuilder",
    "EPICPromptConfig",
    "GReaTTextCodec",
    "HuggingFaceGReaTFineTuner",
    "OpenAICompatibleChatConfig",
    "OpenAICompatibleChatLLM",
    "RevisionReport",
    "UniqueValueMapper",
    "adult_constraints",
    "default_credit_constraints",
    "evaluate_constraints",
    "heloc_constraints",
    "load_adult",
    "load_default_credit",
    "load_heloc",
    "repair_adult",
    "repair_default_credit",
    "repair_heloc",
    "revision_rate",
]
