"""
Judge module for validating predictions in DSPy.
"""

import logging

from dspy.dsp.utils.settings import settings
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.module import Module
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)


class JudgeSignature(Signature):
    """Judge whether a prediction correctly follows requirements."""

    instruction = InputField(desc="Original instruction/task")
    context = InputField(desc="Context about the instruction/task (if any)", default=None)
    prediction = InputField(desc="Model's prediction to evaluate")
    is_correct = OutputField(desc="yes/no - whether prediction is correct")
    feedback = OutputField(desc="Concise and actionable feedback on how to improve. For multiple actionable steps, use a numbered list.")


class Judge(Module):
    """A module that judges whether predictions are correct and provides feedback."""

    def __init__(self):
        super().__init__()
        self.judge = ChainOfThought(JudgeSignature)
        if not settings.judge_lm:
            raise ValueError("Judge module requires a judge_lm to be configured")
        self.judge.predict.lm = settings.judge_lm
        logger.info("Using judge_lm for judge")

    def forward(self, instruction, prediction, context=None):
        """Judge a prediction and return whether it's correct with feedback."""
        return self.judge(instruction=instruction, prediction=prediction, context=context)

    async def aforward(self, instruction, prediction, context=None):
        """Async version of judge forward."""
        return await self.judge.aforward(instruction=instruction, prediction=prediction, context=context)
