"""
Ranking Judge with Feedback module for DSPy.
This judge ranks multiple predictions AND provides detailed feedback on each.
"""

import asyncio
import json
import logging

from dspy.dsp.utils.settings import settings
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.module import Module
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)


class RankingJudgeWithFeedbackSignature(Signature):
    """Judge and rank multiple predictions, providing detailed feedback for each."""

    instruction = InputField(desc="Original instruction/task")
    predictions = InputField(desc="List of predictions to evaluate and rank")
    results = OutputField(
        desc="JSON-formatted string with rankings and feedback. Format: {\"1\": {\"is_correct\": \"yes/no\", \"rank\": 1, \"feedback\": \"...\"}, ...}"
    )


class RankingJudgeWithFeedback(Module):
    """A module that ranks multiple predictions and provides detailed feedback on each."""

    def __init__(self):
        super().__init__()
        self.judge = ChainOfThought(RankingJudgeWithFeedbackSignature)
        if not settings.judge_lm:
            raise ValueError("RankingJudgeWithFeedback module requires a judge_lm to be configured")
        self.judge.predict.lm = settings.judge_lm
        logger.info("Using judge_lm for ranking judge with feedback")

    async def aforward(self, instruction, predictions):
        """Async version of ranking judge forward."""

        # Format predictions as a list
        predictions_str = "\n".join([f"Prediction {i+1}: {pred}" for i, pred in enumerate(predictions)])

        result = await self.judge.aforward(instruction=instruction, predictions=predictions_str)

        # Parse the JSON results
        try:
            results_dict = json.loads(result.results)
        except (json.JSONDecodeError, AttributeError):
            logger.error(f"Failed to parse judge results: {result.results}")
            raise ValueError("Judge results must be valid JSON")

        return results_dict

    def forward(self, instruction, predictions):
        """Sync version of the forward method."""
        return asyncio.run(self.aforward(instruction, predictions))
