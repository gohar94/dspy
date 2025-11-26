"""
Ranking Judge module for ranking multiple predictions in DSPy.
"""

import asyncio
import logging

from dspy.dsp.utils.settings import settings
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.module import Module
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)


class RankingJudgeSignature(Signature):
    """Judge and rank multiple predictions from best to worst."""

    instruction = InputField(desc="Original instruction/task")
    predictions = InputField(desc="List of predictions to rank")
    rankings = OutputField(
        desc="A list of rankings for each prediction, where 1 is best. Format: comma-separated numbers like '1,2,3'"
    )


class RankingJudge(Module):
    """A module that ranks multiple predictions and returns the best one."""

    def __init__(self):
        super().__init__()
        self.judge = ChainOfThought(RankingJudgeSignature)
        if not settings.judge_lm:
            raise ValueError("RankingJudge module requires a judge_lm to be configured")
        self.judge.predict.lm = settings.judge_lm
        logger.info("Using judge_lm for ranking judge")

    async def aforward(self, instruction, predictions):
        """Async version of ranking judge forward."""
        # Format predictions as a list
        predictions_str = "\n".join([f"Prediction {i+1}: {pred}" for i, pred in enumerate(predictions)])

        result = await self.judge.aforward(instruction=instruction, predictions=predictions_str)

        # Parse rankings from the result
        rankings_str = result.rankings
        rankings = [int(x.strip()) for x in rankings_str.split(",")]

        return {"rankings": rankings}

    def forward(self, instruction, predictions):
        """Sync version of the forward method."""
        return asyncio.run(self.aforward(instruction, predictions))
