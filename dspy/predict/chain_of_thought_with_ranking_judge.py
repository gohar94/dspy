"""
Chain of Thought with Ranking Judge module for DSPy.
A module that samples N different responses, then uses a ranking judge to select the best one.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

import dspy
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.ranking_judge import RankingJudge
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ChainOfThoughtWithRankingJudge(Module):
    """
    A DSPy module that samples N different responses from the model,
    then uses a ranking judge to select the best one.

    The module will:
    1. Generate N different responses (using different rollout_ids)
    2. Filter out duplicate responses
    3. If all responses are the same, return that response
    4. Otherwise, use the ranking judge to rank them and select the best (rank 1)
    """

    def __init__(self, n: int = 3):
        """
        Initialize the ChainOfThoughtWithRankingJudge module.

        Args:
            n: Number of responses to sample (default: 3)
        """
        super().__init__()
        self.n = n

        # Initialize the components
        self.chain_of_thought = ChainOfThought("instruction -> response")
        self.ranking_judge = RankingJudge()

    async def aforward(self, instruction: str) -> Prediction:
        """Async version of forward method."""
        lm = self.chain_of_thought.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        rollout_ids = [start + i for i in range(self.n)]

        # Sample N responses with different rollout IDs
        responses = []
        response_predictions = []

        for idx, rid in enumerate(rollout_ids):
            lm_ = lm.copy(rollout_id=rid, temperature=0.6)
            cot = self.chain_of_thought.deepcopy()
            cot.set_lm(lm_)

            try:
                cot_result = await cot.aforward(instruction=instruction)
                responses.append(cot_result.response)
                response_predictions.append(cot_result)
            except Exception as e:
                logger.warning(f"Failed to generate response {idx + 1} with rollout id {rid}: {e}")

        if not responses:
            raise ValueError("Failed to generate any responses")

        # Filter out duplicate responses
        unique_responses = []
        seen = set()
        for resp in responses:
            if resp not in seen:
                seen.add(resp)
                unique_responses.append(resp)

        # If all responses are the same, just return the first one
        if len(unique_responses) == 1:
            logger.debug(f"All {self.n} responses were identical")
            execution_trace = [
                {
                    "attempt": i + 1,
                    "response": resp,
                    "unique": i == 0,  # First occurrence is unique
                }
                for i, resp in enumerate(responses)
            ]

            return Prediction(
                response=unique_responses[0],
                reasoning=response_predictions[0].reasoning if hasattr(response_predictions[0], "reasoning") else None,
                execution_trace=execution_trace,
            )

        # Use ranking judge to rank the responses
        logger.debug(f"Ranking {len(unique_responses)} unique responses")
        judge_result = await self.ranking_judge.aforward(
            instruction=instruction,
            predictions=unique_responses
        )

        rankings = judge_result["rankings"]

        # Find the best response (rank 1)
        best_idx = rankings.index(1)
        best_response = unique_responses[best_idx]

        # Build execution trace
        execution_trace = []
        response_idx_map = {}  # Map from response to indices in original responses list
        for unique_idx, unique_resp in enumerate(unique_responses):
            indices = [i for i, resp in enumerate(responses) if resp == unique_resp]
            response_idx_map[unique_resp] = indices
            execution_trace.append({
                "unique_response_idx": unique_idx,
                "response": unique_resp,
                "rank": rankings[unique_idx],
                "times_generated": len(indices),
            })

        # Get the reasoning from the best response
        best_unique_idx = best_idx
        best_response_in_original = responses.index(unique_responses[best_unique_idx])
        best_prediction = response_predictions[best_response_in_original]

        prediction = Prediction(
            response=best_response,
            reasoning=best_prediction.reasoning if hasattr(best_prediction, "reasoning") else None,
            execution_trace=execution_trace,
        )

        return prediction

    def forward(self, instruction: str) -> Prediction:
        """Sync version of forward method."""
        return asyncio.run(self.aforward(instruction))
