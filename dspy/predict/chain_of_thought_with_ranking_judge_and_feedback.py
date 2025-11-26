"""
Chain of Thought with Ranking Judge and Feedback module for DSPy.
A module that samples N different responses, judges each with feedback, ranks them, 
and retries with feedback if needed.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

import dspy
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.ranking_judge_with_feedback import RankingJudgeWithFeedback
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ChainOfThoughtWithRankingJudgeAndFeedback(Module):
    """
    A DSPy module that samples N different responses, judges each with feedback,
    ranks them, and retries with combined feedback if needed.

    The module will:
    1. Generate N different responses (using different rollout_ids)
    2. Filter out duplicate responses
    3. Judge each unique response (get is_correct + feedback)
    4. Rank all responses
    5. If the best response is correct, return it
    6. If none are correct, retry with combined feedback
    """

    def __init__(self, n: int = 3, max_retries: int = 2):
        """
        Initialize the ChainOfThoughtWithRankingJudgeAndFeedback module.

        Args:
            n: Number of responses to sample per round (default: 3)
            max_retries: Maximum number of retry rounds if judge fails (default: 2)
        """
        super().__init__()
        self.n = n
        self.max_retries = max_retries

        # Initialize the components
        self.chain_of_thought = ChainOfThought("instruction -> response")
        self.ranking_judge = RankingJudgeWithFeedback()

    async def aforward(self, instruction: str) -> Prediction:
        """Async version of forward method."""
        lm = self.chain_of_thought.get_lm() or dspy.settings.lm

        # Try to get a good answer with retries
        final_result = None
        execution_trace = []
        feedback_summary = None

        remaining_retries = self.max_retries + 1
        while remaining_retries > 0:
            start = lm.kwargs.get("rollout_id", 0)
            rollout_ids = [start + i for i in range(self.n)]

            # Sample N responses with different rollout IDs
            responses = []
            response_predictions = []

            for idx, rid in enumerate(rollout_ids):
                lm_ = lm.copy(rollout_id=rid, temperature=0.6)
                cot = self.chain_of_thought.deepcopy()
                cot.set_lm(lm_)

                # Add feedback to instruction if we're retrying
                current_instruction = instruction
                if feedback_summary and remaining_retries < self.max_retries + 1:
                    current_instruction = f"{instruction}\n\nFeedback from previous attempts: {feedback_summary}"

                try:
                    cot_result = await cot.aforward(instruction=current_instruction)
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

            # If all responses are the same, judge just that one
            if len(unique_responses) == 1:
                logger.debug(f"All {self.n} responses were identical")

                # Judge the response
                if remaining_retries > 0:
                    judge_results = await self.ranking_judge.aforward(
                        instruction=instruction,
                        predictions=unique_responses
                    )

                    # Get results for first (and only) prediction
                    result_key = list(judge_results.keys())[0]
                    judge_result = judge_results[result_key]
                    is_correct = judge_result.get("is_correct", "no").lower() in ["yes", "true", "correct"]
                    feedback = judge_result.get("feedback", "No feedback")

                    execution_trace.append({
                        "round": self.max_retries - remaining_retries + 1,
                        "attempts": len(responses),
                        "unique_responses": len(unique_responses),
                        "best_response": unique_responses[0],
                        "judge_verdict": judge_result.get("is_correct"),
                        "judge_rank": judge_result.get("rank", 1),
                        "judge_feedback": feedback,
                        "approved": is_correct
                    })

                    if is_correct:
                        final_result = unique_responses[0]
                        feedback_summary = feedback
                        break
                    else:
                        feedback_summary = feedback
            else:
                # Judge all unique responses
                if remaining_retries > 0:
                    judge_results = await self.ranking_judge.aforward(
                        instruction=instruction,
                        predictions=unique_responses
                    )

                    # Find the best response (rank 1)
                    best_prediction_idx = None
                    best_rank = float('inf')
                    is_best_correct = False
                    best_feedback = None

                    for idx, resp in enumerate(unique_responses):
                        result_key = list(judge_results.keys())[idx]
                        judge_result = judge_results[result_key]
                        rank = judge_result.get("rank", idx + 1)
                        is_correct = judge_result.get("is_correct", "no").lower() in ["yes", "true", "correct"]
                        feedback = judge_result.get("feedback", "No feedback")

                        if rank < best_rank:
                            best_rank = rank
                            best_prediction_idx = idx
                            is_best_correct = is_correct
                            best_feedback = feedback

                    execution_trace.append({
                        "round": self.max_retries - remaining_retries + 1,
                        "attempts": len(responses),
                        "unique_responses": len(unique_responses),
                        "best_response": unique_responses[best_prediction_idx],
                        "judge_verdict": "correct" if is_best_correct else "incorrect",
                        "judge_rank": best_rank,
                        "judge_feedback": best_feedback,
                        "approved": is_best_correct,
                        "all_results": [{
                            "response": resp,
                            "rank": results.get("rank", idx + 1),
                            "is_correct": results.get("is_correct", "no"),
                            "feedback": results.get("feedback", "No feedback")
                        } for idx, (resp, results) in enumerate(zip(unique_responses, judge_results.values()))]
                    })

                    if is_best_correct:
                        final_result = unique_responses[best_prediction_idx]
                        break
                    else:
                        # Combine feedback from all responses
                        all_feedbacks = [
                            f"- {resp[:50]}...: {results.get('feedback', 'No feedback')}" 
                            for resp, results in zip(unique_responses, judge_results.values())
                        ]
                        feedback_summary = "\n".join(all_feedbacks)
                else:
                    # Last attempt, just pick the best ranked response
                    final_result = unique_responses[0]

            remaining_retries -= 1

        # If all attempts failed, use the last best result
        if final_result is None:
            # Get the best from the last round
            if unique_responses:
                final_result = unique_responses[0]
            else:
                final_result = responses[0]

            logger.warning(f"All {self.max_retries + 1} rounds failed, using last best result")

        # Get the reasoning from the best response
        best_response_idx = responses.index(final_result) if final_result in responses else 0
        best_prediction = response_predictions[best_response_idx] if response_predictions else None

        prediction = Prediction(
            response=final_result,
            reasoning=best_prediction.reasoning if best_prediction and hasattr(best_prediction, "reasoning") else None,
            execution_trace=execution_trace,
        )

        return prediction

    def forward(self, instruction: str) -> Prediction:
        """Sync version of forward method."""
        return asyncio.run(self.aforward(instruction))
