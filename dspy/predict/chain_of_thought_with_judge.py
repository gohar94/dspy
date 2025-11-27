"""
Chain of Thought with Judge module for DSPy.
A simple module that uses Chain of Thought to answer questions, then uses a judge to verify the answer.
If the judge rejects the answer, it retries with feedback for a configurable number of attempts.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.judge import Judge
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ChainOfThoughtWithJudge(Module):
    """
    A DSPy module that uses Chain of Thought to answer questions,
    then uses a judge to verify the answer quality.

    If the judge rejects the answer, it retries with feedback for a configurable
    number of attempts.
    """

    def __init__(self, max_retries: int = 2):
        """
        Initialize the ChainOfThoughtWithJudge module.

        Args:
            max_retries: Maximum number of retries if judge fails (default: 2)
        """
        super().__init__()
        self.max_retries = max_retries

        # Initialize the components
        self.chain_of_thought = ChainOfThought("instruction -> response")
        self.judge = Judge()

    async def aforward(self, instruction: str) -> Prediction:
        """Async version of forward method."""
        # Try to get a good answer with retries
        final_result = None
        execution_trace = []
        feedback = None
        remaining_retries = self.max_retries

        while remaining_retries > 0:
            if remaining_retries == self.max_retries:
                # First attempt - no feedback
                cot_result = await self.chain_of_thought.aforward(instruction=instruction)
            else:
                # Retry with feedback
                enhanced_instruction = f"{instruction}\n\nPrevious attempt feedback: {feedback}"
                cot_result = await self.chain_of_thought.aforward(instruction=enhanced_instruction)

            # Judge the result
            judge_result = await self.judge.aforward(
                instruction=instruction,
                prediction=cot_result.response
            )

            # Check if the judge approves
            is_correct = judge_result.is_correct.lower() in ["yes", "true", "correct"]

            execution_trace.append({
                "attempt": self.max_retries - remaining_retries + 1,
                "response": cot_result.response,
                "judge_verdict": judge_result.is_correct,
                "judge_feedback": judge_result.feedback,
                "approved": is_correct
            })

            if is_correct:
                final_result = cot_result.response
                break
            else:
                feedback = judge_result.feedback

            remaining_retries -= 1

        # If all attempts failed, use the last result
        if final_result is None:
            final_result = cot_result.response
            logger.warning(f"All {self.max_retries} attempts failed, using last result")

        # Create the prediction with full trace
        prediction = Prediction(
            response=final_result,
            reasoning=cot_result.reasoning if hasattr(cot_result, "reasoning") else None,
            execution_trace=execution_trace,
        )

        return prediction

    def forward(self, instruction: str) -> Prediction:
        """Sync version of forward method."""
        return asyncio.run(self.aforward(instruction))
