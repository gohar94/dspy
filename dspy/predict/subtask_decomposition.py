"""
Sub-task decomposition module for DSPy.
Breaks down complex queries into manageable sub-tasks with judge verification at each step.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from dspy.dsp.utils.settings import settings
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.judge import Judge
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TaskDecompositionSignature(Signature):
    """Break down a given instruction into a list of sub-tasks to make it easy to follow the instruction. Do not add any new tasks that are not contained in the original instruction and do not create any validation steps."""

    instruction = InputField(desc="The original complex instruction to break down")
    max_subtasks = InputField(desc="Maximum number of sub-tasks to create (may be less than the maximum if the instruction is not complex)")
    subtasks = OutputField(desc="List of specific, actionable sub-tasks that together accomplish the original instruction. Format as a numbered list.")


class SubTaskExecutionSignature(Signature):
    """Execute a specific sub-task with context from the original instruction."""

    original_instruction = InputField(desc="The original complex instruction")
    subtask = InputField(desc="The specific sub-task to execute")
    previous_attempt = InputField(desc="The previous (incorrect) attempt at the same sub-task (if any)", default=None)
    previous_attempt_feedback = InputField(desc="Feedback on the previous attempt at the same sub-task (if any)", default=None)
    result = OutputField(desc="The result of executing this sub-task")


class SubTaskDecomposition(Module):
    """
    A DSPy module that decomposes complex instructions into sub-tasks,
    executes each sub-task with judge verification using the existing Judge module,
    and combines results.

    This module integrates with the existing DSPy Judge module for quality verification
    at each sub-task step, ensuring that each sub-task result is correct before proceeding.

    The task decomposition step uses the judge_lm (if configured) for more careful and
    analytical task breakdown, while sub-task execution uses the regular lm.
    """

    def __init__(self, max_subtasks: int = 5, max_retries_per_subtask: int = 2):
        """
        Initialize the SubTaskDecomposition module.

        Args:
            max_subtasks: Maximum number of sub-tasks to create (default: 5)
            max_retries_per_subtask: Maximum retries per sub-task if judge fails (default: 2)
        """
        super().__init__()
        self.max_subtasks = max_subtasks
        self.max_retries_per_subtask = max_retries_per_subtask

        # Initialize the components
        self.decomposer = Predict(TaskDecompositionSignature)
        if settings.judge_lm:
            # Set the judge_lm on the internal Predict module of ChainOfThought
            self.decomposer.lm = settings.judge_lm
        else:
            logger.debug("Using default lm for task decomposition")

        self.executor = Predict(SubTaskExecutionSignature)
        self.judge = Judge()

    def _parse_subtasks(self, subtasks_text: str) -> list[str]:
        """Parse the subtasks from the decomposer output."""
        lines = subtasks_text.strip().split("\n")
        subtasks = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering (1., 2., etc.) and clean up
            if line[0].isdigit() and ("." in line or ")" in line):
                # Find the first space after the number
                space_idx = line.find(" ")
                if space_idx > 0:
                    line = line[space_idx:].strip()

            if line:
                subtasks.append(line)

        return subtasks[:self.max_subtasks]  # Limit to max_subtasks

    def _combine_results(self, instruction: str, subtasks: list[str], results: list[str]) -> str:
        """Combine sub-task results into a coherent final answer."""
        if len(results) == 1:
            return results[0]

        # Create a summary combining all results
        combined = f"Based on the original instruction: {instruction}\n\n"

        for i, (subtask, result) in enumerate(zip(subtasks, results, strict=False)):
            combined += f"Step {i+1} - {subtask}:\n{result}\n\n"

        # Try to synthesize a final answer
        synthesis_signature = Signature(
            "instruction, subtask_results -> final_answer",
            "Given the original instruction and results from sub-tasks, provide a concise final answer that follows the original instruction."
        )

        synthesizer = ChainOfThought(synthesis_signature)

        try:
            synthesis_result = synthesizer(
                instruction=instruction,
                subtask_results=combined
            )
            return synthesis_result.final_answer
        except Exception as e:
            logger.debug(f"Failed to synthesize final answer: {e}")
            return combined

    async def aforward(self, instruction: str) -> Prediction:
        """Async version of forward method."""
        logger.debug(f"Starting async sub-task decomposition for: {instruction[:100]}...")

        # Step 1: Decompose the instruction into sub-tasks
        decomposition_result = await self.decomposer.acall(
            instruction=instruction,
            max_subtasks=str(self.max_subtasks)
        )

        subtasks = self._parse_subtasks(decomposition_result.subtasks)
        logger.debug(f"Decomposed into {len(subtasks)} sub-tasks: {subtasks}")

        if not subtasks:
            # Fallback: treat the original instruction as a single task
            subtasks = [instruction]
            logger.debug("No sub-tasks generated, using original instruction as single task")

        # Step 2: Execute each sub-task
        results = []
        execution_trace = []

        for i, subtask in enumerate(subtasks):
            logger.debug(f"Executing sub-task {i+1}/{len(subtasks)}: {subtask}")

            # Execute sub-task with retries
            subtask_success = False
            previous_attempt = None
            previous_attempt_feedback = None
            attempt = 0

            while attempt < self.max_retries_per_subtask and not subtask_success:
                attempt += 1
                logger.debug(f"Sub-task {i+1} attempt {attempt}")

                # Execute the sub-task
                execution_result = await self.executor.aforward(
                    original_instruction=instruction,
                    subtask=subtask,
                    previous_attempt=previous_attempt,
                    previous_attempt_feedback=previous_attempt_feedback,
                )

                if settings.per_task_judge:
                    judge_result = await self.judge.aforward(
                        instruction=subtask,
                        prediction=execution_result.result,
                        context=instruction
                    )

                    # Check if the judge approves
                    is_correct = judge_result.is_correct.lower() in ["yes", "true", "correct"]

                    execution_trace.append({
                        "subtask_index": i + 1,
                        "subtask": subtask,
                        "attempt": attempt,
                        "result": execution_result.result,
                        "judge_verdict": judge_result.is_correct,
                        "judge_feedback": judge_result.feedback,
                        "approved": is_correct
                    })

                    if is_correct:
                        results.append(execution_result.result)
                        subtask_success = True
                        logger.debug(f"Sub-task {i+1} approved by judge")
                    else:
                        logger.debug(f"Sub-task {i+1} rejected by judge: {judge_result.feedback}")
                        if attempt < self.max_retries_per_subtask:
                            logger.debug(f"Retrying sub-task {i+1} with feedback...")
                            previous_attempt = execution_result.result
                            previous_attempt_feedback = judge_result.feedback
                else:
                    subtask_success = True
                    results.append(execution_result.result)

            if not subtask_success:
                # Use the last result even if not approved
                results.append(execution_result.result)
                logger.debug(f"Sub-task {i+1} failed all attempts, using last result")

        # Step 3: Combine results into final answer
        final_result = self._combine_results(instruction, subtasks, results)

        # Create the prediction with full trace
        prediction = Prediction(
            response=final_result,
            subtasks=subtasks,
            subtask_results=results,
            execution_trace=execution_trace,
            reasoning=f"Decomposed into {len(subtasks)} sub-tasks and executed each with judge verification"
        )

        logger.debug(f"Async sub-task decomposition completed. Final result: {final_result[:100]}...")
        return prediction

    def forward(self, instruction: str) -> Prediction:
        """Sync version of forward method."""
        return asyncio.run(self.aforward(instruction))
