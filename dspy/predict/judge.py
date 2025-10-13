"""
Judge module for validating predictions in DSPy.
"""

from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.module import Module
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature


class JudgeSignature(Signature):
    """Judge whether a prediction correctly follows requirements."""

    instruction = InputField(desc="Original instruction/task")
    prediction = InputField(desc="Model's prediction to evaluate")
    is_correct = OutputField(desc="yes/no - whether prediction is correct")
    feedback = OutputField(desc="Concise feedback and instructions on how to improve")


class Judge(Module):
    """A module that judges whether predictions are correct and provides feedback."""

    def __init__(self):
        super().__init__()
        self.judge = ChainOfThought(JudgeSignature)

    def forward(self, instruction, prediction):
        """Judge a prediction and return whether it's correct with feedback."""
        return self.judge(instruction=instruction, prediction=prediction)

    async def aforward(self, instruction, prediction):
        """Async version of judge forward."""
        return await self.judge.aforward(instruction=instruction, prediction=prediction)
