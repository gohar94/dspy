from dspy.predict.aggregation import majority
from dspy.predict.best_of_n import BestOfN
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.chain_of_thought_with_judge import ChainOfThoughtWithJudge
from dspy.predict.chain_of_thought_with_ranking_judge import ChainOfThoughtWithRankingJudge
from dspy.predict.chain_of_thought_with_ranking_judge_and_feedback import ChainOfThoughtWithRankingJudgeAndFeedback
from dspy.predict.code_act import CodeAct
from dspy.predict.judge import Judge, JudgeSignature
from dspy.predict.knn import KNN
from dspy.predict.multi_chain_comparison import MultiChainComparison
from dspy.predict.parallel import Parallel
from dspy.predict.predict import Predict
from dspy.predict.program_of_thought import ProgramOfThought
from dspy.predict.ranking_judge import RankingJudge, RankingJudgeSignature
from dspy.predict.ranking_judge_with_feedback import RankingJudgeWithFeedback, RankingJudgeWithFeedbackSignature
from dspy.predict.react import ReAct, Tool
from dspy.predict.refine import Refine
from dspy.predict.subtask_decomposition import SubTaskDecomposition

__all__ = [
    "majority",
    "BestOfN",
    "ChainOfThought",
    "CodeAct",
    "Judge",
    "JudgeSignature",
    "KNN",
    "MultiChainComparison",
    "Predict",
    "ProgramOfThought",
    "RankingJudge",
    "RankingJudgeSignature",
    "RankingJudgeWithFeedback",
    "RankingJudgeWithFeedbackSignature",
    "ReAct",
    "Refine",
    "Tool",
    "Parallel",
    "SubTaskDecomposition",
    "ChainOfThoughtWithJudge",
    "ChainOfThoughtWithRankingJudge",
    "ChainOfThoughtWithRankingJudgeAndFeedback",
]
