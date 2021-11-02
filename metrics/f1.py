import torch

from metrics.base import Metric
from outputs import Output, ClassificationOutput, TopRankingOutput


class F1(Metric):

    nature = 'F1'
    best = max

    @classmethod
    def compute(
        cls,
        outputs: Output,
    ) -> float:
        """
        Computes f1 score from (outputs).
        """
        # Classification or TopRanking
        if isinstance(outputs, ClassificationOutput) or isinstance(outputs, TopRankingOutput):
            predictions = outputs.X if outputs.argmaxed else outputs.X.argmax(1)
            correct = predictions.eq(outputs.Y).double()
            if correct.mean() == 1:
                return 1.            
            positive = predictions.eq(1).double()
            tp = correct.mul(positive).sum().item()
            tn = correct.mul(1 - positive).sum().item()
            fp = (1 - correct).mul(positive).sum().item()
            fn = (1 - correct).mul(1 - positive).sum().item()
            return tp / (tp + (fp + fn) / 2)
        else:
            cls.unsupported_output_types(type(outputs), cls.nature)
