from metrics.base import Metric
from outputs import Output, ClassificationOutput, TopRankingOutput


class Accuracy(Metric):

    nature = 'Accuracy'
    best = max

    @classmethod
    def compute(
        cls,
        outputs: Output,
    ) -> float:
        """
        Computes accuracy from (outputs).
        """
        # Classification or TopRanking
        if isinstance(outputs, ClassificationOutput) or isinstance(outputs, TopRankingOutput):
            predictions = outputs.X if outputs.argmaxed else outputs.X.argmax(1)
            return predictions.eq(outputs.Y).double().mean().item()
        else:
            cls.unsupported_output_types(type(outputs), cls.nature)
