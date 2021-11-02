from metrics.base import Metric
from outputs import Output, ClassificationOutput, TopRankingOutput


class Loss(Metric):

    nature = 'Loss'
    best = min

    @classmethod
    def compute(
        cls,
        outputs : Output,
    ) -> float:
        """
        Computes loss from (outputs).
        """
        # Classification or TopRanking
        if isinstance(outputs, ClassificationOutput) or isinstance(outputs, TopRankingOutput):
            if isinstance(outputs.loss, float):
                return outputs.loss
            else:
                return outputs.loss.item()
        else:
            cls.unsupported_output_types(type(outputs), cls.nature)