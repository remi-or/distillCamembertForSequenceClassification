from metrics.base import Metric
from outputs import Output, ClassificationOutput, TopRankingOutput


class MeanReciprocalRank(Metric):

    nature = 'Mean reciprocal rank'
    best = max

    @classmethod
    def compute(
        cls,
        outputs: Output,
    ) -> float:
        """
        Computes accuracy from (outputs).
        """
        # TopRanking
        if isinstance(outputs, TopRankingOutput):
            ranks = 1 + (outputs.X > outputs.X.gather(1, outputs.Y.long()[:, None])).sum(1)
            reciprocal_ranks = 1 / ranks
            return reciprocal_ranks.mean().item()
        else:
            cls.unsupported_output_types(type(outputs), cls.nature)
