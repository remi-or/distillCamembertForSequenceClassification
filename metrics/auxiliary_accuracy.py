from metrics.base import Metric
from outputs import Output, ClassificationOutput, TopRankingOutput
import logging


class AuxiliaryAccuracy(Metric):

    nature = 'Auxiliary accuracy'
    best = max

    @classmethod
    def compute(
        cls,
        outputs: Output,
    ) -> float:
        """
        Computes aiuxiliary accuracy from (outputs).
        """
        # Classification or TopRanking
        if isinstance(outputs, ClassificationOutput) or isinstance(outputs, TopRankingOutput):
            if outputs.Y2 is None:
                logging.warning('Auxialiary accuracy was used on an input with no auxiliary labels.')
                return 0.
            else:
                predictions = outputs.X if outputs.argmaxed else outputs.X.argmax(1)
                return predictions.eq(outputs.Y2).double().mean().item()
        else:
            cls.unsupported_output_types(type(outputs), cls.nature)