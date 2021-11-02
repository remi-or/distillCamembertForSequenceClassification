from typing import Optional
from torch import Tensor


class TopRankingOutput:

    """
    Output type for top ranking problems.
    
    Notations:
        - N is the batch size
        - D is the number of documents
    
    Inputs:
        - predictions is of shape N * D if the lines are probabily distribution
        - labels is of shape N where each element is an integer in [0, D[
        - loss is an optional tensor of shape 1
    """

    def __init__(
        self,
        predictions: Tensor,
        labels: Tensor,
        loss: Optional[Tensor] = None,
    ) -> None:
        self.X = predictions
        self.Y = labels
        self.loss = loss

    def __len__(self,) -> int:
        """
        Returns the batch's size.
        """
        return self.Y.size()[0]
