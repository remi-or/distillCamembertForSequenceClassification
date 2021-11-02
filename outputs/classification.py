from typing import Optional
from torch import Tensor


class ClassificationOutput:

    """
    Output type for classification problems.
    
    Notations:
        - N is the batch size
        - C is the number of classes
    
    Inputs:
        - predictions is of shape N * C where each line is a probability distribution OR shape N if already argmaxed
        - labels is of shape N where each element is an integer in [0, C[
        - labels_2 is like labels, but is used when we want to measure the accuracy relative to 2 targets
        - loss is an optional tensor of shape 1
    """

    def __init__(
        self,
        predictions: Tensor,
        labels: Tensor,
        labels_2: Optional[Tensor] = None,
        loss: Optional[Tensor] = None,
    ) -> None:
        # print(predictions, labels, predictions.size())
        self.X = predictions
        self.argmaxed = len(self.X.size()) == 1
        self.Y = labels
        self.Y2 = labels_2
        self.loss = loss

    def __len__(self,) -> int:
        """
        Returns the batch's size.
        """
        return self.Y.size()[0]
