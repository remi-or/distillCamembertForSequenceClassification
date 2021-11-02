from __future__ import annotations

from typing import List

from time import perf_counter

from metrics import parse
from outputs import Output

class Logger:

    """
    The base class for logging metrics during training and validation.
    
    Attributes:
        - metrics, a dictionnary with metric_name : metric_object
        - metric_descriptions, the list where metrics' description are stored for clean copies
    """

    def __init__(
        self,
        metric_descriptions : List[str],
    ) -> None:
        self.metric_descriptions = metric_descriptions
        self.metrics = {}
        for metric_description in metric_descriptions:
            metric = parse(metric_description)
            self.metrics[metric.__repr__()] = metric
            

    def clean_copy(
        self,
    ) -> Logger:
        return Logger(self.metric_descriptions)
    
    def training(self) -> None:
        """
        Let the logger know it's going into training.
        """
        for _, metric in self.metrics.items():
            metric.training()
    
    def validation(self) -> None:
        """
        Let the logger know it's going into validation.
        """
        for _, metric in self.metrics.items():
            metric.validation()

    def end_phase(self) -> None:
        """
        Let the logger know the current phase just ended.
        """
        for _, metric in self.metrics.items():
            metric.end_phase()

    def log(
        self,
        outputs : Output,
    ) -> None:
        """
        Logs the (outputs) for the current batch.
        """
        for _, metric in self.metrics.items():
            metric.log(outputs)