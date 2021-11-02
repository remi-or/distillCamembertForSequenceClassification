# General imports
from typing import NoReturn, Callable

# Local imports
from outputs import Output

# Metric class
class Metric:

    """
    Base class for metrics.
    The implementation of a metric subclasses this and adds at least:
        - nature, the metric's nature, like Accuracy
        - best, a function to compare measures, like min or max
        - compute, a function that takes an output and returns a float
    """

    nature : str
    best : Callable

    def __init__(
        self,
        is_training : bool = False,
        is_validation : bool = False,
        ) -> None:
        """
        Initializes the metric as a training or validation metric with either one of (is_training) or (is_validation).
        Has to be one, can't be both.
        """
        if not is_training ^ is_validation:
            raise(ValueError('A metric has to be either a training or validation metric, and cannot be both.'))
        self.phase = 'Training' if is_training else 'Validation'
        self.on = False
        self.acc_measure = 0.
        self.acc_size = 0
        self.history = []

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        return f"{self.phase} {self.nature}"

    def __getitem__(
        self,
        i : int,
        ) -> float:
        return self.history[i]

    # region Phase handling
    def training(self) -> None:
        """
        Let the metric know it's going in training.
        """
        self.on = (self.phase == 'Training')

    def validation(self) -> None:
        """
        Let the metric know it's going in validation.
        """
        self.on = (self.phase == 'Validation')

    def end_phase(self) -> None:
        """
        Ends the current phase.
        """
        if self.on:
            self.on = False
            if self.acc_size != 0:
                self.history.append(self.acc_measure / self.acc_size)
            else:
                self.history.append(None)
            self.reset_accumulators()
    # endregion

    # region Logging
    def reset_accumulators(self) -> None:
        """
        Resets the accumulators for size and measure.
        """
        self.acc_measure = 0.
        self.acc_size = 0

    def log(
        self,
        outputs : Output,
        ) -> None:
        """
        Logs the current batch (outputs).
        """
        # Early stopping if the metric isn't on.
        if not self.on:
            return None
        batch_size = len(outputs)
        self.acc_measure += self.compute(outputs) * batch_size
        self.acc_size += batch_size
    # endregion

    @staticmethod
    def unsupported_output_types(
        output_type : str,
        nature : str,
        ) -> NoReturn:
        """
        Function for raising an error message when a metric with (nature) doesn't support an (output_type).
        """
        raise ValueError(f"Outputs of type {output_type} aren't supported by {nature} metric")