from __future__ import annotations

from typing import Optional, List, Any, Tuple, Union
import numpy as np


Number = Union[int, float]


class Dataset:

    # The yield_order is the common attributes to all datasets in which the dataset is read-through.
    yield_order : List[Any]

    def shuffle(
        self,
        seed : Optional[int] = None,
    ) -> None:
        """
        Shuffles the .yield_order attribute of the dataset.
        A (seed) can be passed for reproducibilty's sake.
        """
        rng = np.random.default_rng(seed)
        rng.shuffle(self.yield_order)

    def get_number_of_batches(
        self,
        **dataset_kwargs,
    ) -> int:
        """
        Counts the number of batches returned when the .batches(**dataset_kwargs) method is called.
        """
        dataset_kwargs['shuffle'] = False
        number_of_batches = 0
        for _ in self.batches(**dataset_kwargs):
            number_of_batches += 1
        return number_of_batches