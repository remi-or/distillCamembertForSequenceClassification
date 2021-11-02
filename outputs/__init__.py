# General imports
from typing import Union

# Output types import
from outputs.classification import ClassificationOutput
from outputs.top_ranking import TopRankingOutput

# Main output type
Output = Union[
    ClassificationOutput,
    TopRankingOutput,
]

#############################################################################
# All outputs must implement a __len__ function that returns the batch size #
#############################################################################