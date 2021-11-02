from metrics.base import Metric
from metrics.loss import Loss
from metrics.accuracy import Accuracy
from metrics.f1 import F1
from metrics.auxiliary_accuracy import AuxiliaryAccuracy
from metrics.mrr import MeanReciprocalRank

def parse(
    description : str,
) -> Metric:
    """
    Parses and returns a metric from a (description).
    """
    phase, nature = description.split(' ')
    is_training = phase in ['train', 'training', 'Train', 'Training']
    is_validation = phase in ['val', 'validation', 'Val', 'Validation']
    if nature in ['loss', 'Loss']:
        return Loss(is_training, is_validation)
    elif nature in ['acc', 'accuracy', 'Acc', 'Accuracy']:
        return Accuracy(is_training, is_validation)
    elif nature in ['aux_acc', 'auxiliary_accuracy', 'AuxAcc']:
        return AuxiliaryAccuracy(is_training, is_validation)
    elif nature in ['f1', 'F1']:
        return F1(is_training, is_validation)
    elif nature in ['mrr', 'MeanReciprocalRank', 'MRR']:
        return MeanReciprocalRank(is_training, is_validation)
    else:
        raise(ValueError(f"Unknown metric nature {nature} that was parsed from {description}."))