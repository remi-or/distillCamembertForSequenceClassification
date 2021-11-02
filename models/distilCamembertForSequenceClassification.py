# region Imports
from typing import Any, Optional, Tuple, Iterator

import torch
import logging
from time import perf_counter

from transformers.models.camembert.modeling_camembert import CamembertForSequenceClassification, CamembertConfig
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaEncoder

from module import Module
from dataset.tanda_dataset import TandaBatch
from outputs.classification import ClassificationOutput

from logger import Logger
from dataset import TandaDataset
from module.utils import flush_gpu
# endregion

# region Types
Tensor = torch.Tensor
Optimizer = torch.optim.Optimizer
# endregion

# region DistilledCamembert class
class DistilledCamembertForSequenceClassification(Module):

    """
    A class to distill a CamemBERT in this way DistilBERT did it.
    """

    student : CamembertForSequenceClassification

    def __init__(
        self,
        teacher : CamembertForSequenceClassification,
        verbose : bool = False,
        temperature : float = 1,
        ) -> None:
        super(DistilledCamembertForSequenceClassification, self).__init__()
        self.teacher = teacher
        self.teacher.eval()
        self.student = self.distill(teacher, verbose)
        self.temperature = temperature
        self.loss_weights = (1, 1, 1)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.parameter.Parameter]:
        return self.student.parameters(recurse=recurse)

    @classmethod
    def distill(
        cls,
        CamemBERT : CamembertForSequenceClassification,
        verbose : bool = False,
        ) -> CamembertForSequenceClassification:
        """
        Distills a given (CamemBERT) model to a smaller distilCamemBERT model as was done with BERT and DistilBERT.
        This means the two model share the same configuration except the number of layers.
        If the (verbose) flag is passed, one can see the details of the copying process
        """
        # Create student configuration
        distilled_config = CamemBERT.config.to_dict()
        distilled_config['num_hidden_layers'] //= 2
        distilled_config = CamembertConfig.from_dict(distilled_config)
        # Create uninitialized student model
        distilCamemBERT = CamembertForSequenceClassification(distilled_config)
        cls.teach_student(CamemBERT, distilCamemBERT, verbose)
        return distilCamemBERT

    @classmethod
    def teach_student(
        cls,
        teacher : Any,
        student : Any,
        verbose : bool = False,
        ) -> None:
        """
        Transfers half the (teacher) encoder layers to the (student), along with all the other weights.
        If the (verbose) flag is passed, the function displays the types of the teacher and the student.
        """
        if verbose:
            print(f"The teach_student method got called on these types:")
            print(f"teacher - {type(teacher)}")
            print(f"student - {type(student)}")
            print()
        # If the part is a supported CamemBERT or an entire RoBERTa model, unpack and iterate
        if isinstance(teacher, RobertaModel) or isinstance(teacher, CamembertForSequenceClassification):
            for old_part, new_part in zip(teacher.children(), student.children()):
                cls.teach_student(old_part, new_part, verbose)
        # Else if the part is an encoder
        elif isinstance(teacher, RobertaEncoder):
                teacher_encoding_layers = [layer for layer in next(teacher.children())]
                student_encoding_layers = [layer for layer in next(student.children())]
                for i in range(len(student_encoding_layers)):
                    student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i].state_dict())
        # Else the part is a regular part
        else:
            student.load_state_dict(teacher.state_dict())

    # region Temperature
    @property
    def temperature(
        self,
    ) -> float:
        return self._temperature if self.training else 1

    @temperature.setter
    def temperature(
        self,
        value : float,
    ) -> None:
        if value < 1:
            raise(ValueError(f"Temperature must be above 1, it cannot be {value}"))
        else:
            self._temperature = value
    # endregion

    def set_loss_weights(
        self,
        cosine : int = 1,
        classification : int = 1,
        teacher_student_cross_entropy : int = 1,
    ) -> None:
        """
        Sets the losses weights. The three weights are:
            - (cosine), the weight of the cosine loss between the two poller outputs
            - (classification), the weight of the classification loss
            - (teacher_student_cross_entropy), the weight of the cross-entropy loss between student and teacher pooler output
        """
        self.loss_weights = (cosine, classification, teacher_student_cross_entropy)

    def distillation_loss(
        self,
        teacher_classification_outputs : Tensor,
        student_classification_outputs : Tensor,
        labels : Tensor,
        ) -> Tensor:
        """
        The custom distillation loss.
        """
        student_classification_outputs = student_classification_outputs / self.temperature
        teacher_classification_outputs = (teacher_classification_outputs / self.temperature).softmax(1)
        # Classification loss
        loss = torch.nn.CrossEntropyLoss()(
                student_classification_outputs,
                labels.to(self.device),
        )* self.loss_weights[1]
        # CrossEntropy teacher-student loss
        loss = loss + torch.nn.CrossEntropyLoss()(
            student_classification_outputs,
            teacher_classification_outputs,
        )* self.loss_weights[2]
        # To probability
        student_classification_outputs = student_classification_outputs.softmax(1)
        # Cosine loss
        loss = loss + torch.nn.CosineEmbeddingLoss()(
            teacher_classification_outputs,
            student_classification_outputs,
            torch.ones(teacher_classification_outputs.size()[0], device=self.device),
        ) * self.loss_weights[0]
        # Average the loss and return it
        loss = loss / sum(self.loss_weights)
        return loss

    def fit(
        self,
        optimizer : Optimizer,
        logger : Logger,
        epochs : int,
        training_dataset : TandaDataset,
        validation_dataset : Optional[TandaDataset] = None,
        force_balance : bool = False,
        batch_size : int = 1,
        max_length : int = 512,
        temperature : Optional[float] = None,
        backpropagation_frequency : int = 1,
        pre_training_validation : bool = False,
        verbose : bool = False,
        ) -> None:
        if temperature is not None:
            self.temperature = temperature
        super().fit(
            optimizer=optimizer,
            logger=logger,
            epochs=epochs,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            dataset_kwargs={'batch_size' : batch_size, 'max_length' : max_length, 'force_balance' : force_balance},
            backpropagation_frequency=backpropagation_frequency,
            pre_training_validation=pre_training_validation,
            verbose=verbose,
        )

    def forward(
        self,
        dataset_batch : TandaBatch,
        with_loss : bool,
    ) -> ClassificationOutput:
        #monitor_tag# flush_gpu('Before forward with batch of size {dataset_batch.X[0].size()}')
        # Teacher part, no gradient 
        with torch.no_grad():
            teacher_classifier_outputs = self.classfiy(
                input_ids=dataset_batch.X[0].to(self.device),
                attention_mask=dataset_batch.X[1].to(self.device),
                teacher=True,
            )
        # Student part, with gradient (except if already in a no gradient context)
        try:
            student_classifier_outputs = self.classfiy(
                    input_ids=dataset_batch.X[0].to(self.device),
                    attention_mask=dataset_batch.X[1].to(self.device),
                    teacher=False,
                )
        except BaseException as error:
            print(f"Error in the student part of the forward pass with batch of size {dataset_batch.X[0].size()}")
            raise(error)
        # Eventual loss computation
        if with_loss:
            loss = self.distillation_loss(
                teacher_classifier_outputs,
                student_classifier_outputs,
                dataset_batch.Y,
            )
        else:
            loss = None
        # Format the outputs and return them
        teacher_prediction = teacher_classifier_outputs.argmax(1)
        #monitor_tag# flush_gpu('After forward')
        return ClassificationOutput(
            predictions=student_classifier_outputs,
            labels=teacher_prediction,
            labels_2=dataset_batch.Y.to(self.device),
            loss=loss,
        )

    def predict(
        self,
        dataset : TandaDataset,
        force_balance : bool = False,
        batch_size : int = 1,
        max_length : int = 512,
        teacher : bool = False,
    ) -> ClassificationOutput:
        dataset_kwargs = {
            'force_balance' : force_balance,
            'batch_size' : batch_size,
            'max_length' : max_length,
        } 
        self.eval()
        predictions, labels = torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        with torch.no_grad():
            for batch in dataset.batches(**dataset_kwargs):
                if teacher:
                    batch_predictions = self.teacher.forward(*batch.to(self.device).X).logits.argmax(1)
                else:
                    batch_predictions = self.student.forward(*batch.to(self.device).X).logits.argmax(1)
                predictions = torch.cat((predictions, batch_predictions))
                labels = torch.cat((labels, batch.Y))
        return ClassificationOutput(
            predictions=predictions,
            labels=labels,
        )

    def classfiy(
        self, 
        input_ids : Tensor,
        attention_mask : Tensor,
        teacher : bool,
        ) -> Tensor:
        if teacher:
            return self.teacher.classifier(
                self.teacher.roberta(input_ids, attention_mask)[0]
            )
        else:
            return self.student.classifier(
                self.student.roberta(input_ids, attention_mask)[0]
            )
# endregion