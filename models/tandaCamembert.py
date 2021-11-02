# region Imports
import torch
from tqdm.auto import tqdm

from transformers.models.camembert.modeling_camembert import CamembertForSequenceClassification

from module import Module
from dataset.tanda_dataset import TandaDataset, TandaBatch
from outputs.top_ranking import TopRankingOutput
# endregion

# region Types
Tensor = torch.Tensor
Optimizer = torch.optim.Optimizer
# endregion

# region DistilledCamembert class
class TandaCamembert(Module):

    """
    A class to distill a CamemBERT in this way DistilBERT did it.
    """

    def __init__(
        self,
        camembert_model : CamembertForSequenceClassification,
        ) -> None:
        super(TandaCamembert, self).__init__()
        self.camembert = camembert_model
        self.loss = torch.nn.NLLLoss()

    def forward(
        self,
        dataset_batch : TandaBatch,
        with_loss : bool,
    ) -> TopRankingOutput:
        """
        Performs a forward pass on the (dataset_batch) maybe (with_loss).
        Each batch corresponds to one question along with its passages, and is of size nb_passages * x where x is the padded length.
        """
        dataset_batch = dataset_batch.to(self.device)
        predictions = self.camembert.forward(*dataset_batch.X).logits[:, 1].softmax(0)[None, :] # Of shape 1 * nb_passages from logits of shape nb_passages * 2
        loss = self.loss(predictions, dataset_batch.Y) if with_loss else None
        return TopRankingOutput(
            predictions=predictions, # Shape: 1 * nb_passages
            labels=dataset_batch.Y, # Shape: 1
            loss=loss, # Shape: 1
        )

    def predict(
        self,
        dataset : TandaDataset,
        max_length : int = 256,
    ) -> TopRankingOutput:
        self.eval()
        predictions = torch.zeros(
            size=(
                len(dataset),
                max(len(paragraph['passages']) for paragraph in dataset.paragraphs.values()), 
            ),
        )
        labels = torch.zeros(size=(len(dataset), ), )
        with torch.no_grad():
            i = 0
            for batch in tqdm(dataset.top_ranking_iterator(max_length), desc='Predicting...', total=len(dataset), leave=False):
                outputs = self.forward(batch, False)
                predictions[i, :outputs.X.size()[1]] = outputs.X[0, :]
                labels[i] = outputs.Y
                i += 1
        return TopRankingOutput(
            predictions=torch.tensor(predictions),
            labels=torch.tensor(labels),
        )


# endregion