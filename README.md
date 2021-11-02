# distilCamemBERT

This code aims to distill CamemBERT, a french model based on RoBERTa, to have a smaller and faster model with hopefully equivalent performances. It does this by using the same approach as DistilBERT with BERT: the distilled model has half the number of layer that the teacher model has.

## Using only the distillation function

If you want to use a distillated model as a normal huggingface model, you can do so by using the .distill @classmethod of DistilledCamembertForSequenceClassification on a CamemBERT model, regardless of its size.  
This is recommended if you want to train your student model like a regular model, to achieve good metrics in a task without regards for imitating the teacher. However, if you want to go for the full distillation process as intented, it is not recommended.

## For full distillation process

This is the recommended way to use this repository. To train the student as was done in DistilBERT, with a loss based on classification itself and on imitating the teacher, follow the same template as in demo.ipynb .  
We recommend checking that notebook for having a good understanding of the pipeline's functionment.
