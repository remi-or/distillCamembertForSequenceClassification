from typing import Optional

import torch
import numpy as np

from time import perf_counter

from logger import Logger
from dataset import Dataset

from tqdm.auto import tqdm

from torch.optim import Optimizer

from module.utils import flush_gpu


class Module(torch.nn.Module):

    """
    The base class for models. Can be used as torch.nn.Module.
    """

    device = 'cpu'

    def cuda(self) -> None:
        """
        Classic cuda method to switch the model to GPU. Also changes the module's device.
        """
        self.device = 'cuda'
        super().cuda()

    def fit(
        self,
        optimizer : Optimizer,
        logger : Logger,
        epochs : int,
        training_dataset : Dataset,
        validation_dataset : Optional[Dataset] = None,
        dataset_kwargs : dict = {},
        backpropagation_frequency : int = 1,
        pre_training_validation : bool = False,
        verbose : bool = False,
        ) -> None:
        """
        Fits the model on the given data.

        Inputs:
            - optimizer, the optimizer that will minimize the loss function,
            - logger, a Logger object to log in the stats,
            - epochs, the number of epochs of the training,
            - training_dataset, the Dataset object containing the training data
            - validation_dataset, the Dataset object containing the validation data
            - dataset_kwargs, a dictionnary containing keywords args for the Datsets objects, such as batch_size
            - backpropagation_frequency, the number of forward passes to wait between updating the weitghs
            - callbacks, a Callbacks object with callbacks such as checkpoints or early stopping
            - pre_training_validation, a boolean that determines if validation is done pre-training
            - verbose, a boolean that determines wether or not stats for each epoch are printed along the way
            
        Outputs: None
        """
        epoch_durations = []
        number_of_training_batches = training_dataset.get_number_of_batches(**dataset_kwargs)
        number_of_validation_batches = validation_dataset.get_number_of_batches(**dataset_kwargs) if validation_dataset is not None else 0
        if pre_training_validation and validation_dataset is not None:
            pre_training_logger = logger.clean_copy()
            self._val(pre_training_logger, validation_dataset, dataset_kwargs, number_of_validation_batches, 'Pre-training ')
            print(f"Pre-traning validation:")
            for name, m in pre_training_logger.metrics.items():
                if m.phase == 'Validation':
                    print(f"{name}: {m[-1]}")
        # Epoch loop
        for i_epoch in range(epochs):
            # Start timing the new epoch
            epoch_durations.append(-perf_counter())
            # Compute ETA
            eta = np.mean(epoch_durations) * (epochs - i_epoch) if epoch_durations[0] > 0 else '?'
            eta = "%.2f" % eta if isinstance(eta, float) else eta
            prefix = f"Epoch {i_epoch+1}/{epochs} - ETA: {eta} - "
            # Training
            self._train(optimizer, logger, training_dataset, dataset_kwargs, backpropagation_frequency, number_of_training_batches, prefix)
            # Validation
            if validation_dataset is None:
                logger.validation()
                logger.end_phase()
            else:
                self._val(logger, validation_dataset, dataset_kwargs, number_of_validation_batches, prefix)
            # Progress display
            if verbose:
                print(f"Epoch {i_epoch+1}/{epochs}:")
                for name, m in logger.metrics.items():
                    print(f"{name}: {m[-1]}")

    def _train(
        self,
        optimizer : Optimizer,
        logger : Logger,
        training_dataset : Dataset,
        dataset_kwargs : dict,
        backpropagation_frequency : int,
        number_of_training_batches : int,
        prefix : str,
        ) -> None:
        """
        Hidden method to train the model over a dataset.
        Inputs:
            - (optimizer), a standart torch Optimizer or a custom otpimizer that subclasses the standart 
            - (logger), a Logger object to log the training
            - (training_dataset), a dataset on which the model trains
            - (dataset_kwargs), a dictionnary containing keyword arguments for the .batches method of the (training_dataset)
            - (number_of_training_batches), an integer representing the number of batches in train_ds for the tqdm bar,
            - (prefix), a string containing the ETA for the tqdm bar,
        Outputs: None.
        """
        # Training mode
        self.train()
        logger.training()
        # Backpropagation timer initiation
        backpropagation_timer = backpropagation_frequency
        # Reset the gradients
        optimizer.zero_grad()
        # Prepare the TQDM bar
        tqdm_bar = tqdm(iterable=training_dataset.batches(**dataset_kwargs),
            total=number_of_training_batches, leave=False, desc=prefix+'Training')
        # Main loop
        for batch in tqdm_bar:
            # Forward pass
            outputs = self.forward(batch, with_loss=True) 
            # Backpropagate the loss
            outputs.loss.backward()
            # Turn the loss into a float
            outputs.loss = outputs.loss.item()
            flush_gpu(None)
            # Log the results
            logger.log(outputs)
            # Update the backpropagation timer
            backpropagation_timer -= 1
            # Backpropagate if the time is right
            if backpropagation_timer == 0:
                optimizer.step()
                optimizer.zero_grad()
                backpropagation_timer = backpropagation_frequency
        # Backpropagate the loss if it hasn't been done
        if backpropagation_timer != backpropagation_frequency:
            optimizer.step()
            optimizer.zero_grad()
        # End the training phase
        logger.end_phase()
        flush_gpu(None)

    def _val(
        self,
        logger : Logger,
        validation_dataset : Dataset,
        dataset_kwargs : dict,
        number_of_validation_batches : int,
        prefix : str,
        ) -> None:
        """
        Hidden method to validate the model over a dataset.
        Inputs:
            - (optimizer), a standart torch Optimizer
            - (logger), a Logger object to log the training
            - (validation_dataset), a dataset on which the model trains
            - (dataset_kwargs), a dictionnary containing keyword arguments for the .batches method of the (training_dataset)
            - (number_of_validation_batches), an integer representing the number of batches in train_ds for the tqdm bar,
            - (prefix), a string containing the ETA for the tqdm bar,
        Outputs: None.
        """
        self.eval()
        logger.validation()
        for batch in tqdm(
            iterable=validation_dataset.batches(**dataset_kwargs),
            total=number_of_validation_batches,
            leave=False,
            desc=prefix+'Validation',
        ):
            outputs = self.forward(batch, with_loss=True) 
            outputs.loss = outputs.loss.item()
            flush_gpu(None)
            logger.log(outputs)
        logger.end_phase()
