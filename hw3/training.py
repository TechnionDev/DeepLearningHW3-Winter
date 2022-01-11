import os
import abc
import sys
import torch
import torch.nn as nn
import tqdm.auto
from torch import Tensor
from typing import Any, Tuple, Callable, Optional, cast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from cs236781.train_results import FitResult, BatchResult, EpochResult


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(
            self, model: nn.Module, device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(
            self,
            dl_train: DataLoader,
            dl_test: DataLoader,
            num_epochs: int,
            checkpoints: str = None,
            early_stopping: int = None,
            print_every: int = 1,
            post_epoch_fn: Callable = None,
            **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """

        actual_num_epochs = 0
        epochs_without_improvement: int = 0

        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        best_acc = None

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.

            if print_every > 0 and (
                    epoch % print_every == 0 or epoch == num_epochs - 1
            ):
                verbose = True
            self._print(f"--- EPOCH {epoch + 1}/{num_epochs} ---", verbose)

            # TODO: Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            # ====== YOUR CODE: ======
            actual_num_epochs = epoch
            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            # print(type(train_result.losses))
            train_loss += train_result.losses
            train_acc += [train_result.accuracy]
            test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            test_loss += test_result.losses
            test_acc += [test_result.accuracy]
            # ========================

            # TODO:
            #  - Optional: Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            #  - Optional: Implement checkpoints. You can use the save_checkpoint
            #    method on this class to save the model to the file specified by
            #    the checkpoints argument.
            if best_acc is None or test_result.accuracy > best_acc:
                # ====== YOUR CODE: ======
                epochs_without_improvement = 0
                best_acc = test_result.accuracy
                if checkpoints is not None:
                    torch.save(self.model, checkpoints)
                # ========================
            else:
                # ====== YOUR CODE: ======
                epochs_without_improvement += 1
                # ========================
            # print(type(epochs_without_improvement))
            # print(type(early_stopping))
            if early_stopping and epochs_without_improvement >= early_stopping:
                break

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def save_checkpoint(self, checkpoint_filename: str):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        torch.save(self.model, checkpoint_filename)
        print(f"\n*** Saved checkpoint {checkpoint_filename}")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and updates weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
            dl: DataLoader,
            forward_fn: Callable[[Any], BatchResult],
            verbose=True,
            max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm.tqdm
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        if not verbose:
            pbar_file.close()

        return EpochResult(losses=losses, accuracy=accuracy)


class LayerTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer):
        # ====== YOUR CODE: ======
        super().__init__(model)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # ========================

    def train_batch(self, batch) -> BatchResult:
        X, y = batch

        # TODO: Train the Layer model on one batch of data.
        #  - Forward pass
        #  - Backward pass
        #  - Optimize params
        #  - Calculate number of correct predictions (make sure it's an int,
        #    not a tensor) as num_correct.
        # ====== YOUR CODE: ======
        # print(X.shape)
        X = X.view(X.shape[0], -1)
        # print(X.shape)
        self.optimizer.zero_grad()
        modelRes = self.model(X)
        loss = self.loss_fn(modelRes, y)
        self.model.backward(self.loss_fn.backward())
        self.optimizer.step()
        y_index = (torch.max(modelRes, 1)[1] == y)
        num_correct = torch.sum(y_index).item()
        # ========================

        return BatchResult(loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch

        # TODO: Evaluate the Layer model on one batch of data.
        # ====== YOUR CODE: ======
        # print(X.shape)
        X = X.view(X.shape[0], -1)
        # print(X.shape)
        modelRes = self.model(X)
        loss = self.loss_fn(modelRes, y)
        y_index = (torch.max(modelRes, 1)[1] == y)
        num_correct = torch.sum(y_index).item()
        # ========================

        return BatchResult(loss, num_correct)


class TorchTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        # TODO: Train the PyTorch model on one batch of data.
        #  - Forward pass
        #  - Backward pass
        #  - Optimize params
        #  - Calculate number of correct predictions
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return BatchResult(loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            # TODO: Evaluate the PyTorch model on one batch of data.
            #  - Forward pass
            #  - Calculate number of correct predictions
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================

        return BatchResult(loss, num_correct)


class RNNTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        self.prev_hidden_state = None
        super().__init__(model, device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # ========================

    def train_epoch(self, dl_train: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        self.prev_hidden_state = None
        # ========================
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        self.prev_hidden_state = None
        # ========================
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        # TODO:
        #  Train the RNN model on one batch of data.
        #  - Forward pass
        #  - Calculate total loss over sequence
        #  - Backward pass: truncated back-propagation through time
        #  - Update params
        #  - Calculate number of correct char predictions
        # ====== YOUR CODE: ======

        #
        # if self.prev_hidden_state is None:
        #     y_pred, state = self.model(x)
        # else:
        #     y_pred, state = self.model(x, self.prev_hidden_state)
        # y_pred_fixed = torch.transpose(y_pred, 1, 2).to(self.device)
        # loss = self.loss_fn(y_pred_fixed, y)
        #
        # self.optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        # self.prev_hidden_state = state.detach().clone()
        # self.optimizer.step()
        #
        # y_pred = torch.argmax(y_pred, dim=2)
        # num_correct = (y_pred.to(device=self.device) == y).sum()
        self.optimizer.zero_grad()
        if self.prev_hidden_state is None:
            output, state = self.model(x)
        else:
            output, state = self.model(x, self.prev_hidden_state)
        loss = self.loss_fn(output.permute(0, 2, 1).to(self.device), y)
        pred = torch.argmax(output, dim=2)
        loss.backward()
        self.prev_hidden_state = state.detach().clone()
        self.optimizer.step()
        num_correct = (pred.to(device=self.device) == y).sum()

        # ========================

        # Note: scaling num_correct by seq_len because each sample has seq_len
        # different predictions.
        return BatchResult(loss.item(), num_correct.item() / seq_len)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        with torch.no_grad():
            # TODO:
            #  Evaluate the RNN model on one batch of data.
            #  - Forward pass
            #  - Loss calculation
            #  - Calculate number of correct predictions
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================

        return BatchResult(loss.item(), num_correct.item() / seq_len)
