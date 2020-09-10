import torch
import numpy as np
from random import randint
from torch.distributions.kl import kl_divergence
from base import BaseTrainer


class NeuralProcessTrainer(BaseTrainer):
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """

    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, data_loader):
        super().__init__(device, neural_process, optimizer, data_loader)
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.data_loader = data_loader

        self.steps = 0
        self.epoch_loss_history = []

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        epoch_loss = 0
        for i, data in enumerate(self.data_loader):
            self.optimizer.zero_grad()

            # Sample number of context and target points
            num_context = randint(
                *self.num_context_range)  # since num_context_range is a tuple, use * to allow variable length of parameters
            num_extra_target = randint(*self.num_extra_target_range)

            # Create context and target points and apply neural process

            x, y = data
            x_context, y_context, x_target, y_target = \
                NeuralProcessTrainer.context_target_split(x, y, num_context, num_extra_target)
            p_y_pred, q_target, q_context = \
                self.neural_process(x_context, y_context, x_target, y_target)

            loss = self._loss(p_y_pred, y_target, q_target, q_context)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            self.steps += 1

        print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(self.data_loader)))
        self.epoch_loss_history.append(epoch_loss / len(self.data_loader))

    def train(self, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            self._train_epoch(epoch)


    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl

    @classmethod
    def context_target_split(cls,x, y, num_context, num_extra_target):
        """Given inputs x and their value y, return random subsets of points for
        context and target. Note that following conventions from "Empirical
        Evaluation of Neural Process Objectives" the context points are chosen as a
        subset of the target points.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim) num_points

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)

        num_context : int
            Number of context points.

        num_extra_target : int
            Number of additional target points.
        """
        num_points = x.shape[1]
        # Sample locations of context and target points
        locations = np.random.choice(num_points,
                                     size=num_context + num_extra_target,
                                     replace=False)
        x_context = x[:, locations[:num_context], :]
        y_context = y[:, locations[:num_context], :]
        x_target = x[:, locations, :]
        y_target = y[:, locations, :]
        return x_context, y_context, x_target, y_target


class MultiTaskNeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """

    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, data_loader, task_num):
        # super().__init__(device, neural_process, optimizer, data_loader, task_num)
        self.device = device
        self.neural_process = neural_process # a dict
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.data_loader = data_loader
        self.task_num = task_num

        self.steps = 0
        self.epoch_loss_history = []

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        epoch_loss = 0
        for i, data in enumerate(self.data_loader):
            self.optimizer.zero_grad()

            # Sample number of context and target points
            num_context = randint(
                *self.num_context_range)  # since num_context_range is a tuple, use * to allow variable length of parameters
            num_extra_target = randint(*self.num_extra_target_range)

            # Create context and target points and apply neural process

            x, y = data
            x_context, y_context, x_target, y_target = \
                NeuralProcessTrainer.context_target_split(x, y, num_context, num_extra_target)

            # Use the same z or different zs for different tasks
            for i in range(self.task_num):     ### different zs
                z_sample, q_target, q_context = \
                    self.neural_process['encoder'](x_context, y_context, x_target, y_target)
                p_y_pred = self.neural_process[i](x_context, y_context, x_target, z_sample)

                if i > 0:
                    loss = loss + self._loss(p_y_pred, y_target, q_target, q_context)
                else:
                    loss = self._loss(p_y_pred, y_target, q_target, q_context)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            self.steps += 1

        print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(self.data_loader)))
        self.epoch_loss_history.append(epoch_loss / len(self.data_loader))

    def train(self, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            self._train_epoch(epoch)


    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl

    @classmethod
    def context_target_split(cls,x, y, num_context, num_extra_target):
        """Given inputs x and their value y, return random subsets of points for
        context and target. Note that following conventions from "Empirical
        Evaluation of Neural Process Objectives" the context points are chosen as a
        subset of the target points.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim) num_points

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)

        num_context : int
            Number of context points.

        num_extra_target : int
            Number of additional target points.
        """
        num_points = x.shape[1]
        # Sample locations of context and target points
        locations = np.random.choice(num_points,
                                     size=num_context + num_extra_target,
                                     replace=False)
        x_context = x[:, locations[:num_context], :]
        y_context = y[:, locations[:num_context], :]
        x_target = x[:, locations, :]
        y_target = y[:, locations, :]
        return x_context, y_context, x_target, y_target