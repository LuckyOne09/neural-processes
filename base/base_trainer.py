import torch
from abc import abstractmethod
from numpy import inf



class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, device, model, optimizer, data_loader):

        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.data_loader = data_loader

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self,epochs):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(1, epochs + 1):
            self._train_epoch(epoch)