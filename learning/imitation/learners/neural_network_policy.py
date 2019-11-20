import numpy as np
from tqdm import tqdm
import cv2

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.tensorboard import SummaryWriter


from .learner import BaseLearner
from ..uncertainty_models import UncertaintyModel


class NeuralNetworkPolicy(BaseLearner):

    def __init__(self, model: UncertaintyModel, optimizer, storage_location, **kwargs):
        print(kwargs)
        self.model = model
        self.optimizer = optimizer
        self.dataset = None
        self.storage_location = storage_location
        self.writer = SummaryWriter(self.storage_location)

        # Optional parameters
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.input_shape = kwargs.get('input_shape', (60, 80))

        # Reserved
        self._train_iteration = 0

    def __del__(self):
        self.writer.close()

    def optimize(self, observations, expert_actions, episode):
        # Transform newly received data
        observations, expert_actions = self._transform(observations, expert_actions)

        # Retrieve data loader
        dataloader = self._get_dataloader(observations, expert_actions)

        # Train model
        for epoch in tqdm(range(1, self.epochs + 1)):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss = self.model.loss(*data)
                loss.backward()
                self.optimizer.step()

                # Statistics
                running_loss += loss.item()

            # Logging
            self._logging(loss=running_loss, epoch=epoch)

        # Post training routine
        self._on_optimization_end()

    def predict(self, observation, metadata):
        # Apply transformations to data
        observation, _ = self._transform([observation], [0])

        # Predict with model
        prediction = self.model.predict_with_uncertainty(observation)

        return prediction

    def save(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, self.storage_location + 'model.pt')

    def _transform(self, observations, expert_actions):
        # Resize images
        observations = [cv2.resize(observation, dsize=self.input_shape) for observation in observations]

        # Transform to tensors
        compose = Compose([
            ToTensor(),
            Normalize((0, 0, 0), (1, 1, 1))
        ])
        observations = torch.stack([compose(observation) for observation in observations])
        expert_actions = torch.stack([torch.tensor(expert_action) for expert_action in expert_actions])

        return observations, expert_actions

    def _get_dataloader(self, observations, expert_actions):
        if self.dataset is None:
            # First time data is received
            self.dataset = TensorDataset(observations, expert_actions)
        else:
            # Just include new experiences
            length = len(self.dataset)
            print(length)
            self.dataset += TensorDataset(observations[length:, ...], expert_actions[length:, ...])
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader

    def _logging(self, **kwargs):
        epoch = kwargs.get('epoch')
        loss = kwargs.get('loss')
        self.writer.add_scalar('Loss/train/{}'.format(self._train_iteration), loss, epoch)

    def _on_optimization_end(self):
        self._train_iteration += 1
