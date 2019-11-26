import numpy as np
from tqdm import tqdm
import cv2

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.tensorboard import SummaryWriter


from .learner import BaseLearner
from ..uncertainty_models import UncertaintyModel
from learning.utils.dataset import MemoryMapDataset

class NeuralNetworkPolicy(BaseLearner):

    def __init__(self, model: UncertaintyModel, optimizer, storage_location, **kwargs):
        # Reserved
        self._train_iteration = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base parameters
        self.model = model.to(self._device)
        self.optimizer = optimizer
        self.storage_location = storage_location
        self.writer = SummaryWriter(self.storage_location)

        # Optional parameters
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.input_shape = kwargs.get('input_shape', (60, 80))

        # Create dataset
        self.dataset = MemoryMapDataset(100000, (3, *self.input_shape), (2,), storage_location)

        # Load previous weights
        if 'model_path' in kwargs:
            self.model.load_state_dict(torch.load(kwargs.get('model_path')))
            print('Loaded ')

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
                # Send data to device
                data = [variable.to(self._device) for variable in data]

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
        observation = torch.tensor(observation)
        # Predict with model
        prediction = self.model.predict_with_uncertainty(observation.to(self._device))

        return prediction

    def save(self):
        torch.save(self.model.state_dict(), self.storage_location + 'model.pt')

    def _transform(self, observations, expert_actions):
        # Resize images
        observations = [cv2.resize(observation, dsize=self.input_shape[::-1]) for observation in observations]

        # Transform to tensors
        compose_obs = Compose([
            ToTensor(),
            Normalize((0, 0, 0), (1, 1, 1))
        ])
        observations = [compose_obs(observation).numpy() for observation in observations]
        expert_actions = [torch.tensor(expert_action).numpy() for expert_action in expert_actions]

        return observations, expert_actions

    def _get_dataloader(self, observations, expert_actions):
        # Include new experiences
        self.dataset.extend(observations, expert_actions)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader

    def _logging(self, **kwargs):
        epoch = kwargs.get('epoch')
        loss = kwargs.get('loss')
        self.writer.add_scalar('Loss/train/{}'.format(self._train_iteration), loss, epoch)

    def _on_optimization_end(self):
        self._train_iteration += 1
