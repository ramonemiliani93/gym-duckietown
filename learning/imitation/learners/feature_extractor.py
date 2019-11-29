import numpy as np
from tqdm import tqdm
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.alexnet import alexnet
from .neural_network_policy import NeuralNetworkPolicy
from ..uncertainty_models import UncertaintyModel
from learning.utils.dataset import MemoryMapDataset
from learning.utils.running_average import RunningAverage


class FeatureExtractor(NeuralNetworkPolicy):

    def __init__(self, model: UncertaintyModel, optimizer, storage_location, **kwargs):
        super(FeatureExtractor, self).__init__(model, optimizer, storage_location, input_shape=(224, 224), **kwargs)
        # Use backbone CNN as feature extractor
        self.backbone = alexnet(pretrained=True)
        self.features = torch.nn.Sequential(*list(self.backbone.children())[:-1])

        # Random pass to find dataset size
        random_tensor = torch.rand((1, 3, *self.input_shape))
        output_shape = self.features(random_tensor).squeeze().shape

        # Overwrite dataset to match features
        if 'no_dataset' not in kwargs:
            self.dataset = MemoryMapDataset(20000, output_shape, (2,), storage_location)

    def _train_transform(self, observations, expert_actions):
        # Use same test transform
        return self._test_transform(observations, expert_actions)

    def _test_transform(self, observations, expert_actions):
        # Resize images
        observations = [cv2.resize(observation, dsize=self.input_shape[::-1]) for observation in observations]

        # Transform to tensors
        compose_obs = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Lambda(self._extract_features)
        ])
        observations = [compose_obs(observation).numpy() for observation in observations]
        expert_actions = [torch.tensor(expert_action).numpy() for expert_action in expert_actions]

        return observations, expert_actions

    def _extract_features(self, observation):
        with torch.no_grad():
            features = self.features(observation.unsqueeze(0))

        return features
