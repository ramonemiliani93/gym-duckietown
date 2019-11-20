import torch
from torch import nn
from torchvision.models.resnet import BasicBlock
import torch.nn.functional as F

from .uncertainty_model import UncertaintyModel
from learning.utils.model import enable_dropout


class MonteCarloResnet(UncertaintyModel):

    def __init__(self, **kwargs):
        super(MonteCarloResnet, self).__init__()
        self.p = kwargs.get('p', 0.2)
        self.num_outputs = kwargs.get('num_outputs', 2)
        self.num_samples = kwargs.get('num_samples', 10)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout2d(p=self.p),
            BasicBlock(32, 32),
            nn.Flatten(),
            nn.Linear(4160, 64),
            nn.Dropout2d(p=self.p),
            nn.Linear(64, 32),
            nn.Dropout2d(p=self.p),
            nn.Linear(32, self.num_outputs)
        )

    def forward(self, images):
        output = self.model(images)

        return output

    def loss(self, *args):
        self.train()
        images, target = args
        prediction = self.forward(images)
        loss = F.mse_loss(prediction, target, reduction='mean')
        return loss

    def predict(self, *args):
        images = args[0]
        output = self.model(images)

        return output

    def predict_with_uncertainty(self, *args):
        # Set model to evaluation except dropout layers
        self.eval()
        enable_dropout(self)

        # Sample multiple times from the ensemble of models
        prediction = []
        for i in range(self.num_samples):
            prediction.append(self.predict(*args))

        # Calculate statistics of the outputs
        prediction = torch.stack(prediction)
        mean = prediction.mean(0)
        var = prediction.var(0)

        return mean.squeeze().tolist(), var.squeeze().tolist()