import torch
from torch import nn

from learning.utils.model import enable_dropout


class IndividualMLP(nn.Module):

    def __init__(self, **kwargs):
        super(IndividualMLP, self).__init__()
        # Extract MCDropout parameters
        self.p = kwargs.get('p', 0.8)
        self.num_samples = kwargs.get('num_samples', 1)

        # MSE loss
        self.loss_fn = nn.MSELoss()

        # Create independent MLP for angular speed and linear speed
        self.angular = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 64),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        angular = self.angular(features)

        return angular

    def loss(self, *args):
        self.angular.train()
        features, target = args
        prediction = self.forward(features)
        total_loss = self.loss_fn(prediction, target[:, 1])

        return total_loss

    def predict(self, *args):
        features = args[0]
        angular = self.angular(features)

        return angular

    def predict_with_uncertainty(self, *args):
        # Set model to evaluation except dropout layers
        self.eval()
        # enable_dropout(self)

        # Sample multiple times from the ensemble of models
        prediction = []
        for i in range(self.num_samples):
            prediction.append(self.predict(*args))

        # Calculate statistics of the outputs
        prediction = torch.stack(prediction)
        mean = prediction.mean(0).squeeze().tolist()
        var = prediction.var(0).squeeze().tolist()

        return [0.4, mean], [0, var]