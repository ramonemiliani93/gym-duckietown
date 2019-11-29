import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

from learning.utils.model import enable_dropout
from torch.nn import SmoothL1Loss
import torch.nn.init as init
import numpy as np


class MonteCarloMnasnet(nn.Module):

    def __init__(self, **kwargs):
        super(MonteCarloMnasnet, self).__init__()
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.p = kwargs.get('p', 0.2)
        self.num_outputs = kwargs.get('num_outputs', 2)
        self.num_samples = kwargs.get('num_samples', 1)

        self.model = models.mnasnet1_3(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=self.p, inplace=True),
            nn.Linear(1280, self.num_outputs),
        )
        self.model.num_classes = self.num_outputs
        self.episode = 0
        self.n_epochs = 0
        self.freeze_pretrained_modules()
        print('Loaded Mnasnet')

        for m in self.model.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                                         nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)

    def forward(self, images):
        output = self.model(images)
        return output

    def freeze_pretrained_modules(self):
        for param in self.model.parameters():
            param.requires_grad = False
        # unfreezing the classifier part
        for module in self.model.classifier:
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_pretrained(self):
        for param in self.model.parameters():
            param.requires_grad = True

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


if __name__ == '__main__':
    # TODO test the model input and output
    pass
