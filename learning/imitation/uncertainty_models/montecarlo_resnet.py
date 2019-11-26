import torch
from torch import nn
from torchvision.models.resnet import BasicBlock
import torch.nn.functional as F

from learning.utils.model import enable_dropout


class MonteCarloResnet(nn.Module):

    def __init__(self, **kwargs):
        super(MonteCarloResnet, self).__init__()
        self.p = kwargs.get('p', 0.05)
        self.num_outputs = kwargs.get('num_outputs', 2)
        self.num_samples = kwargs.get('num_samples', 1000)
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
        self.relu = nn.ReLU()

    def forward(self, images):
        output = self.model(images)
        # velocity = self.relu(output[:, 0])
        # omega = output[:, 1]
        # return torch.stack((velocity, omega), dim=-1)
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


class MonteCarloResnetMLP(nn.Module):

    def __init__(self, **kwargs):
        super(MonteCarloResnetMLP, self).__init__()
        self.p = kwargs.get('p', 0.2)
        self.num_inputs = kwargs.get('num_inputs', 1)
        self.num_outputs = kwargs.get('num_outputs', 1)
        self.num_samples = kwargs.get('num_samples', 10)
        self.model = nn.Sequential(
            nn.Linear(self.num_inputs, 50, bias=True),
            nn.Sigmoid(),
            nn.Dropout(self.p),
            nn.Linear(50, self.num_outputs, bias=True),
            nn.Dropout(self.p),
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
            print(i)
        # Calculate statistics of the outputs
        prediction = torch.stack(prediction)
        mean = prediction.mean(0)
        var = prediction.var(0)

        return mean.squeeze().tolist(), var.squeeze().tolist(), prediction.squeeze().detach().numpy()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader
    from learning.utils.dataset import SineDataset
    from torch.optim import Adam

    model = MonteCarloResnetMLP(p=0.05, num_samples=3000)
    trainloader = DataLoader(SineDataset(500, (0, 10)), batch_size=256)
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=0.000000095)

    for epoch in range(10000):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = model.loss(*data)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))

    print('Finished Training')
    x = np.linspace(-4, 14, 5000)
    x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    y, variance, prediction = model.predict_with_uncertainty((x_tensor))
    x = x[np.newaxis, ...].repeat(len(prediction), axis=0).ravel()
    y = prediction.ravel()
    df = pd.DataFrame({"x": x, "y": y})
    sns.set()
    plt.ylim(-15, 15)
    plt.xlim(-4, 14)
    ax = sns.lineplot(x="x", y="y", data=df, ci='sd')
    func = x * np.sin(x) + 0.3 + 0.3 * x
    sns.lineplot(x, func, color="coral")
    plt.show()
    # error = np.array(variance) ** 0.5
    # plt.plot(x, y, 'k-')
    # plt.fill_between(x, y - error, y + error)
    # plt.show()

    # x = np.random.uniform(-4, 14, (10000))
    # x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    # y, variance = model.predict_with_uncertainty((x_tensor))
    # df = pd.DataFrame({"x": x, "y": y})
    # ax = sns.lineplot(x="x", y="y", data=df)
    # plt.show()