import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

from learning.utils.model import enable_dropout
from torch.nn import SmoothL1Loss
import torch.nn.init as init
import numpy as np

class MonteCarloSqueezenet(nn.Module):

    def __init__(self, **kwargs):
        super(MonteCarloSqueezenet, self).__init__()
        print('Loading Squeeze net')
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p = kwargs.get('p', 0.1)
        self.num_outputs = kwargs.get('num_outputs', 2)
        self.num_samples = kwargs.get('num_samples', 1)
        
        self.model = models.squeezenet1_1(pretrained=True)
        # removing some high level features not needed in this context
        self.model.features = nn.Sequential(*list(self.model.features.children())[:6])
        final_conv = nn.Conv2d(32, self.num_outputs, kernel_size=1, stride=1)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=self.p),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.Dropout(p=self.p),
            final_conv,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.model.num_classes = self.num_outputs
        self.episode = 0
        self.n_epochs = 0
        self.fixed_velocity = 0.65 #TODO add to params space
        self.freeze_pretrained_modules()
        #self.criterion = SmoothL1Loss(reduction='mean')
        self.max_speed = torch.tensor(0.7).to(self._device)
        self.min_speed = torch.tensor(0.3).to(self._device)
        self.speed_threshold = (self.max_speed + self.min_speed ) / 2
        self.speed_selection_threshold = 0.6

        for m in self.model.classifier.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        #TODO add auto tune paramter for the first episode

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
        if self.num_outputs==1:
            loss = F.mse_loss(prediction,target[:,-1].reshape(-1,1), reduction='mean')
        else:
            criterion_v = nn.BCEWithLogitsLoss()
            is_slowing_down = (target[:,0]> self.speed_threshold).float()  # 1 for expert speeding up and 0 for slowing down for a corner or an incoming duckbot
            loss_omega = F.mse_loss(prediction[:,1], target[:,1], reduction='mean')
            loss_v = criterion_v(prediction[:,0], is_slowing_down)
            loss = loss_v + loss_omega
        return loss
    

    def predict(self, *args):
        images = args[0]
        output = self.model(images)
        if self.num_outputs==1:
            # in case of only predicting omega
            v_tensor = torch.tensor([self.fixed_velocity],dtype = output.dtype).to(self._device)
            v_tensor = torch.cat(output.shape[0]*[v_tensor]).unsqueeze(0)
            output = torch.cat((v_tensor, output), 1)
        # post processing v values to its max and min counterparts
        v_tensor = torch.sigmoid(output[:,0])
        v_tensor = torch.where(v_tensor>self.speed_selection_threshold, self.max_speed, self.min_speed ).unsqueeze(0)
        output = torch.cat((v_tensor, output[:,1].unsqueeze(0)), 1)
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
    #TODO test the model input and output
    pass