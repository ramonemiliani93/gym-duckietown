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
        self.model.features = nn.Sequential(*list(self.model.features.children())[:9])
        # we have 256 output channels
        self.velocity_branch = nn.Sequential(
            nn.Dropout(p=self.p),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1)
        )

        self.omega_branch = nn.Sequential(
            nn.Dropout(p=self.p),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1)
        )
        final_conv = nn.Conv2d(32, self.num_outputs, kernel_size=1, stride=1)
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.Dropout(p=self.p),
            final_conv,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.model.num_classes = self.num_outputs
        self.episode = 0
        self.n_epochs = 0 
        self.freeze_pretrained_modules()
        self.init_weights(final_conv)
        self.max_speed = 0.8
        self.min_speed = 0.35

    def init_weights(self, final_conv):
        for subbranch in [self.velocity_branch, self.omega_branch, self.classifier]:
            for m in subbranch.modules():
                if isinstance(m, nn.Conv2d):
                    if m is final_conv:
                        init.normal_(m.weight, mean=0.0, std=0.01)
                    else:
                        init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

    def forward(self, images):
        features_extracted = self.model.features(images) 
        v_feats = self.velocity_branch(features_extracted) 
        omega_feats = self.omega_branch(features_extracted)
        # late_fusion
        feats = torch.cat((v_feats, omega_feats),1)
        output = self.classifier(feats)
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
            loss = F.mse_loss(prediction,target, reduction='mean')
        return loss
    

    def predict(self, *args):
        images = args[0]
        output = self.forward(images)
        if self.num_outputs==1:
            # in case of only predicting omega
            v_tensor = torch.tensor([self.fixed_velocity],dtype = output.dtype).to(self._device)
            v_tensor = torch.cat(output.shape[0]*[v_tensor]).unsqueeze(0)
            output = torch.cat((v_tensor, output), 1)
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